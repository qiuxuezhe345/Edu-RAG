import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("LANGSMITH_TRACING", "false")

from base.config import config
from evaluation.metrics import load_records, mean
from main import build_llm
from rag_qa.core.rag_system import RAGSystem
from rag_qa.core.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "data" / "benchmark_300.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results" / "generation"


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


TARGETS = {
    "faithfulness": 0.907,
    "max_hallucination_rate": 0.08,
    "refusal_accuracy": 0.867,
}


def make_judge_client():
    from openai import OpenAI

    api_key = config.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for generation evaluation")
    return OpenAI(api_key=api_key, base_url=config.DASHSCOPE_BASE_URL)


def extract_json(text):
    text = str(text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    return json.loads(match.group(0))


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "是"}
    return bool(value)


def judge_answer(client, model, record, answer, contexts):
    prompt = f"""你是严格的 RAG 评测裁判。根据问题、检索上下文和系统回答进行判定。

定义：
1. faithfulness：回答中可核验事实被上下文支持的比例，范围 0 到 1。
2. hallucinated：回答是否包含任意无法由上下文支持的事实性断言。
3. refused：回答是否明确表示信息不足或无法回答。
4. answer_correctness：与参考答案语义一致程度，范围 0 到 1；不可回答样本不计算内容正确性。

样本是否可回答：{bool(record.get('answerable', True))}
问题：{record['question']}
参考答案：{record.get('ground_truth', '')}
检索上下文：{json.dumps(contexts, ensure_ascii=False)}
系统回答：{answer}

只输出 JSON：
{{"faithfulness": 0.0, "hallucinated": false, "refused": false, "answer_correctness": 0.0, "reason": "简短理由"}}
"""
    kwargs = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "你是可复现的 RAG 评测裁判，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        response = client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}
        )
    except Exception:
        response = client.chat.completions.create(**kwargs)
    result = extract_json(response.choices[0].message.content)
    return {
        "faithfulness": min(1.0, max(0.0, float(result.get("faithfulness", 0.0)))),
        "hallucinated": as_bool(result.get("hallucinated", False)),
        "refused": as_bool(result.get("refused", False)),
        "answer_correctness": min(
            1.0, max(0.0, float(result.get("answer_correctness", 0.0)))
        ),
        "reason": str(result.get("reason", "")),
    }


def run_answer(rag_system, llm, record, strategy):
    question = record["question"]
    selected_strategy = strategy
    if selected_strategy == "auto":
        selected_strategy = rag_system.strategy_selector.select_strategy(question)
    documents = rag_system.retrieve_and_merge(
        question,
        source_filter=record.get("source"),
        strategy=selected_strategy,
    )
    contexts = [doc.page_content for doc in documents]
    prompt = rag_system.rag_prompt.format(
        context="\n\n".join(contexts),
        history="",
        question=question,
        phone=config.CUSTOMER_SERVICE_PHONE,
    )
    return llm(prompt), contexts, selected_strategy


def write_csv(path, rows):
    fieldnames = [
        "id", "question", "answerable", "strategy", "answer", "faithfulness",
        "hallucinated", "refused", "answer_correctness", "latency_ms", "reason",
        "contexts",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["contexts"] = json.dumps(item["contexts"], ensure_ascii=False)
            writer.writerow(item)


def summarize(rows):
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    faithfulness = (
        mean(row["faithfulness"] for row in answerable) if answerable else None
    )
    hallucination_rate = (
        mean(float(row["hallucinated"]) for row in answerable) if answerable else None
    )
    refusal_accuracy = (
        mean(float(row["refused"]) for row in unanswerable) if unanswerable else None
    )
    return {
        "samples": len(rows),
        "answerable_samples": len(answerable),
        "unanswerable_samples": len(unanswerable),
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate,
        "refusal_accuracy": refusal_accuracy,
        "answer_correctness": (
            mean(row["answer_correctness"] for row in answerable) if answerable else None
        ),
        "mean_latency_ms": mean(row["latency_ms"] for row in rows),
        "target_check": {
            "faithfulness": (
                None if faithfulness is None else faithfulness >= TARGETS["faithfulness"]
            ),
            "hallucination_rate": (
                None if hallucination_rate is None
                else hallucination_rate <= TARGETS["max_hallucination_rate"]
            ),
            "refusal_accuracy": (
                None if refusal_accuracy is None
                else refusal_accuracy >= TARGETS["refusal_accuracy"]
            ),
        },
        "targets": TARGETS,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate faithfulness, hallucination and refusal")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge-model", default=config.LLM_MODEL)
    parser.add_argument(
        "--subset", choices=["all", "answerable", "unanswerable"], default="all"
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        choices=["auto", "直接检索", "假设问题检索", "子查询检索", "回溯问题检索"],
    )
    args = parser.parse_args()

    records = load_records(project_path(args.dataset))
    if args.subset != "all":
        expected = args.subset == "answerable"
        records = [
            record for record in records
            if bool(record.get("answerable", True)) is expected
        ]
    if args.limit:
        records = records[:args.limit]
    if not records:
        parser.error("Dataset is empty")

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = build_llm()
    rag_system = RAGSystem(vector_store=VectorStore(), llm=llm)
    judge_client = make_judge_client()
    rows = []

    for index, record in enumerate(records, 1):
        started = time.perf_counter()
        answer, contexts, selected_strategy = run_answer(
            rag_system, llm, record, args.strategy
        )
        verdict = judge_answer(
            judge_client, args.judge_model, record, answer, contexts
        )
        latency_ms = (time.perf_counter() - started) * 1000
        row = {
            "id": record.get("id", str(index)),
            "question": record["question"],
            "answerable": bool(record.get("answerable", True)),
            "strategy": selected_strategy,
            "answer": answer,
            "latency_ms": round(latency_ms, 3),
            "contexts": contexts,
            **verdict,
        }
        rows.append(row)
        print(
            f"[{index}/{len(records)}] faithfulness={row['faithfulness']:.3f} "
            f"hallucinated={row['hallucinated']} refused={row['refused']} "
            f"{latency_ms / 1000:.1f}s"
        )

    summary = summarize(rows)
    write_csv(output_dir / "details.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
