import argparse
import hashlib
import json
import os
from pathlib import Path

from base.config import config
from evaluation.metrics import load_records, record_contexts


PARAPHRASE_TEMPLATES = [
    "请问{question}",
    "我想了解一下，{question}",
    "能否说明一下：{question}",
    "关于这个课程，{question}",
    "帮我解释一下，{question}",
    "如果换一种说法，{question}",
    "请简要回答：{question}",
    "从学习者角度看，{question}",
]


def stable_id(prefix, question):
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def parse_json_array(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Model response is not a JSON array")
    return [str(item).strip() for item in data if str(item).strip()]


def make_client():
    from openai import OpenAI

    api_key = config.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required unless --no-llm is used")
    return OpenAI(api_key=api_key, base_url=config.DASHSCOPE_BASE_URL)


def llm_list(client, prompt, count, model):
    kwargs = {
        "model": model,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": "只输出合法 JSON，键名必须是 items。"},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        response = client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}
        )
    except Exception:
        response = client.chat.completions.create(**kwargs)
    payload = json.loads(response.choices[0].message.content)
    items = payload.get("items", [])
    return [str(item).strip() for item in items[:count] if str(item).strip()]


def paraphrases(seed, count, client, model):
    if client is None:
        return [
            template.format(question=seed["question"].rstrip("？?")) + "？"
            for template in PARAPHRASE_TEMPLATES[:count]
        ]
    prompt = (
        f"将下面的教育问答问题改写成 {count} 个语义完全一致、表达自然且彼此不同的中文问题。"
        "不得增加原上下文没有的信息。返回 {\"items\": [...]}。\n"
        f"原问题：{seed['question']}\n"
        f"参考答案：{seed.get('ground_truth', '')}\n"
        f"参考上下文：{record_contexts(seed)}"
    )
    return llm_list(client, prompt, count, model)


def negative_questions(count, client, model):
    if client is None:
        topics = ["量子计算", "医学影像诊断", "航空发动机维修", "证券投资", "建筑结构设计"]
        return [f"课程第{i + 1}章是否详细讲解{topics[i % len(topics)]}？" for i in range(count)]
    prompt = (
        f"生成 {count} 个教育课程知识库无法回答的中文问题，用于测试拒答。"
        "问题要自然，但内容应明显超出 Python、Java、测试、运维、大数据和 AI 课程资料范围。"
        "返回 {\"items\": [...]}。"
    )
    return llm_list(client, prompt, count, model)


def build_records(seeds, size, negative_ratio, client, model, source):
    negative_count = round(size * negative_ratio)
    positive_count = size - negative_count
    positives = []
    seen = set()

    for seed_index, seed in enumerate(seeds):
        if len(positives) >= positive_count:
            break
        question = str(seed["question"]).strip()
        contexts = record_contexts(seed)
        record = {
            "id": stable_id("positive", question),
            "question": question,
            "ground_truth": seed.get("ground_truth", seed.get("answer", "")),
            "contexts": contexts,
            "relevant_parent_ids": seed.get("relevant_parent_ids", []),
            "source": seed.get("source", source),
            "answerable": True,
            "kind": "seed",
            "generation_method": "seed",
        }
        positives.append(record)
        seen.add(question)

    seed_cursor = 0
    while len(positives) < positive_count:
        seed = seeds[seed_cursor % len(seeds)]
        remaining = positive_count - len(positives)
        batch_size = min(len(PARAPHRASE_TEMPLATES), remaining)
        generated = paraphrases(seed, batch_size, client, model)
        for question in generated:
            if question in seen:
                continue
            positives.append({
                "id": stable_id("positive", question),
                "question": question,
                "ground_truth": seed.get("ground_truth", seed.get("answer", "")),
                "contexts": record_contexts(seed),
                "relevant_parent_ids": seed.get("relevant_parent_ids", []),
                "source": seed.get("source", source),
                "answerable": True,
                "kind": "paraphrase",
                "generation_method": "template" if client is None else "llm",
                "seed_question": seed["question"],
            })
            seen.add(question)
            if len(positives) >= positive_count:
                break
        seed_cursor += 1
        if seed_cursor > positive_count * 2:
            raise RuntimeError("Could not generate enough unique positive questions")

    negatives = []
    for question in negative_questions(negative_count, client, model):
        negatives.append({
            "id": stable_id("negative", question),
            "question": question,
            "ground_truth": "信息不足，无法回答。",
            "contexts": [],
            "relevant_parent_ids": [],
            "source": None,
            "answerable": False,
            "kind": "unanswerable",
            "generation_method": "template" if client is None else "llm",
        })
    if len(negatives) != negative_count:
        raise RuntimeError(f"Expected {negative_count} negatives, got {len(negatives)}")
    return (positives + negatives)[:size]


def main():
    parser = argparse.ArgumentParser(description="Build a reproducible 300-item EduRAG benchmark")
    parser.add_argument("--seed", default="rag_qa/data/rag_evaluate_data.json")
    parser.add_argument("--output", default="evaluation/data/benchmark_300.jsonl")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--negative-ratio", type=float, default=0.1)
    parser.add_argument("--source", default="ai")
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    if args.size < 1 or not 0 <= args.negative_ratio < 1:
        parser.error("--size must be positive and --negative-ratio must be in [0, 1)")
    seeds = load_records(args.seed)
    if not seeds:
        parser.error("Seed dataset is empty")
    client = None if args.no_llm else make_client()
    records = build_records(
        seeds, args.size, args.negative_ratio, client, args.model, args.source
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    answerable = sum(bool(record["answerable"]) for record in records)
    print(f"Wrote {len(records)} records to {output}")
    print(f"Answerable: {answerable}; unanswerable: {len(records) - answerable}")


if __name__ == "__main__":
    main()
