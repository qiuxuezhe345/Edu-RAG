import argparse
import csv
import json
import time
from pathlib import Path

from evaluation.metrics import load_records, retrieval_metrics, summarize_retrieval
from rag_qa.core.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "data" / "benchmark_300.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results" / "retrieval"


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


VARIANTS = {
    "dense": {"mode": "dense", "rerank": False},
    "hybrid": {"mode": "hybrid", "rerank": False},
    "hybrid_rerank": {"mode": "hybrid", "rerank": True},
}

TARGETS = {"hit@2": 0.911, "recall@5": 0.959, "mrr@5": 0.84}


def serialize_docs(documents):
    return [
        {
            "rank": index + 1,
            "parent_id": doc.metadata.get("parent_id"),
            "source": doc.metadata.get("source"),
            "retrieval_score": doc.metadata.get("retrieval_score"),
            "rerank_score": doc.metadata.get("rerank_score"),
            "text": doc.page_content[:300],
        }
        for index, doc in enumerate(documents)
    ]


def evaluate_variant(store, records, name, options, args):
    rows = []
    for index, record in enumerate(records, 1):
        started = time.perf_counter()
        documents = store.search(
            query=record["question"],
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            mode=options["mode"],
            rerank=options["rerank"],
            sparse_weight=args.sparse_weight,
            dense_weight=args.dense_weight,
            source_filter=record.get("source") if args.use_source_filter else None,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        metrics = retrieval_metrics(documents, record, hit_k=2, recall_k=5)
        row = {
            "variant": name,
            "id": record.get("id", str(index)),
            "question": record["question"],
            "latency_ms": round(latency_ms, 3),
            **metrics,
            "documents": serialize_docs(documents),
        }
        rows.append(row)
        print(
            f"[{name}] {index}/{len(records)} "
            f"hit@2={row['hit@2']:.0f} recall@5={row['recall@5']:.3f} "
            f"mrr@5={row['mrr@5']:.3f} {latency_ms:.0f}ms"
        )
    return rows


def write_csv(path, rows):
    fieldnames = [
        "variant", "id", "question", "latency_ms", "hit@2", "recall@5",
        "mrr@5", "first_relevant_rank", "documents",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["documents"] = json.dumps(item["documents"], ensure_ascii=False)
            writer.writerow(item)


def main():
    parser = argparse.ArgumentParser(description="Run dense/hybrid/rerank retrieval ablations")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--sparse-weight", type=float, default=0.7)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--use-source-filter", action="store_true")
    args = parser.parse_args()

    dataset_path = project_path(args.dataset)
    records = [
        record for record in load_records(dataset_path)
        if record.get("answerable", True)
    ]
    if args.limit:
        records = records[:args.limit]
    if not records:
        parser.error("No answerable records found")

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = VectorStore()
    all_rows = []
    summaries = {}
    for variant in args.variants:
        rows = evaluate_variant(store, records, variant, VARIANTS[variant], args)
        all_rows.extend(rows)
        summaries[variant] = summarize_retrieval(rows)

    optimized = summaries.get("hybrid_rerank", {})
    baseline = summaries.get("dense", {})
    target_check = {
        metric: {
            "target": target,
            "actual": optimized.get(metric),
            "met": optimized.get(metric, 0.0) >= target,
        }
        for metric, target in TARGETS.items()
    }
    report = {
        "dataset": str(dataset_path),
        "answerable_queries": len(records),
        "settings": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "sparse_weight": args.sparse_weight,
            "dense_weight": args.dense_weight,
            "use_source_filter": args.use_source_filter,
        },
        "variants": summaries,
        "optimized_vs_dense_percentage_points": {
            metric: round((optimized.get(metric, 0.0) - baseline.get(metric, 0.0)) * 100, 3)
            for metric in TARGETS
            if metric in optimized and metric in baseline
        },
        "optimized_target_check": target_check,
    }
    write_csv(output_dir / "details.csv", all_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
