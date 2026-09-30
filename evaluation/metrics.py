import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def load_records(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Evaluation data must be a JSON array or JSONL records")
    return data


def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def record_contexts(record):
    contexts = record.get("contexts", record.get("context", []))
    if isinstance(contexts, str):
        contexts = [contexts]
    return [str(item) for item in contexts if str(item).strip()]


def document_data(document):
    if isinstance(document, dict):
        text = document.get("page_content", document.get("text", ""))
        metadata = document.get("metadata", {})
    else:
        text = getattr(document, "page_content", "")
        metadata = getattr(document, "metadata", {}) or {}
    return str(text or ""), metadata


def reference_matches(normalized_doc, reference):
    normalized_reference = normalize_text(reference)
    if not normalized_reference:
        return False
    if normalized_reference in normalized_doc or normalized_doc in normalized_reference:
        return True

    fragments = [
        normalize_text(fragment)
        for fragment in re.split(r"(?:\.{2,}|…+|[\n，,、；;。])", str(reference))
    ]
    fragments = [fragment for fragment in fragments if len(fragment) >= 5]
    if not fragments:
        return False
    matched = 0
    for fragment in fragments:
        if fragment in normalized_doc:
            matched += 1
            continue
        longest = SequenceMatcher(None, fragment, normalized_doc).find_longest_match().size
        if longest / len(fragment) >= 0.65:
            matched += 1
    required = max(1, math.ceil(len(fragments) * 0.5))
    return matched >= required and (len(fragments) > 1 or len(fragments[0]) >= 10)


def matched_reference_indexes(document, record):
    text, metadata = document_data(document)
    parent_id = str(metadata.get("parent_id", ""))
    relevant_ids = {str(item) for item in record.get("relevant_parent_ids", [])}
    if relevant_ids and parent_id in relevant_ids:
        return {("id", parent_id)}

    normalized_doc = normalize_text(text)
    matches = set()
    contexts = record_contexts(record)
    for index, context in enumerate(contexts):
        if reference_matches(normalized_doc, context):
            matches.add(("context", index))
    if not matches and not relevant_ids:
        ground_truth = record.get("ground_truth", "")
        if ground_truth and reference_matches(normalized_doc, ground_truth):
            matches.add(("ground_truth", 0))
    return matches


def retrieval_metrics(documents, record, hit_k=2, recall_k=5):
    references = record.get("relevant_parent_ids", []) or record_contexts(record)
    reference_count = max(1, len(references))
    matches_by_rank = [matched_reference_indexes(doc, record) for doc in documents]
    first_rank = next((index + 1 for index, matches in enumerate(matches_by_rank) if matches), None)
    covered = set()
    for matches in matches_by_rank[:recall_k]:
        covered.update(matches)
    return {
        f"hit@{hit_k}": float(first_rank is not None and first_rank <= hit_k),
        f"recall@{recall_k}": min(1.0, len(covered) / reference_count),
        f"mrr@{recall_k}": 0.0 if first_rank is None or first_rank > recall_k else 1.0 / first_rank,
        "first_relevant_rank": first_rank,
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_retrieval(rows, hit_k=2, recall_k=5):
    return {
        "queries": len(rows),
        f"hit@{hit_k}": mean(row[f"hit@{hit_k}"] for row in rows),
        f"recall@{recall_k}": mean(row[f"recall@{recall_k}"] for row in rows),
        f"mrr@{recall_k}": mean(row[f"mrr@{recall_k}"] for row in rows),
        "mean_latency_ms": mean(row["latency_ms"] for row in rows),
    }
