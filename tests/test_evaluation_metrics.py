import unittest

from evaluation.metrics import normalize_text, retrieval_metrics, summarize_retrieval


def document(text, parent_id=""):
    return {"page_content": text, "metadata": {"parent_id": parent_id}}


class EvaluationMetricsTest(unittest.TestCase):
    def test_normalize_text_handles_width_case_and_punctuation(self):
        self.assertEqual(normalize_text("ＢＧＥ-M3，模型!"), "bgem3模型")

    def test_retrieval_metrics_with_context_references(self):
        record = {"contexts": ["Python函数包含参数和返回值", "支持异常捕获"]}
        documents = [
            document("无关文档"),
            document("课程说明：Python函数包含参数和返回值。"),
            document("本课程支持异常捕获和异常传递。"),
        ]
        metrics = retrieval_metrics(documents, record)
        self.assertEqual(metrics["hit@2"], 1.0)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["mrr@5"], 0.5)

    def test_parent_id_has_priority(self):
        record = {"relevant_parent_ids": ["p-2"], "contexts": ["不应参与分母"]}
        metrics = retrieval_metrics(
            [document("任意内容", "p-1"), document("另一个内容", "p-2")], record
        )
        self.assertEqual(metrics["hit@2"], 1.0)
        self.assertEqual(metrics["recall@5"], 1.0)

    def test_ellipsis_context_matches_by_fragments(self):
        record = {
            "contexts": [
                "课程优势：优势一：热门岗位覆盖...优势二：与大厂深入合作...优势三：垂直行业赋能"
            ]
        }
        docs = [
            document(
                "课程优势包括优势一热门岗位覆盖，优势二与大厂深入合作，"
                "优势三提供垂直行业赋能。"
            )
        ]
        self.assertEqual(retrieval_metrics(docs, record)["hit@2"], 1.0)

    def test_summary(self):
        rows = [
            {"hit@2": 1.0, "recall@5": 1.0, "mrr@5": 1.0, "latency_ms": 10},
            {"hit@2": 0.0, "recall@5": 0.0, "mrr@5": 0.0, "latency_ms": 30},
        ]
        summary = summarize_retrieval(rows)
        self.assertEqual(summary["hit@2"], 0.5)
        self.assertEqual(summary["mean_latency_ms"], 20.0)


if __name__ == "__main__":
    unittest.main()
