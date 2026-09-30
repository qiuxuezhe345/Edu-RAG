# EduRAG 评测脚本

这些脚本真实计算指标，不会把目标百分比写入结果。建议先跑小样本检查环境，再执行 300 条完整评测。

## 1. 生成 300 条测试集

使用 DashScope 生成自然改写，默认包含 270 条可回答问题和 30 条拒答问题：

```powershell
python -m evaluation.build_dataset
```

仅检查流程、不调用 LLM：

```powershell
python -m evaluation.build_dataset --no-llm
```

`--no-llm` 生成的是模板改写，只适合冒烟测试，可能高估检索指标。正式报告请使用默认的 LLM 改写数据，并人工抽检至少 10% 样本。

输出文件为 `evaluation/data/benchmark_300.jsonl`。每条记录包含问题、参考答案、参考上下文、可回答标记及可选父文档 ID。

## 2. 检索消融评测

先跑 10 条冒烟测试：

```powershell
python -m evaluation.run_retrieval_eval --limit 10
```

执行完整评测：

```powershell
python -m evaluation.run_retrieval_eval
```

CPU 上执行 BGE-Reranker-Large 会比较慢，当前机器实测每条约 12 至 15 秒；完整评测建议使用 CUDA。

脚本依次评测：

- `dense`：仅使用 BGE-M3 稠密向量，不重排。
- `hybrid`：BGE-M3 稠密与稀疏向量，由 Milvus `WeightedRanker` 融合，不重排。
- `hybrid_rerank`：混合召回后使用 BGE-Reranker-Large 重排。

结果保存在 `evaluation/results/retrieval/summary.json` 和 `details.csv`，包含 `Hit@2`、`Recall@5`、`MRR@5`、平均延迟及目标差距。

可以调节候选池和融合权重：

```powershell
python -m evaluation.run_retrieval_eval --candidate-k 30 --sparse-weight 0.7 --dense-weight 1.0
```

## 3. 生成质量与拒答评测

该脚本会实际调用问答模型和裁判模型，完整运行 300 条会产生 API 费用。先运行小样本：

```powershell
python -m evaluation.run_generation_eval --limit 10
```

单独检查拒答样本：

```powershell
python -m evaluation.run_generation_eval --subset unanswerable --limit 10 --strategy 直接检索
```

确认结果后执行完整评测：

```powershell
python -m evaluation.run_generation_eval
```

结果保存在 `evaluation/results/generation/summary.json` 和 `details.csv`，包含：

- `faithfulness`：回答中被检索上下文支持的事实比例。
- `hallucination_rate`：可回答样本中出现至少一项无依据事实的比例。
- `refusal_accuracy`：不可回答样本被明确拒答的比例。
- `answer_correctness`：回答与参考答案的语义一致程度。

## 指标口径

- `Hit@2`：前 2 个结果中至少有一个相关父文档的查询比例。
- `Recall@5`：前 5 个结果覆盖的相关父文档或参考上下文比例。
- `MRR@5`：第一个相关结果排名倒数的均值，超过第 5 名记为 0。
- 检索指标只统计 `answerable=true` 的样本。
- 拒答准确率只统计 `answerable=false` 的样本。

目标值只是验收线。数据集、Milvus 集合内容、模型版本或融合权重发生变化时，实际结果也会变化，应以生成的报告为准。

## 一键执行

模板数据加 3 条冒烟测试：

```powershell
.\evaluation\run_full_benchmark.ps1 -UseTemplateDataset -Limit 3
```

生成正式数据并执行 300 条完整评测：

```powershell
.\evaluation\run_full_benchmark.ps1
```

复用已经生成的数据集：

```powershell
.\evaluation\run_full_benchmark.ps1 -SkipBuild
```
