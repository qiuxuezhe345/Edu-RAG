# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

EduRag 是一个面向 IT 培训机构的智能问答系统（教育场景 RAG），由两套**相互独立、尚未在代码中串接**的问答子系统组成：

- **`mysql_qa/`** — 结构化 FAQ 问答：MySQL 存问答对（表 `jpkb`）+ Redis 缓存 + BM25 相似度检索，命中阈值 0.85 返回答案，否则提示回退 RAG 系统。
- **`rag_qa/`** — 非结构化文档 RAG 问答：Milvus 向量库 + BGE-M3 混合检索 + BGE-Reranker 重排 + DashScope(Qwen) 大模型生成答案。

## 运行环境

- Python 3.10，conda 环境名 `edu_rag`，解释器路径 `/opt/anaconda3/envs/edu_rag/bin/python`（已在 `.claude/settings.local.json` 中授权）。
- 没有 `requirements.txt`、`README`、git 仓库、pytest 等测试框架。`tests/` 只是 logger 用法示例。



**文档入库流程**（`document_processor.process_documents` → `VectorStore.add_documents`，无独立脚本，在 `vector_store.py` 的 `__main__` 中组合）：先切分，再 `add_documents` upsert 进 Milvus。

## 架构

### 配置与日志（`base/`）

- `base/config.py`：`Config` 类读取根目录 `config.ini`（`configparser.ExtendedInterpolation`，支持 `${section:key}` 插值）。Milvus 连接、DASHSCOPE_API_KEY、分类器路径可用环境变量覆盖（见 `base/config.py` 中的 `os.getenv`）。
- `base/logger.py`：全局单例 logger `EduRAG`，文件 + 控制台双输出。各模块统一通过 `from base import config, logger` 使用（`base/__init__.py` 导出）。
- `config.ini` 关键段：`[mysql]`、`[redis]`、`[milvus]`（数据库 `itcast`，集合 `edurag_final`）、`[llm]`（model 默认 qwen-plus）、`[retrieval]`（parent/child chunk 大小、RETRIEVAL_K、CANDIDATE_M）、`[app]`（valid_sources 学科白名单）。

### RAG 核心流程（`rag_qa/core/rag_system.py` 的 `generate_answer`）

1. **查询分类**：`QueryClassifier`（BERT 二分类，本地模型 `bert_query_classifier`）判断「通用知识」→ 直接 LLM 回答（不带上下文）；「专业咨询」→ 走 RAG。
2. **策略选择**：`StrategySelector` 用 LLM 从四种策略中选一种 —— 直接检索 / 假设问题检索(HyDE) / 子查询检索 / 回溯问题检索，分别对应 `rag_system.py` 中的 `_retrieve_with_*` 方法。
3. **混合检索 + 重排**：`VectorStore.hybrid_search_with_rerank` 用 BGE-M3 生成 dense+sparse 向量，Milvus `hybrid_search` 配合 `WeightedRanker(0.7, 1.0)`（稀疏/稠密权重），对命中的**子块**按 `parent_content` 去重聚合回**父文档**，再用 BGE-Reranker 重排。
4. **生成答案**：取前 `CANDIDATE_M`（默认 2）个父文档拼成上下文，`RAGPrompts.rag_prompt` 模板调用 LLM。Prompt 模板全部集中在 `core/prompts.py`。

### 本地模型（`rag_qa/models/`，全部离线加载，不联网下载）

- `bert-base-chinese` — 分类器预训练底模
- `bert_query_classifier` — 微调后的查询分类器（不存在时自动基于底模初始化新模型）
- `bge-m3` — 嵌入模型（dense 1024 维 + sparse）
- `bge-reranker-large` — 重排序模型

### MySQL 问答系统（`mysql_qa/`）

- `db/mysql_client.py`：PyMySQL 连接；`create_table`/`insert_data` 从 CSV（`mysql_qa/data/JP学科知识问答.csv`，列名：学科名称/问题/答案）建表导入。
- `retrieval/bm25_search.py`：`rank_bm25` + softmax 归一化，分数 ≥ 0.85 视为命中；答案优先查 Redis 缓存，未命中再查 MySQL。
- `cache/redis_client.py`：值以 JSON 序列化存储（`ensure_ascii=False` 保留中文），答案缓存键 `answer:{query}`。

## 注意事项

- `rag_qa/edu_text_spliter/edu_model_text_spliter.py`（AliTextSplitter）硬编码了 `/Users/ligang/Desktop/...` 的模型路径并在模块级导入 modelscope，**在任何机器上都无法实例化**。它不是默认切分器（`__init__.py` 导入但不使用），不要依赖它。
- 设备选择为自动：CUDA → MPS（Apple Silicon）→ CPU，相关代码见 `vector_store.py` 和 `query_classifier.py`。
