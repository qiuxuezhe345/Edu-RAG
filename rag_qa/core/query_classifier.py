# core/query_classifier.py 源码
# 功能：基于 bert-base-chinese 的查询二分类（通用知识 / 专业咨询）
#       train_model 训练；predict_category 用已训练模型预测
import os
import sys
import json
import random

# --- sys.path 前置段：必须在 base / rag_qa 导入之前，保证脚本可直接运行 ---
local_path = os.path.abspath(os.path.dirname(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
sys.path.insert(0, rag_qa_path)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from base.config import config
from base.logger import logger

# 类别映射：已训练模型的 logit 索引 0 -> 通用知识, 1 -> 专业咨询（经验证）
LABELS = ["通用知识", "专业咨询"]
ID2LABEL = {0: "通用知识", 1: "专业咨询"}
LABEL2ID = {"通用知识": 0, "专业咨询": 1}


class _ClassificationDataset(torch.utils.data.Dataset):
    """把预编码后的 tensors + labels 包装成 torch Dataset，供 Trainer 使用。"""
    def __init__(self, encodings, labels):
        self.encodings = encodings          # dict[str, Tensor]（预 pad 到 max_length）
        self.labels = labels                # list[int]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class QueryClassifier:
    # 初始化方法：设置路径、设备等参数，模型/分词器采用懒加载，首次 predict 才加载
    def __init__(self,
                 bert_model_path=config.BERT_MODEL_PATH,
                 classifier_data_path=config.CLASSIFIER_DATA_PATH,
                 classifier_model_path=config.CLASSIFIER_MODEL_PATH,
                 max_length=128,
                 device=None):
        # 预训练基础模型路径
        self.bert_model_path = bert_model_path
        # 分类器训练数据路径（JSONL 格式）
        self.classifier_data_path = classifier_data_path
        # 训练后分类器模型保存路径
        self.classifier_model_path = classifier_model_path
        # 输入文本最大长度
        self.max_length = max_length
        # 设备自动选择：cuda -> mps -> cpu（与 vector_store.py 一致）
        self.device = device if device else self._detect_device()
        self.logger = logger
        # 懒加载缓存：_ensure_model_loaded 首次调用时填充
        self._tokenizer = None
        self._model = None

    # 定义静态方法，自动选择计算设备
    @staticmethod
    def _detect_device():
        """优先 CUDA，其次 Apple MPS，最后 CPU。"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    # 定义私有方法，懒加载已训练模型与分词器
    def _ensure_model_loaded(self):
        """首次 predict 调用时加载模型，之后复用缓存。"""
        if self._model is not None and self._tokenizer is not None:
            return
        self.logger.info(f"加载查询分类模型: {self.classifier_model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.classifier_model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.classifier_model_path)
        self._model.to(self.device)
        self._model.eval()

    # 定义方法，对单条用户问题做意图预测
    def predict_category(self, query):
        """对单条用户问题预测类别，返回 '通用知识' 或 '专业咨询'。"""
        # 懒加载模型与分词器（首次调用）
        self._ensure_model_loaded()
        query = str(query).strip()
        # 显式指定 max_length 并开启截断/补齐，避免 tokenizer 的巨型默认 model_max_length
        inputs = self._tokenizer(
            query,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        return ID2LABEL[pred_id]

    # 定义方法，读取 JSONL 训练数据
    def _load_records(self):
        """返回 [{'query': ..., 'label': ...}] 列表。"""
        records = []
        with open(self.classifier_data_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        self.logger.info(f"加载分类数据: {len(records)} 条")
        return records

    # 定义方法，用 HuggingFace Trainer 微调二分类模型
    def train_model(self, val_ratio=0.1, epochs=3, batch_size=16,
                    learning_rate=2e-5, seed=42):
        """在本地 bert-base-chinese 上微调，训练完成后保存到已训练模型目录。

        默认按 90/10 划分训练/验证集，指标使用准确率，训练结束后
        将模型与 tokenizer 保存到 classifier_model_path。
        """
        # --- 训练专用依赖放方法内，避免 predict 路径加载训练 API ---
        import numpy as np
        from transformers import (
            BertTokenizer, BertForSequenceClassification,
            TrainingArguments, Trainer,
        )

        # 1. 加载数据并映射为 label id
        records = self._load_records()
        texts = [r["query"] for r in records]
        labels = [LABEL2ID[r["label"]] for r in records]

        # 2. 固定随机种子，按 90/10 划分训练/验证集
        idx = list(range(len(records)))
        random.Random(seed).shuffle(idx)
        n_val = int(len(records) * val_ratio)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        self.logger.info(f"训练集 {len(train_texts)} / 验证集 {len(val_texts)}")

        # 3. 分词：预 pad 到 max_length（一次性）
        tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        train_enc = tokenizer(train_texts, max_length=self.max_length,
                              truncation=True, padding='max_length', return_tensors='pt')
        val_enc = tokenizer(val_texts, max_length=self.max_length,
                            truncation=True, padding='max_length', return_tensors='pt')
        train_dataset = _ClassificationDataset(train_enc, train_labels)
        val_dataset = _ClassificationDataset(val_enc, val_labels)

        # 4. 从基础 BERT 初始化序列分类模型
        model = BertForSequenceClassification.from_pretrained(
            self.bert_model_path, num_labels=2,
            id2label=ID2LABEL, label2id=LABEL2ID)

        # 5. 评估指标：准确率
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": float((preds == labels).mean())}

        # 6. 训练参数与 Trainer
        run_dir = os.path.join(os.path.dirname(self.classifier_model_path),
                               'bert_query_classifier_runs')
        training_args = TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy='epoch',        # transformers>=4.41 的新参数名
            save_strategy='epoch',
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            logging_dir=os.path.join(run_dir, 'logs'),
            logging_steps=10,
            report_to=[],                 # 关闭 wandb/tensorboard 等第三方上报
            seed=seed,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # 7. 训练并保存到已训练模型目录（含 tokenizer）
        trainer.train()
        trainer.save_model(self.classifier_model_path)
        tokenizer.save_pretrained(self.classifier_model_path)
        self.logger.info(f"模型已保存: {self.classifier_model_path}")


if __name__ == '__main__':
    # 冒烟测试：随机抽取 5 条数据，仅预测不训练
    print("=" * 60)
    qc = QueryClassifier()
    records = qc._load_records()
    print(f"数据集总数: {len(records)}")
    samples = random.sample(records, 5)
    for i, rec in enumerate(samples, 1):
        pred = qc.predict_category(rec["query"])
        hit = "命中" if pred == rec["label"] else "未命中"
        print(f"[{i}] query={rec['query']!r}  预测={pred}  真实={rec['label']}  {hit}")
    print("=" * 60)
