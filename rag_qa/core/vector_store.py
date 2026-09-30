# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
from base.config import config
from base.logger import logger
import sys
import os

local_path = os.path.abspath(os.path.dirname(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
sys.path.insert(0, rag_qa_path)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)


class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=config.MILVUS_COLLECTION_NAME,
                 host=config.MILVUS_HOST,
                 port=config.MILVUS_PORT,
                 database=config.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        rerank_model_path = os.path.join(config.MODELS_DIR, 'bge-reranker-large')
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        # TODO device代表设备： mps:m1系列的mac/ cpu: cpu / cuda: nvidia的gpu。和操作系统无关
        device = None
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.reranker = CrossEncoder(rerank_model_path, device=device)
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        bge_m3_model_path = os.path.join(config.MODELS_DIR, 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=bge_m3_model_path
            , use_f16=False,
             device=device
        )
        # 获取稠密向量的维度 bge-m3维度 1024
        self.dense_dim = self.embedding_function.dim["dense"]
        print(f'dense的维度：{self.dense_dim}')
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    # 定义私有方法，创建或加载 Milvus 集合
    def _create_or_load_collection(self):
        # 检查指定集合是否已存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # 定义方法，向向量存储添加文档
    def add_documents(self, documents):
        # 提取所有文档的内容列表
        texts = [doc.page_content for doc in documents]
        # 使用 BGE-M3 嵌入函数生成文档的嵌入
        embeddings = self.embedding_function(texts)
        # 初始化空列表，用于存储插入的数据
        data = []
        # 遍历每个文档，带上索引 i
        for i, doc in enumerate(documents):
            # 生成文档内容的 MD5 哈希值，作为唯一 ID
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # 初始化稀疏向量字典
            sparse_vector = {}
            # 获取第 i 行的稀疏向量数据
            row = embeddings['sparse'][[i], :]
            # 获取稀疏向量的非零值索引
            indices = row.indices
            # 获取稀疏向量的非零值
            values = row.data
            # 将索引和值配对，填充稀疏向量字典
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value
            # 创建数据字典，包含所有字段
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })
        # 检查是否有数据需要插入
        if data:
            # 使用 upsert 操作插入数据，覆盖重复 ID
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 记录插入或更新的文档数量日志
            logger.info(f"已插入或更新 {len(data)} 个文档")

    def _encode_query(self, query):
        """Encode one query into the dense and sparse formats Milvus expects."""
        query_embeddings = self.embedding_function([str(query)])
        dense_vector = query_embeddings["dense"][0]
        sparse_vector = {}
        row = query_embeddings["sparse"][[0], :]
        for idx, value in zip(row.indices, row.data):
            sparse_vector[int(idx)] = float(value)
        return dense_vector, sparse_vector

    @staticmethod
    def _safe_filter(source_filter):
        if not source_filter:
            return ""
        escaped = str(source_filter).replace("\\", "\\\\").replace("'", "\\'")
        return f"source == '{escaped}'"

    def search(self, query, top_k=5, candidate_k=20, mode="hybrid", rerank=True,
               sparse_weight=0.7, dense_weight=1.0, source_filter=None):
        """Run configurable retrieval for production and ablation evaluation."""
        if mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError("mode must be one of: dense, sparse, hybrid")

        top_k = max(1, int(top_k))
        candidate_k = max(top_k, int(candidate_k))
        dense_vector, sparse_vector = self._encode_query(query)
        filter_expr = self._safe_filter(source_filter)
        output_fields = ["text", "parent_id", "parent_content", "source", "timestamp"]

        if mode == "hybrid":
            dense_request = AnnSearchRequest(
                data=[dense_vector], anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=candidate_k, expr=filter_expr,
            )
            sparse_request = AnnSearchRequest(
                data=[sparse_vector], anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {}},
                limit=candidate_k, expr=filter_expr,
            )
            hits = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[sparse_request, dense_request],
                ranker=WeightedRanker(float(sparse_weight), float(dense_weight)),
                limit=candidate_k,
                output_fields=output_fields,
            )[0]
        else:
            is_dense = mode == "dense"
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vector if is_dense else sparse_vector],
                anns_field="dense_vector" if is_dense else "sparse_vector",
                search_params=(
                    {"metric_type": "IP", "params": {"nprobe": 10}}
                    if is_dense else {"metric_type": "IP", "params": {}}
                ),
                filter=filter_expr,
                limit=candidate_k,
                output_fields=output_fields,
            )[0]

        sub_chunks = []
        for hit in hits:
            doc = self._doc_from_hit(hit.get("entity", {}))
            doc.metadata["child_id"] = str(hit.get("id", ""))
            doc.metadata["retrieval_score"] = float(hit.get("distance", 0.0))
            sub_chunks.append(doc)

        parent_docs = self._get_unique_parent_docs(sub_chunks)
        if rerank:
            parent_docs = self.rerank_documents(query, parent_docs)

        return parent_docs[:top_k]

    def rerank_documents(self, query, documents, top_k=None):
        """Apply the BGE cross-encoder to an existing document candidate list."""
        documents = list(documents)
        if len(documents) > 1:
            pairs = [[str(query), doc.page_content] for doc in documents]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(documents, scores):
                doc.metadata["rerank_score"] = float(score)
            documents.sort(
                key=lambda doc: doc.metadata.get("rerank_score", float("-inf")),
                reverse=True,
            )
        return documents if top_k is None else documents[:max(1, int(top_k))]

    # 定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k=config.RETRIEVAL_K, source_filter=None):
        return self.search(
            query=query,
            top_k=config.CANDIDATE_M,
            candidate_k=max(int(k) * 4, 20),
            mode="hybrid",
            rerank=True,
            sparse_weight=0.7,
            dense_weight=1.0,
            source_filter=source_filter,
        )

    # 定义私有方法，从子块中提取去重的父文档
    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs

    # 定义私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )

if __name__ == '__main__':
    # 初始化向量存储，会加载 BGE-M3 和 BGE-Reranker 模型，连接 Milvus
    from document_processor import process_documents
    docs = process_documents('/Users/jencks.gao/heima/EduRag/rag_qa/data')
    vs = VectorStore()
    vs.add_documents(docs)

    # # 可取消注释进行检索测试
    # parent_docs = vs.hybrid_search_with_rerank('大模型课程介绍')
    # print(parent_docs)
