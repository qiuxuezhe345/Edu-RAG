# core/document_processor.py 源码
# 功能：从目录（可多层级）读取多类型文档，并做父子分片，产出可直接入库 Milvus 的子块
import os
import sys

# --- sys.path 前置段：必须在 base / rag_qa 导入之前，保证脚本可直接运行 ---
local_path = os.path.abspath(os.path.dirname(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
sys.path.insert(0, rag_qa_path)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base.config import config
from base.logger import logger

# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 TextLoader 加载原始文本
    ".md": TextLoader,
}


def _derive_source(file_path, directory_path):
    """根据文件相对传入目录的路径推导学科编码（学科子目录名去除 "_data"）。

    支持两种调用方式：
    1. 传入多层级根目录（如 rag_qa/data）：取文件相对路径的第一段作为学科目录；
    2. 传入叶子学科目录（如 ai_data）：回退用目录名本身推导。
    """
    try:
        rel = os.path.relpath(file_path, directory_path)
    except ValueError:
        # 跨盘符等场景下 relpath 不可计算，回退到文件名
        rel = os.path.basename(file_path)
    parts = rel.split(os.sep)
    if len(parts) > 1 and parts[0]:
        subject_dir = parts[0]
    else:
        subject_dir = os.path.basename(os.path.normpath(directory_path))
    source = subject_dir.removesuffix("_data")
    return source or "unknown"


def load_documents_from_directory(directory_path):
    """从指定目录（可多层级）递归读取支持类型的文档，返回 Document 列表。

    每个 Document 的 metadata 含：
    - source: 学科编码，由文件所在学科子目录名去除 "_data" 得到
    - file_path: 文档绝对路径
    - timestamp: 加载时间戳
    """
    # 初始化空列表，用于存储加载的文档
    documents = []
    # 目录不存在时直接返回空列表
    if not os.path.isdir(directory_path):
        logger.warning(f"目录不存在: {directory_path}")
        return documents
    # 获取支持的文件扩展名集合
    supported_extensions = document_loaders.keys()

    # 遍历指定目录及其子目录（覆盖多层级目录结构）
    for root, _, files in os.walk(directory_path):
        for file in files:
            # 跳过隐藏文件（如 .DS_Store），减少噪音
            if file.startswith("."):
                continue
            # 构造文件的绝对路径
            file_path = os.path.abspath(os.path.join(root, file))
            # 获取文件扩展名并转换为小写
            file_extension = os.path.splitext(file_path)[1].lower()
            # 检查文件类型是否在支持的扩展名列表中
            if file_extension in supported_extensions:
                # 使用 try-except 捕获加载过程中的异常，单个文件失败不中断整个批次
                try:
                    loader_class = document_loaders[file_extension]
                    # 实例化加载器对象，传入文件路径
                    if file_extension in (".txt", ".md"):
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    # 调用加载器加载文档内容，返回文档列表
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        # 添加学科编码元数据（学科子目录名去除 "_data"）
                        doc.metadata["source"] = _derive_source(file_path, directory_path)
                        # 添加文件绝对路径元数据
                        doc.metadata["file_path"] = file_path
                        # 添加当前时间戳元数据
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                    # 将加载的文档添加到总列表中
                    documents.extend(loaded_docs)
                    # 记录成功加载文件的日志
                    logger.info(f"成功加载文件: {file_path}")
                except Exception as e:
                    # 记录加载失败的日志，包含错误信息
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            else:
                # 如果文件类型不在支持列表中，记录警告日志
                logger.warning(f"不支持的文件类型: {file_path}")
    # 返回加载的所有文档列表
    return documents


def process_documents(directory_path,
                      parent_chunk_size=config.PARENT_CHUNK_SIZE,
                      child_chunk_size=config.CHILD_CHUNK_SIZE,
                      chunk_overlap=config.CHUNK_OVERLAP):
    """对指定目录下的文档做父子分片，返回子块列表。

    每个子块 metadata 含：
    - parent_id: 所属父分片的唯一 id
    - parent_content: 所属父分片的内容
    - id: 子分片唯一 id
    其余元数据（source/file_path/timestamp）继承自父分片。
    """
    # 从指定目录加载所有文档
    documents = load_documents_from_directory(directory_path)
    # 记录加载的文档总数日志
    logger.info(f"加载的文档数量: {len(documents)}")

    # 初始化父块和子块切分器（通用中文切分）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 Markdown 专用切分器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 初始化空列表，用于存储所有子块
    child_chunks = []
    # 遍历每个原始文档，带上索引 i（i 为全量文档平铺索引，保证 parent_id 跨文件唯一）
    for i, doc in enumerate(documents):
        # 获取文件扩展名
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
        # 选择切分器
        is_markdown = (file_extension == ".md")
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}")

        # 使用父块切分器将文档切分为父块
        parent_docs = parent_splitter_to_use.split_documents([doc])
        # 遍历每个父块，带上索引 j
        for j, parent_doc in enumerate(parent_docs):
            # 为父块生成唯一 ID，格式为 "doc_{i}_parent_{j}"
            parent_id = f"doc_{i}_parent_{j}"
            # 使用子块切分器将父块切分为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            # 遍历每个子块，带上索引 k
            for k, sub_chunk in enumerate(sub_chunks):
                # 为子块添加父块 ID 到元数据
                sub_chunk.metadata["parent_id"] = parent_id
                # 为子块添加父块内容到元数据
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                # 为子块生成唯一 ID，格式为 "parent_id_child_k"
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                # 将子块添加到子块列表中
                child_chunks.append(sub_chunk)

    # 记录子块总数日志
    logger.info(f"子块数量: {len(child_chunks)}")
    # 返回所有子块列表
    return child_chunks


if __name__ == "__main__":
    # 冒烟测试：对 rag_qa/data 目录做文档读取与父子分片测试
    print("=" * 60)
    docs = load_documents_from_directory(config.DATA_DIR)
    print(f"加载文档数量: {len(docs)}")
    for d in docs:
        print(f"  source={d.metadata.get('source')!r}  file_path={d.metadata['file_path']}  内容长度={len(d.page_content)}")

    chunks = process_documents(config.DATA_DIR)  # 内部会再次读取
    print(f"子块数量: {len(chunks)}")
    if chunks:
        c = chunks[0]
        print(f"样例子块 metadata: {c.metadata}")
        assert c.metadata.get("parent_id"), "parent_id 缺失"
        assert c.metadata.get("parent_content"), "parent_content 缺失"
        assert c.metadata.get("source") == "ai", f"source 应为 ai, 实际 {c.metadata.get('source')}"
        print("SMOKE TEST PASSED")
