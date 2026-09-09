# rag_qa/main.py
# 命令行交互式 RAG 问答入口
# 用法: /opt/anaconda3/envs/edu_rag/bin/python rag_qa/main.py
# 运行前提：Milvus 已启动、DASHSCOPE_API_KEY 已配置（见 config.ini）

import os
import sys

# 将项目根目录加入 sys.path，保证能导入项目根目录的 base / rag_qa 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base import config, logger


def build_llm():
    """初始化 DashScope（OpenAI 兼容接口）LLM 调用函数，输入 prompt 返回回答文本。"""
    from openai import OpenAI

    # 与 rag_system.py 的 __main__ 保持一致：LangSmith 包装 OpenAI 客户端，自动追踪 LLM 调用
    try:
        from langsmith.wrappers import wrap_openai
        client = wrap_openai(
            OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)
        )
    except ImportError:
        # langsmith 未安装时降级为普通客户端，不影响核心功能
        client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)

    def llm(prompt):
        completion = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个智能助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content if completion.choices else ""

    return llm


def main():
    print("=" * 70)
    print("EduRag 智能问答系统（RAG）")
    print(f"可选学科: {config.VALID_SOURCES}")
    print("输入问题后按提示输入学科；学科可留空（直接回车）表示不限定学科。")
    print("输入 exit / quit / q 退出。")
    print("=" * 70)

    # 初始化 LLM、向量库、RAG 系统（首次加载本地 BGE-M3 / Reranker / 分类器模型较慢，请耐心等待）
    print("正在初始化 RAG 系统（加载本地模型并连接 Milvus）...")
    try:
        from rag_qa.core.rag_system import RAGSystem
        from rag_qa.core.vector_store import VectorStore

        llm = build_llm()
        vector_store = VectorStore()
        rag_system = RAGSystem(vector_store=vector_store, llm=llm)
    except Exception as e:
        logger.error(f"RAG 系统初始化失败: {e}", exc_info=True)
        print(f"初始化失败: {e}")
        return
    print("初始化完成，开始对话。\n")

    while True:
        try:
            question = input("【问题】: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出程序。")
            break

        if question.lower() in ("exit", "quit", "q"):
            print("退出程序。")
            break
        if not question:
            print("问题不能为空，请重新输入。")
            continue

        # 输入学科（可选）：回车跳过 → 不限定学科；合法学科 → 作为 source_filter 传入
        valid_sources = config.VALID_SOURCES
        source_filter = None
        while True:
            try:
                subject = input(f"【学科】(可选，回车跳过，可选 {valid_sources}): ").strip()
            except (EOFError, KeyboardInterrupt):
                subject = ""
            if not subject:
                source_filter = None
                break
            if subject in valid_sources:
                source_filter = subject
                break
            print(f"无效学科 '{subject}'，可选学科为: {valid_sources}")

        print("正在生成答案（请稍候）...\n")
        try:
            answer = rag_system.generate_answer(question, source_filter=source_filter)
        except Exception as e:
            logger.error(f"生成答案失败: {e}", exc_info=True)
            answer = f"抱歉，生成答案时出错: {e}"

        print("-" * 70)
        print(f"【学科】: {source_filter if source_filter else '不限'}")
        print(f"【回答】: {answer}")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
