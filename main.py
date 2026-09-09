# 导入 MySQL 客户端
from mysql_qa.db.mysql_client import MySQLClient
# 导入 Redis 客户端
from mysql_qa.cache.redis_client import RedisClient
# 导入 BM25 搜索
from mysql_qa.retrieval.bm25_search import BM25Search
# 导入日志
from base import logger
# 导入时间库
import time

class MySQLQASystem:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = MySQLClient()
        # 初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query):
        # 查询 MySQL 系统
        start_time = time.time()
        # 记录查询信息
        self.logger.info(f"处理查询: '{query}'")
        # 执行 BM25 搜索
        answer, rag_flag = self.bm25_search.search(query, threshold=0.85)
        if answer:
            # 记录 MySQL 答案
            self.logger.info(f"MySQL 答案: {answer}")
        elif rag_flag: # 需要RAG
            self.logger.info("当前需要进行RAG检索")
            print("当前需要进行RAG检索")
            print("正在初始化 RAG 系统（加载本地模型并连接 Milvus）...")
            try:
                from rag_qa.core.rag_system import RAGSystem
                from rag_qa.core.vector_store import VectorStore

                llm = build_llm()
                vector_store = VectorStore()
                rag_system = RAGSystem(vector_store=vector_store, llm=llm)
                res = rag_system.generate_answer(query)
                print('*'*80)
                print(res)
                print('*'*80)
            except Exception as e:
                logger.error(f"RAG 系统初始化失败: {e}", exc_info=True)
                print(f"初始化失败: {e}")


        else:
            # 记录无答案
            self.logger.info("SQL中未找到答案, 需要调用RAG系统")
            # 设置默认答案
            answer = "SQL未找到答案"
        # 计算处理时间
        processing_time = time.time() - start_time
        # 记录处理时间
        self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
        # 返回答案
        return answer

def build_llm():
    """初始化 DashScope（OpenAI 兼容接口）LLM 调用函数，输入 prompt 返回回答文本。"""
    from openai import OpenAI
    from base import config

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
    # 初始化 MySQL 系统
    mysql_system = MySQLQASystem()
    try:
        # 打印欢迎信息
        print("\n欢迎使用 MySQL 问答系统！")
        print("输入查询进行问答，输入 'exit' 退出。")
        while True:
            # 获取用户输入
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 记录退出日志
                logger.info("退出 MySQL 系统")
                # 打印退出信息
                print("再见！")
                break
            # 执行查询
            answer = mysql_system.query(query)
            # 打印答案
            print(f"\n答案: {answer}")
    except Exception as e:
        # 记录系统错误
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭 MySQL 连接
        mysql_system.mysql_client.close()

if __name__ == "__main__":
    # 运行主程序
    main()