# core/strategy_selector.py 源码
# 导入 os / sys
import os
import sys

# 保证直接运行脚本（python xxx/strategy_selector.py）时能导入项目根目录的包（base / rag_qa）
# __file__ 的第三级上级目录即项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- LangSmith 配置（必须在创建 OpenAI 客户端之前设置） ----
# 使用 setdefault：若 shell 中已设置同名环境变量，则外部值优先
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "111111")

# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate
# 导入日志和配置
from base.config import config
from base.logger import logger
# 导入 OpenAI
from openai import OpenAI
# 导入 LangSmith 的 OpenAI 包装器，用于自动追踪 LLM 调用
try:
    from langsmith import Client as LangsmithClient
    from langsmith.wrappers import wrap_openai
    LANGSMITH_AVAILABLE = True
except ImportError:
    # langsmith 未安装时降级为普通 OpenAI 客户端，不影响核心功能
    LangsmithClient = None
    wrap_openai = None
    LANGSMITH_AVAILABLE = False
    logger.warning("langsmith 未安装，本次运行将不进行 LangSmith 追踪")


class StrategySelector:
    def __init__(self):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=config.DASHSCOPE_API_KEY,
                             base_url=config.DASHSCOPE_BASE_URL)
        # 用 langsmith 包装客户端，开启对 DashScope/DeepSeek 调用的自动追踪
        if LANGSMITH_AVAILABLE and wrap_openai is not None:
            try:
                self.client = wrap_openai(self.client)
                logger.info("已启用 LangSmith 追踪")
            except Exception as e:
                # 包装失败不阻断主流程，仅记录日志
                logger.error(f"LangSmith 包装 OpenAI 客户端失败，跳过追踪: {e}")
        # 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()

    def call_dashscope(self, prompt):
        # 调用 DashScope API
        try:
            # 创建聊天完成请求
            completion = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1
            )
            # 返回完成结果
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            # 记录 API 调用失败
            logger.error(f"DashScope API 调用失败: {e}")
            # 默认返回直接检索
            return "直接检索"


    def _get_strategy_prompt(self):
        #   定义私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：AI 学科学费是多少？
                        * 策略：直接检索
                    * 查询：JAVA的课程大纲是什么？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )

    #   定义方法，选择检索策略
    def select_strategy(self, query):
        #   调用 LLM 获取检索策略
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy

if __name__ == '__main__':
    ss = StrategySelector()
    ss.select_strategy('大模型课程和JAVA智能应用课程有什么区别？')
    # 校验 LangSmith 连通性，确认 API Key 与网络可用、追踪会上报
    if LANGSMITH_AVAILABLE and LangsmithClient is not None:
        try:
            LangsmithClient().list_projects(limit=1)
            logger.info("LangSmith 连接正常，追踪数据将上报至项目「%s」",
                        os.environ.get("LANGSMITH_PROJECT", "default"))
        except Exception as e:
            logger.error(f"LangSmith 连接失败，请检查 LANGSMITH_API_KEY 与网络: {e}")
    else:
        logger.warning("跳过 LangSmith 连通性校验（langsmith 不可用）")
