# base/config.py
# 导入配置解析库 TextLoader
import configparser
# 导入路径操作库
import os


class Config:
    # 初始化配置，加载 config.init 文件
    def __init__(self, config_file=None):
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/edu_document_loaders')
        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')

        # 创建配置解析器,configparser.ExtendedInterpolation():启用插值，可以ini文件里以${别的块的值 redis:host}
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 读取配置文件 load
        self.config.read(config_file)

        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        # MySQL 用户名
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        # MySQL 密码
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='12345656565')
        # MySQL 数据库名
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')
        self.DEMO = self.config.get('demo', 'test')
        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        # Redis 端口
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        # Redis 密码
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234')
        # Redis 数据库编号
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)
        # 日志文件路径
        self.LOG_FILE = os.path.join(self.PROJECT_ROOT,self.config.get('logger', 'log_file', fallback='logs/app.log'))

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus 端口
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='itcast'))
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name', fallback='edurag_xian_1'))

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key',
                                                                                fallback=''))
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')

        # Query 分类器配置
        # 基础 BERT 模型路径（用于初始化二分类模型）
        self.BERT_MODEL_PATH = os.getenv(
            'BERT_MODEL_PATH',
            self.config.get('classifier', 'bert_model_path',
                            fallback=os.path.join(self.MODELS_DIR, 'bert-base-chinese')))
        # 分类器训练数据路径（JSONL 格式）
        self.CLASSIFIER_DATA_PATH = os.getenv(
            'CLASSIFIER_DATA_PATH',
            self.config.get('classifier', 'classifier_data_path',
                            fallback=os.path.join(self.PROJECT_ROOT, 'rag_qa/classify_data/model_generic_5000.json')))
        # 训练后分类器模型保存路径
        self.CLASSIFIER_MODEL_PATH = os.getenv(
            'CLASSIFIER_MODEL_PATH',
            self.config.get('classifier', 'classifier_model_path',
                            fallback=os.path.join(self.MODELS_DIR, 'bert_query_classifier')))


config = Config()
if __name__ == '__main__':
    conf = Config() # 创建配置类读取对象
    # print(conf.DEMO)
    print(__file__)
    # __file__ 当前代码文件
    # # /Users/jencks.gao/itcast/integrated_qa_system/base/config.py
    abs_path = os.path.abspath(__file__) # 当前源代码的绝对路径
    print('绝对路径：',abs_path)
    print('去除最后一级目录：',os.path.dirname(abs_path))
    dir_path = os.path.dirname(os.path.dirname(abs_path)) # 舍弃目录最后一级
    print(dir_path)
    print(os.path.join(dir_path, 'config.ini'))
    # print('-'*60)
    # print(os.path.basename(dir_path)) # 取目录的最后一级
