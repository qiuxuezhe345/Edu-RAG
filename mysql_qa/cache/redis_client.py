# cache/redis_client.py
# 导入 Redis 客户端
import redis
# 导入 JSON 处理
import json
# 导入配置和日志
from base import config, logger


class RedisClient:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        try:
            # 连接 Redis
            self.client = redis.StrictRedis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                decode_responses=True
            )
            # 记录连接成功
            self.logger.info("Redis 连接成功")
        except redis.RedisError as e:
            # 记录连接失败
            self.logger.error(f"Redis 连接失败: {e}")
            raise

    def set_data(self, key, value):
        # 存储数据到 Redis
        try:
            # 存储 JSON 数据（ensure_ascii=False 保留中文，便于在 Redis 中直接查看）
            self.client.set(key, json.dumps(value, ensure_ascii=False))
            # 记录存储成功
            self.logger.info(f"存储数据到 Redis: {key}")
        except redis.RedisError as e:
            # 记录存储失败
            self.logger.error(f"Redis 存储失败: {e}")

    def get_data(self, key):
        # 从 Redis 获取数据
        try:
            # 获取数据
            data = self.client.get(key)
            # 返回解析后的 JSON 数据或 None
            return json.loads(data) if data else None
        except redis.RedisError as e:
            # 记录获取失败
            self.logger.error(f"Redis 获取失败: {e}")
            # 返回 None
            return None

    def get_answer(self, query):
        # 获取查询的缓存答案
        try:
            # 从 Redis 获取答案
            answer = self.client.get(f"answer:{query}")
            if answer:
                # 记录获取成功
                self.logger.info(f"从 Redis 获取答案: {query}")
                # 反序列化 JSON，还原中文（与 set_data 的 json.dumps 对应）
                return json.loads(answer)
            # 返回 None
            return None
        except redis.RedisError as e:
            # 记录查询失败
            self.logger.error(f"Redis 查询失败: {e}")
            # 返回 None
            return None
        except json.JSONDecodeError as e:
            # 记录 JSON 解析失败
            self.logger.error(f"Redis 答案 JSON 解析失败: {e}")
            # 返回 None
            return None
if __name__ == '__main__':
    redcli = RedisClient()
    print(redcli)