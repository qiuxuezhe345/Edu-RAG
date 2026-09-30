from datetime import datetime
from threading import Lock

from base import config, logger


class IntegratedQASystem:
    """Adapter used by the web API, with lazy loading of heavy dependencies."""

    def __init__(self):
        self.config = config
        self.mysql_client = None
        self.redis_client = None
        self.bm25_search = None
        self.rag_system = None
        self._history = {}
        self._lock = Lock()

    def _ensure_faq_system(self):
        if self.bm25_search is not None:
            return

        from mysql_qa.cache.redis_client import RedisClient
        from mysql_qa.db.mysql_client import MySQLClient
        from mysql_qa.retrieval.bm25_search import BM25Search

        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def _ensure_rag_system(self):
        if self.rag_system is not None:
            return

        from main import build_llm
        from rag_qa.core.rag_system import RAGSystem
        from rag_qa.core.vector_store import VectorStore

        self.rag_system = RAGSystem(vector_store=VectorStore(), llm=build_llm())

    def _append_history(self, session_id, question, answer):
        with self._lock:
            self._history.setdefault(session_id, []).append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    def query(self, query, source_filter=None, session_id=None):
        session_id = session_id or "default"
        answer = None

        try:
            self._ensure_faq_system()
            answer, needs_rag = self.bm25_search.search(query, threshold=0.85)
            if not answer and needs_rag:
                self._ensure_rag_system()
                history = self.get_session_history(session_id)
                answer = self.rag_system.generate_answer(
                    query,
                    source_filter=source_filter,
                    history=history,
                )
        except Exception as exc:
            logger.error("Web 问答处理失败: %s", exc, exc_info=True)
            answer = (
                "后端已启动，但问答依赖尚未就绪。"
                "请检查 MySQL、Redis、Milvus 和模型/API Key 配置。"
            )

        if not answer:
            answer = "暂时没有找到可用答案。"

        answer = str(answer)
        self._append_history(session_id, query, answer)
        yield answer, True

    def get_session_history(self, session_id):
        with self._lock:
            return [dict(item) for item in self._history.get(session_id, [])]

    def clear_session_history(self, session_id):
        with self._lock:
            self._history.pop(session_id, None)
        return True

    def list_sessions(self):
        with self._lock:
            sessions = []
            for session_id, history in self._history.items():
                if not history:
                    continue
                sessions.append({
                    "session_id": session_id,
                    "title": history[0]["question"],
                    "updated_at": history[-1]["timestamp"],
                    "message_count": len(history),
                })
            return sorted(
                sessions,
                key=lambda item: item["updated_at"],
                reverse=True,
            )
