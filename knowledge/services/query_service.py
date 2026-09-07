"""
金融问答服务

职责：
    - 组装查询初始状态并运行查询 LangGraph（非流式同步 / 流式后台+SSE 两种模式）
    - 会话/任务 ID 生成、结果获取
    - 历史记录读写（查看 / 清空）
"""
import uuid
from typing import List, Dict, Any

from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.task_util import (update_task_status, TASK_STATUS_PROCESSING,
                                       TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, get_task_result)


class QueryService:
    """金融问答服务"""

    def run_query_graph(self, original_query: str, session_id: str,
                        task_id: str, is_stream: bool):
        """运行一次问答流程（阻塞式，供后台任务 / 线程池调用）"""
        try:
            init_state = {
                "original_query": original_query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
            }
            update_task_status(task_id, TASK_STATUS_PROCESSING)

            # invoke 执行整图（各节点的进度事件由 BaseNode 推入 SSE 队列）
            final_state = query_app.invoke(init_state)

            update_task_status(task_id, TASK_STATUS_COMPLETED)
            return final_state
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger = __import__("logging").getLogger(__name__)
            logger.error(f"查询流程执行出错: {e}")

    # ==================== ID 生成 ====================

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def get_task_result(self, task_id: str) -> str:
        return get_task_result(task_id, "answer")

    def get_task_citations(self, task_id: str) -> List[Dict[str, Any]]:
        """读取非流式场景持久化的引用来源（JSON 字符串 -> list）"""
        import json
        raw = get_task_result(task_id, "citations", default="[]")
        try:
            return json.loads(raw)
        except Exception:
            return []

    # ==================== 历史 ====================

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        from knowledge.utils.mongo_history_util import get_recent_messages
        records = get_recent_messages(session_id, limit=limit)
        records.reverse()  # Mongo 时间倒序 -> 转为正序展示
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "entity_names": r.get("entity_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        from knowledge.utils.mongo_history_util import clear_history
        return clear_history(session_id)
