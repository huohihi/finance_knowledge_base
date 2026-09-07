"""
MongoDB 对话历史工具

以 session_id 维度存储/读取用户与助手的问答消息，支撑：
    - 多轮追问的上下文（查询图 item_confirm 阶段读取最近 N 条）
    - 历史记录查看 / 清空（前端）
"""
import logging
from datetime import datetime
from typing import List, Dict, Any

from bson import ObjectId
from pymongo import DESCENDING
from pymongo.collection import Collection

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def _get_collection() -> Collection:
    """获取 chat_message 集合"""
    return StorageClients.get_mongo_db()["chat_message"]


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        entity_names: List[str] = None,
        message_id: str = None,
) -> str:
    """
    保存一条对话消息（message_id 为空则新增，否则更新）。
    附带 rewritten_query 与 entity_names 等检索过程元信息。
    """
    ts = datetime.now().timestamp()
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "entity_names": entity_names or [],
        "ts": ts,
    }

    collection = _get_collection()
    if message_id:
        collection.update_one({"_id": ObjectId(message_id)}, {"$set": document})
        return message_id
    else:
        result = collection.insert_one(document)
        return str(result.inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """读取最近 limit 条消息（时间倒序 -> 业务侧再 reverse 为正序）"""
    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        logger.error(f"Error getting recent messages: {e}")
        return []


def clear_history(session_id: str) -> int:
    """清空某会话的全部消息"""
    try:
        result = _get_collection().delete_many({"session_id": session_id})
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error clearing history for session {session_id}: {e}")
        return 0
