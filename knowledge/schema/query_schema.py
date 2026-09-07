"""
查询相关 Schema（Pydantic 请求/响应模型）

citations 对应需求文档 6.2「引用来源」。
"""
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="用户查询内容")
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式返回")


class QueryResponse(BaseModel):
    """非流式响应：完整答案 + 引用来源"""
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    answer: str = Field("", description="生成的答案")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="引用来源列表")


class StreamSubmitResponse(BaseModel):
    """流式响应：先返回任务ID，前端据此建立 SSE 连接"""
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID")


class HistoryItem(BaseModel):
    id: str = Field("", alias="_id")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    entity_names: List[str] = Field(default_factory=list)
    ts: Optional[float] = None


class HistoryResponse(BaseModel):
    session_id: str
    items: List[HistoryItem]
