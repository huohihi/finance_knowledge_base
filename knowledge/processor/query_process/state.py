"""
查询流程状态类型定义

定义完整的金融问答状态结构与辅助函数。
"""
import copy
from typing import TypedDict, List, Dict, Any


class QueryGraphState(TypedDict):
    """查询流程图状态

    包含整个查询流程中传递的所有数据。
    """
    session_id: str              # 会话 ID（多轮历史维度）
    task_id: str                 # 任务 ID（进度/SSE 维度）
    message_id: str              # 消息 ID
    original_query: str          # 原始查询（用户输入）

    # ---- 实体确认产物 ----
    query_type: str              # product/announcement/general（识别的问题类型）
    entity_names: List[str]      # 对齐确认后的实体名（用于正文过滤；为空则全库检索）
    rewritten_query: str         # 重写查询（独立完整问题）

    # ---- 多路检索产物 ----
    embedding_chunks: list       # 向量检索结果
    hyde_embedding_chunks: list  # HyDE 检索结果
    web_search_docs: list        # 网络搜索结果
    rrf_chunks: list             # RRF 融合后切片
    reranked_docs: list          # 重排序后文档（含金融引用元数据）

    # ---- 生成产物 ----
    prompt: str                  # 组装后的提示词
    answer: str                  # 最终答案
    citations: list              # 引用来源列表（需求 6.2 引用来源）

    # ---- 上下文 ----
    history: list                # 历史对话
    is_stream: bool              # 是否流式输出


DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "message_id": "",
    "original_query": "",
    "query_type": "general",
    "entity_names": [],
    "rewritten_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "citations": [],
    "history": [],
    "is_stream": False,
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖"""
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本"""
    return copy.deepcopy(DEFAULT_STATE)
