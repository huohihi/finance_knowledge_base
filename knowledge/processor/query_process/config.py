"""
查询流程配置管理模块

集中管理金融问答链路配置，支持环境变量覆盖（延迟加载 + 热更新）。
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from knowledge.core.paths import ENV_FILE

load_dotenv(dotenv_path=ENV_FILE)


@dataclass
class QueryConfig:
    """金融查询流程配置"""

    # ==================== 文本处理配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "10")))
    rerank_min_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3")))
    rerank_gap_abs: float = field(default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.15")))

    # ==================== RRF 配置 ====================
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    rrf_max_results: int = field(default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10")))

    # ==================== 检索配置 ====================
    embedding_search_limit: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "10")))
    hyde_search_limit: int = field(default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "5")))

    # ==================== 实体确认节点配置 ====================
    entity_high_confidence: float = field(default_factory=lambda: float(os.getenv("ENTITY_HIGH_CONFIDENCE", "0.7")))
    entity_mid_confidence: float = field(default_factory=lambda: float(os.getenv("ENTITY_MID_CONFIDENCE", "0.6")))
    entity_max_options: int = field(default_factory=lambda: int(os.getenv("ENTITY_MAX_OPTIONS", "5")))
    entity_dense_weight: float = field(default_factory=lambda: float(os.getenv("ENTITY_DENSE_WEIGHT", "0.5")))
    entity_sparse_weight: float = field(default_factory=lambda: float(os.getenv("ENTITY_SPARSE_WEIGHT", "0.5")))
    # 兜底：LLM 未识别出实体时是否继续走通用检索（金融知识问答/FAQ 场景需要 True）
    allow_general_query_without_entity: bool = field(
        default_factory=lambda: os.getenv("ALLOW_GENERAL_QUERY_WITHOUT_ENTITY", "true").lower() in ("true", "1")
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    default_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", ""))

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", ""))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", ""))
    entity_collection: str = field(default_factory=lambda: os.getenv("ENTITY_COLLECTION", ""))

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = field(default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", ""))
    mcp_dashscope_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))

    @classmethod
    def from_env(cls) -> "QueryConfig":
        return cls()


# ==================== 全局单例 ====================
_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config
