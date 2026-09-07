"""
导入流程配置管理模块

集中管理金融文档导入链路的所有配置项，支持环境变量覆盖。
字段语义对齐需求文档第 5 节内容字段要求。
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Set

from dotenv import load_dotenv

from knowledge.core.paths import ENV_FILE

load_dotenv(dotenv_path=ENV_FILE)


@dataclass
class ImportConfig:
    """金融文档导入流程配置"""

    # ==================== 文档切分配置 ====================
    max_content_length: int = 1800          # 单个切片最大长度（金融条款较长，控制上下文）
    img_content_length: int = 200           # 图片上下文最大长度
    min_content_length: int = 400           # 触发合并的短正文阈值
    overlap_sentences: int = 1              # 句子级切分重叠句数
    meta_extract_chunk_k: int = 5           # 元数据抽取取前 N 个切片
    meta_extract_chunk_size: int = 2500     # 元数据抽取单切片内容长度
    entity_extract_chunk_k: int = 8         # 实体抽取观察的切片数
    entity_extract_chunk_size: int = 2500   # 实体抽取观察的切片内容长度

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    vl_model: str = field(default_factory=lambda: os.getenv("VL_MODEL", ""))
    default_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", ""))

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", ""))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", ""))
    entity_collection: str = field(default_factory=lambda: os.getenv("ENTITY_COLLECTION", ""))

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", ""))
    minio_access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", ""))
    minio_secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", ""))
    minio_bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", ""))
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    embedding_batch_chunk_size: int = 16

    # ==================== 速率限制 ====================
    requests_per_minute: int = 30           # VLM 图片总结限流

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    def get_minio_base_url(self) -> str:
        base_protocol = "https://" if self.minio_secure else "http://"
        return base_protocol + self.minio_endpoint


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
