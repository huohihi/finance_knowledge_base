"""
AI 模型客户端管理器

统一管理:
    - OpenAI 原生客户端       （供 VLM 图片摘要、图片理解使用）
    - ChatOpenAI 文本/JSON    （供 LLM 抽取元数据、生成答案、HyDE 等）
    - BGE-M3 嵌入模型          （dense + sparse 混合向量）
    - BGE-Reranker 重排模型     （交叉编码器精排）

全部通过 BaseClientManager._get_or_create 实现线程安全懒加载单例。
"""
import threading
from typing import Optional

from langchain_openai import ChatOpenAI
from openai import OpenAI
from FlagEmbedding import FlagReranker

# 注意：BGEM3EmbeddingFunction 仅在 pymilvus 2.x 的 pymilvus.model.hybrid 提供。
# 3.x 已剥离 model 子模块，故 requirements 中锁定 pymilvus>=2.4,<3.0。
try:
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "无法导入 BGEM3EmbeddingFunction：请确认安装 pymilvus 2.x（pip install 'pymilvus>=2.4,<3.0'）"
    ) from _e

from knowledge.utils.client.base import BaseClientManager, logger


class AIClients(BaseClientManager):
    """AI 模型类客户端管理器"""

    # ==================== OpenAI 原生客户端（VLM 等）====================
    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    @classmethod
    def get_openai(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_openai)

    @classmethod
    def _create_openai(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI API 创建成功: {base_url}")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI API 创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ==================== ChatOpenAI 文本 / JSON LLM ====================
    _openai_llm_text_client: Optional[ChatOpenAI] = None
    _openai_llm_text_lock = threading.Lock()

    _openai_llm_json_client: Optional[ChatOpenAI] = None
    _openai_llm_json_lock = threading.Lock()

    @classmethod
    def get_llm_openai(cls, response_format: bool = True) -> ChatOpenAI:
        """
        获取 LLM 客户端。

        Args:
            response_format: True  -> JSON 输出模式客户端（抽取类任务）
                             False -> 自由文本模式客户端（问答类任务）
        """
        if response_format:
            # lambda 延迟执行：避免 _create_llm_openai(response_format) 立即执行
            return cls._get_or_create(
                "_openai_llm_json_client", cls._openai_llm_json_lock,
                lambda: cls._create_llm_openai(response_format)
            )
        return cls._get_or_create(
            "_openai_llm_text_client", cls._openai_llm_text_lock,
            lambda: cls._create_llm_openai(response_format)
        )

    @classmethod
    def _create_llm_openai(cls, response_format: bool) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")

            model_kwargs = {}
            if response_format:
                model_kwargs["response_format"] = {"type": "json_object"}

            client = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0,
                model_kwargs=model_kwargs,
            )
            logger.info("ChatOpenAI LLM 客户端初始化成功")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ChatOpenAI LLM 客户端初始化失败: {e}")
            raise ConnectionError(f"ChatOpenAI LLM 连接失败: {e}") from e

    # ==================== BGE-M3 嵌入模型 ====================
    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    @classmethod
    def get_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        try:
            model_name = cls._require_env("BGE_M3_PATH")
            device = cls._require_env("BGE_DEVICE")
            fp16 = cls._require_env("BGE_FP16").lower() in ("true", "1")

            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=fp16,
            )
            logger.info("bge_m3 客户端初始化成功")
            return bge_m3_ef
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"bge_m3 客户端初始化失败: {e}")
            raise ConnectionError(f"bge_m3 客户端创建失败: {e}") from e

    # ==================== BGE-Reranker 重排模型 ====================
    _bge_m3_rerank_client: Optional[FlagReranker] = None
    _bge_m3_rerank_lock = threading.Lock()

    @classmethod
    def get_bge_m3_rerank_client(cls) -> FlagReranker:
        return cls._get_or_create(
            "_bge_m3_rerank_client", cls._bge_m3_rerank_lock, cls._create_bge_m3_rerank_client
        )

    @classmethod
    def _create_bge_m3_rerank_client(cls) -> FlagReranker:
        try:
            model_name_or_path = cls._require_env("BGE_RERANKER_LARGE")
            device = cls._require_env("BGE_RERANKER_DEVICE")
            fp16 = cls._require_env("BGE_RERANKER_FP16").lower() in ("true", "1")

            reranker = FlagReranker(
                model_name_or_path=model_name_or_path,
                device=device,
                use_fp16=fp16,
            )
            logger.info("bge_rerank 客户端初始化成功")
            return reranker
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"bge_rerank 客户端初始化失败: {e}")
            raise ConnectionError(f"bge_rerank 客户端创建失败: {e}") from e


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print(AIClients.get_bge_m3_client())
