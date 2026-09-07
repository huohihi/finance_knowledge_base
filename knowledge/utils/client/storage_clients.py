"""
存储类客户端管理器

统一管理:
    - Milvus    向量库（chunk 正文混合检索 + 实体集合对齐检索）
    - MongoDB   对话历史 / 会话记录
    - MinIO     原始文件 / 图片对象存储
"""
import logging
import threading
from typing import Optional

from dotenv import load_dotenv
from minio import Minio
from pymilvus import MilvusClient
from pymongo import MongoClient
from pymongo.database import Database

from knowledge.core.paths import ENV_FILE
from knowledge.utils.client.base import BaseClientManager

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=ENV_FILE)


class StorageClients(BaseClientManager):
    """存储类客户端管理器"""

    # ==================== Milvus ====================
    _milvus_client: Optional[MilvusClient] = None
    _milvus_lock = threading.Lock()

    @classmethod
    def get_milvus_client(cls) -> MilvusClient:
        return cls._get_or_create("_milvus_client", cls._milvus_lock, cls._create_milvus_client)

    @classmethod
    def _create_milvus_client(cls) -> MilvusClient:
        try:
            milvus_uri = cls._require_env("MILVUS_URL")
            client = MilvusClient(milvus_uri)
            logger.info(f"Milvus 客户端初始化成功: {milvus_uri}")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"Milvus 客户端初始化失败: {e}")
            raise ConnectionError(f"Milvus 连接失败: {e}") from e

    # ==================== MongoDB ====================
    _mongo_db: Optional[Database] = None
    _mongo_lock = threading.Lock()

    @classmethod
    def get_mongo_db(cls) -> Database:
        return cls._get_or_create("_mongo_db", cls._mongo_lock, cls._create_mongo_db)

    @classmethod
    def _create_mongo_db(cls) -> Database:
        try:
            mongo_url = cls._require_env("MONGO_URL")
            db_name = cls._require_env("MONGO_DB_NAME")
            client = MongoClient(mongo_url)
            db = client[db_name]
            logger.info(f"MongoDB 客户端初始化成功 (db={db_name})")
            return db
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MongoDB 客户端创建失败: {e}")
            raise ConnectionError(f"MongoDB 连接失败: {e}") from e

    # ==================== MinIO ====================
    _minio_client: Optional[Minio] = None
    _minio_lock = threading.Lock()

    @classmethod
    def get_minio_client(cls) -> Minio:
        return cls._get_or_create("_minio_client", cls._minio_lock, cls._create_minio)

    @classmethod
    def _create_minio(cls) -> Minio:
        try:
            endpoint = cls._require_env("MINIO_ENDPOINT")
            access_key = cls._require_env("MINIO_ACCESS_KEY")
            secret_key = cls._require_env("MINIO_SECRET_KEY")
            bucket_name = cls._require_env("MINIO_BUCKET_NAME")

            client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"自动创建存储桶: {bucket_name}")
            else:
                logger.info(f"存储桶已存在: {bucket_name}")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MinIO 客户端初始化失败: {e}")
            raise ConnectionError(f"MinIO 配置错误: {e}") from e
