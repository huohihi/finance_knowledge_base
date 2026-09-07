"""
FastAPI 依赖注入：业务服务单例缓存
"""
from functools import lru_cache

from knowledge.services.query_service import QueryService
from knowledge.services.file_import_service import ImportFileService


@lru_cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService()


@lru_cache
def get_query_service() -> QueryService:
    return QueryService()
