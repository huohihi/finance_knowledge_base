"""
查询流程自定义异常类

统一错误处理，提供更清晰的错误信息。
"""


class QueryProcessError(Exception):
    """查询流程基础异常"""

    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(parts)


class StateFieldError(QueryProcessError):
    """状态字段错误。从 state 中获取必需字段缺失、为空或类型不符时抛出。"""

    def __init__(
            self,
            node_name: str = "",
            field_name: str = "",
            expected_type: type = None,
            message: str = "",
            cause: Exception = None,
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"状态字段 '{field_name}' 缺失或无效"
            if expected_type:
                message += f"，期望类型: {expected_type.__name__}"
        super().__init__(message, node_name=node_name, cause=cause)


class ConfigurationError(QueryProcessError):
    """配置错误"""
    pass


class SearchError(QueryProcessError):
    """搜索错误：向量/混合/网络搜索失败"""
    pass


class EmbeddingError(QueryProcessError):
    """向量化错误"""
    pass


class LLMError(QueryProcessError):
    """LLM 调用错误"""
    pass


class StorageError(QueryProcessError):
    """存储错误"""
    pass


class MilvusError(StorageError):
    """Milvus 存储错误"""
    pass


class MongoDBError(StorageError):
    """MongoDB 存储错误"""
    pass


class ValidationError(QueryProcessError):
    """数据验证错误"""
    pass


class EntityAlignmentError(QueryProcessError):
    """金融实体对齐错误"""
    pass


class RerankError(QueryProcessError):
    """重排序错误"""
    pass


class EntityConfirmError(QueryProcessError):
    """实体确认错误"""
    pass
