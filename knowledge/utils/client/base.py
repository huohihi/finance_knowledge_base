"""
客户端管理器基类

提供:
    _require_env():  环境变量校验（缺失立即抛 EnvironmentError）
    _get_or_create(): 双重检查锁模板方法（懒加载单例）
"""
import logging
import os
import threading

logger = logging.getLogger(__name__)


class BaseClientManager:
    """所有客户端管理器（AI 模型 / 存储）的统一基类"""

    @staticmethod
    def _require_env(key: str) -> str:
        """
        读取必需的环境变量，缺失立即抛异常。
        注：.env 由各 process/config.py 中的 load_dotenv() 完成装载，
            因此此处只需要读取 os.environ。
        """
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(f"缺少必需的环境变量: {key}")
        return value

    @classmethod
    def _get_or_create(cls, attr_name: str, lock: threading.Lock, factory):
        """
        双重检查锁模板方法。

        Args:
            attr_name: 类属性名（存放单例）
            lock: 对应的线程锁
            factory: 工厂函数（不执行调用，仅在真正需要创建时执行）

        Returns:
            单例实例
        """
        # 1. 第一次检查（无锁，快速返回）
        instance = getattr(cls, attr_name, None)
        if instance is not None:
            return instance

        # 2. 加锁后第二次检查（防止并发重复创建）
        with lock:
            instance = getattr(cls, attr_name, None)
            if instance is not None:
                return instance
            # 3. 真正创建
            instance = factory()
            setattr(cls, attr_name, instance)
            return instance
