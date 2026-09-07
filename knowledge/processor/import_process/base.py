"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能：
    - 统一日志（开始/完成/耗时）
    - 任务追踪（running/done 列表，供前端实时进度）
    - 统一异常包装（ImportProcessError）
"""
import datetime
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Optional

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
from knowledge.utils.task_util import add_running_task, add_done_task, add_node_duration

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """导入流程节点基类

    所有节点类都应继承此基类并实现 process 方法。
    基类提供统一的日志、任务追踪与错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口

        LangGraph 调用节点时会调用此方法，提供统一追踪与异常包装。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        task_id = state.get("task_id", "")
        try:
            # 1. 开始准备执行节点
            self.logger.info(f"--- {self.name} 开始 ---")
            if task_id:
                add_running_task(task_id, self.name)

            start = datetime.datetime.now()
            # 2. 执行子节点业务
            result = self.process(state)
            duration = (datetime.datetime.now() - start).total_seconds()

            # 3. 执行成功
            self.logger.info(f"--- {self.name} 完成 ---")
            if task_id:
                add_done_task(task_id, self.name)
                add_node_duration(task_id, self.name, duration)
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(message=str(e), node_name=self.name, cause=e)

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑（子类必须实现）"""
        pass

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志"""
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


def setup_logging(level: int = logging.INFO):
    """配置导入流程日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
