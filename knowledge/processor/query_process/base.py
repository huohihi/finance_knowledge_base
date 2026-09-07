"""
查询流程节点基类

定义统一的节点接口规范，提供通用功能：
    - 统一日志 / 任务追踪（running/done）
    - 流式场景下推送节点进度到 SSE
    - 统一异常包装（QueryProcessError）
"""
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Optional

from knowledge.processor.query_process.config import QueryConfig, get_config
from knowledge.processor.query_process.exceptions import QueryProcessError
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import (add_running_task, add_done_task, get_task_status,
                                       get_running_task_list, get_done_task_list)

T = TypeVar("T")


class BaseNode(ABC):
    """查询流程节点基类

    所有节点类都应继承此基类并实现 process 方法。
    基类提供统一的日志、任务追踪、SSE 进度推送与错误包装。
    """

    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口"""
        is_stream = state.get('is_stream')
        task_id = state.get('task_id')

        try:
            self.logger.info(f"--- {self.name} 开始 ---")
            if task_id:
                add_running_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)

            result = self.process(state)

            if task_id:
                add_done_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)
                    # 答案节点执行完成后推送 FINAL（无论是否流式 LLM）
                    if self.name == "answer_output":
                        push_sse_event(
                            task_id=task_id, event=SSEEvent.FINAL,
                            data={"answer": state.get("answer"), "citations": state.get("citations", [])},
                        )
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise QueryProcessError(message=str(e), node_name=self.name, cause=e)

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑（子类必须实现）"""
        pass

    def log_step(self, step_name: str, message: str = ""):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

    def _push_progress(self, task_id: str):
        """推送当前节点进度（全量推所有节点状态）"""
        push_sse_event(
            task_id=task_id,
            event=SSEEvent.PROGRESS,
            data={
                "status": get_task_status(task_id),
                "done_list": get_done_task_list(task_id),
                "running_list": get_running_task_list(task_id),
            },
        )


def setup_logging(level: int = logging.INFO):
    """配置查询流程日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
    )
