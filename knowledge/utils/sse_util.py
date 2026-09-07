"""
SSE（Server-Sent Events）流式工具

以 task_id 为键维护全局消息队列：
    - 查询图的各节点把「进度事件」「LLM 增量」「最终答案」push 到队列；
    - /stream/{task_id} 端点通过 sse_generator 持续消费并推送前端。
"""
import asyncio
import json
import logging
import queue
from typing import Dict, Any, Optional, AsyncGenerator

from fastapi import Request


class SSEEvent:
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案


# 全局 SSE 队列存储 {task_id: queue.Queue}
_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    q = queue.Queue()
    _task_stream[task_id] = q
    return q


def remove_sse_queue(task_id: str):
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式：event: xxx\ndata: {...}\n\n"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    """通过 task_id 推送事件到 SSE 队列"""
    stream_queue = get_sse_queue(task_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator:
    """SSE 消费者：循环从队列取事件并 yield 给前端"""
    sse_queue = _task_stream.get(task_id)
    if sse_queue is None:
        return

    loop = asyncio.get_event_loop()
    try:
        while True:
            # 前端断连探测
            if await request.is_disconnected():
                return
            try:
                # 阻塞读取（放入线程池避免阻塞事件循环），1 秒超时
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                yield _sse_pack(msg.get('event'), msg.get('data'))
            except queue.Empty:
                continue
    except (ConnectionResetError, BrokenPipeError):
        return
    except asyncio.CancelledError:
        raise
    finally:
        remove_sse_queue(task_id)
