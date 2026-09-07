"""
查询 API（FastAPI 应用，默认端口 8001）

路由：
    GET    /chat                  金融问答页
    POST   /query                 问答（流式：返回 task_id；非流式：返回完整答案+引用）
    GET    /stream/{task_id}      SSE 流式输出（进度事件 + LLM 增量 + 最终答案/引用）
    GET    /history/{session_id}  历史记录查看
    DELETE /history/{session_id}  清空历史

交互模式（与 0525 一致）：
    流式：POST /query -> task_id -> 建立 GET /stream/{task_id} SSE 连接
    非流式：POST /query 在默认线程池中同步执行后返回 answer + citations
"""
import asyncio
import os

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.query_process.base import setup_logging
from knowledge.schema.query_schema import StreamSubmitResponse, QueryResponse, QueryRequest
from knowledge.services.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator


def register_router(app: FastAPI):
    @app.get("/chat")
    def chat_page():
        return FileResponse(path=os.path.join(get_front_page_dir(), "chat.html"))

    @app.post("/query", response_model=QueryResponse | StreamSubmitResponse)
    async def query(
            request: QueryRequest,
            background_tasks: BackgroundTasks,
            service: QueryService = Depends(get_query_service),
    ):
        # ⚠️ 契约提醒：union response_model 依赖 Pydantic 的 isinstance 精确匹配。
        # 两个分支必须 return 对应 Pydantic 模型实例（不能 return dict），
        # 否则 QueryResponse 的 answer/citations 默认值会吞掉 StreamSubmitResponse 的 task_id。
        session_id = request.session_id or service.generate_session_id()
        original_query = request.query
        task_id = service.generate_task_id()
        is_stream = request.is_stream

        if is_stream:
            # 创建 SSE 队列（存放节点进度 / LLM 增量 / 最终答案）
            create_sse_queue(task_id)
            # 后台异步执行（避免阻塞事件循环）
            background_tasks.add_task(
                service.run_query_graph,
                original_query=original_query,
                session_id=session_id,
                task_id=task_id,
                is_stream=is_stream,
            )
            return StreamSubmitResponse(
                message="查询流程已启动，请通过 SSE 接收答案！",
                session_id=session_id,
                task_id=task_id,
            )
        else:
            # 非流式：放入默认线程池执行（LangGraph 同步代码不能阻塞事件循环）
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, service.run_query_graph,
                original_query, session_id, task_id, is_stream,
            )

            answer = service.get_task_result(task_id)
            citations = service.get_task_citations(task_id)
            return QueryResponse(message="最终答案", session_id=session_id,
                                 answer=answer, citations=citations)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """SSE 流式输出"""
        return StreamingResponse(sse_generator(task_id, request), media_type="text/event-stream")

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str,
            limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


def create_app() -> FastAPI:
    app = FastAPI(description="金融知识库-问答服务", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001)
