"""
导入 API（FastAPI 应用，默认端口 8000）

路由：
    GET  /import                导入管理页
    POST /upload                上传金融文档（PDF/MD），后台启动导入流程
    GET  /status/{task_id}      轮询任务进度（running/done 节点 + 耗时）

交互模式（与 0525 一致）：
    前端 POST /upload 立即拿到 task_id，随后 GET /status/{task_id} 轮询进度。
"""
import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.params import Depends
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.services.file_import_service import ImportFileService
from knowledge.utils.task_util import get_task_info


def register_router(app: FastAPI):
    @app.get("/import")
    def import_page():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file(
            background_tasks: BackgroundTasks,
            service: ImportFileService = Depends(get_import_file_service),
            file: UploadFile = File(...),
    ):
        # 双写：本地 + MinIO；立即返回 task_id
        task_id, file_dir, import_file_path = service.upload_file(file)

        # 异步导入：先返回，避免用户长时间等待
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)

        return UploadResponse(task_id=task_id, message="文件上传处理完成")

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status(task_id: str):
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)


def create_app() -> FastAPI:
    app = FastAPI(description="金融知识库-导入服务", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载前端静态资源
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app


if __name__ == "__main__":
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000)
