"""
文件导入服务

职责：
    - 接收上传文件：本地双写（本地临时目录 + MinIO 原始文件桶）
    - 启动导入 LangGraph（后台任务执行，逐节点透传进度）
    - 任务状态管理
"""
import datetime
import logging
import os
import shutil
import uuid

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.main_graph import kb_import_process_graph
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import (add_done_task, add_running_task, update_task_status,
                                       TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, TASK_STATUS_PROCESSING)

logger = logging.getLogger(__name__)


class ImportFileService:
    """金融文件导入服务"""

    def upload_file(self, file: UploadFile):
        """
        文件上传业务处理。
        双写策略：本地（供导入 pipeline 读取）+ MinIO（原始文件归档）。
        返回 (task_id, file_dir, import_file_path)。
        """
        # 1. 生成任务 ID 并创建本地工作目录
        task_id = self._generate_task_id()
        file_dir = os.path.join(get_local_base_dir(), task_id)
        os.makedirs(file_dir, exist_ok=True)
        add_running_task(task_id, "upload_file")

        # 2. 双写：本地 + MinIO
        try:
            import_file_path = self._upload_file_to_local(file, file_dir)
            self._upload_file_to_minio(import_file_path, file.filename)
        finally:
            add_done_task(task_id, "upload_file")

        return task_id, file_dir, import_file_path

    def run_import_graph(self, task_id: str, file_dir: str, import_file_path: str):
        """后台启动导入流程（供 FastAPI BackgroundTasks 调用）"""
        init_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir,
        }
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        try:
            # 逐节点 stream 执行（节点内部的进度追踪由 BaseNode 完成）
            for event in kb_import_process_graph.stream(init_state):
                for node_name, process_state in event.items():
                    print(f"{task_id} - 运行节点：{node_name}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
            logger.info(f"任务 {task_id} 导入完成")
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.error(f"导入流程执行出错: {e}")

    def _generate_task_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _upload_file_to_local(self, file: UploadFile, file_dir: str) -> str:
        os.makedirs(file_dir, exist_ok=True)
        try:
            import_file_path = os.path.join(file_dir, file.filename)
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            logger.info(f"文件已保存本地: {import_file_path}")
            return import_file_path
        except Exception as e:
            logger.error(f"文件保存本地出错: {e}")
            raise FileProcessingError(f"文件保存本地出错: {e}")

    def _upload_file_to_minio(self, import_file_path: str, file_name: str):
        """上传原始文件到 MinIO（失败仅告警，不阻断本地导入）"""
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            logger.warning(f"获取 MinIO 客户端失败，跳过归档: {e}")
            return
        try:
            config = get_config()
            obj_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{file_name}"
            minio_client.fput_object(config.minio_bucket, obj_name, import_file_path)
            logger.info(f"原始文件已归档 MinIO: {obj_name}")
        except Exception as e:
            logger.warning(f"文件上传 MinIO 失败（不影响本地导入）: {e}")
