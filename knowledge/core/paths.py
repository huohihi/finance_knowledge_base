"""
路径常量：工程目录相关
"""
import os

# 项目 knowledge 包根目录
KNOWLEDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 工程根目录（finance_knowledge_base/），.env 位于此处
PROJECT_ROOT = os.path.abspath(os.path.join(KNOWLEDGE_ROOT, ".."))

# .env 环境变量文件（显式定位，不依赖当前工作目录）
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# 本地文件存储基础目录（上传原文件 + 处理中间产物）
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")

# 前端页面静态资源目录
FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")


def get_local_base_dir() -> str:
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    return FRONT_PAGE_DIR
