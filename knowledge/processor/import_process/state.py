"""
导入流程状态类型定义

定义完整的金融文档导入状态结构与辅助函数。
"""
import copy
from typing import TypedDict, List, Dict, Any


class ImportGraphState(TypedDict, total=False):
    """导入流程图状态

    包含整个导入流程中传递的所有数据。
    """
    # ==================== 任务标识 ====================
    task_id: str                     # 任务 ID（web 交互时按任务查询处理进度）
    session_id: str                  # 会话 ID（备用）

    # ==================== 控制标志 ====================
    is_md_read_enabled: bool         # 是否启用 MD 读取
    is_pdf_read_enabled: bool        # 是否启用 PDF 读取

    # ==================== 路径信息 ====================
    import_file_path: str            # 原始上传文件路径
    file_dir: str                    # 任务工作目录（本地 + 输出落盘）
    pdf_path: str                    # PDF 文件路径
    md_path: str                     # 转换后 Markdown 文件路径

    # ==================== 文件信息 ====================
    file_title: str                  # 文件标题（不含扩展名）
    document_title: str              # 资料名称（金融元数据，规范名）

    # ==================== 处理中间数据 ====================
    md_content: str                  # Markdown 文档内容
    chunks: List[Dict[str, Any]]     # 文档切片列表（逐步富化：标题->元数据->向量->id）
    finance_meta: Dict[str, Any]     # 文档级金融元数据（content_type/document_title/risk_level...）
    entities: List[Dict[str, Any]]   # 抽取的金融实体列表（写入实体集合）


# ==================== 默认状态 ====================
GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "session_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "file_dir": "",
    "import_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "document_title": "",
    "md_content": "",
    "chunks": [],
    "finance_meta": {},
    "entities": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态，支持覆盖"""
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """获取默认状态副本（避免全局污染）"""
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
