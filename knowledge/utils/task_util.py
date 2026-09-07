"""
任务状态追踪工具（内存版）

以 task_id 为键，追踪导入/查询流程的节点运行进度，供前端轮询 /status。
    - running_list：正在运行的节点（中文名）
    - done_list：已完成节点
    - durations：各节点耗时
    - result：任务结果（如 answer）
"""
from collections import defaultdict
from typing import Dict, List

# 只要访问不存在的 key，自动初始化为 []
_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
_tasks_duration: Dict[str, Dict[str, float]] = defaultdict(dict)

# 任务结果 {task_id: {"answer": "..."}}
_tasks_result: Dict[str, Dict[str, str]] = defaultdict(dict)

_tasks_status: Dict[str, str] = {}

TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

_NODE_NAME_TO_CN: Dict[str, str] = {
    # --- 导入流程节点 ---
    "upload_file": "上传文件",
    "entry_node": "检查文件",
    "pdf_to_md_node": "PDF转Markdown",
    "md_img_node": "图片处理",
    "document_split_node": "文档切分",
    "finance_meta_extract_node": "金融元数据识别",
    "bge_embedding_node": "向量生成",
    "import_milvus_node": "导入向量库",
    "__end__": "处理完成",
    # --- 查询流程节点 ---
    "entity_confirm": "识别金融实体",
    "answer_output": "生成答案",
    "rerank": "重排序",
    "rrf": "倒排融合",
    "web_search_mcp": "网络搜索",
    "search_embedding": "切片检索",
    "search_embedding_hyde": "假设性文档检索",
}


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)


def get_running_task_list(task_id: str) -> List[str]:
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def get_done_task_list(task_id: str) -> List[str]:
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_task_status(task_id: str) -> str:
    return _tasks_status.get(task_id, "")


def update_task_status(task_id: str, status_name: str) -> None:
    _tasks_status[task_id] = status_name


def set_task_result(task_id: str, key: str, value: str) -> None:
    """存储任务结果字段（如 answer）"""
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    return _tasks_result.get(task_id, {}).get(key, default)


def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    cn_name = _to_cn(node_name)
    _tasks_duration[task_id][cn_name] = round(duration, 2)


def get_node_durations(task_id: str) -> Dict[str, float]:
    return dict(_tasks_duration.get(task_id, {}))


def get_task_info(task_id: str) -> Dict[str, any]:
    """任务全局信息（状态 + 运行中 + 已完成 + 耗时）"""
    return {
        "status": get_task_status(task_id),
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id),
        "durations": get_node_durations(task_id),
    }
