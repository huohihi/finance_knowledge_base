"""
导入流程的入口节点

职责：校验输入（文件路径/工作目录存在），识别文件类型：
    - .pdf -> 开启 PDF 解析链路
    - .md  -> 直接进入 MD 读取链路
提取 file_title（不含扩展名）供下游使用。
"""
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    name: str = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        # 1. 校验入参
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")
        if not import_file_path or not file_dir:
            raise ValidationError("数据校验失败：import_file_path/file_dir 缺失", self.name)

        import_file_path_obj = Path(import_file_path)

        # 2. 识别文件类型
        suffix = import_file_path_obj.suffix.lower()
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise ValidationError("文件类型暂不支持（仅支持 PDF/Markdown）", self.name)

        # 3. 提取文件标题
        file_title = import_file_path_obj.stem
        state["file_title"] = file_title
        state["document_title"] = file_title  # 初始值，后续由元数据抽取节点规范

        return state
