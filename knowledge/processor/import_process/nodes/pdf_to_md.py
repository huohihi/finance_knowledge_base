"""
导入流程的 PDF -> Markdown 转换节点

职责：调用 MinerU（本地 CLI）将金融 PDF（产品说明书、公告、研报、风险揭示书等）
转换为 Markdown，输出路径为 <file_dir>/<文件名>/hybrid_auto/<文件名>.md。

失败（返回码非0 / 未找到 mineru）时抛出 PdfConversionError，
由 import 图捕获后结束任务，不影响整体服务。
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        # 1. 校验路径
        import_file_path_obj, file_dir_obj = self._validate_state_inputs_path(state)

        # 2. 执行 MinerU 转换
        code = self._execute_mineru(import_file_path_obj, file_dir_obj)
        if code != 0:
            raise PdfConversionError("PDF 转换 Markdown 失败", self.name)

        # 3. 定位生成的 MD 路径
        md_path_obj = self._get_md_path(import_file_path_obj, file_dir_obj)
        state["md_path"] = str(md_path_obj)
        return state

    # ------------------------------------------------------------------ #
    #                        私有方法                                     #
    # ------------------------------------------------------------------ #

    def _validate_state_inputs_path(self, state) -> Tuple[Path, Path]:
        import_file_path = state.get("import_file_path", "")
        if not import_file_path:
            raise ValidationError("import_file_path 参数不存在", self.name)

        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError("PDF 文件不存在", self.name)

        file_dir = state.get("file_dir", "")
        if not file_dir:
            file_dir = str(import_file_path_obj.parent)
        file_dir_obj = Path(file_dir)
        return import_file_path_obj, file_dir_obj

    def _execute_mineru(self, import_file_path_obj: Path, file_dir_obj: Path) -> int:
        self.log_step("step2", "执行 MinerU 解析 PDF")
        process_start_time = time.time()

        # 定位 mineru 可执行文件（优先 PATH，其次当前 Python 环境 Scripts）
        mineru_exe = shutil.which("mineru")
        if not mineru_exe:
            candidate = Path(sys.executable).parent / "Scripts" / "mineru.exe"
            if candidate.exists():
                mineru_exe = str(candidate)
        if not mineru_exe:
            raise PdfConversionError(
                "未找到 mineru 可执行文件，请确认 mineru 已安装并处于当前 Python 环境", self.name
            )

        proc = subprocess.Popen(
            args=[mineru_exe, "-p", str(import_file_path_obj), "-o", str(file_dir_obj), "--source=local"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        # 实时透传 MinerU 输出
        for line in proc.stdout:
            print(line.rstrip())
        return_code = proc.wait()
        process_end_time = time.time()

        if return_code == 0:
            self.logger.info(
                f"MinerU 成功解析 PDF 文件：{import_file_path_obj.name} "
                f"耗时:{process_end_time - process_start_time:.2f}s"
            )
        else:
            self.logger.error(f"MinerU 解析 PDF 文件：{import_file_path_obj.name} 失败")
        return return_code

    def _get_md_path(self, import_file_path_obj: Path, file_dir_obj: Path) -> Path:
        """MinerU 默认输出目录结构: <out>/<stem>/hybrid_auto/<stem>.md"""
        return file_dir_obj / import_file_path_obj.stem / "hybrid_auto" / (import_file_path_obj.stem + ".md")


if __name__ == "__main__":
    setup_logging()
    node = PdfToMdNode()
    state = {
        "import_file_path": r"E:\doc\某理财产品说明书.pdf",
        "file_dir": r"E:\tmp\fin_import",
    }
    node.process(state)
