"""
导入流程的 MD 图片处理节点

职责：处理金融 Markdown 文档中的图片引用：
    1. 扫描 images 目录与 MD 中的图片标记，收集每张图的上下文（章节标题/上文/下文）
    2. 调用 VLM 视觉模型为图片生成中文摘要
    3. 上传图片到 MinIO，将 MD 中的图片引用替换为「摘要 + 远程URL」
    （金融文档含大量图表/截图，将其转写为可检索的语义文本描述）

任一子步骤失败均降级处理（保留原文），不阻断整体导入。
"""
import base64
import re
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息"""
    heading: str   # 最近的章节标题
    pre_text: str  # 图片上方正文
    post_text: str # 图片下方正文


@dataclass
class ImageInfo:
    """一张图片的完整信息"""
    name: str                # 图片文件名
    path: str                # 本地完整路径
    context: ImageContext    # 上下文


class MdFileHandler:
    """MD 文件读取与备份"""

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self, state) -> tuple[str, Path, Path]:
        """读取 MD 文件内容，返回 (内容, md路径对象, images目录路径对象)"""
        md_path = state.get("md_path", "")
        if not md_path:
            raise StateFieldError(self.node_name, "md_path", str, "属性不存在")

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(f"路径不存在：{md_path}", self.node_name)

        with open(md_path_obj, 'r', encoding="utf-8") as f:
            md_content = f.read()

        image_path_obj = md_path_obj.parent / "images"
        return md_content, md_path_obj, image_path_obj

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        """将处理后的 MD 另存为 <stem>_new.md"""
        new_file_path = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise
        return str(new_file_path)


class ImageScanner:
    """扫描 images 目录并定位图片在 MD 中的上下文"""

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, image_path_obj: Path, md_content: str,
                     image_extensions, img_content_length: int) -> List[ImageInfo]:
        imageinfo_list = []
        if not image_path_obj.exists():
            return imageinfo_list

        for image_path in image_path_obj.iterdir():
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in image_extensions:
                continue

            ctx = self._find_context(image_path.name, md_content, img_content_length)
            if ctx:
                imageinfo_list.append(ImageInfo(
                    name=image_path.name,
                    path=str(image_path),
                    context=ctx,
                ))
        return imageinfo_list

    def _find_context(self, image_name: str, md_content: str,
                      img_content_length: int) -> ImageContext | None:
        line_list = md_content.split("\n")
        for idx, line in enumerate(line_list):
            if re.search(r"^!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)", line):
                heading, pre_index = self._find_heading_above(idx, line_list)
                post_index = self._find_heading_below(idx, line_list)

                pre_text = self._extract_limited_context(line_list[pre_index + 1:idx], img_content_length, "front")
                post_text = self._extract_limited_context(line_list[idx + 1:post_index], img_content_length, "end")

                return ImageContext(heading=heading, pre_text=pre_text, post_text=post_text)
        return None

    def _find_heading_above(self, idx: int, line_list: List[str]) -> tuple[str, int]:
        for i in range(idx - 1, -1, -1):
            if re.match(r"^\s*#{1,6}\s+", line_list[i]):
                return line_list[i].strip(), i
        return "", -1

    def _find_heading_below(self, idx: int, line_list: List[str]) -> int:
        for i in range(idx + 1, len(line_list), 1):
            if re.match(r"^\s*#{1,6}\s+", line_list[i]):
                return i
        return len(line_list)

    def _extract_limited_context(self, context_list: List[str], limit: int, direction: str) -> str:
        """贪心策略截取最贴近图片的段落，控制总长度不超过 limit"""
        paragraphs: List[str] = []
        current: List[str] = []
        for line in context_list:
            is_blank = not line.strip()
            is_other_image = re.match(r"!\[.*?\]\(.*?\)", line)
            if is_blank or is_other_image:
                if current:
                    paragraphs.append("\n".join(current))
                    current = []
                continue
            current.append(line)
        if current:
            paragraphs.append("\n".join(current))

        if direction == "front":
            paragraphs.reverse()

        total, selected = 0, []
        for paragraph in paragraphs:
            if total + len(paragraph) >= limit and selected:
                break
            total += len(paragraph)
            selected.append(paragraph)

        if direction == "front":
            selected.reverse()
        return "\n\n".join(selected)


class VLMSummarizer:
    """调用 VLM 为图片生成摘要"""

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def summarize_all(self, file_title: str, imageinfo_list: List[ImageInfo],
                      vl_model: str) -> Dict[str, str]:
        summarizes: Dict[str, str] = {}
        try:
            client = AIClients.get_openai()
        except Exception as e:
            self.logger.warning(f"获取 VLM 客户端失败，图片全部降级为默认摘要: {e}")
            for image_info in imageinfo_list:
                summarizes[image_info.name] = "默认摘要"
            return summarizes

        for image_info in imageinfo_list:
            try:
                summarizes[image_info.name] = self._summarize_one(file_title, image_info, vl_model, client)
            except Exception as e:
                self.logger.warning(f"图片 {image_info.name} 摘要失败，降级: {e}")
                summarizes[image_info.name] = "默认摘要"
        return summarizes

    def _summarize_one(self, file_title: str, image_info: ImageInfo, vl_model: str, client: OpenAI) -> str:
        parts = [p for p in (image_info.context.heading, image_info.context.pre_text, image_info.context.post_text) if p]
        final_context = "\n".join(parts)

        with open(image_info.path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        resp = client.chat.completions.create(
            model=vl_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "任务：为金融文档中的图片生成一个简短的中文描述标题。\n"
                        "背景信息：\n"
                        f"  1. 所属文档标题：\"{file_title}\"\n"
                        f"  2. 图片上下文：{final_context}\n"
                        "请结合图片与上下文，用中文简要总结该图内容"
                        "（若是表格/走势图，说明其中反映的关键数据含义），"
                        "生成精准简短的标题（不要包含『图片』二字）。"
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()


class ImageUploader:
    """上传图片到 MinIO 并替换 MD 引用"""

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self, file_title: str, md_content: str,
                           imageinfo_list: List[ImageInfo],
                           summarizes: Dict[str, str],
                           minio_bucket: str, endpoint_minio_url: str) -> str:
        image_name_urls = self._upload_all(file_title, imageinfo_list, minio_bucket, endpoint_minio_url)
        return self._replace_in_md(md_content, summarizes, image_name_urls)

    def _upload_all(self, file_title: str, imageinfo_list: List[ImageInfo],
                    minio_bucket: str, endpoint_minio_url: str) -> Dict[str, str]:
        image_name_urls: Dict[str, str] = {}
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            self.logger.warning(f"获取 MinIO 客户端失败，图片保留本地路径: {e}")
            for image_info in imageinfo_list:
                image_name_urls[image_info.name] = image_info.path
            return image_name_urls

        for image_info in imageinfo_list:
            try:
                minio_client.fput_object(
                    bucket_name=minio_bucket,
                    object_name=f"{file_title}/{image_info.name}",
                    file_path=image_info.path,
                    content_type="image/jpeg",
                )
                image_name_urls[image_info.name] = f"{endpoint_minio_url}/{minio_bucket}/{file_title}/{image_info.name}"
            except Exception as e:
                self.logger.warning(f"图片 {image_info.name} 上传失败，保留本地路径: {e}")
                image_name_urls[image_info.name] = image_info.path
        return image_name_urls

    @staticmethod
    def _replace_in_md(md_content: str, summarizes: Dict[str, str],
                       remote_urls: Dict[str, str]) -> str:
        """将 ![默认](images/xx.jpg) 替换为 ![摘要](远程URL)"""
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            original_path = match.group(2).strip()
            file_name_in_md = Path(original_path).name
            for img_name, summary in summarizes.items():
                if img_name == file_name_in_md:
                    return f"![{summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)


class MarkDownImageNode(BaseNode):
    """MD 图片处理节点"""

    name: str = "md_img_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(self.logger, self.name)
        self.image_scanner = ImageScanner(self.logger, self.name)
        self.vlm_summarizer = VLMSummarizer(self.logger, self.name)
        self.image_uploader = ImageUploader(self.logger, self.name)

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        # 1. 读取 MD 文件
        md_content, md_path_obj, image_path_obj = self.md_file_handler.read_md(state)

        # 2. 无图片目录 -> 直接透传（纯文本金融文档常见）
        if not image_path_obj.exists():
            state['md_content'] = md_content
            return state

        # 3. 扫描图片
        imageinfo_list = self.image_scanner.scan_img_dir(
            image_path_obj, md_content,
            self.config.image_extensions, self.config.img_content_length,
        )

        # 4. VLM 生成摘要
        summarizes = self.vlm_summarizer.summarize_all(
            state['file_title'], imageinfo_list, self.config.vl_model,
        )

        # 5. 上传 MinIO & 替换引用（摘要+URL）
        new_md_content = self.image_uploader.upload_and_replace(
            state['file_title'],
            md_content,
            imageinfo_list,
            summarizes,
            self.config.minio_bucket,
            self.config.get_minio_base_url(),
        )

        # 6. 备份新文件
        self.md_file_handler.backup(md_path_obj, new_md_content)

        # 7. 更新状态
        state['md_content'] = new_md_content
        return state


if __name__ == "__main__":
    setup_logging()
    node = MarkDownImageNode()
    state = {"file_title": "某基金招募说明书", "md_path": r"E:\tmp\fin_import\xx.md"}
    node.process(state)
