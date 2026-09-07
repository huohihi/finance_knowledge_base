"""
导入流程的文档切分节点（金融领域）

职责：把 PDF/MD 转换得到的 Markdown 原文，基于【标题层级】切成带上下文的
章节块（chunk），并对超出长度上限的长章节做二次切分、对过短的章节做同父合并。

金融适配点：
    - 金融文档（说明书/招募书/研报）标题层级清晰：章节切块天然保留
      「产品名/基金类型/风险提示」等结构化上下文；
    - 处理 <table> 表格降维（费率、申赎规则等多为表格）；
    - 长切短合：招募说明书"投资策略"等大章节按标点二次切分；
      风险提示短句等不足阈值时向同父章节合并，避免碎块。

输出 chunk 结构（逐步富化）:
    {
        "title":       章节标题(可含 -N 分段标记),
        "parent_title": 父章节标题,
        "file_title":   文件标题,
        "content":      "标题\n\n正文",
        "part":         可选分段号
    }
"""
import json
import os
import re
from typing import Tuple, Dict, List, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    """文档切分节点"""

    name: str = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        # 1. 获取输入并校验
        md_content, file_title, max_content_length, min_content_length = self._get_input_validation(state)

        # 2. 按标题切分
        sections = self._split_by_title(md_content, file_title)

        # 3. 长切短合
        final_sections = self.split_and_merge(sections, max_content_length, min_content_length)

        # 4. 组装最终 chunk
        chunks = self._assemble_chunk(final_sections)

        # 5. 日志 + 备份
        self._log_summary(md_content, chunks, max_content_length)
        self._backup_chunks(state, chunks)

        state['chunks'] = chunks
        return state

    # ------------------------------------------------------------------ #
    #               Step 1: 输入校验                                      #
    # ------------------------------------------------------------------ #

    def _get_input_validation(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(self.name, "md_content", str)

        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(self.name, "file_title", str)

        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length

        if max_content_length < 0 or min_content_length < 0 or max_content_length <= min_content_length:
            raise ValidationError("切分最大值和最小值参数错误", self.name)

        return md_content, file_title, max_content_length, min_content_length

    # ------------------------------------------------------------------ #
    #               Step 2: 基于标题层级切分                               #
    # ------------------------------------------------------------------ #

    def _split_by_title(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        """
        根据标题切分文档，形成 section 列表。
        每个 section 携带 parent_title（最近的祖先标题），用于保持层级上下文。
        """
        sections: List[Dict[str, Any]] = []
        in_fence = False                       # 是否处于代码围栏内
        heading_re = re.compile(r"^\s*(#{1,6})\s+.+")
        body: List[str] = []
        content_lines = md_content.split("\n")
        current_title = ""
        current_level = 0
        hierarchy = [""] * 7                   # 各层级最近标题（下标即标题级别）

        def _flush():
            """把当前标题及其正文封装成一个 section"""
            if current_title or body:
                # 寻找最近的祖先标题作为 parent_title
                parent_title = ""
                for lev in range(current_level - 1, 0, -1):
                    if hierarchy[lev]:
                        parent_title = hierarchy[lev]
                        break
                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "parent_title": parent_title,
                    "title": current_title if current_title else file_title,
                    "body": "\n".join(body),
                    "file_title": file_title,
                })

        for line in content_lines:
            if line.startswith("~~~~") or line.startswith("```"):
                in_fence = not in_fence

            match = heading_re.match(line) if not in_fence else None

            if match:
                _flush()
                level = len(match.group(1))
                current_level = level
                current_title = line
                hierarchy[level] = current_title
                # 只清理比当前级别更深的子级别（同级别或祖先级别需保留给后续兄弟节使用）
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
                body = []
            else:
                body.append(line.strip())

        # 收尾：处理最后一个标题后的段落
        _flush()
        return sections

    # ------------------------------------------------------------------ #
    #               Step 3: 长切短合                                      #
    # ------------------------------------------------------------------ #

    def split_and_merge(self, sections: List[Dict[str, Any]],
                        max_content_length: int, min_content_length: int):
        """二次切分（长） + 同父合并（短）"""
        self.log_step("step3", "切分及合并...")

        # 1. 长章节切分
        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2. 短章节合并
        final_sections = self.merge_short_section(current_sections, min_content_length)
        return final_sections

    def split_long_section(self, section: Dict[str, Any],
                           max_content_length: int = 1000) -> List[Dict[str, Any]]:
        """切分超长章节（表格先降维、标题超长截断）"""
        title = section.get("title")
        parent_title = section.get("parent_title")
        body = section.get("body")
        file_title = section.get("file_title")

        # 表格降维
        if "<table>" in body or re.search(r"^\s*\|.*\|\s*$", body, re.MULTILINE):
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        # 标题长度限制
        MAX_TITLE_LENGTH = 50
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH]

        title_prefix = title + "\n\n"

        # 未超长直接返回（仍以列表返回以统一接口）
        if len(title_prefix) + len(body) <= max_content_length:
            return [section]

        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        sub_sections = []
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", ".", ",", "!", " ", ""],
            chunk_size=body_length,
            chunk_overlap=0,
            keep_separator=False,
        )
        texts = splitter.split_text(body)
        if len(texts) <= 1:
            return [section]

        for index, text in enumerate(texts):
            sub_sections.append({
                "parent_title": parent_title,
                "title": section.get("title") + f"-{index + 1}",
                "body": text,
                "file_title": file_title,
                "part": f"{index + 1}",
            })
        return sub_sections

    def merge_short_section(self, current_sections: List[Dict[str, Any]],
                            min_content_length: int) -> List[Dict[str, Any]]:
        """将同父标题下的过短正文合并为完整章节"""
        final_sections: List[Dict[str, Any]] = []

        current_section = current_sections[0]
        for next_section in current_sections[1:]:
            same_parent = current_section["parent_title"] == next_section["parent_title"]
            if same_parent and len(current_section['body']) < min_content_length:
                current_section['body'] = current_section['body'].rstrip() + "\n\n" + next_section['body'].lstrip()
                current_section['title'] = current_section['parent_title']
                current_section['part'] = 0
            else:
                final_sections.append(current_section)
                current_section = next_section
        final_sections.append(current_section)

        # 对带 part 标记的 section 重新编号
        part_counter: Dict[str, int] = {}
        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                final_section['title'] = final_section['title'] + f"- {new_part}"

        return final_sections

    # ------------------------------------------------------------------ #
    #               Step 4: 组装最终 chunk                                #
    # ------------------------------------------------------------------ #

    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组装最终的 chunk 列表"""
        self.log_step("step4", "组装最终的切片信息...")
        chunks = []

        for chunk in final_chunks:
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{title}\n\n{body}"

            assemble_chunk = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }
            if "part" in chunk:
                assemble_chunk['part'] = chunk.get('part')

            chunks.append(assemble_chunk)
        return chunks

    # ------------------------------------------------------------------ #
    #                       日志 & 备份                                   #
    # ------------------------------------------------------------------ #

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")
        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件（便于人工核查切片质量）"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()
    node = DocumentSplitNode()
    state = {
        "file_title": "某基金招募说明书",
        "md_content": "# 第一章 概览\n\n本基金为混合型基金...\n\n## 风险提示\n\n基金有风险，投资需谨慎...",
        "file_dir": r"E:\tmp\fin_import",
    }
    node.process(state)
