"""
Markdown 表格降维工具

金融文档（产品说明书/招募书/费率表）含大量 HTML/原生 Markdown 表格，
直接切片会破坏表格结构导致检索与问答质量下降。本工具将表格"降维"
为一行一条的键值式自然语言，便于向量化与检索。

支持：
    - HTML 表格（含合并单元格 rowspan/colspan、无表头 K-V 表、左上角空置交叉表）
    - 原生 Markdown 管道表格
"""
import re
from typing import List

from bs4 import BeautifulSoup


class MarkdownTableLinearizer:
    """表格降维器"""

    HTML_TABLE_PATTERN = re.compile(r"<table.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
    MD_TABLE_PATTERN = re.compile(
        r'((?:^[ \t]*\|.*\|[ \t]*\n)'
        r'(?:^[ \t]*\|[ \t]*[-:]+[-| :]*\|[ \t]*\n)'
        r'(?:^[ \t]*\|.*\|[ \t]*(?:\n|$))*)',
        re.MULTILINE
    )

    @classmethod
    def process(cls, content: str) -> str:
        """对输入文本中的表格做降维处理"""
        if not content:
            return content

        if "<table" in content.lower():
            content = cls.HTML_TABLE_PATTERN.sub(cls._replace_html_table, content)

        if "|" in content:
            content = cls.MD_TABLE_PATTERN.sub(cls._replace_md_table, content)

        return content

    @classmethod
    def _replace_html_table(cls, match) -> str:
        html_content = match.group(0)
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table")
        if not table:
            return html_content

        rows = table.find_all("tr")
        if not rows:
            return html_content

        # 嗅探是否使用 <th> 表头标签
        has_th = len(table.find_all("th")) > 0

        grid: List[List] = [[] for _ in rows]
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(['td', 'th']):
                while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                    col_idx += 1

                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                text = cell.get_text(separator=" ", strip=True)

                for r in range(row_idx, row_idx + rowspan):
                    while len(grid) <= r:
                        grid.append([])
                    while len(grid[r]) < col_idx + colspan:
                        grid[r].append(None)
                    for c in range(col_idx, col_idx + colspan):
                        grid[r][c] = text
                col_idx += colspan

        return cls._grid_to_text(grid, is_md=False, has_th=has_th)

    @classmethod
    def _replace_md_table(cls, match) -> str:
        md_text = match.group(0).strip()
        lines = md_text.split('\n')
        grid = []
        for line in lines:
            if re.match(r'^[ \t]*\|[ \t\-|:]+\|[ \t]*$', line):
                continue
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            grid.append(cells)
        return cls._grid_to_text(grid, is_md=True, has_th=False)

    @classmethod
    def _grid_to_text(cls, grid: List[List[str]], is_md: bool, has_th: bool) -> str:
        if not grid or not grid[0]:
            return ""

        cols_count = max(len(r) for r in grid)
        for r in grid:
            while len(r) < cols_count:
                r.append("")

        # 判断首行是否为表头
        is_header_row = is_md or has_th or (grid[0][0] == "") or (cols_count > 2)

        res = []
        if not is_header_row and cols_count == 2:
            # 策略1：纯 K-V 表格（如费率、要素速览）
            for r in grid:
                k, v = r[0], r[1]
                if k or v:
                    res.append(f"- 【{k if k else '未知属性'}】：{v if v else '无'}。")
        else:
            # 策略2：标准/交叉表头表格
            headers = grid[0]
            for r in grid[1:]:
                if not any(r):
                    continue
                subject = r[0] if r[0] else "未知项目"
                subject_header = headers[0] if headers[0] else ""

                props = []
                for c in range(1, cols_count):
                    head = headers[c] if headers[c] else f"属性{c}"
                    val = r[c] if r[c] else ""
                    if val and val not in ('-', '/', '\\', '无'):
                        props.append(f"{head}为{val}")
                if props:
                    prop_str = "，".join(props)
                    if subject_header:
                        res.append(f"- 【{subject}】(对应{subject_header})：{prop_str}。")
                    else:
                        res.append(f"- 【{subject}】：{prop_str}。")
                else:
                    if subject != "未知项目":
                        res.append(f"- 【{subject}】")

        return "\n\n" + "\n".join(res) + "\n\n"
