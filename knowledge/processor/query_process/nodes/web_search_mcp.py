"""
查询流程的联网检索节点（三路检索第三路，可选）

职责：当本地知识库无法覆盖（如最新政策/行情/公告时效性内容）时，
通过 DashScope MCP WebSearch 服务补充实时信息。

金融合规说明：需求文档 7.4「不保证实时性」——联网结果仅作补充参考，
answer 节点会提示用户以官方渠道为准；同时 Web 结果走 rerank 时标 source="web"。
"""
import asyncio
import json
from typing import Dict, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState

try:
    from agents.mcp import MCPServerStreamableHttp
except ImportError:  # 未安装 openai-agents 时降级：节点直接返回空
    MCPServerStreamableHttp = None


class WebSearchMcpNode(BaseNode):
    """网络搜索（MCP）节点"""

    name = "web_search_mcp"

    def process(self, state: QueryGraphState) -> Dict:
        # 未配置 MCP 服务或未安装依赖时安全降级
        if MCPServerStreamableHttp is None:
            return {"web_search_docs": []}

        validated_rewritten_query, _ = self._validate_input(state)

        # 仅在配置了 MCP 地址且不是空值时才真正联网
        if not self.config.mcp_dashscope_base_url or not self.config.mcp_dashscope_api_key:
            return {"web_search_docs": []}

        try:
            web_search_docs = asyncio.run(self._web_mcp(validated_rewritten_query))
        except Exception as e:
            self.logger.warning(f"网络搜索失败（不影响本地检索）: {e}")
            web_search_docs = []
        return {"web_search_docs": web_search_docs}

    def _validate_input(self, state) -> Tuple[str, list]:
        rewritten_query = state.get("rewritten_query") or state.get("original_query", "")
        entity_names = state.get("entity_names") or []
        return rewritten_query, entity_names

    async def _web_mcp(self, query: str):
        if MCPServerStreamableHttp is None:
            return []

        async with MCPServerStreamableHttp(
                name="金融联网搜索",
                params={
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": {"Authorization": f"Bearer {self.config.mcp_dashscope_api_key}"},
                    "timeout": 60,
                    "terminate_on_close": True,
                },
                max_retry_attempts=1,
                client_session_timeout_seconds=30,
                cache_tools_list=True,
        ) as client:
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 3},
            )
            if not execute_tool_result or not execute_tool_result.content or not execute_tool_result.content[0]:
                return []

            text_json = json.loads(execute_tool_result.content[0].text)
            pages = (text_json or {}).get("pages") or []
            return [
                {
                    "snippet": page.get("snippet", ""),
                    "title": page.get("title", ""),
                    "url": page.get("url", ""),
                }
                for page in pages
            ]
