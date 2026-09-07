"""
查询流程的答案输出节点

职责：将重排序后的金融证据文档 + 历史对话组装为提示词，调用 LLM 生成最终答案。

金融领域关键实现点：
    1. 证据格式化带 [N] 编号与引用元数据标签，约束 LLM 用 [N] 标注引用，
       满足需求文档 6.1（结构清晰）/6.2（引用来源）/6.3（无资料不编造）；
    2. 无检索证据（reranked_docs 为空）时，不调用 LLM，直接返回
       "当前知识库中未检索到足够信息..." 的合规话术；
    3. 流式输出：逐 token 通过 SSE push（LLM 输出增量走队列）；
    4. 回答与引用来源写入 Mongo 历史，支撑多轮追问（需求 3.6 / 8）；
    5. 引用来源结构化为 citations 列表（含 document_title/content_type/product_code/...），
       随 SSE FINAL 事件与 /query 响应回传前端（前端展示引用来源）。
"""
import json
from typing import List, Dict, Tuple

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    """生成答案节点"""

    name = "answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id = state.get("session_id")
        task_id = state.get("task_id")

        # 1. 已有答案（实体澄清/无法识别短路路径）-> 直接推送
        if state.get("answer"):
            self._push_existing_answer(state)
        else:
            # 2. 无检索证据 -> 返回合规兜底话术（需求 6.3）
            if not state.get("reranked_docs"):
                state["answer"] = (
                    "当前知识库中未检索到足够信息，建议查看正式产品文件、"
                    "公告原文或咨询相关工作人员。"
                )
                set_task_result(task_id, "answer", state["answer"])
            else:
                # 3. 有证据 -> 组装提示词并生成答案
                prompt = self._build_prompt(state)
                self._generate_answer(state, prompt)

        # 4. 结构化引用来源（供前端展示），非流式场景持久化供 /query 返回
        state["citations"] = self._collect_citations(state.get("reranked_docs") or [])
        if not state.get("is_stream") and task_id:
            set_task_result(task_id, "citations", json.dumps(state["citations"], ensure_ascii=False))

        # 5. 写入历史（用户问题 + 助手答案，供多轮追问）
        self._write_history(state)

        return state

    def _push_existing_answer(self, state):
        """处理已在实体确认节点生成的答案（澄清话术）"""
        if not state.get("is_stream"):
            set_task_result(state['task_id'], "answer", state['answer'])

    # ================================================================== #
    #                    提示词组装                                        #
    # ================================================================== #

    def _build_prompt(self, state: QueryGraphState) -> str:
        char_budget = self.config.max_context_chars

        question = state.get("rewritten_query") or state.get("original_query", "")
        entity_names = state.get("entity_names") or []

        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            entity_names="、".join(entity_names) if entity_names else "（通用金融知识，未限定具体产品）",
            question=question,
        )

    def _format_reranked_docs(self, reranked_docs: List[Dict],
                              char_budget: int) -> Tuple[str, int]:
        """证据文档带编号与元数据标签格式化，并受字符预算控制"""
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = (doc.get("content") or "").strip()
            if not content:
                continue

            # 编号 + 元数据标签
            meta_tags = [f"[{idx}]", f"[source={doc.get('source', 'local')}]"]
            if doc.get("source") == "web":
                if doc.get("url"):
                    meta_tags.append(f"[url={doc['url']}]")
            else:
                for field, template in [
                    ("chunk_id", "[chunk_id={}]"),
                    ("document_title", "[资料={}]"),
                    ("product_name", "[产品={}]"),
                    ("product_code", "[代码={}]"),
                    ("institution_name", "[机构={}]"),
                    ("risk_level", "[风险={}]"),
                    ("content_type", "[类型={}]"),
                    ("publish_date", "[发布={}]"),
                    ("entry_name", "[条目={}]"),
                    ("source_file", "[来源文件={}]"),
                ]:
                    value = str(doc.get(field, "")).strip()
                    if value:
                        meta_tags.append(template.format(value))

            score = doc.get("score")
            if score is not None:
                meta_tags.append(f"[score={float(score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break
            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict],
                             char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0
        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue
            line = f"{role_label_map[role]}: {text}"
            if used_chars + len(line) > char_budget:
                break
            formatted_lines.append(line)
            used_chars += len(line) + 1

        return "\n".join(formatted_lines), char_budget - used_chars

    # ================================================================== #
    #                    答案生成                                          #
    # ================================================================== #

    def _generate_answer(self, state, prompt: str):
        try:
            llm_client = AIClients.get_llm_openai(response_format=False)
        except Exception as e:
            state["answer"] = "抱歉，获取 LLM 客户端失败，无法生成答案！"
            return

        if state.get("is_stream"):
            state["answer"] = self._generate_answer_stream(llm_client, state, prompt)
        else:
            state["answer"] = self._generate_answer_invoke(llm_client, state, prompt)
            set_task_result(state['task_id'], "answer", state['answer'])

    def _generate_answer_stream(self, llm_client, state, prompt: str) -> str:
        try:
            answer = ""
            for chunk in llm_client.stream(prompt):
                delta_text = chunk.content
                answer += delta_text
                push_sse_event(
                    task_id=state.get("task_id"),
                    event=SSEEvent.DELTA,
                    data={"delta": delta_text},
                )
            return answer
        except Exception as e:
            self.logger.warning(f"stream 生成答案失败: {e}")
            return "抱歉，stream 生成答案失败！"

    def _generate_answer_invoke(self, llm_client, state, prompt: str) -> str:
        try:
            llm_response = llm_client.invoke(prompt)
            return llm_response.content.strip()
        except Exception as e:
            self.logger.warning(f"invoke 生成答案失败: {e}")
            return "抱歉，invoke 生成答案失败！"

    # ================================================================== #
    #                    引用来源 / 历史                                  #
    # ================================================================== #

    def _collect_citations(self, reranked_docs: List[Dict]) -> List[Dict]:
        """把本地证据的引用元数据整理成结构化的 citations 列表（去重）"""
        seen = set()
        citations = []
        for doc in reranked_docs:
            if doc.get("source") != "local":
                continue
            citation = {
                "document_title": doc.get("document_title", ""),
                "content_type": doc.get("content_type", ""),
                "product_name": doc.get("product_name", ""),
                "product_code": doc.get("product_code", ""),
                "institution_name": doc.get("institution_name", ""),
                "risk_level": doc.get("risk_level", ""),
                "publish_date": doc.get("publish_date", ""),
                "entry_name": doc.get("entry_name", ""),
                "source_file": doc.get("source_file", ""),
                "title": doc.get("title", ""),
            }
            dedup_key = (citation["document_title"], citation["entry_name"], citation["source_file"])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            citations.append(citation)
        return citations

    def _write_history(self, state):
        """把用户问题与助手答案写入 Mongo，支撑多轮追问"""
        try:
            session_id = state.get("session_id")
            save_chat_message(
                session_id=session_id,
                role="user",
                text=state.get("original_query", ""),
                rewritten_query=state.get("rewritten_query") or state.get("original_query", ""),
                entity_names=state.get("entity_names", []),
            )
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state.get("answer", ""),
                rewritten_query=state.get("rewritten_query") or state.get("original_query", ""),
                entity_names=state.get("entity_names", []),
            )
        except Exception as e:
            self.logger.error(f"保存会话失败: {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    mock_state = {
        "task_id": "test-fin-001",
        "session_id": "test-session-001",
        "is_stream": False,
        "original_query": "某某纯债基金的风险等级？",
        "rewritten_query": "某某纯债债券型证券投资基金的风险等级是几级？",
        "entity_names": ["某某纯债债券型证券投资基金"],
        "reranked_docs": [
            {
                "content": "本基金为债券型基金，产品风险等级为 R2，属于中低风险等级，适合稳健型投资者。",
                "source": "local",
                "chunk_id": 101,
                "document_title": "某某纯债基金产品资料概要",
                "product_name": "某某纯债债券型证券投资基金",
                "product_code": "000001",
                "institution_name": "某某基金管理有限公司",
                "risk_level": "R2",
                "content_type": "基金产品资料概要",
                "source_file": "某某纯债基金产品资料概要.pdf",
                "score": 0.9234,
            }
        ],
        "history": [],
    }
    result = AnswerOutputNode()(mock_state)
    print(result.get("answer"))
