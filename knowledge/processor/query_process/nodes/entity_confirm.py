"""
查询流程的金融实体识别与对齐节点

职责：把用户自然语言问题转换为可用于 Milvus 过滤检索的【金融实体】键。

三步式（对齐 0525 item_name_confirm 的 Extract -> Align -> Decide 模式）：
    Step1 意图/实体抽取：LLM 从「当前问题 + 最近历史」识别
          query_type（product/announcement/industry/general）、
          entity_names、rewritten_query；
    Step2 实体对齐：把 LLM 抽取的实体与 Milvus 实体集合做混合向量检索，
          按分数分为 confirmed（高置信，直接作为过滤键）与 options（待澄清候选）；
    Step3 决策：
          - confirmed 非空  -> 走三路检索（按实体过滤正文）
          - confirmed 空且无候选 -> 若 query_type=general（金融知识/FAQ/流程），
            降级为全库通用检索（需求 2.3/2.7 场景必须支持）
          - options 非空 -> 生成澄清答案，跳过检索
          - 完全无法理解 -> 礼貌话术

金融适配点：产品代码/机构等多字段冗余在实体行，便于对齐后直接展示。
"""
import json
import re
from json import JSONDecodeError
from typing import List, Dict, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import AnnSearchRequest, MilvusClient

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import (
    ENTITY_EXTRACT_SYSTEM_PROMPT, ENTITY_EXTRACT_TEMPLATE,
    ENTITY_CLARIFY_TEMPLATE, ENTITY_NOT_FOUND_TEMPLATE,
)
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


class EntityExtractor:
    """LLM 意图识别 + 实体抽取 + 问题改写"""

    def __init__(self, logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def extract(self, original_query: str, history_context: str) -> Dict[str, Any]:
        """
        返回:
            {
                "query_type": "product" | "announcement" | "industry" | "general",
                "entity_names": ["..."],
                "rewritten_query": "..."
            }
        """
        default_result = {
            "query_type": "general",
            "entity_names": [],
            "rewritten_query": original_query,
        }
        llm_client = AIClients.get_llm_openai(response_format=True)
        if llm_client is None:
            return default_result

        human_prompt = ENTITY_EXTRACT_TEMPLATE.format(
            history_text=history_context or "暂无历史对话",
            query=original_query,
        )
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=ENTITY_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=human_prompt),
            ])
            content = llm_response.content.strip()
            parsed = self._clean_parse(content)

            result = {
                "query_type": parsed.get("query_type") or "general",
                "entity_names": parsed.get("entity_names") or [],
                "rewritten_query": parsed.get("rewritten_query") or original_query,
            }
            # 合法性收口
            if result["query_type"] not in ("product", "announcement", "industry", "general"):
                result["query_type"] = "general"
            return result
        except Exception as e:
            self.logger.error(f"LLM 实体抽取失败，降级: {e}")
            return default_result

    @staticmethod
    def _clean_parse(llm_content: str) -> Dict[str, Any]:
        """清洗 JSON 围栏并反序列化"""
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(content)
        except JSONDecodeError as e:
            raise ValueError(f"LLM 输出 JSON 反序列化失败: {e}")

        raw_names = parsed.get('entity_names') or []
        clean_names = [n.strip() for n in raw_names if isinstance(n, str) and n.strip()]
        raw_query = parsed.get('rewritten_query') or ""
        raw_type = parsed.get('query_type') or "general"

        return {
            "query_type": raw_type.strip(),
            "entity_names": clean_names,
            "rewritten_query": raw_query.strip(),
        }


class EntityAligner:
    """实体对齐器：与 Milvus 实体集合做向量匹配 -> 评分对齐 -> 候选收口"""

    def __init__(self, logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def match_align_filter(self, entity_names: List[str],
                           entity_collection: str) -> Tuple[List[str], List[str]]:
        """返回 (confirmed, options)"""
        if not entity_names:
            return [], []

        search_results = self._match_vector(entity_names, entity_collection)
        confirmed, options = self._score_align(search_results)
        if len(confirmed) > 1:
            confirmed = self._score_filter(confirmed, search_results)
        return confirmed, options

    def _match_vector(self, entity_names: List[str],
                      entity_collection: str) -> List[Dict[str, Any]]:
        """将 LLM 抽取的实体向量化并逐一对齐实体集合"""
        result: List[Dict[str, Any]] = []
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            milvus_client: MilvusClient = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"实体对齐客户端获取失败: {e}")
            return result
        if bge_m3_client is None or milvus_client is None:
            return result

        try:
            embedding_result = generate_bge_m3_hybrid_vectors(bge_m3_client, entity_names, is_query=True)
        except Exception as e:
            self.logger.error(f"实体向量化失败: {e}")
            return result

        for index, extracted_name in enumerate(entity_names):
            try:
                hybrid_search_requests: List[AnnSearchRequest] = create_hybrid_search_requests(
                    dense_vector=embedding_result['dense'][index],
                    sparse_vector=embedding_result['sparse'][index],
                )
                hits = execute_hybrid_search_query(
                    milvus_client,
                    collection_name=entity_collection,
                    search_requests=hybrid_search_requests,
                    ranker_weights=(0.5, 0.5),
                    norm_score=True,
                    limit=5,
                    output_fields=["entity_name", "entity_type", "product_code",
                                   "institution_name", "risk_level"],
                )
            except Exception as e:
                self.logger.error(f"实体 [{extracted_name}] 对齐检索失败: {e}")
                continue

            result.append({
                "extracted_name": extracted_name,
                "matches": [
                    {"entity_name": h['entity']['entity_name'],
                     "score": h['distance'],
                     "product_code": h['entity'].get('product_code', ''),
                     "risk_level": h['entity'].get('risk_level', '')}
                    for h in hits[0] if hits and hits[0]
                ],
            })
        return result

    def _score_align(self, search_results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """评分对齐：高置信->confirmed；中置信->options"""
        confirmed: List[str] = []
        options: List[str] = []
        high_conf = 0.7
        mid_conf = 0.6

        for search_result in search_results:
            extracted_name = search_result.get("extracted_name")
            sorted_matches = sorted(search_result.get("matches") or [],
                                    key=lambda x: x['score'], reverse=True)
            high = [m for m in sorted_matches if m.get('score') >= high_conf]

            if high:
                # 场景1：库里名称与抽取名一致 -> 直接确认
                exact = next((h for h in high if h['entity_name'] == extracted_name), None)
                if exact and exact['entity_name'] not in confirmed:
                    confirmed.append(exact['entity_name'])
                # 场景2：唯一高置信
                elif len(high) == 1 and high[0]['entity_name'] not in confirmed:
                    confirmed.append(high[0]['entity_name'])
                # 场景3：多个高置信候选 -> 进 options 供用户澄清
                else:
                    for h in high[:3]:
                        name = h['entity_name']
                        if name not in confirmed and name not in options:
                            options.append(name)
            else:
                mid = [m for m in sorted_matches
                       if m.get('score') >= mid_conf
                       and m['entity_name'] not in confirmed
                       and m['entity_name'] not in options]
                for m in mid[:3]:
                    options.append(m['entity_name'])
        return confirmed, options[:3]

    def _score_filter(self, confirmed: List[str],
                      search_results: List[Dict[str, Any]]) -> List[str]:
        """多确认时按最高分与断崖阈值收口（score 差距 > 0.15 剔除低分）"""
        item_score: Dict[str, float] = {}
        for search_result in search_results:
            for m in search_result.get("matches") or []:
                name = m.get('entity_name')
                if name in confirmed:
                    item_score[name] = max(item_score.get(name, 0), m.get('score', 0))

        sorted_scores = sorted(item_score.items(), key=lambda x: x[1], reverse=True)
        if not sorted_scores:
            return confirmed
        max_score = sorted_scores[0][1]
        return [name for name, score in item_score.items() if max_score - score <= 0.15]


class EntityConfirmNode(BaseNode):
    """金融实体确认节点"""

    name = "entity_confirm"

    def __init__(self):
        super().__init__()
        self.entity_extractor = EntityExtractor(self.logger, self.name)
        self.entity_aligner = EntityAligner(self.logger, self.name)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 历史上下文
        session_id = state.get("session_id")
        original_query = state.get("original_query")

        history_messages = get_recent_messages(session_id)
        history_messages.reverse()
        history_context = "暂无历史对话信息"
        for message in history_messages:
            history_context += f"{message['role']}:{message['text']}\n"

        # 2. LLM 抽取（意图/实体/改写）
        extract_result = self.entity_extractor.extract(original_query, history_context)
        query_type = extract_result['query_type']
        raw_entities = extract_result['entity_names']
        rewritten_query = extract_result['rewritten_query']

        # 3. 实体对齐（非 general 类型才尝试对齐）
        confirmed: List[str] = []
        options: List[str] = []
        if raw_entities:
            confirmed, options = self.entity_aligner.match_align_filter(
                raw_entities, self.config.entity_collection
            )

        # 4. 决策写回
        self._decide(state, query_type, raw_entities, confirmed, options, rewritten_query)

        # 5. 历史回填供下游/答案引用
        state["history"] = history_messages
        return state

    def _decide(self, state, query_type, raw_entities, confirmed,
                options, rewritten_query):
        # 分支1：有确认实体 -> 带实体过滤检索
        if confirmed:
            state["query_type"] = query_type
            state["entity_names"] = confirmed
            state["rewritten_query"] = rewritten_query
            return

        # 分支2：无确认实体，但 general 型（知识/FAQ/流程）-> 全库通用检索
        if query_type == "general" and self.config.allow_general_query_without_entity:
            state["query_type"] = query_type
            state["entity_names"] = []
            state["rewritten_query"] = rewritten_query
            return

        # 分支3：industry/announcement 等无库内实体但仍是合法主题检索 -> 通用检索
        if query_type in ("industry", "announcement") and not options:
            state["query_type"] = query_type
            state["entity_names"] = []
            state["rewritten_query"] = rewritten_query
            return

        # 分支4：存在待澄清候选 -> 澄清话术（短路，不检索）
        if options:
            state["answer"] = ENTITY_CLARIFY_TEMPLATE.format(options="、".join(options))
            return

        # 分支5：product 型却对齐不上 -> 礼貌提示
        state["answer"] = ENTITY_NOT_FOUND_TEMPLATE


if __name__ == '__main__':
    node = EntityConfirmNode()
    init_state = {
        "session_id": "123456",
        # "original_query": "某某纯债基金的风险等级是多少？"
        "original_query": "什么是基金净值？",
    }
    result = node(init_state)
    print(result)
