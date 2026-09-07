"""
查询流程的 HyDE 检索节点（三路检索第二路）

职责：先用 LLM 生成一段「假设性金融文档」（Hypothetical Document），
    再把「原问题 + 假设文档」拼接后向量化检索，弥补 query-document 词汇鸿沟
    （需求中"最大回撤/风险等级/到账"等口语与文本质地的差距）。

输出写入 state["hyde_embedding_chunks"]，交给下游 RRF 融合。
"""
from typing import Dict, Tuple, List

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import (create_hybrid_search_requests,
                                         execute_hybrid_search_query, entity_confirm_expr)


class HyDeSearchNode(BaseNode):
    """HyDE 假设性文档检索"""

    name = "search_embedding_hyde"

    OUTPUT_FIELDS = [
        "chunk_id", "content", "title", "parent_title",
        "entity_name", "content_type", "document_title",
        "product_name", "product_code", "institution_name",
        "risk_level", "publish_date", "entry_name", "source_file",
    ]

    def process(self, state: QueryGraphState) -> Dict:
        # 1. 校验
        validated_query, validated_entities = self._validate_query_inputs(state)

        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validated_entities)

        # 3. 嵌入 + Milvus 客户端
        embedding_model = AIClients.get_bge_m3_client()
        milvus_client = StorageClients.get_milvus_client()
        if not embedding_model or not milvus_client:
            return state

        # 4. 拼接嵌入
        embedding_document = f"{validated_query}\n{hy_document}"
        try:
            embedding_result = generate_bge_m3_hybrid_vectors(
                embedding_model, [embedding_document], is_query=True
            )
        except Exception as e:
            self.logger.error(f"HyDE 嵌入失败: {e}")
            return state
        if not embedding_result:
            return state

        # 5. 实体过滤表达式（字符串直拼）
        filter_expr = entity_confirm_expr(validated_entities) if validated_entities else None

        try:
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=embedding_result['dense'][0],
                sparse_vector=embedding_result['sparse'][0],
                expr=filter_expr,
                limit=self.config.hyde_search_limit,
            )
            reps = execute_hybrid_search_query(
                milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_requests,
                norm_score=True,
                output_fields=self.OUTPUT_FIELDS,
            )
            if not reps or not reps[0]:
                return state
            return {"hyde_embedding_chunks": reps[0]}
        except Exception as e:
            self.logger.error(f"HyDE 混合检索失败: {e}")
            return state

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get("rewritten_query") or state.get("original_query", "")
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)
        entity_names = state.get("entity_names") or []
        return rewritten_query, entity_names

    def _generate_hy_document(self, validated_query: str, validated_entities: List[str]) -> str:
        llm_client = AIClients.get_llm_openai(response_format=False)
        if llm_client is None:
            return ""

        entity_desc = "、".join(validated_entities) if validated_entities else "通用金融知识"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            entity_names=entity_desc,
            rewritten_query=validated_query,
        )
        system_prompt = f"你是一位关于「{entity_desc}」的金融领域专家，擅长撰写金融产品说明、投研资料与业务规则说明。"
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            content = getattr(llm_response, 'content', "") or ""
            return content.strip()
        except Exception as e:
            self.logger.error(f"HyDE LLM 调用失败: {e}")
            return ""


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()
    node = HyDeSearchNode()
    mock_state = {
        "rewritten_query": "某某纯债基金的最大回撤是多少？",
        "entity_names": ["某某纯债债券型证券投资基金"],
    }
    result = node(mock_state)
    print(len(result.get("hyde_embedding_chunks", [])))
