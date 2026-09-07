"""
查询流程的向量检索节点（三路检索第一路）

职责：对改写后问题做 BGE-M3 混合向量（dense+sparse）检索金融切片集合，
    命中结果放入 state["embedding_chunks"]。

金融适配：若已确认实体（state["entity_names"] 非空）则附加 entity_name 过滤，
    无实体（general 型知识问答）则全库检索。

输出结果结构与 0525 一致：Milvus hybrid_search 的 hits 列表，每项为
    {"id": ..., "distance": ..., "entity": {...标量字段...}}
"""
from typing import Dict, List

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, entity_names_filter


class VectorSearchNode(BaseNode):
    """向量检索（稠密+稀疏混合）"""

    name = "search_embedding"

    # 金融切片集合需回传的字段（含引用元数据，供 rrf/rerank/answer 使用）
    OUTPUT_FIELDS = [
        "chunk_id", "content", "title", "parent_title",
        "entity_name", "content_type", "document_title",
        "product_name", "product_code", "institution_name",
        "risk_level", "publish_date", "entry_name", "source_file",
    ]

    def process(self, state: QueryGraphState) -> Dict:
        # 1. 参数校验
        validated_query, validated_entities = self._validate_state(state)

        # 2. 客户端获取
        try:
            embedding_model = AIClients.get_bge_m3_client()
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"嵌入/Milvus 客户端获取失败: {e}")
            return state
        if embedding_model is None or milvus_client is None:
            return state

        # 3. 问题向量化
        try:
            embed_query = generate_bge_m3_hybrid_vectors(embedding_model, [validated_query], is_query=True)
        except Exception as e:
            self.logger.error(f"问题嵌入失败: {e}")
            return state

        # 4. 实体过滤（无实体则全库）
        filter_expr, filter_expr_param = entity_names_filter(validated_entities) if validated_entities else ("", {})

        try:
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=embed_query['dense'][0],
                sparse_vector=embed_query['sparse'][0],
                expr=filter_expr or None,
                expr_params=filter_expr_param or None,
                limit=self.config.embedding_search_limit,
            )
            reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_requests,
                output_fields=self.OUTPUT_FIELDS,
            )
            if not reps or not reps[0]:
                return state
            return {"embedding_chunks": reps[0]}
        except Exception as e:
            self.logger.error(f"混合检索失败: {e}")
            return state

    def _validate_state(self, state):
        rewritten_query = state.get("rewritten_query") or state.get("original_query")
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)
        entity_names = state.get("entity_names") or []
        return rewritten_query, entity_names


if __name__ == '__main__':
    node = VectorSearchNode()
    state = {"original_query": "某某基金的赎回规则", "rewritten_query": "某某纯债债券型证券投资基金赎回几天到账",
             "entity_names": ["某某纯债债券型证券投资基金"]}
    result = node(state)
    print(len(result.get('embedding_chunks', [])))
