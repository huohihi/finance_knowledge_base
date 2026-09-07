"""
查询流程的 RRF（倒序融合）节点

职责：把「向量检索 embedding_chunks」与「HyDE 检索 hyde_embedding_chunks」
两路结果按 Recursive Rank Fusion 公式融合去重，抑制单路噪声、放大多路共识。

RRF 公式：score(doc) = Σ_weight weight / (k + rank(doc))

金融适配：融合对象为金融 chunk 的完整元数据（chunk_id 去重键），
输出保留全部标量字段供下游重排与答案引用。
"""
from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    """倒序融合节点"""

    name = "rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 取两路结果（至少一路非空才继续）
        embedding_chunks = state.get("embedding_chunks") or []
        hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []

        # 2. 统一格式化为纯实体字典（去掉 Milvus 外壳 id/distance）
        embedding_entities = self._normalize_input(embedding_chunks)
        hyde_entities = self._normalize_input(hyde_embedding_chunks)

        # 3. 两路权重相同（均为本地知识库检索）
        rrf_inputs: List[tuple] = [(embedding_entities, 1.0), (hyde_entities, 1.0)]

        # 4. RRF 融合
        rrf_results = self._rrf_merge(
            rrf_inputs=rrf_inputs,
            rrf_k=self.config.rrf_k,
            top_k=self.config.rrf_max_results,
        )

        # 5. 回填
        state["rrf_chunks"] = [entity for entity, _ in rrf_results]
        return state

    def _normalize_input(self, chunks_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """剥掉 Milvus 结果外壳（pk/distance/entity），只留 entity 标量字段"""
        entity_results = []
        for chunk in chunks_input:
            if not isinstance(chunk, dict):
                continue
            entity = chunk.get("entity")
            if not isinstance(entity, dict):
                continue
            entity_results.append(entity)
        return entity_results

    def _rrf_merge(self, rrf_inputs: List[tuple], rrf_k: int = 60,
                   top_k: int = 10) -> List[tuple]:
        """
        按 RRF 公式融合多路结果，返回 [(entity, score)] 按分数降序截断。

        rrf_inputs: [(doc_list, weight), ...]
        """
        entity_scores: Dict[Any, float] = {}   # chunk_id -> 累计分
        entity_data: Dict[Any, Dict] = {}      # chunk_id -> 完整文档

        for chunks_input, weight in rrf_inputs:
            for index, entity in enumerate(chunks_input):
                chunk_id = entity.get("chunk_id")
                if chunk_id is None:
                    continue
                entity_scores[chunk_id] = entity_scores.get(chunk_id, 0.0) + weight / (rrf_k + (index + 1))
                entity_data.setdefault(chunk_id, entity)

        # 按分数降序排序并截断
        rrf_result = sorted(entity_scores.items(), key=lambda item: item[1], reverse=True)
        return [(entity_data[chunk_id], score) for chunk_id, score in rrf_result[:top_k]]


if __name__ == "__main__":
    setup_logging()
    node = RrfNode()
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": 1, "content": "向量结果#1"}},
            {"entity": {"chunk_id": 2, "content": "向量结果#2"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": 2, "content": "HyDE结果#2"}},
            {"entity": {"chunk_id": 3, "content": "HyDE结果#3"}},
        ],
    }
    result = node(mock_state)
    print([c.get("chunk_id") for c in result["rrf_chunks"]])
