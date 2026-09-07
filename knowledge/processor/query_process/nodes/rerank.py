"""
查询流程的重排序（Rerank）节点

职责：使用 BGE-Reranker 交叉编码器对 RRF 融合结果（本地）与网络结果
按「查询-文档」相关性精排，并采用【第一断崖检测】动态截断，
避免固定 Top-K 造成的相关/无关边界一刀切问题。

金融适配：本地 chunk 带完整引用元数据；网络文档标记 source="web"。
"""
from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankNode(BaseNode):
    """重排序节点（交叉编码器精排 + 断崖截断）"""

    name = "rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 精排依据：改写问题优先，退化为原始问题
        rewritten_query = state.get("rewritten_query") or state.get("original_query")

        # 2. 合并多源文档（RRF 融合结果 + Web 结果）
        merge_docs: List[Dict[str, Any]] = self._merge_multi_source_docs(state)

        # 3. 交叉编码器精排
        rerank_docs = self._rerank_merged_docs(rewritten_query, merge_docs)

        # 4. 第一断崖截断
        cutoff_docs = self.cliff_cutoff(
            rerank_docs,
            self.config.rerank_max_top_k,
            self.config.rerank_min_top_k,
            self.config.rerank_gap_abs,
        )

        # 5. 回填
        state["reranked_docs"] = cutoff_docs
        return state

    def _merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地（rrf_chunks）+ 网络（web_search_docs）两类来源"""
        merge_docs: List[Dict[str, Any]] = []

        for doc in state.get("rrf_chunks") or []:
            merge_docs.append({
                "chunk_id": doc.get("chunk_id"),
                "content": doc.get("content", ""),
                "title": doc.get("title", ""),
                "entity_name": doc.get("entity_name", ""),
                # 金融引用元数据透传（answer 引用来源展示）
                "content_type": doc.get("content_type", ""),
                "document_title": doc.get("document_title", ""),
                "product_name": doc.get("product_name", ""),
                "product_code": doc.get("product_code", ""),
                "institution_name": doc.get("institution_name", ""),
                "risk_level": doc.get("risk_level", ""),
                "publish_date": doc.get("publish_date", ""),
                "entry_name": doc.get("entry_name", ""),
                "source_file": doc.get("source_file", ""),
                "source": "local",
            })

        for doc in state.get("web_search_docs") or []:
            merge_docs.append({
                "content": doc.get("content", "") or doc.get("snippet", ""),
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "source": "web",
            })

        return merge_docs

    def _rerank_merged_docs(self, rewritten_query: str,
                            merge_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """BGE-Reranker 交叉编码器打分排序；失败降级保留原序"""
        if not merge_docs:
            return []

        try:
            reranker_client = AIClients.get_bge_m3_rerank_client()
            pairs = [[rewritten_query, doc["content"]] for doc in merge_docs]
            scores = reranker_client.compute_score(sentence_pairs=pairs, normalize=True)
            # 单文档返回标量时的防护（FlagEmbedding 版本差异兼容）
            if isinstance(scores, (float, int)):
                scores = [scores]

            rerank_docs = [{**doc, "score": score} for doc, score in zip(merge_docs, scores)]
            return sorted(rerank_docs, key=lambda doc: doc["score"], reverse=True)
        except Exception as e:
            self.logger.warning(f"BGE-Reranker 调用失败，降级返回原序: {e}")
            return [{**doc, "score": None} for doc in merge_docs]

    def cliff_cutoff(self, reranked_docs: List[Dict[str, Any]],
                     rerank_max_top_k: int = 10, rerank_min_top_k: int = 3,
                     rerank_gap_abs: float = 0.15) -> List[Dict[str, Any]]:
        """
        第一断崖检测截断：从高分向低分扫描，遇到相邻分差 > gap 处截断，
        且截断点不早于 lower_bound、不晚于 upper_bound。
        """
        upper_bound = min(rerank_max_top_k, len(reranked_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)

        if upper_bound <= 1:
            return reranked_docs[:upper_bound]

        cut_off = upper_bound
        for i in range(0, upper_bound - 1):
            current_score = reranked_docs[i].get("score")
            next_score = reranked_docs[i + 1].get("score")
            if current_score is None or next_score is None:
                continue
            gap = current_score - next_score
            if gap > rerank_gap_abs:
                cut_off = i + 1
                self.logger.info(f"第一断崖位置={cut_off}, gap={gap:.4f}")
                break

        cut_off = max(cut_off, lower_bound)
        return reranked_docs[:cut_off]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()
    node = RerankNode()
    mock_state = {
        "rewritten_query": "某某纯债基金的风险等级？",
        "rrf_chunks": [
            {"chunk_id": 1, "content": "本基金为债券型基金，风险等级R2，适合稳健型投资者。",
             "title": "风险提示", "entity_name": "某某纯债基金"},
            {"chunk_id": 2, "content": "招募说明书全文较长与问题关系不大，介绍公司历史沿革。",
             "title": "公司概况", "entity_name": "某某纯债基金"},
        ],
        "web_search_docs": [],
    }
    result = node(mock_state)
    for doc in result["reranked_docs"]:
        print(doc.get("score"), doc.get("content", "")[:30])
