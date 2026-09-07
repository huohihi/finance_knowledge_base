"""
导入流程的向量化节点

职责：对每个 chunk 生成 BGE-M3 混合向量（dense + sparse），
并把向量回填进 chunk，供 import_milvus 节点统一入库。

关键设计（与 0525 对齐）：
    1. 嵌入文本 = f"{entity_name}\n{content}"，让实体名参与语义匹配，
       提升金融产品类问题的召回准确率；
    2. 分批处理（embedding_batch_chunk_size），防止长文本爆显存/内存；
    3. 单批失败降级：返回未加向量的 batch，由 import_milvus 过滤掉无向量 chunk，
       不影响整体导入（“导入失败不影响整体服务”质量要求）。
"""
from typing import List, Dict, Any

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class BgeEmbeddingNode(BaseNode):
    name = "bge_embedding_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        # 1. 校验输入
        validated_chunks = self._validate_get_inputs(state)

        # 2. 分批向量化
        batch_size = self.config.embedding_batch_chunk_size
        final_chunks = []
        total_length = len(validated_chunks)

        for i in range(0, total_length, batch_size):
            batch = validated_chunks[i:i + batch_size]
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)

        # 3. 更新状态
        state["chunks"] = final_chunks
        return state

    def _validate_get_inputs(self, state: ImportGraphState) -> List[Dict[str, Any]]:
        self.log_step("step1", "参数校验")
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunks 为空或无效", self.name)
        self.logger.info(f"待嵌入的块数：{len(chunks)}")
        return chunks

    def _process_batch_chunks(self, batch: List[Dict[str, Any]],
                              start_index: int, total_length: int) -> List[Dict[str, Any]]:
        self.log_step("step2", f"批量处理 chunk 嵌入: 批次{start_index + 1}-{start_index + len(batch)}")

        # 1. 组装嵌入文本（实体名 + 正文）
        embedding_contents = []
        for chunk in batch:
            content = chunk.get("content")
            entity_name = chunk.get("entity_name") or ""
            embedding_contents.append(f"{entity_name}\n{content}")

        # 2. 批量向量化
        try:
            bge_m3_model = AIClients.get_bge_m3_client()
            embedding_result = bge_m3_model.encode_documents(embedding_contents)
            if not embedding_result:
                self.logger.warning("嵌入结果为空，跳过本批")
                return self._mark_batch_failed(batch, "嵌入结果为空")
        except Exception as e:
            self.logger.warning(f"嵌入向量失败...{str(e)}，跳过本批")
            return self._mark_batch_failed(batch, str(e))

        # 3. 解构向量并回填 chunk
        for index, chunk in enumerate(batch):
            # 3.1 稠密向量
            dense_vector = embedding_result["dense"][index].tolist()

            # 3.2 CSR 稀疏矩阵 -> {token_id: weight}
            csr = embedding_result['sparse']
            start_ptr = csr.indptr[index]
            end_ptr = csr.indptr[index + 1]
            token_ids = csr.indices[start_ptr:end_ptr].tolist()
            weights = csr.data[start_ptr:end_ptr].tolist()
            sparse_vector = dict(zip(token_ids, weights))

            # 3.3 回填（并清除可能的历史失败标记）
            chunk["dense_vector"] = dense_vector
            chunk["sparse_vector"] = sparse_vector
            chunk.pop("embed_error", None)

        self.logger.info(f"完成批次嵌入: {start_index + 1}-{start_index + len(batch)}/{total_length}")
        return batch

    @staticmethod
    def _mark_batch_failed(batch: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
        """整批失败时把原因写入每个 chunk，供 import_milvus 报错时透传根因"""
        for chunk in batch:
            chunk["embed_error"] = chunk.get("embed_error") or reason
        return batch


if __name__ == '__main__':
    setup_logging()
    node = BgeEmbeddingNode()
    state = {
        "chunks": [
            {"content": "基金有风险，投资需谨慎。本基金为混合型证券投资基金。", "entity_name": "某某混合基金"},
        ],
    }
    node.process(state)
