"""
导入流程的金融元数据抽取节点
（0525 item_name_recognition 的金融领域升级版）

职责（一次 LLM 调用完成两件事）：
    1. 抽取【文档级金融元数据】finance_meta —— 对齐需求文档第 5 节字段：
       content_type / document_title / product_name / product_code /
       institution_name / risk_level / industry / market / publish_date / entry_name
    2. 抽取【金融实体】entities —— 产品/机构/概念，写入 Milvus 实体集合，
       供查询链路做「用户问题 → 实体」对齐确认（等价 0525 的 item_name_collection）。

实体集合 Milvus schema 字段（见 schema/finance_meta.py ENTITY_FIELDS）：
    pk / entity_name / entity_type / product_code / institution_name /
    risk_level / content_type / document_title / source_file /
    dense_vector(1024) / sparse_vector

为控制调用成本，实体向量由下游 bge_embedding 统一生成？—— 否。
本节点内直接对 entity_name 做 BGE-M3 向量化并写实体集合（独立于 chunk 集合），
保持与 0525 item_name_recognition 节点一致的单点职责。

步骤：
    validate -> 组装上下文 -> LLM 抽取(meta + entities)
    -> 逐个实体向量化 -> 写实体集合 -> 元数据回填全部 chunk -> 备份
"""
import json
import os
from typing import Tuple, List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import FINANCE_META_SYSTEM_PROMPT, FINANCE_META_USER_PROMPT_TEMPLATE
from knowledge.schema.finance_meta import ENTITY_FIELDS, ENTITY_TYPE_PRODUCT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients

# LLM 抽取字段 -> chunk 回填字段（与 finance_meta.py 中 FINANCE_SCALAR_FIELDS 一致）
_META_KEYS = [
    "content_type", "document_title", "product_name", "product_code",
    "institution_name", "risk_level", "industry", "market", "publish_date", "entry_name",
]


class FinanceMetaExtractNode(BaseNode):
    """金融元数据 + 实体识别节点"""

    name = "finance_meta_extract_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        # 1. 参数校验
        file_title, chunks, meta_chunk_k, meta_chunk_size = self._validate_state(state)

        # 2. 组装用于抽取的上下文（前 k 个切片）
        context = self._prepare_extract_context(chunks, meta_chunk_k, meta_chunk_size)

        # 3. LLM 抽取 finance_meta + entities
        finance_meta, entities = self._extract_meta_and_entities(file_title, context)

        # 4. 实体向量化并写 Milvus 实体集合（全局降级：单个失败不影响整体）
        self._persist_entities(file_title, entities, state)

        # 5. 元数据回填全部 chunk
        self._fill_chunks(finance_meta, state, chunks)

        # 6. 备份（供人工核查 / 下游测试）
        self._backup(state)

        return state

    # ------------------------------------------------------------------ #
    #                      Step 1: 校验                                  #
    # ------------------------------------------------------------------ #

    def _validate_state(self, state: ImportGraphState) -> Tuple[str, List, int, int]:
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)

        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        meta_chunk_k = self.config.meta_extract_chunk_k
        if not meta_chunk_k or meta_chunk_k <= 0:
            raise ValidationError(message="meta_extract_chunk_k 为空或无效", node_name=self.name)

        meta_chunk_size = self.config.meta_extract_chunk_size
        if not meta_chunk_size or meta_chunk_size <= 0:
            raise ValidationError(message="meta_extract_chunk_size 为空或无效", node_name=self.name)

        return file_title, chunks, meta_chunk_k, meta_chunk_size

    # ------------------------------------------------------------------ #
    #                      Step 2: 上下文组装                             #
    # ------------------------------------------------------------------ #

    def _prepare_extract_context(self, chunks: List[Dict], chunk_k: int, chunk_size: int) -> str:
        """取文档前 k 个切片做抽取依据（标题通常覆盖产品名/机构/风险等级）"""
        total = 0
        final_context = []
        for index, chunk in enumerate(chunks[:chunk_k]):
            if not isinstance(chunk, dict):
                continue
            chunk_content = chunk.get("content")
            if not chunk_content:
                continue
            context = f"【切片-{index}】\n{chunk_content}"
            if total + len(context) > chunk_size:
                break
            total += len(context)
            final_context.append(context)
        return "\n".join(final_context)

    # ------------------------------------------------------------------ #
    #                      Step 3: LLM 抽取                              #
    # ------------------------------------------------------------------ #

    def _extract_meta_and_entities(self, file_title: str, context: str) -> Tuple[Dict, List[Dict]]:
        """LLM 抽取元数据与实体；失败时返回空元数据与空实体（不阻断导入）"""
        empty_meta = {k: "" for k in _META_KEYS}
        try:
            llm_client = AIClients.get_llm_openai(response_format=True)
            user_prompt = FINANCE_META_USER_PROMPT_TEMPLATE.format(
                file_title=file_title, context=context or "（文档正文前段为空）"
            )
            llm_response = llm_client.invoke([
                SystemMessage(content=FINANCE_META_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            llm_result = llm_response.content.strip()
            if not llm_result:
                self.logger.warning("LLM 未返回元数据，降级为空元数据")
                return empty_meta, []

            parsed = json.loads(self._strip_code_fence(llm_result))
            meta = parsed.get("finance_meta") or {}
            clean_meta = {}
            for k in _META_KEYS:
                v = meta.get(k)
                clean_meta[k] = str(v).strip() if v else ""

            entities = parsed.get("entities") or []
            clean_entities = []
            for ent in entities[:3]:
                if not isinstance(ent, dict):
                    continue
                ent_name = str(ent.get("entity_name") or "").strip()
                if not ent_name:
                    continue
                clean_entities.append({
                    "entity_name": ent_name,
                    "entity_type": str(ent.get("entity_type") or ENTITY_TYPE_PRODUCT).strip(),
                    "product_code": str(ent.get("product_code") or "").strip(),
                    # 机构/风险等级等上下文也冗余到实体行，便于查询时展示
                    "institution_name": clean_meta.get("institution_name", ""),
                    "risk_level": clean_meta.get("risk_level", ""),
                    "content_type": clean_meta.get("content_type", ""),
                    "document_title": clean_meta.get("document_title") or file_title,
                })
            self.logger.info(f"LLM 抽取元数据: {clean_meta}")
            self.logger.info(f"LLM 抽取实体: {[e['entity_name'] for e in clean_entities]}")
            return clean_meta, clean_entities

        except Exception as e:
            self.logger.error(f"LLM 抽取失败，降级为空元数据: {e}")
            return empty_meta, []

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            # 去除可能的 json 标记行
            if content.startswith("json"):
                content = content[4:]
        return content.strip()

    # ------------------------------------------------------------------ #
    #               Step 4: 实体向量化 + 写实体集合                        #
    # ------------------------------------------------------------------ #

    def _persist_entities(self, file_title: str, entities: List[Dict], state: ImportGraphState):
        """将每个实体向量化并插入 Milvus 实体集合（单点失败不影响整体）"""
        if not entities:
            self.logger.info(f"文档 {file_title} 未识别到实体，跳过实体集合写入")
            state["entities"] = []
            return

        milvus_client = None
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端创建失败: {e}")
            state["entities"] = entities
            return

        entity_collection = self.config.entity_collection
        try:
            if not milvus_client.has_collection(entity_collection):
                self._create_entity_collection(entity_collection, milvus_client)
        except Exception as e:
            self.logger.error(f"实体集合检查/创建失败: {e}")
            state["entities"] = entities
            return

        # 按配置容量逐个处理实体
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"BGE-M3 客户端获取失败: {e}")
            state["entities"] = entities
            return

        for ent in entities:
            entity_name = ent["entity_name"]
            dense_vector, sparse_vector = None, None
            try:
                vector_result = bge_m3_client.encode_documents([entity_name])
                dense_vector = vector_result["dense"][0].tolist()
                # CSR 稀疏矩阵解析
                start_index = vector_result["sparse"].indptr[0]
                end_index = vector_result["sparse"].indptr[1]
                token_ids = vector_result["sparse"].indices[start_index:end_index].tolist()
                weights = vector_result["sparse"].data[start_index:end_index].tolist()
                sparse_vector = dict(zip(token_ids, weights))
            except Exception as e:
                self.logger.error(f"实体[{entity_name}] 向量化失败: {e}")
                continue

            if not dense_vector or not sparse_vector:
                continue
            try:
                row = {
                    "entity_name": entity_name,
                    "entity_type": ent.get("entity_type", ENTITY_TYPE_PRODUCT),
                    "product_code": ent.get("product_code", ""),
                    "institution_name": ent.get("institution_name", ""),
                    "risk_level": ent.get("risk_level", ""),
                    "content_type": ent.get("content_type", ""),
                    "document_title": ent.get("document_title", file_title),
                    "source_file": file_title,
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                }
                result = milvus_client.insert(collection_name=entity_collection, data=[row])
                self.logger.info(f"实体[{entity_name}] 已写入实体集合, ID:{result['ids'][0]}")
            except Exception as e:
                self.logger.error(f"实体[{entity_name}] 写入实体集合失败: {e}")

        state["entities"] = entities

    def _create_entity_collection(self, collection_name: str, milvus_client):
        """构建实体集合 schema + 双索引（dense COSINE / sparse IP）"""
        self.logger.info(f"创建实体集合 {collection_name} ...")
        schema = milvus_client.create_schema()

        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        for field_name in ENTITY_FIELDS:
            schema.add_field(field_name=field_name, datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        milvus_client.create_collection(collection_name=collection_name,
                                        schema=schema, index_params=index_param)
        self.logger.info(f"实体集合 {collection_name} 创建成功并构建索引")

    # ------------------------------------------------------------------ #
    #               Step 5: 元数据回填 + 备份                             #
    # ------------------------------------------------------------------ #

    def _fill_chunks(self, finance_meta: Dict, state: ImportGraphState, chunks: List[Dict]):
        """把文档级元数据与实体回填到每个 chunk（供检索过滤与引用来源展示）"""
        for chunk in chunks:
            for key, value in finance_meta.items():
                if key == "content" or not isinstance(value, str):
                    continue
                chunk[key] = value
            chunk["entity_name"] = self._primary_entity(finance_meta, state)
            chunk["source_file"] = state.get("file_title", "")
        state["finance_meta"] = finance_meta

    @staticmethod
    def _primary_entity(finance_meta: Dict, state: ImportGraphState) -> str:
        """主实体选择：优先产品名 -> 机构名 -> entry_name(术语/FAQ) -> 文件标题"""
        if finance_meta.get("product_name"):
            return finance_meta["product_name"]
        if finance_meta.get("institution_name"):
            return finance_meta["institution_name"]
        if finance_meta.get("entry_name"):
            return finance_meta["entry_name"]
        return state.get("file_title", "")

    def _backup(self, state: ImportGraphState):
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks_meta_entities.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "finance_meta": state.get("finance_meta", {}),
                    "entities": state.get("entities", []),
                    "chunks": state.get("chunks", []),
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")


if __name__ == '__main__':
    setup_logging()
    node = FinanceMetaExtractNode()
    state = {
        "file_dir": r"E:\tmp\fin_import",
        "file_title": "某纯债债券型证券投资基金招募说明书",
        "chunks": [
            {"content": "基金全称：某某纯债债券型证券投资基金\n基金代码：000001\n风险等级：R2\n管理人中某基金管理有限公司"},
            {"content": "投资目标：在严格控制风险的前提下，追求基金资产的长期稳健增值"},
        ],
    }
    node.process(state)
