"""
导入流程的 Milvus 入库节点

职责：把带混合向量的 chunk 写入 Milvus 金融切片集合；集合不存在则按需创建。

金融集合 schema（标量字段对齐需求文档第 5 节，见 finance_meta.py）：
    chunk_id        INT64  主键自增
    dense_vector    FLOAT_VECTOR  dim=1024
    sparse_vector   SPARSE_FLOAT_VECTOR
    + FINANCE_SCALAR_FIELDS（content/title/parent_title/content_type/document_title/
      product_name/product_code/institution_name/risk_level/industry/market/
      publish_date/entry_name/entity_name/source_file/source_path）VARCHAR
    启用动态字段 enable_dynamic_field=True，便于后续无感加字段。

索引：dense AUTOINDEX/COSINE，sparse SPARSE_INVERTED_INDEX/IP。

质量约束：无混合向量的 chunk 直接过滤（“导入失败不影响整体服务”）。
"""
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional, Sequence, Dict, Any

from pymilvus import MilvusClient, DataType, CollectionSchema

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.schema.finance_meta import FINANCE_SCALAR_FIELDS
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


# 预定义金融标量字段（全部 VARCHAR 大字段 + 可选 max_length）
_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = tuple(
    ScalarFieldSpec(field_name=f, datatype=DataType.VARCHAR, max_length=65535)
    for f in FINANCE_SCALAR_FIELDS
)


# --------------------------------------------------------------------- #
# 建造者：Schema 构建                                                    #
# --------------------------------------------------------------------- #
class _MilvusSchemaBuilder:
    """职责：构建切片集合 schema"""

    @staticmethod
    def build(client: MilvusClient, dim: int) -> CollectionSchema:
        logger.info("开始构建 schema...")
        schema = client.create_schema(enable_dynamic_field=True)

        # 1. 主键
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 2. 向量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 3. 标量字段
        for scalar_field in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {
                "field_name": scalar_field.field_name,
                "datatype": scalar_field.datatype,
            }
            if scalar_field.max_length is not None:
                kwargs["max_length"] = scalar_field.max_length
            schema.add_field(**kwargs)
        logger.info("schema 构建完成")
        return schema


# --------------------------------------------------------------------- #
# 建造者：索引构建                                                        #
# --------------------------------------------------------------------- #
class _MilvusIndexBuilder:
    """职责：构建集合索引"""

    @staticmethod
    def build(client: MilvusClient, collection_name: str):
        logger.info(f"开始构建集合 {collection_name} 索引...")
        index = client.prepare_index_params(collection_name=collection_name)
        # 稠密向量索引
        index.add_index(field_name="dense_vector", index_name="dense_vector_index",
                        index_type="AUTOINDEX", metric_type="COSINE")
        # 稀疏向量索引
        index.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        logger.info(f"集合 {collection_name} 索引构建完成")
        return index


# --------------------------------------------------------------------- #
# 插入器                                                                    #
# --------------------------------------------------------------------- #
class _MilvusInserter:
    """职责：数据插入 Milvus + 回填 chunk_id"""

    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"开始插入 {len(chunks)} 块到 Milvus...")
        inserted_result = self._client.insert(collection_name=self._collection_name, data=chunks)
        ids = inserted_result.get("ids", [])
        self._fill_chunk_ids(chunks, ids)
        logger.info(f"完成插入 {inserted_result.get('insert_count')} 条记录，并回填 chunk_id")
        return chunks

    @staticmethod
    def _fill_chunk_ids(chunks: List[Dict[str, Any]], ids: List[Any]):
        for chunk, chunk_id in zip(chunks, ids):
            chunk["chunk_id"] = chunk_id


# --------------------------------------------------------------------- #
# 节点                                                                     #
# --------------------------------------------------------------------- #
class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        # 1. 参数校验
        validated_chunks, dim, config = self._validate_get_inputs(state)

        # 2. 获取 Milvus 客户端（失败则安全返回，不阻断）
        milvus_client = StorageClients.get_milvus_client()
        if milvus_client is None:
            return state

        # 3. 集合名
        collection = config.chunks_collection

        # 4. 确保集合存在
        self._ensure_has_collection(milvus_client, collection, dim)

        # 5. 插入 + 回填 chunk_id
        inserter = _MilvusInserter(client=milvus_client, collection_name=collection)
        final_chunks = inserter.insert(chunks=validated_chunks)

        # 6. 更新状态
        state["chunks"] = final_chunks
        return state

    def _validate_get_inputs(self, state: ImportGraphState) -> Tuple[List, int, ImportConfig]:
        self.log_step("step1", "参数校验")
        config = get_config()
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("待入库的切块 chunk 不存在", self.name)

        # 过滤掉无向量的 chunk
        validated_chunks = []
        skipped_reason = None
        for chunk in chunks:
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                # 透传嵌入失败原因（bge_embedding 节点写入 state），便于定位根因
                if skipped_reason is None:
                    skipped_reason = chunk.get("embed_error")
                self.logger.error("发现无混合向量的 chunk，跳过入库")

        if not validated_chunks:
            hint = ""
            if skipped_reason:
                hint = f"；嵌入阶段失败原因: {skipped_reason}"
            raise ValidationError(
                f"入库的 chunk 均无效（缺少混合向量）{hint}。"
                f"请检查 BGE-M3 环境配置（BGE_M3_PATH/BGE_DEVICE/BGE_FP16）与模型路径是否有效。",
                self.name,
            )

        # 补齐 schema 声明的全部标量字段（缺省置空串），防止 Milvus 报
        # "Insert missed an field xxx ... nullable/default_value" 错误。
        # 各字段语义见 finance_meta.py 的 FINANCE_SCALAR_FIELDS 注释。
        for chunk in validated_chunks:
            for field_name in FINANCE_SCALAR_FIELDS:
                if field_name not in chunk or chunk[field_name] is None:
                    chunk[field_name] = ""

        dim = len(validated_chunks[0].get("dense_vector"))
        self.logger.info(f"导入 Milvus 的有效块：{len(validated_chunks)}，向量维度 {dim}")
        return validated_chunks, dim, config

    def _ensure_has_collection(self, milvus_client: MilvusClient, collection_name: str,
                               dim: int, delete_flag: bool = False):
        self.log_step("step2", f"准备集合 {collection_name} 创建")

        if delete_flag and milvus_client.has_collection(collection_name=collection_name):
            milvus_client.drop_collection(collection_name=collection_name)
            self.logger.info(f"Milvus 集合 {collection_name} 已删除")

        if milvus_client.has_collection(collection_name=collection_name):
            return

        schema = _MilvusSchemaBuilder.build(milvus_client, dim)
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)
        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index)
        self.logger.info(f"集合 {collection_name} 创建成功")


if __name__ == "__main__":
    setup_logging()
    node = ImportMilvusNode()
    state = {
        "chunks": [{"content": "测试", "entity_name": "某某基金", "dense_vector": [0.0] * 1024, "sparse_vector": {}}],
    }
    # 仅演示：真实执行需 Milvus 环境
    print(node)
