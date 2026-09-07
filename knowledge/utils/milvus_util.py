"""
Milvus 混合检索工具

封装稠密/稀疏混合检索的公共操作：
    - create_hybrid_search_requests: 构造 AnnSearchRequest 对（dense + sparse）
    - execute_hybrid_search_query: 执行 hybrid_search（WeightedRanker 融合）
    - entity_names_filter: 实体过滤表达式（等价 0525 item_names_filter）
    - risk_meta_filter: 金融标量过滤（risk_level / content_type / product_name...）
"""
import logging
from typing import Optional, List, Tuple, Any, Dict

from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# 构造混合检索请求                                                          #
# --------------------------------------------------------------------- #
def create_hybrid_search_requests(dense_vector,
                                  sparse_vector,
                                  dense_params=None,
                                  sparse_params=None,
                                  expr=None,
                                  expr_params=None,
                                  limit=5) -> List[AnnSearchRequest]:
    """
    创建混合搜索请求（稠密 + 稀疏两个 AnnSearchRequest）。

    Args:
        dense_vector: 稠密向量
        sparse_vector: 稀疏向量
        dense_params:  稠密检索参数（默认 COSINE）
        sparse_params: 稀疏检索参数（默认 IP）
        expr: 过滤表达式（可空）
        expr_params: 过滤表达式参数（可空）
        limit: 每路返回条数
    """
    if dense_vector is None or sparse_vector is None:
        raise ValueError("dense_vector 和 sparse_vector 不能为 None")

    try:
        if dense_params is None:
            dense_params = {"metric_type": "COSINE"}
        if sparse_params is None:
            sparse_params = {"metric_type": "IP"}

        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param=dense_params,
            expr=expr,
            expr_params=expr_params,
            limit=limit,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param=sparse_params,
            expr=expr,
            expr_params=expr_params,
            limit=limit,
        )
        return [dense_req, sparse_req]
    except Exception as e:
        raise RuntimeError(f"创建混合搜索请求失败: {e}") from e


# --------------------------------------------------------------------- #
# 执行混合检索                                                              #
# --------------------------------------------------------------------- #
def execute_hybrid_search_query(milvus_client: MilvusClient,
                                collection_name,
                                search_requests,
                                ranker_weights=(0.5, 0.5),
                                norm_score=True,
                                limit=5,
                                output_fields=None,
                                search_params=None):
    """
    执行 Milvus 混合搜索（WeightedRanker 融合 dense/sparse 得分）。

    Args:
        ranker_weights: 稠密/稀疏权重（默认 0.5/0.5）
        norm_score: 分数归一化（RRF 场景建议 True）
        output_fields: 需要回传的标量字段
    """
    if milvus_client is None:
        raise ValueError("milvus_client 不能为 None")
    if search_requests is None or len(search_requests) == 0:
        raise ValueError("search_requests 不能为 None 或空列表")

    try:
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        if output_fields is None:
            output_fields = ["entity_name"]

        res = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=search_requests,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )
        total_hits = sum(len(hits) for hits in res) if res else 0
        logger.info(f"Milvus 混合搜索完成，{len(res) if res else 0} 个查询共 {total_hits} 条结果")
        return res
    except Exception as e:
        raise RuntimeError(f"执行 Milvus 混合搜索失败 (collection={collection_name}): {e}") from e


# --------------------------------------------------------------------- #
# 过滤表达式构造                                                            #
# --------------------------------------------------------------------- #
def entity_names_filter(entity_names: List[str]) -> Tuple[str, Dict[str, Any]]:
    """
    实体名过滤表达式。金融查询链路：用户问题先对齐实体，再用实体过滤正文。

    expr: entity_name in {entity_names}
    """
    if not entity_names:
        return "", {}
    expr = "entity_name in {entity_names}"
    expr_params = {"entity_names": entity_names}
    return expr, expr_params


def risk_meta_filter(entity_names: List[str] = None,
                     content_type: str = None,
                     risk_level: str = None,
                     product_code: str = None) -> Tuple[str, Dict[str, Any]]:
    """
    金融标量组合过滤表达式（按需拼接 AND 条件）。

    Args:
        entity_names: 实体名列表
        content_type: 内容类型（产品说明书/FAQ/公告...）
        risk_level:   风险等级（R1-R5）
        product_code: 产品代码
    """
    conditions = []
    params: Dict[str, Any] = {}

    if entity_names:
        conditions.append("entity_name in {entity_names}")
        params["entity_names"] = entity_names
    if content_type:
        conditions.append("content_type == {content_type}")
        params["content_type"] = content_type
    if risk_level:
        conditions.append("risk_level == {risk_level}")
        params["risk_level"] = risk_level
    if product_code:
        conditions.append("product_code == {product_code}")
        params["product_code"] = product_code

    if not conditions:
        return "", {}
    return " and ".join(conditions), params


def entity_confirm_expr(validate_entities: List[str]) -> str:
    """字符串直拼实体过滤表达式（Milvus filter 场景，等价 0525 _item_name_filte_expr）"""
    quoted = ", ".join(f'"{v}"' for v in validate_entities)
    return f" entity_name in [{quoted}]"
