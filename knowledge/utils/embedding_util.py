"""
BGE-M3 混合向量生成工具

统一生成稠密 + 稀疏向量，供查询侧所有节点复用。
关键差异（与 0525 一致）：
    encode_queries   编码"用户问题"（查询侧）
    encode_documents 编码"文档正文"（入库侧）
BGE-M3 为非对称检索，两方法内部会加不同的指令前缀。
"""
from typing import List

try:
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "无法导入 BGEM3EmbeddingFunction：请确认安装 pymilvus 2.x（pip install 'pymilvus>=2.4,<3.0'）"
    ) from _e


def generate_bge_m3_hybrid_vectors(model: BGEM3EmbeddingFunction,
                                   embedding_documents: List[str],
                                   is_query: bool = True):
    """
    为文本列表生成混合向量。

    Args:
        model: BGE-M3 嵌入模型
        embedding_documents: 文本列表
        is_query: True 按查询编码（encode_queries）；False 按文档编码（encode_documents）

    Returns:
        {"dense": [dense...], "sparse": [sparse_dict...]}
    """
    if not embedding_documents:
        raise ValueError("embedding_documents 不能为空")
    if not all(isinstance(doc, str) and doc.strip() for doc in embedding_documents):
        raise ValueError("embedding_documents 中存在无效元素")

    try:
        if is_query:
            embedding_result = model.encode_queries(embedding_documents)
        else:
            embedding_result = model.encode_documents(embedding_documents)
    except Exception as e:
        raise RuntimeError(f"BGE-M3 嵌入生成失败: {e}") from e

    if 'dense' not in embedding_result or 'sparse' not in embedding_result:
        raise RuntimeError(f"嵌入结果缺少必要字段: {list(embedding_result.keys())}")

    # CSR 稀疏矩阵 -> [{token_id: weight}]
    try:
        processed_sparse = []
        csr_array = embedding_result['sparse']
        for index in range(len(embedding_documents)):
            start = csr_array.indptr[index]
            end = csr_array.indptr[index + 1]
            token_ids = csr_array.indices[start:end].tolist()
            weights = csr_array.data[start:end].tolist()
            processed_sparse.append(dict(zip(token_ids, weights)))
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"稀疏向量解析失败（CSR 矩阵结构异常）: {e}") from e

    return {
        "dense": [den.tolist() for den in embedding_result["dense"]],
        "sparse": processed_sparse,
    }
