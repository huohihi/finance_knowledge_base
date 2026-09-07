"""
导入流程主图（LangGraph 有向图）

串联金融文档摄取与索引全链路节点：
    entry_node ──(按文件类型条件路由)──> pdf_to_md_node / md_img_node / END
        pdf_to_md_node -> md_img_node
        md_img_node    -> document_split_node
        document_split_node    -> finance_meta_extract_node
        finance_meta_extract_node -> bge_embedding_node
        bge_embedding_node      -> import_milvus_node -> END

节点职责总览：
    entry_node                文件类型识别（pdf/md），提取文件标题
    pdf_to_md_node            MinerU：PDF -> Markdown
    md_img_node               MD 图片：VLM 摘要 + MinIO 上传 + 引用替换
    document_split_node       Markdown 标题层级切块（长切短合、表格降维）
    finance_meta_extract_node LLM 抽取金融元数据 + 实体（实体写集合）
    bge_embedding_node        BGE-M3 dense+sparse 混合向量化
    import_milvus_node        chunk 写 Milvus 金融切片集合

实现模式与 0525 工程一致：
    - 节点实例化即注入配置；LangGraph stream 执行，任务状态由 base.py 统一追踪。
"""
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingNode
from knowledge.processor.import_process.nodes.document_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.finance_meta_extract import FinanceMetaExtractNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_process.nodes.md_img import MarkDownImageNode
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def load_file_route(state: ImportGraphState) -> str:
    """入口节点后的文件类型路由"""
    if state.get("is_pdf_read_enabled", False):
        return "pdf_to_md_node"
    elif state.get("is_md_read_enabled", False):
        return "md_img_node"
    else:
        return "END"


def create_import_graph() -> CompiledStateGraph:
    nodes = {
        "entry_node": EntryNode(),
        "md_img_node": MarkDownImageNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "document_split_node": DocumentSplitNode(),
        "finance_meta_extract_node": FinanceMetaExtractNode(),
        "bge_embedding_node": BgeEmbeddingNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    import_graph = StateGraph(ImportGraphState)
    for key, node in nodes.items():
        import_graph.add_node(key, node)

    import_graph.set_entry_point("entry_node")

    import_graph.add_conditional_edges(
        "entry_node",
        load_file_route,
        {
            "pdf_to_md_node": "pdf_to_md_node",
            "md_img_node": "md_img_node",
            "END": END,
        },
    )

    import_graph.add_edge("pdf_to_md_node", "md_img_node")
    import_graph.add_edge("md_img_node", "document_split_node")
    import_graph.add_edge("document_split_node", "finance_meta_extract_node")
    import_graph.add_edge("finance_meta_extract_node", "bge_embedding_node")
    import_graph.add_edge("bge_embedding_node", "import_milvus_node")
    import_graph.add_edge("import_milvus_node", END)

    return import_graph.compile()


# 全局单例：进程内复用编译图
kb_import_process_graph = create_import_graph()


def run_import_graph(input_state: ImportGraphState) -> ImportGraphState:
    """以 stream 方式运行导入图（便于逐节点观察）"""
    init_state = create_default_state(
        import_file_path=input_state.get("import_file_path"),
        file_dir=input_state.get("file_dir"),
    )

    final_state = None
    for event in kb_import_process_graph.stream(init_state):
        for node_name, process_state in event.items():
            print(f"运行节点：{node_name}")
            final_state = process_state
    return final_state


if __name__ == "__main__":
    setup_logging()

    input_state = {
        "task_id": "demo-finance-001",
        "import_file_path": r"E:\doc\某基金招募说明书.pdf",
        "file_dir": r"E:\tmp\fin_import",
    }

    final_state = run_import_graph(input_state)
    kb_import_process_graph.get_graph().print_ascii()
