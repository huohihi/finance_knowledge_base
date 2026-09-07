"""
查询流程主图（LangGraph）

金融问答检索增强生成链路：
    entity_confirm ──路由──> (有 answer: 澄清/无法识别) answer_output
                          └──(无 answer) multi_search
    multi_search ──并行──> search_embedding / search_embedding_hyde / web_search_mcp
                        └─ join -> rrf -> rerank -> answer_output -> END

与 0525 模式差异（金融领域适配）：
    - item_name_confirm 升级为 entity_confirm：
      当问题为 general（知识/FAQ/流程）型且无库内实体时，不打断用户，
      走全库通用检索（需求 2.3 金融知识问答、2.7 业务问答必须支持）；
    - 答案节点额外生成 citations（引用来源），随 SSE FINAL 返回前端。
"""
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.nodes.answer_output import AnswerOutputNode
from knowledge.processor.query_process.nodes.entity_confirm import EntityConfirmNode
from knowledge.processor.query_process.nodes.hyde_search import HyDeSearchNode
from knowledge.processor.query_process.nodes.rerank import RerankNode
from knowledge.processor.query_process.nodes.rrf import RrfNode
from knowledge.processor.query_process.nodes.vector_search import VectorSearchNode
from knowledge.processor.query_process.nodes.web_search_mcp import WebSearchMcpNode


def route_after_entity_confirm(state: QueryGraphState) -> bool:
    """实体确认后的路由：已有 answer（澄清/未识别）则跳过检索直接出答案"""
    return True if state.get("answer") else False


def create_query_graph() -> CompiledStateGraph:
    workflow = StateGraph(QueryGraphState)

    nodes = {
        "entity_confirm": EntityConfirmNode(),
        "multi_search": lambda x: x,          # 虚拟分发节点
        "search_embedding": VectorSearchNode(),
        "search_embedding_hyde": HyDeSearchNode(),
        "web_search_mcp": WebSearchMcpNode(),
        "join": lambda x: {},                 # 虚拟汇合节点
        "rrf": RrfNode(),
        "rerank": RerankNode(),
        "answer_output": AnswerOutputNode(),
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("entity_confirm")

    workflow.add_conditional_edges(
        "entity_confirm",
        route_after_entity_confirm,
        {
            False: "multi_search",
            True: "answer_output",
        },
    )

    # 三路检索并行分发
    workflow.add_edge("multi_search", "search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "web_search_mcp")

    # 三路汇合
    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("web_search_mcp", "join")

    # 顺序链路：融合 -> 精排 -> 生成
    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", "answer_output")
    workflow.add_edge("answer_output", END)

    return workflow.compile()


# 全局图实例
query_app = create_query_graph()


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()
    print("=" * 60)
    print("测试: 查询流程主图")
    print("=" * 60)

    mock_state = {
        "original_query": "某某纯债基金的风险等级是什么？",
        "session_id": "test_session_main",
        "task_id": "test_task_001",
        "is_stream": False,
    }

    result = query_app.invoke(mock_state)
    print(f"\n【结果】")
    print(f"query_type: {result.get('query_type')}")
    print(f"entity_names: {result.get('entity_names')}")
    print(f"rewritten_query: {result.get('rewritten_query')}")
    answer = result.get("answer", "")
    print(f"答案: {answer[:300]}...")
    query_app.get_graph().print_ascii()
