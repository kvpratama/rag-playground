from langgraph.graph import END, StateGraph, START
from crag.nodes import (
    grade_documents,
    generate,
    retrieve,
    transform_query,
    web_search,
    decide_to_generate,
    init_retriever_node,
)
from crag.state import GraphState, GraphStateInput, GraphStateOutput

workflow = StateGraph(GraphState, input=GraphStateInput, output=GraphStateOutput)

# Define the nodes
workflow.add_node("init_retriever_node", init_retriever_node)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)

# Build graph
workflow.add_edge(START, "init_retriever_node")
workflow.add_edge("init_retriever_node", "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)
