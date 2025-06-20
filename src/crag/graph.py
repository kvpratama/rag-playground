from langgraph.graph import END, StateGraph, START
from crag.nodes import (
    grade_documents,
    generate,
    retrieve,
    transform_query,
    web_search,
    decide_to_generate,
    init_retriever_node,
    should_continue,
)
from crag.state import GraphState, GraphStateInput, GraphStateOutput
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(GraphState, input=GraphStateInput, output=GraphStateOutput)

# Define the nodes
builder.add_node("init_retriever_node", init_retriever_node)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("generate", generate)
builder.add_node("transform_query", transform_query)
builder.add_node("web_search_node", web_search)

# Build graph
builder.add_edge(START, "init_retriever_node")
builder.add_conditional_edges("init_retriever_node", should_continue, {
    "retrieve": "retrieve",
    END: END,
})
builder.add_edge("retrieve", "grade_documents")
builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
builder.add_edge("transform_query", "web_search_node")
builder.add_edge("web_search_node", "generate")
builder.add_edge("generate", END)

# memory = MemorySaver()
workflow = builder.compile()