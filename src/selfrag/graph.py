from langgraph.graph import END, StateGraph, START
from selfrag.nodes import (
    grade_documents,
    generate,
    retrieve,
    transform_query,
    decide_to_generate,
    init_retriever_node,
    should_continue,
    grade_generation_v_documents_and_question,
)
from selfrag.state import GraphState, GraphStateInput, GraphStateOutput

builder = StateGraph(GraphState, input=GraphStateInput, output=GraphStateOutput)

# Define the nodes
builder.add_node("init_retriever_node", init_retriever_node)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("generate", generate)
builder.add_node("transform_query", transform_query)

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
        # "__end__": END,
    },
)
builder.add_edge("transform_query", "retrieve")
builder.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# memory = MemorySaver()
workflow = builder.compile()