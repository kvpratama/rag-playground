from langgraph.graph import END, StateGraph, START
from selfrag.nodes import (
    grade_documents,
    generate,
    retrieve,
    transform_query,
    decide_to_generate,
    grade_generation_v_documents_and_question,
)
from adaptiverag.nodes import init_retriever_node, should_continue, question_grader
from adaptiverag.state import GraphState, GraphStateInput, GraphStateOutput, SubGraphStateInput, SubGraphStateOutput


# Self-RAG subgraph
selfrag_subgraph = StateGraph(GraphState, input=SubGraphStateInput, output=SubGraphStateOutput)
selfrag_subgraph.add_node("retrieve", retrieve)
selfrag_subgraph.add_node("grade_documents", grade_documents)
selfrag_subgraph.add_node("generate", generate)
selfrag_subgraph.add_node("transform_query", transform_query)

selfrag_subgraph.add_edge(START, "retrieve")
selfrag_subgraph.add_edge("retrieve", "grade_documents")
selfrag_subgraph.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
selfrag_subgraph.add_edge("transform_query", "retrieve")
selfrag_subgraph.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

selfrag_workflow = selfrag_subgraph.compile()
# selfrag_workflow = selfrag_subgraph.compile(checkpointer=False) # For Langgraph Studio

# Main graph
builder = StateGraph(GraphState, input=GraphStateInput, output=GraphStateOutput)
builder.add_node("init_retriever_node", init_retriever_node)
builder.add_node("question_grader", question_grader)
builder.add_node("subgraph", selfrag_workflow)

builder.add_edge(START, "init_retriever_node")
builder.add_conditional_edges("init_retriever_node", should_continue, {
    "question_grader": "question_grader",
    END: END,
})
builder.add_edge("question_grader", "subgraph")
builder.add_edge("subgraph", END)

# memory = MemorySaver()
workflow = builder.compile()