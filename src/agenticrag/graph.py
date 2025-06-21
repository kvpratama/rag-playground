from langgraph.graph import END, StateGraph, START

from langgraph.prebuilt import tools_condition
from agenticrag.nodes import (
    init_retriever_node,
    grade_documents,
    generate_query_or_respond,
    rewrite_question,
    generate_answer,
    should_continue,
    get_tool_node,
)
from agenticrag.state import GraphState, GraphStateInput, GraphStateOutput

builder = StateGraph(GraphState, input=GraphStateInput, output=GraphStateOutput)

builder.add_node(init_retriever_node)
builder.add_node(generate_query_or_respond)
builder.add_node("tools", get_tool_node)
builder.add_node(rewrite_question)
builder.add_node(generate_answer)

builder.add_edge(START, "init_retriever_node")
builder.add_conditional_edges("init_retriever_node", should_continue, {
    "generate_query_or_respond": "generate_query_or_respond",
    END: END,
})
# Decide whether to retrieve
builder.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "tools",
        END: END,
    },
)

# Edges taken after the `action` node is called.
builder.add_conditional_edges(
    "tools",
    # Assess agent decision
    grade_documents,
)
builder.add_edge("generate_answer", END)
builder.add_edge("rewrite_question", "generate_query_or_respond")

workflow = builder.compile()