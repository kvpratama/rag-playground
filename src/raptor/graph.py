from langgraph.graph import START, END, StateGraph
from raptor.state import GraphState, GraphStateInput, GraphStateOutput
from raptor.nodes import init_retriever_node, rag_node
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(GraphState, input=GraphStateInput, output=GraphStateOutput)
builder.add_node("init_retriever_node", init_retriever_node)
builder.add_node("rag_node", rag_node)

builder.add_edge(START, "init_retriever_node")
builder.add_edge("init_retriever_node", "rag_node")
builder.add_edge("rag_node", END)

# Compile
memory = MemorySaver()
workflow = builder.compile(checkpointer=memory)