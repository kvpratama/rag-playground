from typing import List
from langgraph.graph import MessagesState


class GraphState(MessagesState):
    question: str
    generation: str
    urls: List[str]
    documents: List[str]
    iteration: int

class GraphStateInput(MessagesState):
    question: str
    urls: List[str]

class GraphStateOutput(MessagesState):
    generation: str