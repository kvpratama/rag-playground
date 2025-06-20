from typing import List, Annotated
from langgraph.graph import MessagesState
from operator import add


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
    generate: str