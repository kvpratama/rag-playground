from typing import List, Annotated
from langgraph.graph import MessagesState
from operator import add


class GraphState(MessagesState):
    question: str
    answer: str
    urls: List[str]
    documents: List[str]
    relevant_docs: Annotated[List[str], add]

class GraphStateInput(MessagesState):
    question: str
    urls: List[str]

class GraphStateOutput(MessagesState):
    answer: str