from typing import List, Optional
from langgraph.graph import MessagesState
from langchain.schema import BaseRetriever


class GraphState(MessagesState):
    question: str
    generation: str
    urls: List[str]
    documents: List[str]
    iteration: int
    retriever: Optional[BaseRetriever]

class GraphStateInput(MessagesState):
    question: str
    urls: List[str]

class GraphStateOutput(MessagesState):
    generation: str