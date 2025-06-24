from typing import List, Annotated, Optional
from langgraph.graph import MessagesState
from operator import add
from langchain.schema import BaseRetriever


class GraphState(MessagesState):
    question: str
    generation: str
    urls: List[str]
    documents: List[str]
    relevant_documents: Annotated[List[str], add]
    min_relevant_documents: int
    iteration: int
    max_iterations: int
    retriever: Optional[BaseRetriever]

class GraphStateInput(MessagesState):
    question: str
    urls: List[str]

class GraphStateOutput(MessagesState):
    generation: str

class SubGraphStateInput(MessagesState):
    question: str
    retriever: Optional[BaseRetriever]
    max_iterations: int

class SubGraphStateOutput(MessagesState):
    generation: str