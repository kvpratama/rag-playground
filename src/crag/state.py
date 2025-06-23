from typing import List, Annotated, Optional
from langgraph.graph import MessagesState
from operator import add
from langchain.schema import BaseRetriever

class GraphState(MessagesState):
    question: str
    answer: str
    urls: List[str]
    documents: List[str]
    relevant_docs: Annotated[List[str], add]
    retriever: Optional[BaseRetriever]

class GraphStateInput(MessagesState):
    question: str
    urls: List[str]

class GraphStateOutput(MessagesState):
    answer: str