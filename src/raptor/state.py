from langgraph.graph import MessagesState

class GraphState(MessagesState):
    url: str
    max_depth: int
    question: str
    answer: str

class GraphStateInput(MessagesState):
    url: str
    max_depth: int
    question: str

class GraphStateOutput(MessagesState):
    answer: str
