from typing import Dict, List, Optional, Tuple
from langchain_core.output_parsers import StrOutputParser
from raptor.raptor import build_vectorstore
from raptor.state import GraphState, GraphStateOutput
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model
import logging

logger = logging.getLogger(__name__)


# def init_retriever_node(state: GraphState, configurable: Dict):
#     """
#     Initialize the retriever node for the graph state.
#     """

#     logger.info("Retriever initialized successfully.")
#     return state


def rag_node(state: GraphState, config: Dict):
    """
    RAG node that retrieves relevant documents based on the question.
    """
    logger.info(f'Running RAG node with question: {state["question"]}')

    logger.info(f'Initializing retriever node with URL: {state["url"]} and max depth: {state["max_depth"]}')
    retriever = build_vectorstore(config["configurable"]["thread_id"], state["url"], state["max_depth"])

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | hub.pull("rlm/rag-prompt")
        | init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")
        | StrOutputParser()
    )

    # Question
    answer = rag_chain.invoke(state["question"])

    logger.info(f"RAG node completed with answer: {answer}")
    return {"answer": answer}
