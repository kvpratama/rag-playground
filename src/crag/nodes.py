from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
from crag.state import GraphState
from commonlib.vectorstore_utils import build_vectorstore
from langchain_tavily import TavilySearch
import logging


logger = logging.getLogger(__name__)


def init_retriever_node(state: GraphState, config: Dict):
    """
    Initialize the retriever node for the graph state.
    """

    logger.info(f'Initializing retriever node with URL: {state["urls"]}')
    _ = build_vectorstore(config["configurable"]["thread_id"], state["urls"])

    logger.info("Retriever initialized successfully.")
    return {}


def should_continue(state: GraphState, config: Dict):
    """
    Determine whether to continue the workflow.

    Args:
        state (dict): The current graph state

    Returns:
        str: The next node to call
    """

    logger.info(f"Checking if should continue with question: {state['question']}")
    if "question" not in state or state["question"] == "":
        logger.info("Question is empty or not in state, ending workflow.")
        return "__end__"

    logger.info("Question is not empty, continuing workflow.")
    return "retrieve"


def retrieve(state: GraphState, config: Dict):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logger.info("---RETRIEVE---")
    question = state["question"]
    logger.info(f"Question: {question}")

    # Retrieval
    logger.info(f"Retrieving documents for thread_id: {config['configurable']['thread_id']}")
    retriever = build_vectorstore(config["configurable"]["thread_id"], state["urls"])
    documents = retriever.invoke(question)
    logger.info(f"Retrieved documents: {len(documents)}")
    return {"documents": documents}


def grade_documents(state: GraphState, config: Dict):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )


    # LLM with function call
    llm = init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    # Score each doc
    filtered_docs = []
    
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            logger.info("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs}


def decide_to_generate(state: GraphState, config: Dict):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    logger.info("---ASSESS GRADED DOCUMENTS---")

    if len(state["documents"]) == 0:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.info(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"

    else:
        # We have relevant documents, so generate answer
        logger.info("---DECISION: GENERATE---")
        return "generate"


def transform_query(state: GraphState, config: Dict):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    logger.info("---TRANSFORM QUERY---")
    question = state["question"]
    # documents = state["documents"]

    class RewriteQuestion(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        reasoning: str = Field(
            description="Reasoning behind the re-written question, explaining how it improves searchability.")
        improved_question: str = Field(
            description="The final improved question that is optimized for web search and ready to be use as a search query.")
    
    # LLM
    llm = init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")
    structured_llm_rewriter = llm.with_structured_output(RewriteQuestion)

    # Prompt
    system = """You are a question re-writer that converts an input question to a better version that is optimized for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n\n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | structured_llm_rewriter

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question.improved_question}


def web_search(state: GraphState, config: Dict):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    logger.info("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    web_search_tool = TavilySearch(max_results=3)
    docs = web_search_tool.invoke({"query": question})

    if "results" in docs and docs["results"]:
        # Extract meaningful content from each result
        for d in docs["results"]:
            web_result = f"Title: {d.get('title', 'N/A')}\n" + \
            f"URL: {d.get('url', 'N/A')}\n" + \
            f"Content: {d.get('content', d.get('snippet', 'No content available'))}"
            
            web_document = Document(
                page_content=web_result,
                metadata={"source": "tavily_search", "query": question}
            )
            documents.append(web_document)
    else:
        logger.info("No search results found")
    return {"documents": documents, "question": question}


def generate(state: GraphState, config: Dict):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, answer, that contains LLM generation
    """
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # RAG generation
    answer = rag_chain.invoke({"context": documents, "question": question})
    return {"answer": answer}
