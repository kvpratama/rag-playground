from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Literal
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from agenticrag.state import GraphState
from commonlib.vectorstore_utils import build_vectorstore
from langgraph.config import get_stream_writer
import logging


logger = logging.getLogger(__name__)

    
def init_retriever_node(state: GraphState, config: Dict):
    """
    Initialize the retriever node for the graph state.
    """

    logger.info(f'Initializing retriever node with URL: {state["urls"]}')
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Building vectorstore...*\n"})

    _ = build_vectorstore(config["configurable"]["thread_id"], state["urls"])

    logger.info("Retriever initialized successfully.")
    stream_writer({"custom_key": "*Retriever initialized successfully.*\n"})
    return {}


def should_continue(state: GraphState, config: Dict):
    """
    Determine whether to continue the workflow.

    Args:
        state (dict): The current graph state

    Returns:
        str: The next node to call
    """

    logger.info("Checking if should continue with question")
    if "question" not in state or state["question"] == "":
        logger.info("Question is empty or not in state, ending workflow.")
        return "__end__"
    
    logger.info(f"Question is not empty, continuing workflow with question: {state['question']}")
    return "generate_query_or_respond"


def get_retriever_tool(retriever):
    document_prompt = PromptTemplate.from_template(
        " Content: {page_content}\n Source: {source} \n"
    )

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_documents",
        "Search and return information from Vectorstore based on the user query.",
        document_prompt=document_prompt,
    )
    return retriever_tool


def get_tool_node(state: GraphState, config: Dict):
    retriever = build_vectorstore(config["configurable"]["thread_id"], state["urls"])
    return ToolNode([get_retriever_tool(retriever)])


def generate_query_or_respond(state: GraphState, config: Dict):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    retriever = build_vectorstore(config["configurable"]["thread_id"], state["urls"])
    retriever_tool = get_retriever_tool(retriever)
    iteration = state.get("iteration", 0)
    stream_writer = get_stream_writer()

    if len(state["messages"]) == 0:
        logger.info("Initializing assistant with system message")
        stream_writer({"custom_key": "*Initializing assistant with system message...*\n"})
        state["messages"].append(SystemMessage(content="You are a helpful assistant. Whenever a user asks a question, you will always try to retrieve relevant information from the vectorstore."))

    logger.info("Generating query or response")
    stream_writer({"custom_key": "*Generating query or response...*\n"})
    state["messages"].append(HumanMessage(content=state["question"]))
    response_model = init_chat_model("gemini-2.0-flash", temperature=0, model_provider="google_genai")
    response = (
        response_model
        # highlight-next-line
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    stream_writer({"custom_key": f"*{response.content}*\n"})
    return {"messages": [response], "iteration": iteration + 1}


def grade_documents(state: GraphState, config: Dict) -> Literal["generate_answer", "rewrite_question"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["messages"][-1].content
    iteration = state.get("iteration", 0)
    stream_writer = get_stream_writer()

    logger.info("Checking document relevance to question")
    stream_writer({"custom_key": "*Checking document relevance to question...*\n"})

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )
        reasoning: str = Field(
            description="Reasoning for the relevance score"
        )


    # LLM with function call
    llm = init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", GRADE_PROMPT),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    response = retrieval_grader.invoke({"question": question, "context": documents})
    score = response.binary_score
    
    if score == "yes" or iteration >= 3:
        logger.info(f"Documents are relevant to the question, generating answer. Iteration: {iteration}")
        stream_writer({"custom_key": "*Documents are relevant to the question, generating answer...*\n"})
        return "generate_answer"
    else:
        logger.info(f"Documents are not relevant to the question, rephrasing question. Iteration: {iteration}")
        stream_writer({"custom_key": "*Documents are not relevant to the question, rephrasing question...*\n"})
        return "rewrite_question"


def rewrite_question(state: GraphState, config: Dict):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    logger.info("---Rewrite Question---")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Rewriting question...*\n"})
    question = state["question"]
    # documents = state["documents"]

    class RewriteQuestion(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        reasoning: str = Field(
            description="Reasoning behind the re-written question, explaining how it improves searchability.")
        improved_question: str = Field(
            description="The final improved question that is optimized for vectorstore retrieval and ready to be use as a search query.")
    
    # LLM
    llm = init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")
    structured_llm_rewriter = llm.with_structured_output(RewriteQuestion)

    # Prompt
    REWRITE_PROMPT = """You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. Here is the initial question: \n\n {question} \n\n Formulate an improved question."""
    re_write_prompt = ChatPromptTemplate.from_messages([("human", REWRITE_PROMPT),])
    question_rewriter = re_write_prompt | structured_llm_rewriter
    better_question = question_rewriter.invoke({"question": question})
    
    ai_message = AIMessage(content="The retrieved documents are not relevant to the question. Try to rephrase the question to make it more search-friendly.")

    logger.info(f"Rewritten question: {better_question.improved_question}")
    stream_writer({"custom_key": f"*Rewritten question: {better_question.improved_question}*\n"})
    return {"messages": [ai_message], "question": better_question.improved_question}


def generate_answer(state: GraphState, config: Dict):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, answer, that contains LLM generation
    """

    logger.info("---GENERATE ANSWER---")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Generating answer...*\n"})
    question = state["question"]
    documents = state["messages"][-1].content

    # Prompt
    GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}")
    generate_prompt = ChatPromptTemplate.from_messages([("human", GENERATE_PROMPT),])

    # LLM
    llm = init_chat_model("gemma-3-12b-it", temperature=0, model_provider="google_genai")

    # Chain
    rag_chain = generate_prompt | llm #| StrOutputParser()

    # RAG generation
    answer = rag_chain.invoke({"context": documents, "question": question})

    logger.info(f"Generated answer: {answer.content}")
    stream_writer({"custom_key": f"*{answer.content}*\n"})
    return {"messages": [answer], "generation": answer.content}
