from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
from selfrag.state import GraphState
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
    retriever = build_vectorstore(config["configurable"]["thread_id"], state["urls"])

    logger.info("Retriever initialized successfully.")
    stream_writer({"custom_key": "*Retriever initialized successfully.*\n"})
    return {"retriever": retriever}


def should_continue(state: GraphState, config: Dict):
    """
    Determine whether to continue the workflow.

    Args:
        state (dict): The current graph state

    Returns:
        str: The next node to call
    """

    logger.info("Checking if should continue with question")
    stream_writer = get_stream_writer()
    if "question" not in state or state["question"] == "":
        logger.info("Question is empty or not in state, ending workflow.")
        stream_writer({"custom_key": "*Question is empty or not in state, ending workflow.*\n"})
        return "__end__"
    
    logger.info(f"Question is not empty, continuing workflow with question: {state['question']}")
    stream_writer({"custom_key": f"*Question is not empty, continuing workflow with question: {state['question']}*\n"})
    return "retrieve"


def retrieve(state: GraphState, config: Dict):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    iteration = state.get("iteration", 0)
    stream_writer = get_stream_writer()
    logger.info("---RETRIEVE---")
    question = state["question"]
    logger.info(f"Question: {question}")
    stream_writer({"custom_key": f"*Question: {question}*\n"})

    # Retrieval
    logger.info(f"Retrieving documents for thread_id: {config['configurable']['thread_id']}")
    retriever = state["retriever"]
    documents = retriever.invoke(question)
    logger.info(f"Retrieved documents: {len(documents)}")
    stream_writer({"custom_key": f"*Retrieved documents: {len(documents)}*\n"})
    return {"documents": documents, "iteration": iteration + 1}


def grade_documents(state: GraphState, config: Dict):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Checking document relevance to question...*\n"})
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
            stream_writer({"custom_key": "*Document is relevant to question.*\n"})
            filtered_docs.append(d)
        else:
            logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
            stream_writer({"custom_key": "*Document is not relevant to question.*\n"})
            continue
    return {"relevant_documents": filtered_docs}


def decide_to_generate(state: GraphState, config: Dict):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    logger.info("---ASSESS GRADED DOCUMENTS---")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Assessing graded documents...*\n"})
    min_relevant_documents = state.get("min_relevant_documents", 3)
    max_iterations = state.get("max_iterations", 5)
    
    if state["iteration"] > max_iterations:
        logger.info("---DECISION: MAX ITERATIONS REACHED, END---")
        stream_writer({"custom_key": "*Max iterations reached, ending workflow.*\n"})
        return "generate"
    
    elif len(state["relevant_documents"]) < min_relevant_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.info(
            f"---DECISION: RELEVANT DOCUMENTS COUNT < {min_relevant_documents}, TRANSFORM QUERY---"
        )
        stream_writer({"custom_key": f"*Relevant documents count < {min_relevant_documents}, transforming query.*\n"})
        return "transform_query"

    else:
        # We have relevant documents, so generate answer
        logger.info("---DECISION: GENERATE---")
        stream_writer({"custom_key": "*We have relevant documents, generating answer.*\n"})
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
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Transforming query...*\n"})
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
    system = """You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
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
    logger.info(f"Improved question: {better_question.improved_question}")
    stream_writer({"custom_key": f"*Improved question: {better_question.improved_question}*\n"})
    return {"question": better_question.improved_question}


def generate(state: GraphState, config: Dict):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, answer, that contains LLM generation
    """
    iteration = state.get("iteration", 0)
    logger.info("---GENERATE---")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Generating answer...*\n"})
    question = state["question"]
    documents = state["relevant_documents"]

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # RAG generation
    answer = rag_chain.invoke({"context": documents, "question": question})
    logger.info(f"Answer: {answer}")
    stream_writer({"custom_key": f"*{answer}*"})
    return {"generation": answer, "iteration": iteration + 1}


def hallucination_grader(documents, generation):
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )

    logger.info("---HALLUCINATION GRADER---")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Checking for hallucination...*\n"})
    
    # LLM with function call
    llm = init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader_chain = hallucination_prompt | structured_llm_grader

    score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score
    logger.info(f"Is generation grounded in facts: {grade}")
    stream_writer({"custom_key": f"*Is generation grounded in facts: {grade}*\n"})
    return grade


def answer_grader(question, generation):
    # Data model
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )

    logger.info("---ANSWER GRADER---")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Checking if answer addresses question...*\n"})

    # LLM with function call
    llm = init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader_chain = answer_prompt | structured_llm_grader
    answer_grade = answer_grader_chain.invoke({"question": question, "generation": generation})
    logger.info(f"Is answer addresses question: {answer_grade.binary_score}")
    stream_writer({"custom_key": f"*Is answer addresses question: {answer_grade.binary_score}*\n"})
    return answer_grade.binary_score


def grade_generation_v_documents_and_question(state: GraphState, config: Dict):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    logger.info("---GRADE GENERATION vs DOCUMENTS AND QUESTION---")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Checking if generation is grounded in documents and addresses question...*\n"})
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)

    if iteration > max_iterations:
        logger.info("Iteration limit reached, ending workflow.")
        stream_writer({"custom_key": "*Iteration limit reached, ending workflow.*\n"})
        stream_writer({"custom_key": f"*{state['generation']}*"})
        return "useful"
    
    question = state["question"]
    documents = state["relevant_documents"]
    generation = state["generation"]

    logger.info("---CHECK HALLUCINATIONS---")
    hallucination_grade = hallucination_grader(documents, generation)

    if hallucination_grade == "yes":
        logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        stream_writer({"custom_key": "*Generation is grounded in documents.*\n"})
        # Check question-answering
        logger.info("---GRADE GENERATION vs QUESTION---")
        answer_grade = answer_grader(question, generation)
        if answer_grade == "yes":
            logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
            stream_writer({"custom_key": "*Generation addresses question.*\n"})
            stream_writer({"custom_key": f"*{generation}*"})
            return "useful"
        else:
            logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            stream_writer({"custom_key": "*Generation does not address question.*\n"})
            return "not useful"
    else:
        logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        stream_writer({"custom_key": "*Generation is not grounded in documents, re-try.*\n"})
        return "not supported"