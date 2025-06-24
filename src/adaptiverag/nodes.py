from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from adaptiverag.state import GraphState
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
    return "question_grader"


def question_grader(state: GraphState, config: Dict):
    """
    Grade the question.

    Args:
        state (dict): The current graph state

    Returns:
        str: The next node to call
    """
    logger.info("Grading question")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Grading question...*\n"})

    class GradeQuery(BaseModel):
        """Grade a user query for difficulty. The grade will range from 0-5. """

        grade: Literal["0", "1", "2", "3", "4", "5"] = Field(
            ...,
            description="Given a user question grade the difficulty for a RAG system.",
        )
        reasoning: str = Field(
            ...,
            description="Provide a reasoning for the grade you assigned to the user question.",
        )

    # LLM with function call
    llm = init_chat_model("gemini-2.0-flash-lite", temperature=0, model_provider="google_genai")
    structured_llm_router = llm.with_structured_output(GradeQuery)

    # Prompt
    system = """You are an expert at grading the difficulty of a user question for a RAG system. Your grade will range from 0-5. Grade 0 means no retrieval is needed. Grade 1 is a question that can be answered with a single shot retrieval from a vectorstore, while grade 5 is a question that might require up to 5 retrievals querying a vectorstore with each retrieval being a follow-up to the previous one."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Rate the difficulty of the following question: \n\n {question}"),
        ]
    )

    grade_question = grade_prompt | structured_llm_router
    grade = grade_question.invoke({"question": state["question"]})

    logger.info(f"Question graded with grade: {grade.grade} and reasoning: {grade.reasoning}")
    stream_writer({"custom_key": f"*Question graded with grade: {grade.grade}*\n"})
    return {"max_iterations": int(grade.grade)}


def adaptive_routing_node(state: GraphState, config: Dict):
    """
    Determine the next node to call based on the question grade.
    """
    logger.info("Adaptive routing node")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Adaptive routing node...*\n"})
    
    if state["max_iterations"] == 0:
        logger.info("Question grade is 0, answering with no retrieval.")
        stream_writer({"custom_key": "*Question grade is 0, answering with no retrieval.*\n"})
        return "direct_generation"
    
    logger.info("Question grade is not 0, continuing workflow.")
    stream_writer({"custom_key": "*Question grade is not 0, continuing workflow.*\n"})
    return "subgraph"
    

def direct_generation(state: GraphState, config: Dict):
    """
    Generate a response without retrieval.
    """
    logger.info("Answering with no retrieval")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Answering with no retrieval...*\n"})
    
    # LLM with function call
    llm = init_chat_model("gemma-3-12b-it", temperature=0, model_provider="google_genai")

    # Prompt
    prompt = """You are a helpful RAG assistant. Answer the following question: \n\n {question} \n\n Make sure to be respectful and concise. End your response with a courteous offer to assist further using your retrieval-augmented capabilities."""
    generation_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", prompt),
        ]
    )
    
    generation_chain = generation_prompt | llm | StrOutputParser()
    generation = generation_chain.invoke({"question": state["question"]})
    logger.info(f"Generated answer: {generation}")
    stream_writer({"custom_key": f"*{generation}*\n"})
    return {"generation": generation}