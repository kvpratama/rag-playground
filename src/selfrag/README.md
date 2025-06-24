# Self-RAG Implementation

This module implements a Self-Reflective Retrieval-Augmented Generation (Self-RAG) system using LangGraph. The implementation is divided into three main components:

## Workflow Diagram

![Self-RAG Workflow](../../imgs/selfrag.png)

*Figure 1: Self-RAG workflow showing the interaction between different components*

---

## 1. state.py

Defines the state management for the RAG workflow:

- `GraphState`: The main state class that maintains the workflow state including:
  - `question`: The user's query
  - `generation`: The generated response
  - `urls`: List of source URLs
  - `documents`: Retrieved documents
  - `relevant_documents`: Filtered relevant documents
  - `min_relevant_documents`: Minimum documents required before generation
  - `iteration`: Current iteration count
  - `max_iterations`: Maximum allowed iterations
  - `retriever`: Document retriever instance

- `GraphStateInput`: Input state structure
- `GraphStateOutput`: Output state structure

## 2. nodes.py

Contains the core node implementations for the RAG workflow:

- `init_retriever_node`: Initializes the document retriever
- `retrieve`: Retrieves documents based on the query
- `grade_documents`: Evaluates document relevance to the question
- `decide_to_generate`: Determines whether to generate an answer or refine the query
- `transform_query`: Rewrites the query for better results
- `generate`: Generates answers using the LLM
- `hallucination_grader`: Checks for hallucinations in the generation
- `answer_grader`: Evaluates if the answer addresses the question
- `grade_generation_v_documents_and_question`: Main grading function for the generation quality

## 3. graph.py

Defines the workflow graph using LangGraph:

- Sets up the state graph with input/output types
- Defines the following nodes:
  - `init_retriever_node`
  - `retrieve`
  - `grade_documents`
  - `generate`
  - `transform_query`
- Implements conditional edges for workflow control
- Compiles the workflow into an executable graph

## Workflow

The RAG workflow follows these steps:
1. Initialize retriever with source URLs
2. Retrieve relevant documents
3. Grade document relevance
4. Decide whether to generate an answer or refine the query
5. If needed, transform the query and repeat from step 2
6. Generate the final answer
7. Check for hallucination and answer relevance
8. If needed, refine the answer and repeat from step 6 or transform the query and repeat from step 2
9. Return the response

The system includes self-reflection mechanisms to ensure answer quality and relevance.
