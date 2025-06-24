# Adaptive RAG Implementation

This module implements an Adaptive Retrieval-Augmented Generation (RAG) system using LangGraph. The system intelligently routes queries based on their complexity, choosing between direct generation and a more sophisticated Self-RAG approach. The implementation is divided into three main components:

## Workflow Diagram

![Adaptive RAG Workflow](../../imgs/adaptiverag.png)

*Figure 1: Adaptive RAG workflow showing the interaction between different components*

---

## 1. state.py

Defines the state management for the Adaptive RAG workflow:

- `GraphState`: The main state class that maintains the workflow state including:
  - `question`: The user's query
  - `generation`: The generated response
  - `urls`: List of source URLs
  - `documents`: Retrieved documents
  - `relevant_documents`: Filtered relevant documents
  - `min_relevant_documents`: Minimum documents required before generation
  - `iteration`: Current iteration count
  - `max_iterations`: Maximum allowed iterations (determined by question complexity)
  - `retriever`: Document retriever instance

- `GraphStateInput`: Input state structure
- `GraphStateOutput`: Output state structure
- `SubGraphStateInput`: Input structure for the Self-RAG subgraph
- `SubGraphStateOutput`: Output structure for the Self-RAG subgraph

## 2. nodes.py

Contains the core node implementations for the Adaptive RAG workflow:

- `init_retriever_node`: Initializes the document retriever with source URLs
- `should_continue`: Validates if the workflow should proceed
- `question_grader`: Grades the question complexity (0-5)
- `adaptive_routing_node`: Routes the query based on the question grade
- `direct_generation`: Generates a response without retrieval for simple queries

## 3. graph.py

Defines the workflow graph using LangGraph:

- Sets up two main components:
  1. **Self-RAG Subgraph**: For complex queries requiring retrieval
     - `retrieve`: Fetches relevant documents
     - `grade_documents`: Evaluates document relevance
     - `generate`: Creates responses
     - `transform_query`: Refines queries when needed

  2. **Main Graph**: Handles adaptive routing
     - `init_retriever_node`: Initializes the retriever
     - `question_grader`: Assesses question complexity
     - `selfrag_workflow`: Handles complex queries
     - `direct_generation`: Handles simple queries

## Workflow

The Adaptive RAG workflow follows these steps:
1. Initialize the retriever with source URLs
2. Grade the question complexity (0-5):
   - Grade 0: Simple question → Direct generation
   - Grades 1-5: Complex question → Self-RAG workflow
3. For complex questions:
   - Set the maximum iterations based on question grade
   - Enter the Self-RAG subgraph for iterative retrieval and generation
   - Continue until a satisfactory answer is found or max iterations reached
4. Return the generated response

The system optimizes performance by only using the full RAG pipeline when necessary, falling back to direct generation for simple queries.
