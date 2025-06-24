# Agentic RAG Implementation

This module implements an Agentic Retrieval-Augmented Generation (RAG) system using LangGraph. The system combines the power of LLM agents with RAG, enabling dynamic decision-making during the retrieval and generation process. The implementation is divided into three main components:

## Workflow Diagram

![Agentic RAG Workflow](../../imgs/agenticrag.png)

*Figure 1: Agentic RAG workflow showing the interaction between different components*

---

## 1. state.py

Defines the state management for the RAG workflow:

- `GraphState`: The main state class that maintains the workflow state including:
  - `question`: The user's query
  - `generation`: The generated response
  - `urls`: List of source URLs
  - `documents`: Retrieved documents
  - `iteration`: Current iteration count
  - `retriever`: Document retriever instance

- `GraphStateInput`: Input state structure
- `GraphStateOutput`: Output state structure

## 2. nodes.py

Contains the core node implementations for the Agentic RAG workflow:

- `init_retriever_node`: Initializes the document retriever with source URLs
- `get_retriever_tool`: Creates a retriever tool for the agent
- `get_tool_node`: Sets up the tool node for the agent
- `generate_query_or_respond`: Decides whether to retrieve or respond directly
- `grade_documents`: Evaluates document relevance to the question
- `rewrite_question`: Rewrites the query for better retrieval
- `generate_answer`: Generates the final answer using the LLM

## 3. graph.py

Defines the workflow graph using LangGraph:

- Sets up the state graph with input/output types
- Defines the following nodes:
  - `init_retriever_node`
  - `generate_query_or_respond`
  - `tools` (for document retrieval)
  - `rewrite_question`
  - `generate_answer`
- Implements conditional edges for dynamic workflow control
- Compiles the workflow into an executable graph

## Workflow

The Agentic RAG workflow follows these steps:
1. Initialize retriever with source URLs
2. Generate a query or respond directly based on the question
3. If retrieval is needed:
   - Use the retriever tool to fetch documents
   - Grade the relevance of retrieved documents
   - If documents are relevant, generate an answer
   - If documents are not relevant, rewrite the question and retry (up to 3 times)
4. Generate and return the final answer

The system includes an agentic component that can decide when to retrieve information and when to respond directly, making it more flexible than traditional RAG systems.
