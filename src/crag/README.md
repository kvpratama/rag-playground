# Corrective RAG (CRAG) Implementation

This module implements a Corrective Retrieval-Augmented Generation (CRAG) system using LangGraph. The system enhances traditional RAG with web search capabilities for handling out-of-distribution queries and correcting potential inaccuracies in the retrieved information. The implementation is divided into three main components:

## Workflow Diagram

![CRAG Workflow](../../imgs/crag.png)

*Figure 1: CRAG workflow showing the interaction between different components*

---

## 1. state.py

Defines the state management for the RAG workflow:

- `GraphState`: The main state class that maintains the workflow state including:
  - `question`: The user's query
  - `answer`: The generated response
  - `urls`: List of source URLs
  - `documents`: Retrieved documents
  - `relevant_docs`: Filtered relevant documents
  - `retriever`: Document retriever instance

- `GraphStateInput`: Input state structure
- `GraphStateOutput`: Output state structure

## 2. nodes.py

Contains the core node implementations for the RAG workflow:

- `init_retriever_node`: Initializes the document retriever
- `retrieve`: Retrieves documents based on the query
- `grade_documents`: Evaluates document relevance to the question
- `decide_to_generate`: Determines whether to generate an answer or refine the query
- `transform_query`: Rewrites the query for better web search results
- `web_search`: Performs web search using the transformed query
- `generate`: Generates answers using the LLM

## 3. graph.py

Defines the workflow graph using LangGraph:

- Sets up the state graph with input/output types
- Defines the following nodes:
  - `init_retriever_node`
  - `retrieve`
  - `grade_documents`
  - `generate`
  - `transform_query`
  - `web_search_node`
- Implements conditional edges for workflow control
- Compiles the workflow into an executable graph

## Workflow

The CRAG workflow follows these steps:
1. Initialize retriever with source URLs
2. Retrieve relevant documents
3. Grade document relevance
4. If no relevant documents found:
   - Transform the query for web search
   - Perform web search with the transformed query
5. Generate the final answer using the retrieved documents
6. Return the response

The system includes web search fallback to handle cases where the initial document retrieval doesn't find relevant information.
