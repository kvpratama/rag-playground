# RAPTOR Implementation

This module implements the RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) system, which provides advanced document processing and retrieval capabilities. The implementation is divided into four main components:

## Workflow Diagram

![RAPTOR Workflow](../../imgs/raptor.png)

*Figure 1: RAPTOR workflow showing the interaction between different components*

---

## 1. state.py

Defines the state management for the RAPTOR workflow:

- `GraphState`: The main state class that maintains the workflow state including:
  - `url`: The source URL for document retrieval
  - `max_depth`: Maximum depth for recursive URL crawling
  - `question`: The user's query
  - `answer`: The generated response

- `GraphStateInput`: Input state structure
- `GraphStateOutput`: Output state structure

## 2. raptor.py

Core implementation of the RAPTOR algorithm with these key functions:

- **Embedding and Clustering**:
  - `global_cluster_embeddings`: Performs global dimensionality reduction using UMAP
  - `local_cluster_embeddings`: Handles local dimensionality reduction
  - `get_optimal_clusters`: Determines optimal cluster count using BIC
  - `GMM_cluster`: Clusters embeddings using Gaussian Mixture Models
  - `perform_clustering`: Orchestrates the clustering process

- **Document Processing**:
  - `embed`: Generates embeddings for text documents
  - `embed_cluster_texts`: Combines embedding and clustering
  - `recursive_embed_cluster_summarize`: Recursively processes text with increasing abstraction
  - `url_loader`: Loads documents from URLs with recursive crawling
  - `build_vectorstore`: Creates a vector store from URL content

## 3. nodes.py

Contains the node implementations for the RAG workflow:

- `init_retriever_node`: Initializes the RAPTOR retriever with URL and depth parameters
- `rag_node`: Implements the RAG (Retrieval-Augmented Generation) functionality:
  - Formats documents for retrieval
  - Sets up the RAG chain with prompt and model
  - Generates answers based on retrieved content

## 4. graph.py

Defines the workflow graph using LangGraph:

- Sets up a simple linear workflow:
  1. `init_retriever_node`: Initializes the RAPTOR retriever
  2. `rag_node`: Processes the question and generates answers

## Workflow

The RAPTOR workflow follows these steps:
1. Initialize the RAPTOR retriever with a source URL and maximum crawl depth
2. Load and process documents from the URL using recursive crawling
3. For each document:
   - Split into chunks
   - Generate embeddings
   - Perform hierarchical clustering
   - Create multi-level summaries
4. When a question is received:
   - Retrieve relevant document chunks using the RAPTOR index
   - Generate a response using the RAG pattern
   - Return the answer to the user

The system is particularly effective for processing and retrieving information from large, complex documents or websites with deep hierarchical structures.
