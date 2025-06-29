# RAG Playground

A collection of advanced Retrieval-Augmented Generation (RAG) implementations using LangGraph. This repository showcases different RAG architectures, each designed to handle various retrieval and generation scenarios.

## Implementations

### 1. RAPTOR

**Overview**: Implements Recursive Abstractive Processing for Tree-Organized Retrieval, providing advanced document processing and hierarchical information retrieval.

**Key Features**:
- Hierarchical document clustering
- Multi-level summarization
- Efficient retrieval from large document collections

[RAPTOR paper](https://arxiv.org/abs/2401.18059)

![RAPTOR Workflow](./imgs/raptor.png)

### 2. Corrective RAG (CRAG)

**Overview**: An enhanced RAG system that incorporates web search capabilities to handle out-of-distribution queries and correct potential inaccuracies in retrieved information.

**Key Features**:
- Web search fallback for unknown queries
- Automatic correction of retrieved information
- Dynamic query transformation

[CRAG paper](https://arxiv.org/abs/2401.15884)

![CRAG Workflow](./imgs/crag.png)

### 3. Self-RAG

**Overview**: A Self-Reflective RAG system that enhances traditional RAG with self-assessment capabilities. It evaluates the quality of retrieved documents and generated responses, enabling iterative improvements.

**Key Features**:
- Self-assessment of document relevance
- Iterative response refinement
- Quality control for generated outputs

[Self-RAG paper](https://arxiv.org/abs/2310.11511)

![Self-RAG Workflow](./imgs/selfrag.png)

### 4. Agentic RAG

**Overview**: Combines LLM agents with RAG, enabling dynamic decision-making during the retrieval and generation process through an agent-based architecture.

**Key Features**:
- Autonomous decision-making for retrieval
- Dynamic tool usage
- Flexible response generation

[Agentic RAG](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_agentic_rag.ipynb)

![Agentic RAG Workflow](./imgs/agenticrag.png)

### 5. Adaptive RAG

**Overview**: Intelligently routes queries based on complexity, choosing between direct generation and a more sophisticated RAG approach.

**Key Features**:
- Automatic query complexity assessment
- Dynamic routing between generation strategies
- Optimized performance for different query types

[Adaptive RAG paper](https://arxiv.org/abs/2403.14403)

![Adaptive RAG Workflow](./imgs/adaptiverag.png)

## Streamlit Frontend

This project includes an intuitive Streamlit-based frontend to streamline experimentation and enhance usability.
Check out our demo [here.](https://rag-playground-kvpratama.streamlit.app/) (*Note: You may encounter errors if the API rate limit is exceeded.*).

## Installation

```bash
# Install uv if you haven't already
pip install uv

# Create a virtual environment
uv venv

# Install the package with development dependencies
uv sync --extra dev

# Activate the virtual environment
source .venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

## Acknowledgments

This project is heavily influenced by excelent RAG tutorials from [LangGraph](https://github.com/langchain-ai/langgraph/tree/main/docs/docs/tutorials/rag)

## License

This project is licensed under the MIT License - see the LICENSE file for details.