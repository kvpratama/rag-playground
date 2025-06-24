import streamlit as st
from langgraph_client import run_adaptiverag, run_adaptiverag_stream
import uuid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("Adaptive-RAG")

st.markdown("""
Adaptive RAG is a retrieval-augmented generation approach that dynamically selects between no retrieval, single-shot RAG, or iterative RAG based on query analysis to optimize response quality.
""")

st.markdown("""
**Key Features**:
- Automatic query complexity assessment
- Dynamic routing between generation strategies
- Optimized performance for different query types
""")
st.image("imgs/adaptiverag.png", caption="Adaptive-RAG Workflow")

if 'thread_id_adaptiverag' not in st.session_state:
    with st.form("adaptiverag_form"):
        url1 = st.text_input("Enter URL to index:", value="https://lilianweng.github.io/posts/2023-06-23-agent/")
        url2 = st.text_input("Enter URL to index:", value="https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/")
        url3 = st.text_input("Enter URL to index:", value="https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/")
        submitted = st.form_submit_button("Run Adaptive-RAG")

        if submitted:
            if not url1 or not url2 or not url3:
                st.error("Please provide a URL.")
            else:
                with st.spinner("Running workflow..."):
                    try:
                        # Generate a unique thread_id per session
                        st.session_state["urls_adaptiverag"] = [url1, url2, url3]
                        st.session_state['thread_id_adaptiverag'] = str(uuid.uuid4())
                        thread_id = st.session_state['thread_id_adaptiverag']
                        
                        # Initialize vectorstore
                        _ = run_adaptiverag(st.session_state["urls_adaptiverag"], "", thread_id)
                    except Exception as e:
                        st.session_state['chat_history_adaptiverag'].append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()

else:
    st.markdown("---")
    st.markdown(f"Ask your question about: {st.session_state['urls_adaptiverag']}")
    # Initialize chat history in session state
    if 'chat_history_adaptiverag' not in st.session_state:
        st.session_state['chat_history_adaptiverag'] = []

    # Render chat history as LLM chat bubbles
    for msg in st.session_state['chat_history_adaptiverag']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input(f"Ask your question"):
        st.session_state['chat_history_adaptiverag'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking..."):
                with st.empty():
                    response_stream = run_adaptiverag_stream(st.session_state['urls_adaptiverag'], prompt, st.session_state['thread_id_adaptiverag'])
                    response = ""
                    for chunk in response_stream:
                        st.write(chunk)
                    response = chunk
        st.session_state['chat_history_adaptiverag'].append({"role": "assistant", "content": response})