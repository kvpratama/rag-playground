import streamlit as st
from langgraph_client import run_raptor
import uuid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("RAPTOR")

st.markdown("""
Implements Recursive Abstractive Processing for Tree-Organized Retrieval (RAPTOR), providing advanced document processing and hierarchical information retrieval.
""")
st.markdown("""
**Key Features**:
- Hierarchical document clustering
- Multi-level summarization
- Efficient retrieval from large document collections
""")
st.image("imgs/raptor.png", caption="RAPTOR Workflow")

if 'thread_id_raptor' not in st.session_state:
    with st.form("rag_form"):
        url = st.text_input("Enter URL to index:", value="https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/")
        max_depth = st.number_input("Max Depth", min_value=1, max_value=10, value=2)
        submitted = st.form_submit_button("Run RAG")

        if submitted:
            if not url:
                st.error("Please provide a URL.")
            else:
                with st.spinner("Running workflow..."):
                    try:
                        # Generate a unique thread_id_raptor per session
                        st.session_state["url_raptor"] = url
                        st.session_state['max_depth_raptor'] = max_depth
                        st.session_state['thread_id_raptor'] = str(uuid.uuid4())
                        thread_id_raptor = st.session_state['thread_id_raptor']
                        
                        # Initialize vectorstore
                        _ = run_raptor(url, max_depth, "", thread_id_raptor)
                    except Exception as e:
                        st.session_state['chat_history_raptor'].append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()

else:
    st.markdown("---")
    st.markdown(f"Ask your question about: {st.session_state['url_raptor']}")
    # Initialize chat history in session state
    if 'chat_history_raptor' not in st.session_state:
        st.session_state['chat_history_raptor'] = []

    # Render chat history as LLM chat bubbles
    for msg in st.session_state['chat_history_raptor']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input(f"Ask your question"):
        st.session_state['chat_history_raptor'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = run_raptor(st.session_state['url_raptor'], st.session_state['max_depth_raptor'], prompt, st.session_state['thread_id_raptor'])
            st.markdown(response)
        
        st.session_state['chat_history_raptor'].append({"role": "assistant", "content": response})
