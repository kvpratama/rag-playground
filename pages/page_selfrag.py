import streamlit as st
from langgraph_client import run_selfrag, run_selfrag_stream
import uuid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("Self-RAG")

st.markdown("""
Self-RAG is a Retrieval-Augmented Generation (RAG) approach that integrates self-evaluation or self-assessment of retrieved documents and generation.
""")

if 'thread_id_selfrag' not in st.session_state:
    with st.form("selfrag_form"):
        url1 = st.text_input("Enter URL to index:", value="https://lilianweng.github.io/posts/2023-06-23-agent/")
        url2 = st.text_input("Enter URL to index:", value="https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/")
        url3 = st.text_input("Enter URL to index:", value="https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/")
        submitted = st.form_submit_button("Run Selfrag")

        if submitted:
            if not url1 or not url2 or not url3:
                st.error("Please provide a URL.")
            else:
                with st.spinner("Running workflow..."):
                    try:
                        # Generate a unique thread_id per session
                        st.session_state["urls_selfrag"] = [url1, url2, url3]
                        st.session_state['thread_id_selfrag'] = str(uuid.uuid4())
                        thread_id = st.session_state['thread_id_selfrag']
                        
                        # Initialize vectorstore
                        _ = run_selfrag(st.session_state["urls_selfrag"], "", thread_id)
                    except Exception as e:
                        st.session_state['chat_history_selfrag'].append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()

else:
    st.markdown("---")
    st.markdown(f"Ask your question about: {st.session_state['urls_selfrag']}")
    # Initialize chat history in session state
    if 'chat_history_selfrag' not in st.session_state:
        st.session_state['chat_history_selfrag'] = []

    # Render chat history as LLM chat bubbles
    for msg in st.session_state['chat_history_selfrag']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input(f"Ask your question"):
        st.session_state['chat_history_selfrag'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking..."), st.empty():
                response_stream = run_selfrag_stream(st.session_state['urls_selfrag'], prompt, st.session_state['thread_id_selfrag'])
                response = ""
                for chunk in response_stream:
                    st.write(chunk)
                response = chunk
        st.session_state['chat_history_selfrag'].append({"role": "assistant", "content": response})
