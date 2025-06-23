import streamlit as st
from langgraph_client import run_crag, run_crag_stream
import uuid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("Corrective RAG (CRAG)")

st.markdown("""
Corrective-RAG (CRAG) is a Retrieval-Augmented Generation (RAG) approach that integrates self-evaluation or self-assessment of retrieved documents.
""")

if 'thread_id_crag' not in st.session_state:
    with st.form("crag_form"):
        url1 = st.text_input("Enter URL to index:", value="https://lilianweng.github.io/posts/2023-06-23-agent/")
        url2 = st.text_input("Enter URL to index:", value="https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/")
        url3 = st.text_input("Enter URL to index:", value="https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/")
        submitted = st.form_submit_button("Run CRAG")

        if submitted:
            if not url1:
                st.error("Please provide a URL.")
            else:
                with st.spinner("Running workflow..."):
                    try:
                        # Generate a unique thread_id per session
                        st.session_state["urls_crag"] = [url1, url2, url3]
                        st.session_state['thread_id_crag'] = str(uuid.uuid4())
                        thread_id = st.session_state['thread_id_crag']
                        
                        # Initialize vectorstore
                        _ = run_crag(st.session_state["urls_crag"], "", thread_id)
                    except Exception as e:
                        st.session_state['chat_history_crag'].append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()

else:
    st.markdown("---")
    st.markdown(f"Ask your question about: {st.session_state['urls_crag']}")
    # Initialize chat history in session state
    if 'chat_history_crag' not in st.session_state:
        st.session_state['chat_history_crag'] = []

    # Render chat history as LLM chat bubbles
    for msg in st.session_state['chat_history_crag']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input(f"Ask your question"):
        st.session_state['chat_history_crag'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking..."), st.empty():
                response_stream = run_crag_stream(st.session_state['urls_crag'], prompt, st.session_state['thread_id_crag'])
                response = ""
                for chunk in response_stream:
                    st.write(chunk)
                response = chunk
        st.session_state['chat_history_crag'].append({"role": "assistant", "content": response})
