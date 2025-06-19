import streamlit as st
from langgraph_client import run_workflow
import uuid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("RAG Playground")

st.markdown("""
This app lets you run a RAG (Retrieval-Augmented Generation) workflow using your own URL and question.
""")

if 'thread_id' not in st.session_state:
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
                        # Generate a unique thread_id per session
                        st.session_state["url"] = url
                        st.session_state['max_depth'] = max_depth
                        st.session_state['thread_id'] = str(uuid.uuid4())
                        thread_id = st.session_state['thread_id']
                        
                        # Initialize vectorstore
                        _ = run_workflow(url, max_depth, "", thread_id)
                    except Exception as e:
                        st.session_state['chat_history'].append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()

else:
    st.markdown("---")
    st.markdown(f"Ask your question about: {st.session_state['url']}")
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Render chat history as LLM chat bubbles
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input(f"Ask your question about {st.session_state['url']}"):
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = run_workflow(st.session_state['url'], st.session_state['max_depth'], prompt, st.session_state['thread_id'])
            st.markdown(response)
        
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
