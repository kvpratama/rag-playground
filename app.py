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

with st.form("rag_form"):
    url = st.text_input("Enter URL to index:", value="https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/")
    max_depth = st.number_input("Max Depth", min_value=1, max_value=10, value=2)
    question = st.text_area("Your question:", value="What is LangGraph CLI?")
    submitted = st.form_submit_button("Run RAG")

if submitted:
    if not url or not question:
        st.error("Please provide both a URL and a question.")
    else:
        with st.spinner("Running workflow..."):
            # Prepare initial state
            try:
                # Generate a unique thread_id per session
                if 'thread_id' not in st.session_state:
                    st.session_state['thread_id'] = str(uuid.uuid4())
                thread_id = st.session_state['thread_id']
                # Run the workflow with thread_id
                answer = run_workflow(url, max_depth, question, thread_id)
                if answer:
                    st.success("Answer:")
                    st.write(answer)
                else:
                    st.warning("No answer returned.")
            except Exception as e:
                st.error(f"Error: {e}")
