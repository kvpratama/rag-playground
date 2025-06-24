import streamlit as st

st.title("RAG Playground")

st.markdown("""
This app lets you run different RAG (Retrieval-Augmented Generation) workflows using your own URL and question.
""")

st.page_link("./pages/page_raptor.py", label="RAPTOR", icon=":material/delete:")
st.page_link("./pages/page_crag.py", label="CRAG", icon=":material/add_circle:")
st.page_link("./pages/page_selfrag.py", label="Selfrag", icon=":material/refresh:")
st.page_link("./pages/page_agenticrag.py", label="Agentic-RAG", icon=":material/robot_2:")
st.page_link("./pages/page_adaptiverag.py", label="Adaptive-RAG", icon=":material/tactic:")
