import streamlit as st

st.set_page_config(page_title="RAG Playground", page_icon=":material/edit:")

page_home = st.Page("./pages/page_home.py", title="Home", icon=":material/home:")
page_crag = st.Page("./pages/page_crag.py", title="CRAG", icon=":material/add_circle:")
page_raptor = st.Page("./pages/page_raptor.py", title="RAPTOR", icon=":material/delete:")
page_selfrag = st.Page("./pages/page_selfrag.py", title="Selfrag", icon=":material/refresh:")
page_agenticrag = st.Page("./pages/page_agenticrag.py", title="Agentic-RAG", icon=":material/robot_2:")
page_adaptiverag = st.Page("./pages/page_adaptiverag.py", title="Adaptive-RAG", icon=":material/tactic:")

pg = st.navigation([page_home, page_crag, page_raptor, page_selfrag, page_agenticrag, page_adaptiverag])

pg.run()