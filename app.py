__import__('pysqlite3')
import streamlit as st
import sys
import os
sys.path.append(os.path.abspath("src"))
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.set_page_config(page_title="RAG Playground", page_icon=":material/edit:")

page_home = st.Page("./pages/page_home.py", title="Home", icon=":material/home:")
page_raptor = st.Page("./pages/page_raptor.py", title="RAPTOR", icon=":material/delete:")
page_crag = st.Page("./pages/page_crag.py", title="CRAG", icon=":material/add_circle:")
page_selfrag = st.Page("./pages/page_selfrag.py", title="Selfrag", icon=":material/refresh:")
page_agenticrag = st.Page("./pages/page_agenticrag.py", title="Agentic-RAG", icon=":material/robot_2:")
page_adaptiverag = st.Page("./pages/page_adaptiverag.py", title="Adaptive-RAG", icon=":material/tactic:")

pg = st.navigation([page_home, page_raptor, page_crag, page_selfrag, page_agenticrag, page_adaptiverag])

pg.run()