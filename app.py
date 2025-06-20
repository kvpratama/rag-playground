import streamlit as st

st.set_page_config(page_title="RAG Playground", page_icon=":material/edit:")

page_home = st.Page("./pages/page_home.py", title="Home", icon=":material/home:")
page_crag = st.Page("./pages/page_crag.py", title="CRAG", icon=":material/add_circle:")
page_raptor = st.Page("./pages/page_raptor.py", title="RAPTOR", icon=":material/delete:")

pg = st.navigation([page_home, page_crag, page_raptor])

pg.run()