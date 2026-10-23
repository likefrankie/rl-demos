"""RL demos. Run with `streamlit run Home.py` from this folder.

The menu on the left lists the demos by environment (first level) and method (second level), from lib/catalog.py.
"""

import streamlit as st

from lib import catalog

st.set_page_config(page_title="RL Demos", page_icon="▣", layout="wide", initial_sidebar_state="expanded")

pages = {"": [st.Page(catalog.LANDING, title="All demos", url_path="home", default=True)]}
for env, demos in catalog.by_env().items():
    pages[env] = [st.Page(d["page"], title=d["method"], url_path=d["url"]) for d in demos]
st.navigation(pages).run()
