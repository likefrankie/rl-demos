"""RL demos: landing page, the cards of all demos grouped by environment."""

import streamlit as st

from lib import catalog, theme

theme.apply()

theme.bar("RL ▸ REINFORCEMENT LEARNING ▸ DEMONSTRATIONS")

st.markdown('<div class="note">Pick a demo in the menu on the left or click <b>▶ OPEN DEMO</b>. '
            'On the demo page, set the parameters at the top and click <b>▶ RUN</b>.</div>', unsafe_allow_html=True)

for env, demos in catalog.by_env().items():
    theme.side_header(f"{env.upper()}  ·  {demos[0]['kind']}")
    cols = st.columns(3)
    for k, d in enumerate(demos):
        with cols[k % 3].container(border=True):
            st.markdown(f'<div class="card-id">{d["lab"]}</div>'
                        f'<div class="card-title">{d["method"]}</div>'
                        f'<div class="card-text">{d["text"]}</div>', unsafe_allow_html=True)
            if st.button("▶  OPEN DEMO", key=f"open_{d['page']}", type="primary", width="stretch"):
                st.switch_page(d["page"])
