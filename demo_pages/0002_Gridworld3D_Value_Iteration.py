"""Demo 0002: value iteration on the 3D gridworld, built from the data of manual/gridworld3D_value-iteration."""

import streamlit as st

from lib import gridworld_page, theme

st.set_page_config(page_title="3D Gridworld · Value Iteration · RL Demos", page_icon="▣", layout="wide")
theme.apply()
gridworld_page.main("gridworld3D_value-iteration")
