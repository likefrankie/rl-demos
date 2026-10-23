"""Terminal-style look: black background, ice-white accent, IBM Plex Sans for text and IBM Plex Mono for numbers,
dense panels."""

import streamlit as st

C = dict(
    bg="#000000",
    panel="#0a0c0f",
    line="#1c232c",
    grid="#141a21",
    amber="#eaf6ff",        # the accent color (ice white); the key keeps its old name for the other pages
    txt="#c3ccd6",
    dim="#6b7785",
    green="#19e68c",
    red="#ff4d4d",
    cyan="#38bdf8",
    violet="#b388ff",
)

PARAM_COLORS = [C["cyan"], C["amber"], C["violet"], C["green"]]

FONT = "'IBM Plex Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'IBM Plex Mono', Consolas, Menlo, monospace"
TINT = "rgba(234,246,255,.10)"   # background of a highlighted row

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;600;700&display=swap');
html, body, .stApp, .stMarkdown, .stButton button, .stSlider, .stRadio, .stNumberInput input,
[data-testid="stSidebar"] * {{ font-family: {FONT} !important; }}
.stApp {{ background: {C['bg']}; }}
.block-container {{ padding: 0.8rem 1.2rem 1rem 1.2rem; max-width: 100%; }}
header[data-testid="stHeader"] {{ background: transparent; }}
footer {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{ background: #06080b; border-right: 1px solid {C['line']}; }}
.stApp label p {{ color: {C['dim']}; font-size: 11px; letter-spacing: .04em; }}
.stButton button {{ background: {C['panel']}; color: {C['amber']}; border: 1px solid {C['amber']};
    border-radius: 0; font-weight: 700; letter-spacing: .08em; }}
.stButton button:hover {{ background: {C['amber']}; color: #000; border-color: {C['amber']}; }}
.stButton button[kind="primary"] {{ background: {C['amber']}; color: #000; font-size: 15px; padding: 10px 0; }}
.stButton button[kind="primary"]:hover {{ background: #ffffff; color: #000; }}
.stButton button[kind="primary"] p {{ font-size: 15px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 0 !important; border-color: {C['line']} !important;
    background: {C['panel']}; }}
.card-id {{ color: {C['amber']}; font-size: 11px; letter-spacing: .12em; }}
.card-title {{ color: {C['txt']}; font-size: 17px; font-weight: 700; margin: 4px 0 6px 0; }}
.card-text {{ color: {C['dim']}; font-size: 12px; line-height: 1.6; min-height: 58px; }}

.term-bar {{ display: flex; justify-content: space-between; align-items: center; background: {C['amber']};
    color: #000; font-weight: 700; padding: 5px 10px; letter-spacing: .08em; font-size: 13px; }}
.term-sub {{ display: flex; justify-content: space-between; color: {C['dim']}; font-size: 11px;
    padding: 4px 2px 0 2px; letter-spacing: .06em; }}
.side-h {{ color: {C['amber']}; font-size: 11px; letter-spacing: .14em; border-bottom: 1px solid {C['line']};
    padding: 10px 0 3px 0; margin-bottom: 6px; font-weight: 700; }}
.formula {{ color: {C['txt']}; background: {C['panel']}; border: 1px solid {C['line']}; padding: 6px 8px; font-size: 13px; }}

.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 1px;
    background: {C['line']}; border: 1px solid {C['line']}; margin: 8px 0 10px 0; }}
.kpi {{ background: {C['panel']}; padding: 6px 9px; }}
.kpi .k {{ color: {C['dim']}; font-size: 10px; letter-spacing: .14em; }}
.kpi .v {{ color: {C['txt']}; font-size: 15px; font-weight: 600; white-space: nowrap; }}
.kpi .v.a {{ color: {C['amber']}; }}

.panel {{ border: 1px solid {C['line']}; background: {C['panel']}; margin-bottom: 10px; }}
.panel-h {{ display: flex; justify-content: space-between; padding: 5px 9px; border-bottom: 1px solid {C['line']};
    color: {C['amber']}; font-size: 11px; letter-spacing: .14em; font-weight: 700; }}
.panel-h .r {{ color: {C['dim']}; font-weight: 400; }}
.panel-b {{ padding: 4px 6px 6px 6px; }}

table.term {{ width: 100%; border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }}
table.term th {{ color: {C['dim']}; font-weight: 600; text-align: right; padding: 4px 8px;
    border-bottom: 1px solid {C['line']}; font-size: 10.5px; letter-spacing: .1em; }}
table.term td {{ color: {C['txt']}; text-align: right; padding: 3px 8px; border-bottom: 1px solid #10151b; }}
table.term .l {{ text-align: left; }}
table.term td.fixed {{ color: {C['dim']}; }}
table.term td.fresh {{ color: {C['amber']}; animation: flash .7s ease-out; }}
table.term.mat th, table.term.mat td {{ padding: 2px 5px; font-size: 11px; }}
table.term tr.gap td {{ color: {C['dim']}; text-align: center; letter-spacing: .3em; }}
@keyframes flash {{ from {{ background: rgba(234,246,255,.30); }} to {{ background: transparent; }} }}
table.term, .kpi .v, .term-bar, .stNumberInput input {{ font-family: {MONO} !important; }}
[data-testid="stIconMaterial"], [data-testid="stSidebar"] [data-testid="stIconMaterial"],
span[class*="material-symbols"] {{ font-family: 'Material Symbols Rounded' !important; }}
[data-testid="stNavSectionHeader"] {{ color: {C['amber']} !important; font-weight: 700; font-size: 11px;
    letter-spacing: .12em; text-transform: uppercase; margin-top: 10px; }}
[data-testid="stNavSectionHeader"] ~ li [data-testid="stSidebarNavLink"],
[data-testid="stSidebarNavItems"] ul li [data-testid="stSidebarNavLink"] {{ padding-left: 1.4rem; }}
.tag {{ display: inline-block; min-width: 46px; text-align: center; color: #000; font-weight: 700; padding: 0 6px; }}
.tag.in {{ background: {C['green']}; }}
.tag.out {{ background: {C['red']}; }}
.tag.warn {{ background: {C['amber']}; }}
.note {{ color: {C['dim']}; font-size: 11.5px; line-height: 1.6; padding: 2px 2px 0 2px; }}
.note b {{ color: {C['txt']}; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def bar(left: str, right: str = ""):
    st.markdown(f'<div class="term-bar"><span>{left}</span><span>{right}</span></div>', unsafe_allow_html=True)


def side_header(text: str):
    st.markdown(f'<div class="side-h">{text}</div>', unsafe_allow_html=True)


def panel(title: str, body: str, right: str = "") -> str:
    return (f'<div class="panel"><div class="panel-h"><span>{title}</span><span class="r">{right}</span></div>'
            f'<div class="panel-b">{body}</div></div>')


def kpis(items) -> str:
    cells = "".join(
        f'<div class="kpi"><div class="k">{k}</div><div class="v{" a" if hi else ""}">{v}</div></div>'
        for k, v, hi in items
    )
    return f'<div class="kpis">{cells}</div>'


def tag(ok: bool, yes: str = "IN", no: str = "OUT") -> str:
    return f'<span class="tag {"in" if ok else "out"}">{yes if ok else no}</span>'


def style_figure(fig, height: int):
    fig.update_layout(
        paper_bgcolor=C["panel"], plot_bgcolor=C["panel"], height=height,
        font=dict(family=FONT, size=11, color=C["txt"]),
        margin=dict(l=48, r=14, t=34, b=34), showlegend=False,
        hoverlabel=dict(bgcolor="#111", font_family=FONT),
    )
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["line"], linecolor=C["line"], tickcolor=C["line"], color=C["dim"])
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["line"], linecolor=C["line"], tickcolor=C["line"], color=C["dim"])
    fig.update_annotations(font=dict(family=FONT, size=11, color=C["amber"]))
    return fig
