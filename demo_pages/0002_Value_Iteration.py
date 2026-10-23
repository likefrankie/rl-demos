"""Demo 0002: value iteration on the ice cream storage MDP (lab 0002-01-02, exercise 11).

Notation follows the MDP lecture. Each iteration computes, for every state (e, s) and feasible action a in A(e, s),
    Q_{k+1}((e, s), a) = sum_{(e', s')} p((e', s') | (e, s), a) [ r((e, s), a, (e', s')) + beta v_k(e', s') ],
    v_{k+1}(e, s) = max_a Q_{k+1}((e, s), a),   pi_{k+1}(e, s) = arg max_a Q_{k+1}((e, s), a),
one state at a time or all at once. The page shows the backup of one state term by term, the Q table with the
policy and the value, the value function, the three Q functions and the greedy policy in 3D, the policy of every
state over the iterations, and the sup-norm distances between consecutive iterates.
"""

import time
from datetime import datetime

import numpy as np
import streamlit as st

from lib import icecream as ic, plots, rundata, theme

st.set_page_config(page_title="Value Iteration · RL Demos", page_icon="▣", layout="wide")
theme.apply()

S = st.session_state
BACKUP_STATE = (1, 2)       # (e, s) preselected in the backup table; None follows the update
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
FOLLOW = "FOLLOW THE UPDATE"
LABELS = [f"state {i + 1}  ·  (e, s) = ({e}, {s})" for i, (e, s) in enumerate(ic.STATES)]
ACOL = [theme.C["cyan"], theme.C["violet"], theme.C["green"]]
BL = f'style="border-left:1px solid {theme.C["line"]}"'
Z_ZERO, Z_FIT, Z_MAN = "0 to max of the run", "min to max, current iteration", "manual"
VIEW_3D, VIEW_HEAT = "3D", "heatmap"


def sub(n):
    return str(n).translate(SUB)



# --------------------------------------------------------------------- inputs, grouped by what they belong to
ph_bar = st.empty()

# The precomputed run of manual/ice-cream-scoop_value-iteration. The page only replays it: β, P(e′ | e), and every
# iterate come from this file, and nothing is computed while it plays.
RUN = rundata.load("ice-cream-scoop_value-iteration", "run/value_iteration_run.npz")
if RUN is None:
    st.error("No precomputed run found at " + rundata.path("ice-cream-scoop_value-iteration", "run") + ". Run "
             "run_all.py in manual/ice-cream-scoop_value-iteration first.")
    st.stop()
PE = RUN["P_E"]
beta = float(RUN["meta"]["beta"])

with st.expander("1 · ENVIRONMENT", expanded=False):
    st.markdown(
        '<div class="note">A child has s scoops of ice cream available every day. The child stores a scoops for the '
        'next day and eats the remainder, c = s − a. Overnight the parents add e′ scoops, so the scoops available the '
        'next day are s′ = a + e′; e′ is known only after a is chosen. The fridge holds up to 2 scoops, so '
        'a ∈ {0, 1, 2} and s ∈ {0, …, 4}. How many scoops the parents add tonight depends on how many they added last '
        'night, through the transition matrix P on the right (row: e last night; column: e′ tonight). To avoid '
        'confusion with the transition matrices of the MDP, its entries are written P(e′ | e).</div>',
        unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2], gap="medium")
    par = [("Scoops available, s", "{0, 1, 2, 3, 4}"), ("Parents' scoops, e", "{0, 1, 2}"),
           ("Scoops stored, a", "{0, 1, 2}, with a ≤ s"), ("Scoops eaten, c", "s − a"),
           ("Fridge capacity", "2 scoops")]
    c1.markdown(theme.panel("TABLE 1  ▸  PARAMETERS", '<table class="term">' + "".join(
        f'<tr><td class="l fixed">{k}</td><td class="l">{v}</td></tr>' for k, v in par) + '</table>'),
        unsafe_allow_html=True)
    c2.markdown(theme.panel("P(e′ | e)  ▸  ROW e LAST NIGHT, COLUMN e′ TONIGHT", '<table class="term">'
                            '<tr><th class=l></th>' + "".join(f'<th>e′ = {k}</th>' for k in ic.E) + '</tr>' + "".join(
                                f'<tr><td class="l fixed">e = {e}</td>' + "".join(f'<td>{PE[e, k]:.2f}</td>'
                                                                                 for k in ic.E) + '</tr>'
                                for e in ic.E) + '</table>'), unsafe_allow_html=True)
    c2.markdown('<div class="note">The values the precomputed run was generated with.</div>', unsafe_allow_html=True)


def item(title):
    st.markdown(f'<div class="side-h">{title}</div>', unsafe_allow_html=True)


def chart_controls(key, what):
    """Display controls of one block of charts: 3D or heatmap, and the z range (the z axis of a 3D chart, the color
    scale of a heatmap). They change only the display, never the computation."""
    c = st.columns([1.1, 2.8, 0.8, 0.8, 2.5])
    view = c[0].radio("chart type", [VIEW_3D, VIEW_HEAT], horizontal=True, key=f"view_{key}",
                      index=1 if st.query_params.get("view") == "heatmap" else 0)
    mode = c[1].radio(f"z range of the {what} charts", [Z_FIT, Z_ZERO, Z_MAN], horizontal=True,
                      key=f"zmode_{key}")
    lo = c[2].number_input("z min (blank = auto)", value=None, step=1.0, format="%.2f", disabled=mode != Z_MAN,
                           key=f"zmin_{key}")
    hi = c[3].number_input("z max (blank = auto)", value=None, step=1.0, format="%.2f", disabled=mode != Z_MAN,
                           key=f"zmax_{key}")
    c[4].markdown('<div class="note">Min to max: fit the current iteration across the states, to see its shape. '
                  '0 to max: a fixed scale from 0 to the largest value of the run (the converged one), shared by the '
                  'charts of this block, so the values rise from v₀ = 0 toward it. Manual: values outside the range '
                  'are clipped (3D) or saturate the color (heatmap).</div>', unsafe_allow_html=True)
    return dict(view=view, mode=mode, lo=lo, hi=hi)


with st.expander("2 · AGENT", expanded=False):
    st.markdown('<div class="note">The agent is the child, described by its preferences: the utility of the scoops '
                'eaten and the discount factor. The child maximizes the expected discounted sum of utilities.</div>',
                unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        item("DISCOUNT FACTOR  ▸  β")
        b1, b2 = st.columns([1, 3], gap="medium")
        b1.markdown(theme.kpis([("β  DISCOUNT FACTOR", f"{beta:g}", True)]), unsafe_allow_html=True)
        b2.markdown('<div class="note">β &lt; 1 makes a scoop today worth more than a scoop tomorrow: a utility '
                    'received t days from now counts βᵗ times. This is the value the precomputed run was generated '
                    'with.</div>', unsafe_allow_html=True)

        item("UTILITY  ▸  u(c)")
        st.latex(r"u(c)=\begin{cases}\log(c+1) & c\ge 0,\\ -\infty & c<0\end{cases}")
        st.markdown('<table class="term"><tr><th class=l>c  scoops eaten</th>'
                    + "".join(f'<th>{c}</th>' for c in ic.SC) + '</tr><tr><td class="l fixed">u(c)</td>'
                    + "".join(f'<td>{np.log(c + 1):.4f}</td>' for c in ic.SC) + '</tr></table>'
                    '<div class="note">Each extra scoop adds less utility than the previous one (u is concave, see '
                    'the chart), so the child prefers to spread scoops over days. Eating a negative number of scoops '
                    'is impossible: u = −∞.</div>', unsafe_allow_html=True)

        item("OPTIMIZATION PROBLEM")
        st.latex(r"\max\ \mathbb{E}\bigg[\sum_{t=0}^{\infty}\beta^{t}\,u(c_t)\bigg],\qquad c_t=s_t-a_t")
        st.markdown('<div class="note">The child chooses how many scoops to store each day to maximize the expected '
                    'discounted sum of utilities. Storing a scoop gives up utility today for utility tomorrow, '
                    'discounted by β and uncertain because the parents’ scoops e′ are random.</div>',
                    unsafe_allow_html=True)
    right.plotly_chart(plots.utility(int(ic.SC.max()), height=420), width="stretch",
                       config={"displayModeBar": False})


with st.expander("3 · MARKOV DECISION PROCESS", expanded=False):
    item("STATES  ▸  (e, s)")
    st.markdown('<div class="note">The state is the pair (e, s): the scoops the parents added last night, which '
                'predict the scoops they add tonight, and the scoops available this morning. There are 3 × 5 = 15 '
                'states, ordered by e and then by s and numbered 1 to 15; the tables and charts use these '
                'numbers.</div>', unsafe_allow_html=True)
    st.markdown('<table class="term mat"><tr><th class=l>STATE</th>'
                + "".join(f'<th>{i + 1}</th>' for i in range(ic.N)) + '</tr>'
                + '<tr><td class="l fixed">e</td>' + "".join(f'<td>{e}</td>' for e, _ in ic.STATES) + '</tr>'
                + '<tr><td class="l fixed">s</td>' + "".join(f'<td>{s}</td>' for _, s in ic.STATES) + '</tr></table>',
                unsafe_allow_html=True)

    item("ACTIONS  ▸  A(e, s)")
    st.markdown('<div class="note">The action is the number of scoops stored, a ∈ {0, 1, 2}. Only a ≤ s is feasible: '
                'the child cannot store more scoops than it has. Infeasible actions get the reward −∞, so that they '
                'are never chosen.</div>', unsafe_allow_html=True)
    st.latex(r"A(e,s)=\{0,\dots,\min(s,2)\}")
    st.markdown('<table class="term mat"><tr><th class=l>STATE</th>'
                + "".join(f'<th>{i + 1}<br>({e},{s})</th>' for i, (e, s) in enumerate(ic.STATES)) + '</tr>'
                + '<tr><td class="l fixed">A(e, s)</td>'
                + "".join(f'<td>{{{", ".join(str(a) for a in range(min(s, 2) + 1))}}}</td>' for _, s in ic.STATES)
                + '</tr></table>', unsafe_allow_html=True)

    item("TRANSITION PROBABILITIES  ▸  p((e′, s′) | (e, s), a)")
    st.markdown('<div class="note">Given (e, s) and a, the parents add e′ with probability P(e′ | e) and the next '
                'state is (e′, s′) = (e′, a + e′). For each action a, the probabilities form a 15 × 15 matrix, row = State i = (e, s), '
                'column = State j = (e′, s′); each row has three nonzero entries, one for each e′, and sums to '
                'one.</div>', unsafe_allow_html=True)
    st.latex(r"p\Big((e',s')\mid(e,s),a\Big)=\begin{cases}P(e'\mid e) & s'=a+e',\\ 0 & \text{otherwise}\end{cases}")
    show_zero = st.toggle("show the all-zero columns (next states that action a never reaches)",
                          value=False)
    ph_p = st.empty()

    item("REWARDS  ▸  r((e, s), a, (e′, s′))")
    st.markdown('<div class="note">The reward is the utility u(c) of the scoops eaten, c = s − a '
                '(section 2). It depends only on s and a, so each row of the reward matrix of action a is constant. · = an entry with p = 0, which drops out of every sum.</div>',
                unsafe_allow_html=True)
    st.latex(r"r\Big((e,s),a,(e',s')\Big)=\begin{cases}\log(s-a+1) & a\le s,\\ -\infty & a>s\end{cases}")
    ph_r = st.empty()

    item("EXPECTED IMMEDIATE REWARD  ▸  q((e, s), a)")
    st.markdown('<div class="note">The conditional expected immediate reward weights each reward by its '
                'probability; only the transitions with p &gt; 0 enter the sum. <b>It does not depend on v<sub>k</sub>, so it is '
                'computed once and is the same at every iteration</b>; in Q<sub>k+1</sub> (section 6) only the term with v<sub>k</sub> '
                'changes with k.</div>', unsafe_allow_html=True)
    st.latex(r"q\Big((e,s),a\Big)=\sum_{(e',s')}p\Big((e',s')\mid(e,s),a\Big)\,r\Big((e,s),a,(e',s')\Big)")
    ph_qr = st.empty()

with st.expander("4 · VALUE ITERATION", expanded=False):
    c1, c2 = st.columns([1, 5], gap="medium")
    tol = c1.number_input(r"tolerance on $\lVert v_{k+1}-v_k\rVert_\infty$", min_value=1e-10, max_value=1.0, value=1e-4,
                          step=1e-5, format="%.1e")
    c2.markdown('<div class="note"><b>The Bellman equation.</b> The optimal value function v(e, s), the largest '
                'expected discounted sum of utilities attainable from (e, s), satisfies</div>', unsafe_allow_html=True)
    c2.latex(r"v(e,s)=\max_{a\le\min(s,2)}\Big\{\log(s-a+1)+\beta\sum_{e'\in\{0,1,2\}}P(e'\mid e)\,"
             r"v(e',a+e')\Big\}")
    c2.markdown('<div class="note">Value iteration solves it by putting the current guess v<sub>k</sub> on the right-hand side '
                'and reading off v<sub>k+1</sub> on the left.</div>', unsafe_allow_html=True)
    c2.markdown(
        '<div class="note"><b>1. Initialization.</b> v₀ = 0. <b>2. Update.</b> For every state (e, s), compute '
        'Q<sub>k+1</sub>((e, s), a) for each feasible a from v<sub>k</sub> (sections 6 and 7), then v<sub>k+1</sub>(e, s) = max_a Q<sub>k+1</sub>((e, s), a) and '
        'π<sub>k+1</sub>(e, s) = arg max_a (section 8). <b>3. Termination.</b> If ‖v<sub>k+1</sub> − v<sub>k</sub>‖∞ is below the tolerance, stop '
        '(section 9). Otherwise return to step 2.</div>', unsafe_allow_html=True)
    c2.latex(r"v_{k+1}(e,s)=\max_{a\in A(e,s)}\sum_{(e',s')}p\Big((e',s')\mid(e,s),a\Big)"
             r"\bigg[r\Big((e,s),a,(e',s')\Big)+\beta\,v_k(e',s')\bigg]")

with st.expander("5 · DEMO SETTINGS", expanded=False):
    c = st.columns([1, 1, 2, 1, 1, 1.4])
    max_iter = int(c[0].number_input("max iterations", min_value=1, max_value=5000, value=600, step=50))
    unit = c[1].radio("one step updates", ["all states", "one state"])
    pick = c[2].selectbox("state in the backup table", [FOLLOW] + LABELS,
                          index=0 if BACKUP_STATE is None else ic.idx(*BACKUP_STATE) + 1)
    delay = c[3].number_input("pause per step (s)", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                              format="%.2f")
    n_prev = int(c[4].number_input("earlier iterations drawn", min_value=0, max_value=30, value=10, step=1))
    spread = c[5].radio("earlier iterations", ["spread over all", "most recent"]) == "spread over all"

with st.container(border=True):
    c = st.columns([1.2, 0.6, 0.6, 5])
    run = c[0].button("▶  RUN", type="primary", width="stretch")
    step = c[1].button("+1", width="stretch")
    reset = c[2].button("RESET", width="stretch")
    c[3].markdown('<div class="note">+1 applies one step: all 15 states (one iteration) or the next state only. RUN '
                  'repeats until ‖v<sub>k+1</sub> − v<sub>k</sub>‖∞ is below the tolerance or the max iterations are reached. Any click '
                  'stops a run.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------- state
st.markdown(f'<div class="note"><b>Data:</b> replaying the precomputed run of '
            f'<code>manual/ice-cream-scoop_value-iteration</code> ({RUN["meta"]["iterations"]:,} iterations, '
            f'β = {beta:g}). Nothing is computed while it plays.</div>', unsafe_allow_html=True)
LAST = len(RUN["Q"]) - 1        # the last iteration stored


def stored_q(i=None):
    """Q of iteration k + 1 = len(S.V) from the precomputed run (row i only if given)."""
    k = len(S.V)
    return RUN["Q"][k] if i is None else RUN["Q"][k][i]


cfg = (beta, PE.tobytes())
if reset or S.get("vi_cfg") != cfg:
    S.vi_cfg = cfg
    S.V = [np.zeros(ic.N)]      # V[k] = v_k
    S.Q = [None]                # Q[k] = Q_k, computed from v_{k-1}
    S.part = None               # iteration in progress, one state at a time: {"i": next state, "q": rows so far}
    S.last = None               # last state updated
    S.fresh = set()             # states updated by the last step
    S.draws = 0


def n_done():
    return len(S.V) - 1


def finish(q):
    S.Q.append(q)
    S.V.append(q.max(axis=1))
    S.part = None


def step_state():
    if S.part is None:
        S.part = dict(i=0, q=np.full((ic.N, ic.K), np.nan))
    i = S.part["i"]
    S.part["q"][i] = stored_q(i)
    S.part["i"] += 1
    S.last, S.fresh = i, {i}
    if S.part["i"] == ic.N:
        finish(S.part["q"])


def step_iteration():
    if S.part is not None:
        q = S.part["q"]
        S.fresh = set(range(S.part["i"], ic.N))
        stored = stored_q()
        for i in range(S.part["i"], ic.N):
            q[i] = stored[i]
    else:
        q = stored_q().copy()
        S.fresh = set(range(ic.N))
    finish(q)


def advance():
    if unit == "one state":
        step_state()
    else:
        step_iteration()


def dist_v():
    return np.array([np.abs(S.V[k] - S.V[k - 1]).max() for k in range(1, len(S.V))])


def dist_q():
    """||Q_{k+1} - Q_k|| over feasible actions, for k >= 1 (Q_0 is not defined)."""
    out = []
    for k in range(2, len(S.Q)):
        ok = np.isfinite(S.Q[k])
        out.append(np.abs(S.Q[k][ok] - S.Q[k - 1][ok]).max())
    return np.array(out)


def policy_changes():
    pis = [q.argmax(axis=1) for q in S.Q[1:]]
    return [k + 1 for k in range(1, len(pis)) if (pis[k] != pis[k - 1]).any()]


def converged():
    d = dist_v()
    return len(d) > 0 and d[-1] < tol


def done():
    return S.part is None and (converged() or n_done() >= min(max_iter, LAST))


# --------------------------------------------------------------------- layout: one collapsible section per output
ph_kpi = st.empty()
with st.expander("6 · Q FUNCTION", expanded=False):
    st.latex(r"Q_{k+1}\Big((e,s),a\Big)=q\Big((e,s),a\Big)+\beta\sum_{(e',s')}"
             r"p\Big((e',s')\mid(e,s),a\Big)\,v_k(e',s')")
    st.markdown('<div class="note">The backup of all 15 states at once; section 7 works through one state term by term. q((e, s), a) is the expected immediate reward '
                'of section 3, the same at every iteration; the second term uses v<sub>k</sub> and changes with k.</div>',
                unsafe_allow_html=True)
    ph_q = st.empty()
    qc = chart_controls("q", "Q")
    qa = st.columns(3, gap="small")
    ph_qa = [col.empty() for col in qa]
    st.markdown('<div class="note"><b>Q table:</b> amber = the largest Q in the row; its action is '
                'π<sub>k+1</sub>(e, s) and its value is v<sub>k+1</sub>(e, s) (section 8). −∞ = infeasible action, '
                'a &gt; s.<br><b>3D charts:</b> solid surface = iteration k + 1, dashed wireframes = earlier '
                'iterations, ◆ = states where this action is the arg max.<br><b>Heatmaps:</b> one cell per '
                'state with its value; ◆ = the arg max.</div>', unsafe_allow_html=True)
with st.expander("7 · BACKUP OF ONE STATE", expanded=False):
    st.latex(r"Q_{k+1}\Big((e,s),a\Big)=\sum_{(e',s')}p\Big((e',s')\mid(e,s),a\Big)"
             r"\bigg[r\Big((e,s),a,(e',s')\Big)+\beta\,v_k(e',s')\bigg]")
    ph_b = st.empty()
    ph_bl = st.empty()
    st.markdown('<div class="note">The backup table computes the three numbers Q<sub>k+1</sub>((e, s), a) of one state: each '
                'row is a next state (e′, s′) = (e′, a + e′), the last column of an action is p × [r + β v<sub>k</sub>], and '
                'its sum over the rows is Q<sub>k+1</sub>. The formulas below the table write out the same sum with the '
                'numbers.</div>', unsafe_allow_html=True)
with st.expander("8 · VALUE FUNCTION AND POLICY", expanded=False):
    st.latex(r"v_{k+1}(e,s)=\max_{a\in A(e,s)}Q_{k+1}\Big((e,s),a\Big),\qquad "
             r"\pi_{k+1}(e,s)\in\arg\max_{a\in A(e,s)}Q_{k+1}\Big((e,s),a\Big)")
    vc = chart_controls("v", "v")
    va, pa, pb = st.columns([1, 1, 1.3], gap="small")
    ph_v3 = va.empty()
    ph_pi3 = pa.empty()
    ph_pih = pb.empty()
    st.markdown('<div class="note">The value function and the policy read off the Q table, row by row: v<sub>k+1</sub> is the '
                'largest Q and π<sub>k+1</sub> the action that attains it. The policy history shows, for every state, the '
                'action at each iteration.</div>', unsafe_allow_html=True)
with st.expander("9 · CONVERGENCE", expanded=False):
    q1, q2 = st.columns(2)
    q1.latex(r"\lVert v_{k+1}-v_k\rVert_\infty=\max_{(e,s)}\Big|v_{k+1}(e,s)-v_k(e,s)\Big|")
    q1.latex(r"\lVert Q_{k+1}-Q_k\rVert_\infty=\max_{(e,s)}\ \max_{a\in A(e,s)}"
             r"\Big|Q_{k+1}\Big((e,s),a\Big)-Q_k\Big((e,s),a\Big)\Big|")
    q2.latex(r"\text{stop when }\lVert v_{k+1}-v_k\rVert_\infty<\text{tolerance},\qquad "
             r"\lVert v_{k+1}-v_k\rVert_\infty\le\beta\,\lVert v_k-v_{k-1}\rVert_\infty")
    q2.latex(r"\pi_{k+1}\text{ is }\varepsilon\text{-optimal when }\lVert v_{k+1}-v_k\rVert_\infty"
             r"<\varepsilon(1-\beta)/(2\beta)")
    cl, cr = st.columns([6, 5], gap="small")
    ph_ct = cl.empty()
    ph_dist = cr.empty()
    st.markdown('<div class="note">The table computes both distances for the last iteration: one row per state, the '
                'max over the rows (amber) is the sup norm. The chart shows that both distances shrink at about the '
                'rate β per iteration, as the contraction property predicts (dotted line). Green verticals mark '
                'iterations where the policy changed: it settles long before v stops moving.</div>',
                unsafe_allow_html=True)
with st.expander("10 · DEPLOYMENT", expanded=False):
    DEP = rundata.load_csv("ice-cream-scoop_value-iteration", "deployment/policy_rollout.csv")
    if DEP is None:
        st.markdown('<div class="note">No deployment data found: run optimal_policy_player.py in the code base.'
                    '</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="note">The optimal policy deployed: {len(DEP)} days simulated by '
                    '<code>optimal_policy_player.py</code> in <code>manual/ice-cream-scoop_value-iteration</code>, '
                    'replayed one day at a time. Each morning the child sees (e, s), stores a = π(e, s) scoops and '
                    'eats c = s − a; overnight the parents add e′ scoops, so tomorrow s′ = a + e′.</div>',
                    unsafe_allow_html=True)
        c = st.columns([0.8, 0.6, 0.6, 4])
        dep_play = c[0].button("▶  PLAY", type="primary", width="stretch", key="ic_play")
        if c[1].button("+1", width="stretch", key="ic_dplus"):
            S.dep_t = min(len(DEP), S.get("dep_t", 0) + 1)
        if c[2].button("RESET", width="stretch", key="ic_dreset"):
            S.dep_t = 0
        S.setdefault("dep_t", 0)
        ph_dep, ph_dep_tab = st.empty(), st.empty()


def draw_deployment(t):
    """Days 1, ..., t of the deployment: e, s, a, c and the utility, one point per day."""
    d = DEP.iloc[:t].rename(columns={"day": "t", "stored": "a", "eaten": "c"})
    ph_dep.plotly_chart(plots.deploy_series(d, len(DEP), [
        ("e", "e  parents", ACOL[0]), ("s", "s  available", theme.C["amber"]), ("a", "a  stored", ACOL[1]),
        ("c", "c  eaten", ACOL[2])], height=560), width="stretch", config={"displayModeBar": False},
        key=f"dep_{t}_{time.time_ns()}")
    rows = "".join(f'<tr><td class="l fixed">{r.day}</td><td>({r.e}, {r.s})</td><td>{r.stored}</td>'
                   f'<td>{r.eaten}</td><td>{r.utility:.4f}</td><td>{r.parents_add}</td><td>({r.e_next}, {r.s_next})'
                   f'</td></tr>' for r in DEP.iloc[:t].tail(6).itertuples())
    total = DEP["utility"].iloc[:t].sum()
    ph_dep_tab.markdown(theme.panel(f"DAY {t} / {len(DEP)}  ▸  TOTAL UTILITY {total:.4f}",
                                    '<table class="term mat"><tr><th class=l>DAY</th><th>(e, s)</th><th>a = π(e, s)'
                                    '</th><th>c</th><th>u(c)</th><th>e′</th><th>(e′, s′)</th></tr>' + rows + '</table>'),
                        unsafe_allow_html=True)


def fq(x, d=4):
    return "−∞" if x == -np.inf else f"{x:,.{d}f}"


def backup_state():
    if pick != FOLLOW:
        return LABELS.index(pick)
    return S.last if S.last is not None else ic.idx(1, 2)


def q_table():
    """Rows: the 15 states. In the middle of an iteration (one state at a time) the rows not yet updated show
    the previous iteration in grey."""
    part = S.part
    n = len(S.V) if part else n_done()            # iteration shown, k + 1
    sel = backup_state()
    heads = (["STATE", "e", "s"] + [f"Q{sub(n)}((e,s), {a})" for a in ic.A]
             + [f"π{sub(n)}(e,s)", f"v{sub(n)}(e,s)"])
    head = "".join(f'<th{" class=l" if h == "STATE" else ""}>{h}</th>' for h in heads)
    rows = []
    for i, (e, s) in enumerate(ic.STATES):
        sep = f' style="border-top:1px solid {theme.C["line"]}"' if s == 0 and e > 0 else ""
        mark = "▶ " if i == sel else ""
        lead = f'<td class="l fixed">{mark}{i + 1}</td><td class="fixed">{e}</td><td class="fixed">{s}</td>'
        new = n >= 1 and (part is None or i < part["i"])
        if n == 0 or (part and not new and len(S.Q) < 2):
            cells = "".join('<td class="fixed">—</td>' for _ in range(5))
        else:
            q = (part["q"][i] if part else S.Q[-1][i]) if new else S.Q[-1][i]
            best = int(q.argmax())
            fresh = ' class="fresh"' if i in S.fresh and new else ""
            if not new:
                qc = "".join(f'<td class="fixed">{fq(x)}</td>' for x in q)
                cells = qc + f'<td class="fixed">{best}</td><td class="fixed">{fq(q.max())}</td>'
            else:
                qc = "".join(f'<td style="color:{theme.C["amber"]};font-weight:700">{fq(x)}</td>' if a == best
                             else f'<td>{fq(x)}</td>' for a, x in enumerate(q))
                cells = qc + f'<td{fresh}>{best}</td><td{fresh} style="color:{theme.C["amber"]}">{fq(q.max())}</td>'
        rows.append(f'<tr{sep}>{lead}{cells}</tr>')
    body = f'<table class="term"><tr>{head}</tr>{"".join(rows)}</table>'
    right = f"UPDATED {part['i']} / {ic.N} STATES" if part else "AMBER = ARG MAX  ·  ▶ = STATE IN THE BACKUP TABLE"
    return theme.panel(f"Q TABLE  ▸  ITERATION k + 1 = {n}", body, right=right)


def stored_backup(vk, i, n):
    """The terms of the backup of state i shown in the table, from the run's data: P(e′ | e) over e′ and, per action,
    None if infeasible, else the next states (e′, a + e′), the reward r, β v_k(s′), the terms p [r + β v_k(s′)], and
    Q_n(i, a) as stored in the run (the sum of the terms)."""
    e, s = ic.STATES[i]
    p = PE[e]
    acts = []
    for a in ic.A:
        if not RUN["feasible"][i, a]:
            acts.append(None)
            continue
        nxt = [ic.idx(ep, a + ep) for ep in ic.E]
        r = float(RUN["R"][i, a, nxt[0]])
        bv = beta * vk[nxt]
        acts.append(dict(nxt=nxt, r=r, bv=bv, term=p * (r + bv), q=float(RUN["Q"][n][i, a])))
    return p, acts


def backup():
    """Backup of one state, term by term. Shows the latest computation of that state, or the next one."""
    i = backup_state()
    e, s = ic.STATES[i]
    if S.part is not None and i < S.part["i"]:
        n, vk, note = len(S.V), S.V[-1], ""
    elif n_done() >= 1:
        n, vk, note = n_done(), S.V[-2], ""
    else:
        n, vk, note = 1, S.V[0], "  ·  NEXT"
    k = n - 1
    p, acts = stored_backup(vk, i, n)
    q = np.array([x["q"] if x else -np.inf for x in acts])
    best = int(q.argmax())

    g1 = '<th class=l rowspan=2>e′</th>'
    g2 = ""
    for a in ic.A:
        txt = (f"a = {a}  ·  store {a}, eat {s - a}  ·  r = log({s - a + 1}) = {acts[a]['r']:.4f}" if acts[a]
               else f"a = {a}  ·  NOT IN A({e},{s}): a > s")
        g1 += (f'<th colspan=5 style="text-align:center;color:{ACOL[a]};border-left:1px solid {theme.C["line"]}">'
               f'{txt}</th>')
        g2 += "".join(f'<th {BL if j == 0 else ""}>{h}</th>' for j, h in enumerate(
            ["(e′, s′)", "p", f"v{sub(k)}(e′, s′)", f"β v{sub(k)}(e′, s′)", f"p × [r + β v{sub(k)}]"]))
    rows = []
    for ep in ic.E:
        cells = f'<td class="l fixed">{ep}</td>'
        for a, x in enumerate(acts):
            if x is None:
                cells += f'<td class="fixed" {BL}>—</td>' + '<td class="fixed">—</td>' * 4
                continue
            cells += (f'<td class="fixed" {BL}>({ep}, {a + ep})</td><td>{p[ep]:.2f}</td>'
                      f'<td>{vk[x["nxt"][ep]]:,.4f}</td><td>{x["bv"][ep]:,.4f}</td><td>{x["term"][ep]:,.4f}</td>')
        rows.append(f'<tr>{cells}</tr>')
    tot = '<td class="l" style="color:#fff">Σ over (e′, s′)</td>'
    for a, x in enumerate(acts):
        if x is None:
            tot += f'<td {BL} colspan=5 style="text-align:center;border-left:1px solid {theme.C["line"]}">' \
                   f'Q{sub(n)}(({e},{s}), {a}) = −∞</td>'
            continue
        hi = f'color:{theme.C["amber"]};font-weight:700' if a == best else "font-weight:700"
        tot += f'<td {BL}></td><td>{p.sum():.2f}</td><td></td><td></td><td style="{hi}">{x["q"]:,.4f}</td>'
    rows.append(f'<tr style="border-top:1px solid {theme.C["amber"]}">{tot}</tr>')
    body = (f'<div class="note">The rows are the only next states with p &gt; 0 from ({e}, {s}): '
            f'(e′, s′) = (e′, a + e′) with p = p((e′, s′) | ({e}, {s}), a) = P(e′ | e = {e}). The other '
            f'{ic.N - ic.NE} probabilities in row {i + 1} of the matrix of action a (section 3) are 0, so the sum over all next states has {ic.NE} terms. Σ of the last column = '
            f'Q{sub(n)}(({e}, {s}), a); β = {beta:g}.</div>'
            f'<table class="term"><tr>{g1}</tr><tr>{g2}</tr>{"".join(rows)}</table>')
    panel = theme.panel(f"BACKUP  ▸  STATE {i + 1} = (e, s) = ({e}, {s})  ▸  ITERATION k + 1 = {n}{note}", body,
                        right=f"USES v{sub(k)} ONLY")

    lines = []
    for a, x in enumerate(acts):
        if x is None:
            lines.append(rf"Q_{{{n}}}\Big(({e},{s}),{a}\Big)=-\infty\quad(a={a}>s={s})")
            continue
        terms = "+".join(rf"{p[ep]:.2f}\,[\,{x['r']:.4f}+{beta:g}\times{vk[x['nxt'][ep]]:.4f}\,]" for ep in ic.E)
        sums = "+".join(f"{t:.4f}" for t in x["term"])
        lines.append(rf"Q_{{{n}}}\Big(({e},{s}),{a}\Big)={terms}={sums}={x['q']:.4f}")
    qs = ",\\ ".join("-\\infty" if v == -np.inf else f"{v:.4f}" for v in q)
    lines.append(rf"v_{{{n}}}({e},{s})=\max\{{{qs}\}}={q.max():.4f},\qquad \pi_{{{n}}}({e},{s})={best}"
                 rf"\ \ (\text{{store }}{best},\ \text{{eat }}{s - best})")
    return panel, lines


def conv_table():
    """How the two sup norms of the last full iteration are computed: one row per state, the max over the rows."""
    n = n_done()
    title = f"DISTANCES  ▸  ITERATION k + 1 = {n}"
    if n == 0:
        return theme.panel(title, '<div class="note">No iteration yet: press +1 or RUN.</div>')
    k = n - 1
    v1, v0 = S.V[n], S.V[k]
    dv = np.abs(v1 - v0)
    iv = int(dv.argmax())
    dq = None
    if S.Q[k] is not None:
        ok = np.isfinite(S.Q[n]) & np.isfinite(S.Q[k])
        dq = np.where(ok, np.abs(np.where(ok, S.Q[n], 0) - np.where(ok, S.Q[k], 0)), np.nan)
        dqs = np.nanmax(dq, axis=1)
        iq = int(dqs.argmax())
        aq = int(np.nanargmax(dq[iq]))
    amb = f' style="color:{theme.C["amber"]};font-weight:700"'
    heads = (["STATE", "e", "s", f"v{sub(n)}(e,s)", f"v{sub(k)}(e,s)", f"|v{sub(n)} − v{sub(k)}|"]
             + [f"|Q{sub(n)} − Q{sub(k)}|, a = {a}" for a in ic.A] + ["max over a"])
    head = "".join(f'<th{" class=l" if h == "STATE" else (" " + BL if h.endswith("a = 0") else "")}>{h}</th>'
                   for h in heads)
    rows = []
    for i, (e, s) in enumerate(ic.STATES):
        sep = f' style="border-top:1px solid {theme.C["line"]}"' if s == 0 and e > 0 else ""
        cells = (f'<td class="l fixed">{i + 1}</td><td class="fixed">{e}</td><td class="fixed">{s}</td>'
                 f'<td>{fq(v1[i])}</td><td>{fq(v0[i])}</td><td{amb if i == iv else ""}>{dv[i]:.2e}</td>')
        if dq is None:
            cells += f'<td class="fixed" {BL}>—</td>' + '<td class="fixed">—</td>' * 3
        else:
            for a in ic.A:
                x = dq[i, a]
                style = amb if (i, a) == (iq, aq) else ""
                cells += (f'<td class="fixed"{" " + BL if a == 0 else ""}>—</td>' if np.isnan(x)
                          else f'<td{" " + BL if a == 0 else ""}{style}>{x:.2e}</td>')
            cells += f'<td{amb if i == iq else ""}>{dqs[i]:.2e}</td>'
        rows.append(f'<tr{sep}>{cells}</tr>')
    tot = (f'<td class="l" colspan=5 style="color:#fff">max over (e, s) = ‖·‖∞</td>'
           f'<td{amb}>{dv[iv]:.2e}</td>')
    tot += (f'<td {BL} colspan=4 style="text-align:center">Q₀ is not defined: ‖Q₁ − Q₀‖∞ starts at k + 1 = 2</td>'
            if dq is None else f'<td {BL} colspan=3></td><td{amb}>{dqs[iq]:.2e}</td>')
    rows.append(f'<tr style="border-top:1px solid {theme.C["amber"]}">{tot}</tr>')
    d = dist_v()
    ratio = f"{d[-1] / d[-2]:.4f}" if len(d) > 1 and d[-2] > 0 else "—"
    verdict = "below the tolerance: STOP" if dv[iv] < tol else "not below the tolerance: CONTINUE"
    note = (f'<div class="note">‖v{sub(n)} − v{sub(k)}‖∞ = {dv[iv]:.3e} at state {iv + 1} = ({ic.STATES[iv][0]}, {ic.STATES[iv][1]}), '
            f'tolerance = {tol:.1e} → {verdict}.  Ratio to the previous distance = {ratio} (→ β = {beta:g}).</div>')
    return theme.panel(title, note + f'<table class="term"><tr>{head}</tr>{"".join(rows)}</table>',
                       right="AMBER = THE MAX")


def mat_table(M, sel, d, P, last=None, cols=None):
    """One (N, N) matrix over (state, next state), rows and columns in State order. Entries where P = 0
    are a dot; in the row of the backup state the others are amber. last = (header, column) appends a column;
    cols lists the columns shown (default all)."""
    cols = range(ic.N) if cols is None else cols
    head = ('<th class=l>STATE (e, s)</th>' + "".join(f'<th>{j + 1}<br>({ic.STATES[j][0]},{ic.STATES[j][1]})</th>'
                                                       for j in cols)
            + (f'<th {BL}>{last[0]}</th>' if last else ""))
    rows = []
    for i, (e, s) in enumerate(ic.STATES):
        tint = f' style="background:{theme.TINT}"' if i == sel else ""
        sep = f' style="border-top:1px solid {theme.C["line"]}"' if s == 0 and e > 0 else ""
        hi = f' style="color:{theme.C["amber"]};font-weight:700"' if i == sel else ""
        cells = f'<td class="l fixed">{"▶ " if i == sel else ""}{i + 1} ({e},{s})</td>'
        cells += "".join('<td class="fixed">·</td>' if P[i, j] == 0 else f'<td{hi}>{fq(M[i, j], d)}</td>'
                         for j in cols)
        if last:
            cells += f'<td {BL}{hi}>{fq(last[1][i], d)}</td>'
        rows.append(f'<tr{tint or sep}>{cells}</tr>')
    return f'<div style="overflow-x:auto"><table class="term mat"><tr>{head}</tr>{"".join(rows)}</table></div>'


def matrices():
    """Section 3: the probability and reward matrices of each action side by side, and q((e, s), a) computed term by term, with the row of
    the backup state highlighted."""
    sel = backup_state()
    P, R = RUN["P"].transpose(1, 0, 2), RUN["R"].transpose(1, 0, 2)     # P[a, s, s′], R[a, s, s′] from the run
    # each action reaches only NE next states, (e', a + e'); the other columns are all zero and hidden by default
    cols = [None if show_zero else [j for j in range(ic.N) if P[a][:, j].any()] for a in ic.A]
    tail = (f'<div class="note">All {ic.N} columns shown; scroll each matrix sideways. ▶ = the state in the backup '
            f'table (section 7).</div>' if show_zero else
            f'<div class="note">Only the columns with a nonzero entry are shown: action a reaches only the next states '
            f'(e′, a + e′). ▶ = the state in the backup table (section 7).</div>')
    for ph, M, name, d, right in ((ph_p, P, "PROBABILITY MATRIX", 2, "ROWS SUM TO 1"),
                                  (ph_r, R, "REWARD MATRIX", 4, "CONSTANT ROWS")):
        with ph.container():
            for a, col in zip(ic.A, st.columns(ic.K, gap="small")):
                col.markdown(theme.panel(f"{name}, a = {a}",
                                         mat_table(M[a], sel, d, P[a], cols=cols[a]), right=right),
                             unsafe_allow_html=True)
            st.markdown(tail, unsafe_allow_html=True)

    heads = "".join(f'<th {BL} style="text-align:center;color:{ACOL[a]}" colspan=2>a = {a}  ·  store {a}</th>'
                    for a in ic.A)
    sub_h = "".join(f'<th {BL}>Σ p × r over (e′, s′)</th><th>q((e, s), {a})</th>' for a in ic.A)
    rows = []
    for i, (e, s) in enumerate(ic.STATES):
        tint = f' style="background:{theme.TINT}"' if i == sel else ""
        sep = f' style="border-top:1px solid {theme.C["line"]}"' if s == 0 and e > 0 else ""
        cells = f'<td class="l fixed">{"▶ " if i == sel else ""}{i + 1} ({e},{s})</td>'
        for a in ic.A:
            if a > s:
                cells += f'<td class="fixed" {BL}>a &gt; s: not feasible</td><td class="fixed">−∞</td>'
                continue
            r = float(RUN["q"][i, a])
            calc = " + ".join(f"{PE[e, ep]:.2f} × {r:.4f}" for ep in ic.E)
            cells += f'<td class="fixed" {BL}>{calc}</td><td>{r:.4f}</td>'
        rows.append(f'<tr{tint or sep}>{cells}</tr>')
    ph_qr.markdown(theme.panel(
        "q((e, s), a)  ▸  COMPUTED ONCE, THE SAME AT EVERY ITERATION",
        f'<table class="term mat"><tr><th class=l rowspan=2>STATE (e, s)</th>{heads}</tr><tr>{sub_h}</tr>'
        f'{"".join(rows)}</table>'
        '<div class="note">Each sum has one term per e′, P(e′ | e) × log(s − a + 1). The reward is the same for '
        'every e′ and the probabilities sum to one, so q((e, s), a) = log(s − a + 1).</div>',
        right="CONSTANT ACROSS ITERATIONS"), unsafe_allow_html=True)


def earlier(start, n):
    """Iterations drawn as dashed wireframes before the current iteration n: spread over start..n-1 or the latest."""
    if n_prev == 0 or n - start <= 0:
        return []
    if spread:
        return sorted(set(np.round(np.linspace(start, n - 1, min(n_prev, n - start))).astype(int)))
    return list(range(max(start, n - n_prev), n))


def zrange(grids):
    """Padded [min, max] of the finite values of the grids."""
    if not grids:
        return [0.0, 1.0]
    vals = np.concatenate([g.ravel() for g in grids])
    vals = vals[np.isfinite(vals)]
    lo, hi = (vals.min(), vals.max()) if vals.size else (0.0, 1.0)
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return [lo - pad, hi + pad]


def manual(auto, zmin, zmax):
    """The z range typed in the controls; a blank bound keeps the automatic one."""
    lo = auto[0] if zmin is None else zmin
    hi = auto[1] if zmax is None else zmax
    return [lo, hi] if hi > lo else [lo, lo + 1.0]


def run_max(name):
    """The largest finite value of v (name "V") or Q (name "Q") over the run: the converged one."""
    arr = RUN[name][-1]
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else 1.0


def zranges(charts_grids, c, top):
    """One z range per chart, from the controls c of its block. charts_grids holds, per chart, the grids drawn
    (earlier iterations and the current one); top is the largest value of the run."""
    if c["mode"] == Z_FIT:
        return [zrange(g[-1:]) for g in charts_grids]
    vals = np.concatenate([g.ravel() for gs in charts_grids for g in gs]) if any(charts_grids) else np.zeros(1)
    vals = vals[np.isfinite(vals)]
    low = min(0.0, float(vals.min())) if vals.size else 0.0
    z = [low, top * 1.04 if top > 0 else 1.0]
    if c["mode"] == Z_MAN:
        z = manual(z, c["lo"], c["hi"])
    return [z] * len(charts_grids)


def charts():
    """The chart blocks of sections 6 (Q) and 8 (v and the policy). Each block is drawn as 3D charts or heatmaps,
    with the z range set by its own controls (the z axis of a 3D chart, the color scale of a heatmap)."""
    n = n_done()
    S.draws += 1
    kv = earlier(0, n)
    kq = earlier(1, n)
    vg = [ic.grid(S.V[k]) for k in kv] + [ic.grid(S.V[n])]
    qg = [[ic.grid(S.Q[k][:, a]) for k in kq] + ([ic.grid(S.Q[n][:, a])] if n else []) for a in ic.A]
    zq = zranges(qg, qc, run_max("Q"))
    zv = zranges([vg], vc, run_max("V"))[0]
    pi = S.Q[n].argmax(axis=1) if n else None
    pis = [S.Q[k].argmax(axis=1) for k in range(1, n + 1)]
    pg = [ic.grid(pis[k - 1]) for k in kq] + ([ic.grid(pis[-1])] if n else [])
    labels = [f"({e}, {s})" for e, s in ic.STATES]
    opt = dict(width="stretch", config={"displayModeBar": False})

    # section 6: Q
    for a in ic.A:
        best = ic.grid(pi == a) > 0 if n else None
        title = f"Q{sub(n)}((e, s), a = {a})  ▸  STORE {a}"
        if qc["view"] == VIEW_3D:
            fig = plots.surface3d(qg[a][-1] if n else None, qg[a][:-1] if n else [], ACOL[a], title, f"Q{sub(n)}",
                                  zq[a], f"Q{sub(n)}((e,s),{a})", best=best, height=400)
        else:
            fig = plots.heatmap(qg[a][-1] if n else None, ACOL[a], title, z_range=zq[a], best=best, height=400)
        ph_qa[a].plotly_chart(fig, **opt, key=f"q_{a}_{S.draws}")

    # section 8: v, the policy, and the policy over the iterations
    if vc["view"] == VIEW_3D:
        shown = f"DASHED: k = {', '.join(map(str, kv[-6:]))}{' …' if len(kv) > 6 else ''}" if kv else "NO EARLIER"
        fig_v = plots.surface3d(vg[-1], vg[:-1], theme.C["amber"], f"v{sub(n)}(e, s)  ▸  {shown}", f"v{sub(n)}", zv,
                                f"v{sub(n)}(e,s)", height=400)
        fig_pi = plots.surface3d(pg[-1] if n else None, pg[:-1] if n else [], theme.C["green"],
                                 f"π{sub(n)}(e, s)  ▸  SCOOPS STORED", f"π{sub(n)}", [-0.2, ic.K - 0.8],
                                 f"π{sub(n)}(e,s)", height=400, zticks=list(ic.A), fmt=".0f")
        fig_hist = plots.policy_history(pis, ACOL, labels, height=400)
    else:
        fig_v = plots.heatmap(vg[-1], theme.C["amber"], f"v{sub(n)}(e, s)", z_range=zv, height=400)
        fig_pi = plots.heatmap(pg[-1] if n else None, theme.C["green"], f"π{sub(n)}(e, s)  ▸  SCOOPS STORED",
                               discrete=ACOL, height=400)
        fig_hist = plots.policy_heatmap(pis, ACOL, labels, height=400)
    ph_v3.plotly_chart(fig_v, **opt, key=f"v_{S.draws}")
    ph_pi3.plotly_chart(fig_pi, **opt, key=f"pi_{S.draws}")
    ph_pih.plotly_chart(fig_hist, **opt, key=f"hist_{S.draws}")
    ph_dist.plotly_chart(plots.closeness(dist_v(), dist_q(), tol, beta, policy_changes(), height=380), **opt,
                         key=f"d_{S.draws}")


def render(status, draw=True):
    n = n_done()
    dv = dist_v()
    d = dv[-1] if len(dv) else np.nan
    ch = policy_changes()
    now = datetime.now().strftime("%H:%M:%S")
    source = "REPLAYING THE PRECOMPUTED RUN"
    ph_bar.markdown(
        f'<div class="term-bar"><span>RL ▸ 0002 ▸ VALUE ITERATION ▸ ICE CREAM STORAGE</span>'
        f'<span>{source}  ●  ITERATION {n:,} / {max_iter:,}  ●  {status}  ●  {now}</span></div>', unsafe_allow_html=True)
    ratio = dv[-1] / dv[-2] if len(dv) > 1 and dv[-2] > 0 else np.nan
    dq = dist_q()
    ph_kpi.markdown(theme.kpis([
        ("STATES × ACTIONS", f"{ic.N} × {ic.K}", False),
        ("β", f"{beta:g}", False),
        ("ITERATION k + 1", f"{n:,}", True),
        ("‖v<sub>k+1</sub> − v<sub>k</sub>‖∞", f"{d:.3e}", True),
        ("‖Q<sub>k+1</sub> − Q<sub>k</sub>‖∞", f"{dq[-1]:.3e}" if len(dq) else "—", True),
        ("RATIO OF DISTANCES (→ β)", f"{ratio:.4f}", False),
        ("ε-OPTIMAL WITH ε =", f"{2 * beta * d / (1 - beta):.3e}" if beta < 1 and n else "—", False),
        ("POLICY LAST CHANGED AT", f"{ch[-1]}" if ch else ("1" if n else "—"), False),
    ]), unsafe_allow_html=True)
    ph_q.markdown(q_table(), unsafe_allow_html=True)
    ph_ct.markdown(conv_table(), unsafe_allow_html=True)
    panel, lines = backup()
    ph_b.markdown(panel, unsafe_allow_html=True)
    with ph_bl.container():
        for line in lines:
            st.latex(line)
    if draw:
        charts()


# --------------------------------------------------------------------- run
if step and (S.part is not None or n_done() < min(max_iter, LAST)):
    advance()

if run and not done():
    last, drawn = 0.0, -1
    while not done():
        advance()
        n = n_done()
        draw = n != drawn and (n <= 5 or time.time() - last > 0.5 or done())
        render("RUNNING", draw)
        if draw:
            last, drawn = time.time(), n
        if delay:
            time.sleep(delay)

status = "CONVERGED" if converged() else "MAX ITERATIONS" if n_done() >= max_iter else "IDLE"
render(status)
matrices()
if DEP is not None:
    draw_deployment(S.dep_t)
    if dep_play:
        for day in range(0 if S.dep_t >= len(DEP) else S.dep_t, len(DEP) + 1):
            draw_deployment(day)
            S.dep_t = day
            time.sleep(max(delay, 0.15))
