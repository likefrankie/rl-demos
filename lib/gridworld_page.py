"""Value iteration demo for the 1D, 2D, and 3D gridworlds, built only from the precomputed data of
manual/gridworld{1,2,3}D_value-iteration (run/value_iteration_run.npz and deployment/policy_rollout.csv).

The page mirrors the ice cream demo: environment, agent, MDP, value iteration, demo settings, Q function, backup of
one state, value function and policy, convergence, and deployment. It only replays the stored iterates: nothing is
computed while it runs.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from lib import plots, rundata, theme

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
ACOL = ["#38bdf8", "#b388ff", "#19e68c", "#f59e0b", "#f472b6", "#94a3b8"]
ARROW = {1: ["←", "→"], 2: ["←", "→", "↑", "↓"], 3: ["←", "→", "▲", "▼", "↑", "↓"]}
BL = f'style="border-left:1px solid {theme.C["line"]}"'
VIEW_3D, VIEW_HEAT = "3D", "heatmap"
Z_FIT, Z_ZERO, Z_MAN = "min to max, current iteration", "0 to max of the run", "manual"
FOLLOW = "FOLLOW THE UPDATE"
# display width (px) of the environment's own render: about its native size (1D 800 x 200, 2D 400 x 400,
# 3D 1020 x 830), never stretched to the column
IMG_W = {1: 700, 2: 380, 3: 760}

SPECS = {
    "gridworld1D_value-iteration": dict(
        title="1D GRIDWORLD", lab="0002-01-02 lab exercise 02",
        story="A robot travels along a straight path divided into 21 segments and starts at the center segment, 10. "
              "Every step it tries to move left or right: with probability 0.8 it moves in the intended direction and "
              "with probability 0.2 in the opposite one. Segment 0 is a hole (reward −1) and segment 20 a prize "
              "(reward +1); reaching either ends the episode. Every other move costs fuel (reward −0.05).",
        params=[("Segments", "21, numbered 0 to 20"), ("Start", "segment 10"), ("Actions", "left, right"),
                ("Intended move", "probability 0.8 (0.2 the other way)"), ("Hole, segment 0", "reward −1, ends"),
                ("Prize, segment 20", "reward +1, ends"), ("Any other move", "reward −0.05")]),
    "gridworld2D_value-iteration": dict(
        title="2D GRIDWORLD", lab="0002-01-02 lab exercises 02–03",
        story="A robot moves in a 4 × 4 grid and starts at cell (3, 1). It tries to move left, right, up, or down: it "
              "moves in the intended direction with probability 0.7 and to each side with probability 0.15. Cells "
              "(1, 2) and (2, 1) are blocked: a move into one, or off the grid, leaves the robot in place. The cheese "
              "(3, 3) gives +1 and the trap (3, 2) gives −1; reaching either ends the episode. Every other move "
              "costs fuel (reward −0.05).",
        params=[("Grid", "4 × 4, cells (x, y)"), ("Start", "(3, 1)"), ("Actions", "left, right, up, down"),
                ("Intended move", "probability 0.7 (0.15 to each side)"), ("Blocked", "(1, 2), (2, 1)"),
                ("Cheese (3, 3)", "reward +1, ends"), ("Trap (3, 2)", "reward −1, ends"),
                ("Any other move", "reward −0.05")]),
    "gridworld3D_value-iteration": dict(
        title="3D GRIDWORLD", lab="0002-01-02 lab exercises 04–05",
        story="The 2D gridworld extended through four layers: a 4 × 4 × 4 grid, starting at (3, 1, 2). The robot "
              "tries to move left, right, up or down a layer, forward, or back: it moves in the intended direction "
              "with probability 0.68 and in each of the four perpendicular directions with probability 0.08. Two "
              "pillars of blocked cells, (1, 2, l) and (2, 1, l), span every layer, with a window at (1, 2, 2). The "
              "trap is a pillar of terminal cells (3, 2, l) with reward −1; the cheese (3, 3, 3) gives +1. Every "
              "other move costs fuel (reward −0.05).",
        params=[("Grid", "4 × 4 × 4, cells (x, y, layer)"), ("Start", "(3, 1, 2)"),
                ("Actions", "left, right, up a layer, down a layer, forward, back"),
                ("Intended move", "probability 0.68 (0.08 each perpendicular)"),
                ("Blocked", "pillars (1, 2, l) with a window at l = 2, and (2, 1, l)"),
                ("Cheese (3, 3, 3)", "reward +1, ends"), ("Trap pillar (3, 2, l)", "reward −1, ends"),
                ("Any other move", "reward −0.05")]),
}


def sub(n):
    return str(n).translate(SUB)


def fq(x, d=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return "−∞" if x == -np.inf else f"{x:,.{d}f}".replace("-", "−")


def item(title):
    st.markdown(f'<div class="side-h">{title}</div>', unsafe_allow_html=True)


def note(html):
    st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)


def main(code_base):
    spec = SPECS[code_base]
    RUN = rundata.load(code_base, "run/value_iteration_run.npz")
    if RUN is None:
        st.error(f"No precomputed run found at {rundata.path(code_base, 'run')}. Run run_all.py in "
                 f"manual/{code_base} first.")
        return
    meta = RUN["meta"]
    P, Rw, qv = RUN["P"], RUN["R"], RUN["q"]
    blocked, terminal, states = RUN["blocked"], RUN["terminal"], RUN["states"]
    shape = [int(x) for x in meta["shape"]]
    D = len(shape)
    N, K = P.shape[0], P.shape[1]
    names = meta["action_names"]
    arrows = ARROW[D]
    moves, a2m, start = RUN["moves"], RUN["action2move"], [int(x) for x in RUN["start"]]
    LBL = [str(int(c[0])) if D == 1 else "(" + ", ".join(str(int(v)) for v in c) + ")" for c in states]
    LABELS = [f"state {i}  ·  {LBL[i]}" for i in range(N) if not blocked[i] and not terminal[i]]
    OPEN = [i for i in range(N) if not blocked[i] and not terminal[i]]
    # a terminal cell is the goal if entering it from an open cell pays a positive reward (the self-loop of an
    # absorbing cell has reward 0 and does not count)
    enter = (P > 0) & ~(blocked | terminal)[:, None, None]
    goal = [i for i in range(N) if terminal[i] and enter[:, :, i].any() and (Rw[:, :, i][enter[:, :, i]] > 0).all()]
    kind = np.array(["blocked" if blocked[i] else ("goal" if i in goal else ("trap" if terminal[i] else "open"))
                     for i in range(N)])
    AX = (dict(x="segment", y="", hover="segment %{x}") if D == 1 else
          dict(x="x", y="y", hover="(x, y) = (%{x}, %{y})"))
    SS = st.session_state.setdefault(f"gw_{code_base}", {})

    def grid(vals, layer=None):
        """A vector over the states as the grid shown in the charts: (1, 21) for 1D, z[y, x] for 2D, and the layer's
        z[y, x] for 3D."""
        vals = np.asarray(vals, dtype=object if isinstance(vals[0], str) else float)
        if D == 1:
            return vals.reshape(1, -1)
        out = np.empty((shape[1], shape[0]), dtype=vals.dtype)
        for i, c in enumerate(states):
            if D == 2 or int(c[2]) == layer:
                out[int(c[1]), int(c[0])] = vals[i]
        return out

    # ----------------------------------------------------------------- inputs
    ph_bar = st.empty()
    with st.expander("1 · ENVIRONMENT", expanded=False):
        note(spec["story"])
        c1, c2 = st.columns([2, 3], gap="medium")
        c1.markdown(theme.panel("TABLE 1  ▸  PARAMETERS", '<table class="term">' + "".join(
            f'<tr><td class="l fixed">{k}</td><td class="l">{v}</td></tr>' for k, v in spec["params"]) + '</table>'),
            unsafe_allow_html=True)
        with c2:
            # the environment's own render() of the start, saved by optimal_policy_player.py
            first = rundata.path(code_base, "deployment/frames/episode_0/step_000.png")
            if os.path.exists(first):
                st.image(first, caption="The environment at the start (its own render). Green = goal, red = trap, "
                                        "purple = blocked, white = the robot.", width=IMG_W[D])
            else:
                note("No rendered frames found: run optimal_policy_player.py in the code base.")
        mv_names = names
        note("Slip probabilities: the row is the action chosen, the column the move that actually happens.")
        st.markdown('<table class="term mat"><tr><th class=l>action ╲ move</th>' + "".join(
            f'<th>{arrows[m]} {mv_names[m]}</th>' for m in range(K)) + '</tr>' + "".join(
            f'<tr><td class="l fixed">{a} · {arrows[a]} {names[a]}</td>' + "".join(
                f'<td{" style=&quot;font-weight:700&quot;" if a2m[a, m] > 0.5 else ""}>{a2m[a, m]:.2f}</td>'
                for m in range(K)) + '</tr>' for a in range(K)) + '</table>', unsafe_allow_html=True)

    with st.expander("2 · AGENT", expanded=False):
        left, right = st.columns([3, 2], gap="medium")
        with left:
            item("DISCOUNT FACTOR  ▸  β")
            b1, b2 = st.columns([1, 3], gap="medium")
            beta = float(meta["beta"])
            b1.markdown(theme.kpis([("β  DISCOUNT FACTOR", f"{beta:g}", True)]), unsafe_allow_html=True)
            b2.markdown(f'<div class="note">The value of the lab, which the precomputed run was generated with. '
                        + ('With β = 1 value iteration still converges, because every episode under a good policy '
                           'ends in a terminal cell.' if beta >= 1 else '') + '</div>', unsafe_allow_html=True)
            item("OPTIMIZATION PROBLEM")
            st.latex(r"\max\ \mathbb{E}\Big[\sum_{t=0}^{T-1}\beta^{t}\,r(s_t,a_t,s_{t+1})\Big],\qquad "
                     r"T=\text{the step at which the robot reaches a terminal cell}")
            note("The robot wants to reach the goal while paying as little fuel as possible and avoiding the trap; "
                 "with slips, a short route next to the trap can be worse than a longer, safer one.")
        right.markdown(theme.panel("REWARDS  ▸  r(s, a, s′) = REWARD OF ENTERING s′", '<table class="term">'
                                   + "".join(f'<tr><td class="l fixed">{k}</td><td>{v}</td></tr>' for k, v in [
                                       ("goal cell", "+1, the episode ends"), ("trap cell", "−1, the episode ends"),
                                       ("any other cell", "−0.05 (fuel)"),
                                       ("out of a terminal cell", "0 (absorbing)")]) + '</table>'),
                       unsafe_allow_html=True)

    with st.expander("3 · MARKOV DECISION PROCESS", expanded=False):
        item("STATES  ▸  s")
        note(f"The state is the robot's cell. There are {N} cells: {int(blocked.sum())} blocked (not states the "
             f"robot can be in) and {int(terminal.sum())} terminal. The tables and charts number them as below.")
        st.markdown('<table class="term mat"><tr><th class=l>STATE</th>' + "".join(f'<th>{i}</th>' for i in range(N))
                    + '</tr><tr><td class="l fixed">cell</td>' + "".join(f'<td>{LBL[i]}</td>' for i in range(N))
                    + '</tr><tr><td class="l fixed">kind</td>' + "".join(
                        f'<td class="fixed">{kind[i]}</td>' for i in range(N)) + '</tr></table>', unsafe_allow_html=True)
        item("ACTIONS  ▸  A(s)")
        note("Every action is available in every open cell: " + ", ".join(f"{a} = {arrows[a]} {names[a]}"
                                                                       for a in range(K)) + ".")
        item("TRANSITION PROBABILITIES  ▸  p(s′ | s, a)")
        note("From each open cell, the next cells of each action with their probabilities: the intended move or a "
             "slip; a move off the grid or into a blocked cell leaves the robot in place. Terminal cells are "
             "absorbing.")
        st.latex(r"p(s'\mid s,a)=\sum_{m:\ \text{move }m\text{ from }s\text{ lands in }s'}P(\text{move }m\mid a)")
        ph_p = st.empty()
        item("REWARDS  ▸  r(s, a, s′)")
        note("The reward depends only on the cell entered, s′.")
        st.latex(r"r(s,a,s')=\begin{cases}+1 & s'=\text{goal}\\-1 & s'=\text{trap}\\-0.05 & \text{otherwise}"
                 r"\end{cases}\qquad(0\text{ out of a terminal cell})")
        item("EXPECTED IMMEDIATE REWARD  ▸  q(s, a)")
        note("<b>It does not depend on v<sub>k</sub>, so it is computed once and is the same at every iteration</b>; "
             "in Q<sub>k+1</sub> (section 6) only the term with v<sub>k</sub> changes with k.")
        st.latex(r"q(s,a)=\sum_{s'}p(s'\mid s,a)\,r(s,a,s')")
        ph_qr = st.empty()

    with st.expander("4 · VALUE ITERATION", expanded=False):
        c1, c2 = st.columns([1, 5], gap="medium")
        tol = c1.number_input(r"tolerance on $\lVert v_{k+1}-v_k\rVert_\infty$", min_value=1e-10, max_value=1.0,
                              value=1e-4, step=1e-5, format="%.1e", key=f"{code_base}_tol")
        c2.markdown('<div class="note"><b>The Bellman equation.</b> The optimal value function v(s), the largest '
                    'expected discounted sum of rewards from cell s, satisfies</div>', unsafe_allow_html=True)
        c2.latex(r"v(s)=\max_{a}\sum_{s'}p(s'\mid s,a)\Big[r(s,a,s')+\beta\,v(s')\Big]")
        c2.markdown('<div class="note">Value iteration puts the current guess v<sub>k</sub> on the right-hand side and '
                    'reads off v<sub>k+1</sub>. <b>1. Initialization.</b> v₀ = 0. <b>2. Update.</b> Q<sub>k+1</sub> '
                    'for every open cell and action (sections 6 and 7), then v<sub>k+1</sub> = max and π<sub>k+1</sub> '
                    '= arg max (section 8). <b>3. Termination.</b> Stop when ‖v<sub>k+1</sub> − v<sub>k</sub>‖∞ is '
                    'below the tolerance (section 9).</div>', unsafe_allow_html=True)

    with st.expander("5 · DEMO SETTINGS", expanded=False):
        c = st.columns([1, 1, 2, 1])
        max_iter = int(c[0].number_input("max iterations", min_value=1, max_value=5000, value=500, step=50,
                                         key=f"{code_base}_maxit"))
        unit = c[1].radio("one step updates", ["all states", "one state"], key=f"{code_base}_unit")
        start_i = int(np.nonzero((states == start).all(axis=1))[0][0])
        pick = c[2].selectbox("state in the backup table", [FOLLOW] + LABELS,
                              index=LABELS.index(f"state {start_i}  ·  {LBL[start_i]}") + 1, key=f"{code_base}_pick")
        delay = c[3].number_input("pause per step (s)", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                                  format="%.2f", key=f"{code_base}_delay")

    with st.container(border=True):
        c = st.columns([1.2, 0.6, 0.6, 5])
        run = c[0].button("▶  RUN", type="primary", width="stretch", key=f"{code_base}_run")
        step = c[1].button("+1", width="stretch", key=f"{code_base}_step")
        reset = c[2].button("RESET", width="stretch", key=f"{code_base}_reset")
        c[3].markdown('<div class="note">+1 applies one step: all states (one iteration) or the next state only. RUN '
                      'repeats until ‖v<sub>k+1</sub> − v<sub>k</sub>‖∞ is below the tolerance or the max iterations '
                      'are reached. Any click stops a run.</div>', unsafe_allow_html=True)

    LAST = len(RUN["Q"]) - 1        # the last iteration stored
    note(f"<b>Data:</b> replaying the precomputed run of <code>manual/{code_base}</code> ({meta['iterations']:,} "
         f"iterations, β = {beta:g}). Nothing is computed while it plays.")

    # ----------------------------------------------------------------- state
    cfg = (beta,)
    if reset or SS.get("cfg") != cfg:
        SS.update(cfg=cfg, V=[np.where(blocked, np.nan, 0.0)], Q=[None], part=None, last=None, fresh=set(), draws=0)

    def q_of(i, k):
        """Q_k of state i, from the precomputed run."""
        return RUN["Q"][k][i]

    def v_of(q):
        return np.where(blocked, np.nan, np.nanmax(np.where(np.isnan(q), -np.inf, q), axis=1))

    def finish(q):
        SS["Q"].append(q)
        SS["V"].append(v_of(q))
        SS["part"] = None

    def advance():
        k = len(SS["V"])
        if unit == "one state":
            if SS["part"] is None:
                SS["part"] = dict(i=0, q=np.full((N, K), np.nan))
            while SS["part"]["i"] < N and blocked[SS["part"]["i"]]:
                SS["part"]["i"] += 1
            i = SS["part"]["i"]
            SS["part"]["q"][i] = q_of(i, k)
            SS["part"]["i"] += 1
            SS["last"], SS["fresh"] = i, {i}
            while SS["part"]["i"] < N and blocked[SS["part"]["i"]]:
                SS["part"]["i"] += 1
            if SS["part"]["i"] >= N:
                finish(SS["part"]["q"])
        else:
            q = SS["part"]["q"] if SS["part"] else np.full((N, K), np.nan)
            first = SS["part"]["i"] if SS["part"] else 0
            for i in range(first, N):
                q[i] = q_of(i, k)
            SS["fresh"] = set(range(first, N))
            finish(q)

    def n_done():
        return len(SS["V"]) - 1

    def policy(q):
        p = np.nan_to_num(q, nan=-np.inf).argmax(axis=1)
        return np.where(blocked | terminal, -1, p)

    def dist_v():
        return np.array([np.nanmax(np.abs(SS["V"][k] - SS["V"][k - 1])) for k in range(1, len(SS["V"]))])

    def dist_q():
        return np.array([np.nanmax(np.abs(SS["Q"][k] - SS["Q"][k - 1])) for k in range(2, len(SS["Q"]))])

    def changes():
        pis = [policy(q) for q in SS["Q"][1:]]
        return [k + 1 for k in range(1, len(pis)) if (pis[k] != pis[k - 1]).any()]

    def converged():
        d = dist_v()
        return len(d) > 0 and d[-1] < tol

    def done():
        return SS["part"] is None and (converged() or n_done() >= min(max_iter, LAST))

    def backup_state():
        if pick != FOLLOW:
            return int(pick.split()[1])
        return SS["last"] if SS["last"] is not None and not terminal[SS["last"]] else start_i

    # ----------------------------------------------------------------- output sections
    ph_kpi = st.empty()
    nb = int(blocked.sum())
    blocked_note = ("" if nb == 0 else
                    f'<div class="note">The {N} states include {nb} blocked cell{"s" if nb > 1 else ""} the robot cannot '
                    f'enter: ' + ", ".join(f"state {i} = {LBL[i]}" for i in np.nonzero(blocked)[0]) + '. As in the lab, '
                    f'{"they are" if nb > 1 else "it is"} kept in the state space with no transitions (zero rows of P); '
                    'the tables and charts leave them out.</div>')

    def chart_controls(key, what, views):
        c = st.columns([1.1, 2.8, 0.8, 0.8, 1.2])
        view = c[0].radio("chart type", views, horizontal=True, key=f"{code_base}_view_{key}")
        mode = c[1].radio(f"z range of the {what} charts", [Z_FIT, Z_ZERO, Z_MAN], horizontal=True,
                          key=f"{code_base}_zmode_{key}")
        lo = c[2].number_input("z min (blank = auto)", value=None, step=0.1, format="%.2f", disabled=mode != Z_MAN,
                               key=f"{code_base}_zmin_{key}")
        hi = c[3].number_input("z max (blank = auto)", value=None, step=0.1, format="%.2f", disabled=mode != Z_MAN,
                               key=f"{code_base}_zmax_{key}")
        layer = (c[4].selectbox("layer", list(range(shape[2])), index=start[2], key=f"{code_base}_layer_{key}")
                 if D == 3 else None)
        return dict(view=view, mode=mode, lo=lo, hi=hi, layer=layer)

    views = [VIEW_HEAT, VIEW_3D] if D == 2 else [VIEW_HEAT]
    with st.expander("6 · Q FUNCTION", expanded=False):
        st.latex(r"Q_{k+1}(s,a)=q(s,a)+\beta\sum_{s'}p(s'\mid s,a)\,v_k(s')")
        note("The backup of all open cells at once; section 7 works through one cell term by term. q(s, a) is the "
             "expected immediate reward of section 3, the same at every iteration.")
        ph_q = st.empty()
        qc = chart_controls("q", "Q", views)
        cols = st.columns(min(K, 3), gap="small")
        ph_qa = [cols[a % len(cols)].empty() for a in range(K)]
        note("<b>Q table:</b> bold = the largest Q in the row; its action is π<sub>k+1</sub>(s) and its value is "
             "v<sub>k+1</sub>(s). Blocked cells are left out; terminal cells keep Q = 0.")
    with st.expander("7 · BACKUP OF ONE STATE", expanded=False):
        st.latex(r"Q_{k+1}(s,a)=\sum_{s'}p(s'\mid s,a)\Big[r(s,a,s')+\beta\,v_k(s')\Big]")
        ph_b = st.empty()
        ph_bl = st.empty()
        note("One table per action: each row is a next cell s′ with p &gt; 0, the last column is p × [r + β v<sub>k</sub>"
             "(s′)], and its sum is Q<sub>k+1</sub>(s, a).")
    with st.expander("8 · VALUE FUNCTION AND POLICY", expanded=False):
        st.latex(r"v_{k+1}(s)=\max_{a}Q_{k+1}(s,a),\qquad \pi_{k+1}(s)\in\arg\max_{a}Q_{k+1}(s,a)")
        vc = chart_controls("v", "v", views)
        va, pa = st.columns(2, gap="small")
        ph_v, ph_pi = va.empty(), pa.empty()
        ph_hist = st.empty()
        note("The policy chart writes the arrow of π<sub>k+1</sub> in each cell (T = terminal); the history shows, for "
             "every open cell, the action at each iteration.")
    with st.expander("9 · CONVERGENCE", expanded=False):
        q1, q2 = st.columns(2)
        q1.latex(r"\lVert v_{k+1}-v_k\rVert_\infty=\max_{s}\Big|v_{k+1}(s)-v_k(s)\Big|")
        q1.latex(r"\lVert Q_{k+1}-Q_k\rVert_\infty=\max_{s}\ \max_{a}\Big|Q_{k+1}(s,a)-Q_k(s,a)\Big|")
        q2.latex(r"\text{stop when }\lVert v_{k+1}-v_k\rVert_\infty<\text{tolerance}")
        note("With β &lt; 1 the distances shrink at least at the rate β. With β = 1 there is no such guarantee from β: "
             "they shrink because every episode under a good policy ends in a terminal cell.")
        cl, cr = st.columns([6, 5], gap="small")
        ph_ct, ph_dist = cl.empty(), cr.empty()
    with st.expander("10 · DEPLOYMENT", expanded=False):
        deployment(code_base, D, shape, states, kind, AX, grid, names, arrows, delay)

    # ----------------------------------------------------------------- drawing
    def mdp_tables(sel):
        rows = []
        for i in range(N):
            if blocked[i]:
                continue
            tint = f' style="background:{theme.TINT}"' if i == sel else ""
            cells = f'<td class="l fixed">{"▶ " if i == sel else ""}{i} {LBL[i]}</td>'
            for a in range(K):
                js = np.nonzero(P[i, a] > 0)[0]
                cells += f'<td class="l" {BL if a == 0 else ""}>' + ", ".join(f"{LBL[j]}: {P[i, a, j]:.2f}"
                                                                          for j in js) + '</td>'
            rows.append(f'<tr{tint}>{cells}</tr>')
        head = '<th class=l>STATE</th>' + "".join(f'<th class=l>a = {a} · {arrows[a]} {names[a]}</th>' for a in range(K))
        ph_p.markdown(f'<div style="overflow-x:auto"><table class="term mat"><tr>{head}</tr>{"".join(rows)}</table></div>',
                      unsafe_allow_html=True)
        rows = []
        for i in range(N):
            if blocked[i]:
                continue
            tint = f' style="background:{theme.TINT}"' if i == sel else ""
            cells = f'<td class="l fixed">{"▶ " if i == sel else ""}{i} {LBL[i]}</td>'
            for a in range(K):
                js = np.nonzero(P[i, a] > 0)[0]
                calc = " + ".join(f"{P[i, a, j]:.2f} × {fq(Rw[i, a, j], 2)}" for j in js)
                cells += f'<td class="fixed" {BL}>{calc}</td><td>{fq(qv[i, a])}</td>'
            rows.append(f'<tr{tint}>{cells}</tr>')
        head = ('<th class=l>STATE</th>' + "".join(f'<th {BL}>Σ p × r, a = {a} {arrows[a]}</th><th>q(s, {a})</th>'
                                                 for a in range(K)))
        ph_qr.markdown(theme.panel("q(s, a)  ▸  COMPUTED ONCE, THE SAME AT EVERY ITERATION",
                                   f'<div style="overflow-x:auto"><table class="term mat"><tr>{head}</tr>'
                                   f'{"".join(rows)}</table></div>', right="CONSTANT ACROSS ITERATIONS"),
                       unsafe_allow_html=True)

    def q_table():
        part = SS["part"]
        n = len(SS["V"]) if part else n_done()
        sel = backup_state()
        heads = ["STATE"] + [f"Q{sub(n)}(s, {a}) {arrows[a]}" for a in range(K)] + [f"π{sub(n)}(s)", f"v{sub(n)}(s)"]
        head = "".join(f'<th{" class=l" if h == "STATE" else ""}>{h}</th>' for h in heads)
        rows = []
        for i in range(N):
            if blocked[i]:
                continue
            lead = f'<td class="l fixed">{"▶ " if i == sel else ""}{i} {LBL[i]}</td>'
            new = n >= 1 and (part is None or i < part["i"])
            if n == 0 or (part and not new and len(SS["Q"]) < 2):
                cells = "".join('<td class="fixed">—</td>' for _ in range(K + 2))
            else:
                q = (part["q"][i] if part else SS["Q"][-1][i]) if new else SS["Q"][-1][i]
                if terminal[i]:
                    cells = "".join(f'<td class="fixed">{fq(x)}</td>' for x in q) + '<td class="fixed">T</td>' \
                            f'<td class="fixed">{fq(0.0)}</td>'
                else:
                    best = int(np.nanargmax(q))
                    cls = "" if new else ' class="fixed"'
                    cells = "".join(f'<td style="color:{theme.C["amber"]};font-weight:700">{fq(x)}</td>'
                                    if a == best and new else f'<td{cls}>{fq(x)}</td>' for a, x in enumerate(q))
                    fresh = ' class="fresh"' if i in SS["fresh"] and new else ""
                    cells += f'<td{fresh}>{arrows[best]}</td><td{fresh}>{fq(np.nanmax(q))}</td>'
            rows.append(f'<tr>{lead}{cells}</tr>')
        right = f"UPDATED {part['i']} / {N} STATES" if part else "BOLD = ARG MAX  ·  ▶ = STATE IN THE BACKUP TABLE"
        return theme.panel(f"Q TABLE  ▸  ITERATION k + 1 = {n}", f'<div style="overflow-x:auto"><table class="term mat">'
                                                                f'<tr>{head}</tr>{"".join(rows)}</table></div>', right=right)

    def backup():
        i = backup_state()
        if SS["part"] is not None and i < SS["part"]["i"]:
            n, vk = len(SS["V"]), SS["V"][-1]
        elif n_done() >= 1:
            n, vk = n_done(), SS["V"][-2]
        else:
            n, vk = 1, SS["V"][0]
        k = n - 1
        vk0 = np.nan_to_num(vk)
        tables, lines, qs = [], [], []
        for a in range(K):
            js = np.nonzero(P[i, a] > 0)[0]
            terms = P[i, a, js] * (Rw[i, a, js] + beta * vk0[js])
            qs.append(terms.sum())
            rows = "".join(f'<tr><td class="l">{LBL[j]}</td><td>{P[i, a, j]:.2f}</td><td>{fq(Rw[i, a, j], 2)}</td>'
                           f'<td>{fq(vk0[j])}</td><td>{fq(beta * vk0[j])}</td><td>{fq(t)}</td></tr>'
                           for j, t in zip(js, terms))
            rows += (f'<tr style="border-top:1px solid {theme.C["amber"]}"><td class="l" style="color:#fff">Σ</td>'
                     f'<td>{P[i, a, js].sum():.2f}</td><td></td><td></td><td></td>'
                     f'<td style="font-weight:700">{fq(terms.sum())}</td></tr>')
            tables.append(theme.panel(f"a = {a}  ·  {arrows[a]} {names[a].upper()}",
                                      '<table class="term mat"><tr><th class=l>s′</th><th>p</th><th>r</th>'
                                      f'<th>v{sub(k)}(s′)</th><th>β v{sub(k)}(s′)</th><th>p × [r + β v{sub(k)}]</th>'
                                      f'</tr>{rows}</table>'))
            lines.append(rf"Q_{{{n}}}({LBL[i]},{a})=" + "+".join(
                rf"{P[i, a, j]:.2f}\,[{Rw[i, a, j]:.2f}+{beta:g}\times{vk0[j]:.4f}]" for j in js)
                + rf"={terms.sum():.4f}")
        best = int(np.argmax(qs))
        qs_txt = r",\ ".join(f"{x:.4f}" for x in qs)
        lines.append(rf"v_{{{n}}}({LBL[i]})=\max\{{{qs_txt}\}}={max(qs):.4f},\qquad"
                     rf"\pi_{{{n}}}({LBL[i]})={best}\ (\text{{{names[best]}}})")
        head = theme.panel(f"BACKUP  ▸  STATE {i} = {LBL[i]}  ▸  ITERATION k + 1 = {n}",
                           f'<div class="note">The rows are the only next cells with p &gt; 0 from {LBL[i]}; β = '
                           f'{beta:g}. Σ of the last column = Q{sub(n)}({LBL[i]}, a).</div>', right=f"USES v{sub(k)} ONLY")
        return head, tables, lines

    def conv_table():
        n = n_done()
        title = f"DISTANCES  ▸  ITERATION k + 1 = {n}"
        if n == 0:
            return theme.panel(title, '<div class="note">No iteration yet: press +1 or RUN.</div>')
        k = n - 1
        v1, v0 = SS["V"][n], SS["V"][k]
        dv = np.abs(v1 - v0)
        iv = int(np.nanargmax(dv))
        dq = None if SS["Q"][k] is None else np.abs(SS["Q"][n] - SS["Q"][k])
        if dq is not None:
            dq[blocked] = 0.0
        amb = f' style="color:{theme.C["amber"]};font-weight:700"'
        heads = (["STATE", f"v{sub(n)}(s)", f"v{sub(k)}(s)", f"|v{sub(n)} − v{sub(k)}|"]
                 + [f"|ΔQ| a = {a}" for a in range(K)] + ["max over a"])
        rows = []
        if dq is not None:
            dqs = np.nanmax(dq, axis=1)
            iq = int(np.nanargmax(dqs))
        for i in range(N):
            if blocked[i]:
                continue
            cells = (f'<td class="l fixed">{i} {LBL[i]}</td><td>{fq(v1[i])}</td><td>{fq(v0[i])}</td>'
                     f'<td{amb if i == iv else ""}>{dv[i]:.2e}</td>')
            if dq is None:
                cells += "".join('<td class="fixed">—</td>' for _ in range(K + 1))
            else:
                cells += "".join(f'<td>{x:.2e}</td>' for x in dq[i]) + f'<td{amb if i == iq else ""}>{dqs[i]:.2e}</td>'
            rows.append(f'<tr>{cells}</tr>')
        tot = f'<td class="l" colspan=3 style="color:#fff">max over s = ‖·‖∞</td><td{amb}>{dv[iv]:.2e}</td>'
        tot += ('<td colspan={} class="fixed">Q₀ is not defined</td>'.format(K + 1) if dq is None
                else f'<td colspan={K}></td><td{amb}>{dqs[iq]:.2e}</td>')
        rows.append(f'<tr style="border-top:1px solid {theme.C["amber"]}">{tot}</tr>')
        verdict = "below the tolerance: STOP" if dv[iv] < tol else "not below the tolerance: CONTINUE"
        return theme.panel(title, f'<div class="note">‖v{sub(n)} − v{sub(k)}‖∞ = {dv[iv]:.3e} at state {iv} = '
                                  f'{LBL[iv]}, tolerance = {tol:.1e} → {verdict}.</div><div style="overflow-x:auto">'
                                  f'<table class="term mat"><tr>{"".join(f"<th>{h}</th>" for h in heads)}</tr>'
                                  f'{"".join(rows)}</table></div>', right="AMBER = THE MAX")

    def zrange(vals, c, top):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            return [0.0, 1.0]
        if c["mode"] == Z_FIT:
            lo, hi = float(vals.min()), float(vals.max())
            pad = 0.05 * (hi - lo) if hi > lo else 0.5
            z = [lo - pad, hi + pad]
        else:
            z = [min(0.0, float(vals.min())), max(top, float(vals.max())) * 1.04 if max(top, vals.max()) > 0 else 1.0]
        if c["mode"] == Z_MAN:
            z = [z[0] if c["lo"] is None else c["lo"], z[1] if c["hi"] is None else c["hi"]]
        return z

    def run_top(name):
        arr = np.asarray(RUN[name][-1], dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.abs(arr).max()) if arr.size else 1.0

    def charts():
        n = n_done()
        SS["draws"] += 1
        opt = dict(width="stretch", config={"displayModeBar": False})
        q = SS["Q"][n] if n else None
        pi = policy(q) if n else None
        top_q, top_v = run_top("Q"), run_top("V")
        h = 200 if D == 1 else 380
        for a in range(K):
            vals = q[:, a] if n else np.full(N, np.nan)
            g = grid(np.where(blocked, np.nan, vals), qc["layer"])
            z = zrange(vals if n else [], qc, top_q)
            title = f"Q{sub(n)}(s, a = {a})  ▸  {arrows[a]} {names[a].upper()}" + (f"  ·  LAYER {qc['layer']}" if D == 3 else "")
            if qc["view"] == VIEW_3D:
                fig = plots.surface3d(g if n else None, [], ACOL[a], title, f"Q{sub(n)}", z, f"Q{sub(n)}(s,{a})",
                                      best=(grid(pi == a) > 0) if n else None, height=h, axes=AX)
            else:
                lab = None
                if n:
                    lab = grid(["#" if blocked[i] else f"{vals[i]:.3f}" + (" ◆" if pi[i] == a else "")
                                for i in range(N)], qc["layer"])
                fig = plots.heatmap(g if n else None, ACOL[a], title, z_range=z, height=h, axes=AX, labels=lab)
            ph_qa[a].plotly_chart(fig, **opt, key=f"{code_base}_q_{a}_{SS['draws']}")
        vals = SS["V"][n]
        gv = grid(vals, vc["layer"])
        z = zrange(vals, vc, top_v)
        tv = f"v{sub(n)}(s)" + (f"  ·  LAYER {vc['layer']}" if D == 3 else "")
        if vc["view"] == VIEW_3D:
            fig_v = plots.surface3d(gv, [], theme.C["amber"], tv, f"v{sub(n)}", z, f"v{sub(n)}(s)", height=h, axes=AX)
        else:
            lab = grid(["#" if blocked[i] else f"{vals[i]:.3f}" for i in range(N)], vc["layer"])
            fig_v = plots.heatmap(gv, theme.C["amber"], tv, z_range=z, height=h, axes=AX, labels=lab)
        if n:
            pv = np.where(pi < 0, np.nan, pi).astype(float)
            lab = grid(["#" if blocked[i] else ("T" if terminal[i] else arrows[pi[i]]) for i in range(N)], vc["layer"])
            fig_pi = plots.heatmap(grid(pv, vc["layer"]), theme.C["green"], f"π{sub(n)}(s)" +
                                   (f"  ·  LAYER {vc['layer']}" if D == 3 else ""), discrete=ACOL[:K], height=h,
                                   axes=AX, labels=lab)
        else:
            fig_pi = plots.heatmap(None, theme.C["green"], "π(s)", height=h, axes=AX)
        ph_v.plotly_chart(fig_v, **opt, key=f"{code_base}_v_{SS['draws']}")
        ph_pi.plotly_chart(fig_pi, **opt, key=f"{code_base}_pi_{SS['draws']}")
        pis = [policy(x)[OPEN] for x in SS["Q"][1:]]
        ph_hist.plotly_chart(plots.policy_heatmap(pis, ACOL[:K], [LBL[i] for i in OPEN],
                                                  height=max(280, 14 * len(OPEN) + 80)), **opt,
                             key=f"{code_base}_hist_{SS['draws']}")
        ph_dist.plotly_chart(plots.closeness(dist_v(), dist_q(), tol, beta, changes(), height=380), **opt,
                             key=f"{code_base}_d_{SS['draws']}")

    def render(status, draw=True):
        n = n_done()
        dv = dist_v()
        d = dv[-1] if len(dv) else np.nan
        ch = changes()
        source = "REPLAYING THE PRECOMPUTED RUN"
        ph_bar.markdown(f'<div class="term-bar"><span>RL ▸ {spec["title"]} ▸ VALUE ITERATION</span><span>{source}  ●  '
                        f'ITERATION {n:,} / {max_iter:,}  ●  {status}  ●  {datetime.now():%H:%M:%S}</span></div>',
                        unsafe_allow_html=True)
        dq = dist_q()
        ph_kpi.markdown(theme.kpis([
            ("STATES × ACTIONS", f"{N} × {K}", False), ("β", f"{beta:g}", False),
            ("ITERATION k + 1", f"{n:,}", True), ("‖v<sub>k+1</sub> − v<sub>k</sub>‖∞", f"{d:.3e}", True),
            ("‖Q<sub>k+1</sub> − Q<sub>k</sub>‖∞", f"{dq[-1]:.3e}" if len(dq) else "—", True),
            ("v(START)", fq(SS["V"][n][start_i]), False),
            ("POLICY LAST CHANGED AT", f"{ch[-1]}" if ch else ("1" if n else "—"), False),
        ]) + blocked_note, unsafe_allow_html=True)
        ph_q.markdown(q_table(), unsafe_allow_html=True)
        head, tables, lines = backup()
        with ph_b.container():
            st.markdown(head, unsafe_allow_html=True)
            cols = st.columns(min(K, 3), gap="small")
            for a, tb in enumerate(tables):
                cols[a % len(cols)].markdown(tb, unsafe_allow_html=True)
        with ph_bl.container():
            for line in lines:
                st.latex(line)
        ph_ct.markdown(conv_table(), unsafe_allow_html=True)
        if draw:
            charts()

    # ----------------------------------------------------------------- run
    if step and (SS["part"] is not None or n_done() < min(max_iter, LAST)):
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
    mdp_tables(backup_state())
    play_deployment(code_base)


# --------------------------------------------------------------------- deployment
def deployment(code_base, D, shape, states, kind, AX, grid, names, arrows, delay):
    """Section 10: the precomputed episodes of the optimal policy, replayed step by step."""
    df = rundata.load_csv(code_base, "deployment/policy_rollout.csv")
    if df is None:
        note("No deployment data found: run optimal_policy_player.py in the code base.")
        return
    note("The optimal policy deployed in the environment: the episodes simulated by "
         f"<code>optimal_policy_player.py</code> in <code>manual/{code_base}</code>, shown with the environment's own "
         "render() of every step, saved as images by the same script. "
         "Every step the robot takes the action π(s) of its cell; slips make each episode different.")
    eps = sorted(df["episode"].unique())
    c = st.columns([1.2, 0.8, 0.6, 0.6, 3])
    ep = c[0].selectbox("episode", eps, key=f"{code_base}_ep")
    d = df[df["episode"] == ep].reset_index(drop=True)
    key = f"{code_base}_dt"
    st.session_state.setdefault(key, 0)
    if st.session_state.get(f"{key}_ep") != ep:
        st.session_state[key], st.session_state[f"{key}_ep"] = 0, ep
    play = c[1].button("▶  PLAY", type="primary", width="stretch", key=f"{code_base}_play")
    plus = c[2].button("+1", width="stretch", key=f"{code_base}_dplus")
    rst = c[3].button("RESET", width="stretch", key=f"{code_base}_dreset")
    if rst:
        st.session_state[key] = 0
    if plus:
        st.session_state[key] = min(len(d), st.session_state[key] + 1)
    cl, cr = st.columns([3, 2] if D == 3 else [1, 1], gap="medium")
    ph_chart, ph_tab = cl.empty(), cr.empty()

    def draw(t):
        steps = d.iloc[:t]
        frame = rundata.path(code_base, f"deployment/frames/episode_{ep}/step_{t:03d}.png")
        caption = f"Episode {ep}, step {t} / {len(d)}, total reward {steps['reward'].sum():+.2f}"
        if os.path.exists(frame):
            ph_chart.image(frame, caption=caption, width=IMG_W[D])
        else:
            ph_chart.markdown(f'<div class="note">{caption}: frame not found ({frame}).</div>', unsafe_allow_html=True)
        rows = "".join(f'<tr><td class="l fixed">{r.step}</td><td>{r.state}</td><td>{arrows[r.action]} '
                       f'{names[r.action]}</td><td>{r.next_state}</td><td>{r.reward:+.2f}</td>'
                       f'<td>{r.cumulative_reward:+.2f}</td></tr>' for r in steps.tail(8).itertuples())
        ph_tab.markdown(theme.panel("LAST STEPS", '<table class="term mat"><tr><th class=l>STEP</th><th>s</th>'
                                                  '<th>a = π(s)</th><th>s′</th><th>r</th><th>Σ r</th></tr>'
                                                  f'{rows}</table>'), unsafe_allow_html=True)

    st.session_state[f"{code_base}_deploy"] = dict(draw=draw, n=len(d), play=play, delay=delay, key=key)
    draw(st.session_state[key])


def play_deployment(code_base):
    dep = st.session_state.get(f"{code_base}_deploy")
    if not dep or not dep["play"]:
        return
    key = dep["key"]
    t0 = 0 if st.session_state[key] >= dep["n"] else st.session_state[key]
    for t in range(t0 + 1, dep["n"] + 1):
        dep["draw"](t)
        st.session_state[key] = t
        time.sleep(max(dep["delay"], 0.15))
