"""Plotly figures shared by the demos."""

import numpy as np
import pandas  # noqa: F401  imported in full before plotly checks for pandas objects (a half-imported
               # pandas, loaded lazily elsewhere, makes plotly fail with "module pandas has no attribute Series")
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import C, FONT, style_figure

STRATEGY_COLORS = {"Optimal": C["amber"], "Equal split": C["cyan"], "All at once": C["violet"]}


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _waiting(fig, text="NO RUNS YET  ▸  PRESS ▶ RUN OR +1 ABOVE"):
    """Message shown before the first run, so an empty chart doesn't look broken."""
    fig.add_annotation(text=text, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(color=C["amber"], size=14),
                       bgcolor="rgba(0,0,0,0.85)", bordercolor=C["amber"], borderwidth=1, borderpad=8)


def _many(t, rows):
    """Several paths as one trace, separated by None, so hundreds of runs stay fast."""
    xs, ys = [], []
    for r in rows:
        xs.extend(t.tolist() + [None])
        ys.extend(np.asarray(r).tolist() + [None])
    return xs, ys


def paths_grid(runs, name, p, max_paths):
    """2 x 2 grid of the paths of one strategy: price, information variable, trades, shares remaining.
    Earlier runs are faint; the latest run is bright. Dashed lines show the equal split for reference."""
    titles = [f"PRICE  P_t  ▸  {name.upper()}", "INFORMATION VARIABLE  X_t  ▸  SAME FOR ALL STRATEGIES",
              f"SHARES BOUGHT  S_t  ▸  {name.upper()}", f"SHARES REMAINING  W_t  ▸  {name.upper()}"]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.07, subplot_titles=titles)
    t = np.arange(1, p.T + 1)
    color = STRATEGY_COLORS[name]
    panels = [("P", 1, 1, "price"), ("X", 1, 2, "X"), ("S", 2, 1, "shares"), ("W", 2, 2, "shares")]
    past = runs[-(max_paths + 1):-1]
    for attr, row, col, unit in panels:
        c = C["txt"] if attr == "X" else color
        if past:
            xs, ys = _many(t, [getattr(r[name], attr) for r in past])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_rgba(c, 0.13), width=1),
                                     hoverinfo="skip"), row=row, col=col)
        if runs:
            y = getattr(runs[-1][name], attr)
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers", line=dict(color=c, width=2.4),
                                     marker=dict(size=5, color=c),
                                     hovertemplate=f"t = %{{x}}<br>{attr}_t = %{{y:,.4f}}<extra>latest run</extra>"),
                          row=row, col=col)
    eq = p.Sbar / p.T
    fig.add_hline(y=eq, line=dict(color=STRATEGY_COLORS["Equal split"], width=1.2, dash="dash"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=p.Sbar * (p.T - t + 1) / p.T, mode="lines",
                             line=dict(color=STRATEGY_COLORS["Equal split"], width=1.2, dash="dash"),
                             hovertemplate="t = %{x}<br>equal split W_t = %{y:,.0f}<extra></extra>"), row=2, col=2)
    fig.add_hline(y=p.P0, line=dict(color=C["dim"], width=1, dash="dot"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=C["dim"], width=1, dash="dot"), row=1, col=2)
    if not runs:
        _waiting(fig)
    fig.update_xaxes(range=[0.5, p.T + 0.5], title_text="period t", title_font=dict(size=10, color=C["dim"]))
    return style_figure(fig, height=640)


def results_grid(runs, p, expected):
    """2 x 2 grid of the results accumulated over runs.
    1: running average excess cost of each strategy with a +/- 2 SE band, and its expected value (dotted).
    2: histogram of the excess cost per run for each strategy, with the expected values.
    3: histogram of the saving of the optimal strategy over the equal split, per run.
    4: average trade schedule of the optimal strategy with a 10-90% band, against the equal split."""
    n = len(runs)
    titles = ["RUNNING AVERAGE EXCESS COST  vs  EXPECTED  (± 2 SE)", "EXCESS COST PER RUN  ▸  DISTRIBUTION",
              "SAVING PER RUN  ▸  EQUAL SPLIT COST − OPTIMAL COST", "AVERAGE TRADE SCHEDULE  ▸  OPTIMAL (10–90% BAND)  vs  EQUAL SPLIT"]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.13, horizontal_spacing=0.07, subplot_titles=titles)
    runs_axis = np.arange(1, n + 1)
    t = np.arange(1, p.T + 1)

    for name, color in STRATEGY_COLORS.items():
        fig.add_hline(y=expected[name], line=dict(color=color, width=1, dash="dot"), row=1, col=1)
        if not n:
            continue
        ex = np.array([r[name].excess for r in runs])
        avg = np.cumsum(ex) / runs_axis
        sq = np.cumsum(ex ** 2) / runs_axis
        se = np.sqrt(np.maximum(sq - avg ** 2, 0) / np.maximum(runs_axis - 1, 1))
        fig.add_trace(go.Scatter(x=runs_axis, y=avg + 2 * se, mode="lines", line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=runs_axis, y=avg - 2 * se, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=_rgba(color, 0.12), hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=runs_axis, y=avg, mode="lines", name=name, line=dict(color=color, width=2),
                                 hovertemplate=f"{name}<br>run %{{x}}<br>avg excess = %{{y:,.0f}}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Histogram(x=ex, nbinsx=40, marker=dict(color=_rgba(color, 0.5), line=dict(color=C["panel"], width=0.5)),
                                   showlegend=False, hovertemplate=f"{name}<br>%{{x}}<br>runs = %{{y}}<extra></extra>"),
                      row=1, col=2)
        fig.add_vline(x=expected[name], line=dict(color=color, width=1.2, dash="dot"), row=1, col=2)

    if n:
        saving = np.array([r["Equal split"].excess - r["Optimal"].excess for r in runs])
        fig.add_trace(go.Histogram(x=saving, nbinsx=40, marker=dict(color=_rgba(C["green"], 0.55),
                                   line=dict(color=C["panel"], width=0.5)), showlegend=False,
                                   hovertemplate="saving %{x}<br>runs = %{y}<extra></extra>"), row=2, col=1)
        fig.add_vline(x=saving.mean(), line=dict(color=C["green"], width=1.6), row=2, col=1)
        S = np.array([r["Optimal"].S for r in runs])
        lo, hi = np.quantile(S, 0.1, axis=0), np.quantile(S, 0.9, axis=0)
        color = STRATEGY_COLORS["Optimal"]
        fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=_rgba(color, 0.15),
                                 hoverinfo="skip", showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=S.mean(axis=0), mode="lines+markers", line=dict(color=color, width=2),
                                 marker=dict(size=5), showlegend=False,
                                 hovertemplate="t = %{x}<br>avg S_t = %{y:,.0f}<extra>optimal</extra>"), row=2, col=2)
    fig.add_vline(x=0, line=dict(color=C["dim"], width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=p.Sbar / p.T, line=dict(color=STRATEGY_COLORS["Equal split"], width=1.4, dash="dash"), row=2, col=2)
    if not n:
        _waiting(fig)

    fig.update_xaxes(title_text="run", title_font=dict(size=10, color=C["dim"]), row=1, col=1)
    fig.update_xaxes(title_text="cost above P0 · Sbar", title_font=dict(size=10, color=C["dim"]), row=1, col=2)
    fig.update_xaxes(title_text="saving (positive = optimal cheaper)", title_font=dict(size=10, color=C["dim"]), row=2, col=1)
    fig.update_xaxes(title_text="period t", range=[0.5, p.T + 0.5], title_font=dict(size=10, color=C["dim"]), row=2, col=2)
    fig = style_figure(fig, height=700)
    fig.update_layout(barmode="overlay", showlegend=True,
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.07, font=dict(size=11)))
    return fig


# --------------------------------------------------------------------- value iteration (0002)

def _nums(z):
    return [float(x) if np.isfinite(x) else None for x in np.ravel(z)]


def _wire(z):
    """Wireframe of a grid z[e, s] as one line trace: a line along s for each e, then along e for each s.
    None separates the lines and marks cells where z is undefined."""
    ne, ns = z.shape
    X, Y, Z = [], [], []
    for e in range(ne):
        X += list(range(ns)) + [None]
        Y += [e] * ns + [None]
        Z += _nums(z[e]) + [None]
    for s in range(ns):
        X += [s] * ne + [None]
        Y += list(range(ne)) + [None]
        Z += _nums(z[:, s]) + [None]
    return X, Y, Z


def _empty3d(fig):
    """An invisible 3D point, so a chart with no data yet still draws its 3D box, plus the waiting message."""
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=0, opacity=0), hoverinfo="skip"))
    _waiting(fig, "NO ITERATIONS YET  ▸  PRESS ▶ RUN OR +1 ABOVE")


ICE_AXES = dict(x="s  scoops available", y="e  parents' scoops", hover="(e, s) = (%{y}, %{x})")


def _style3d(fig, title, zlabel, z_range, height, zticks=None, axes=None, shape=(3, 5)):
    ax = dict(backgroundcolor=C["panel"], gridcolor=C["grid"], zerolinecolor=C["line"], linecolor=C["line"],
              color=C["dim"], showbackground=True, title_font=dict(size=10, color=C["dim"]), tickfont=dict(size=9))
    fig.update_layout(
        paper_bgcolor=C["panel"], height=height, showlegend=False,
        font=dict(family=FONT, size=11, color=C["txt"]), margin=dict(l=0, r=0, t=34, b=0),
        hoverlabel=dict(bgcolor="#111", font_family=FONT),
        title=dict(text=title, x=0.02, y=0.98, yanchor="top", font=dict(family=FONT, size=11, color=C["amber"])),
        scene=dict(xaxis=dict(ax, title=(axes or ICE_AXES)["x"], tickvals=list(range(shape[1]))),
                   yaxis=dict(ax, title=(axes or ICE_AXES)["y"], tickvals=list(range(shape[0]))),
                   zaxis=dict(ax, title=zlabel, range=z_range, tickvals=zticks),
                   camera=dict(eye=dict(x=1.2, y=-1.4, z=0.6), center=dict(x=0, y=0, z=-0.12)),
                   domain=dict(x=[0, 1], y=[0, 0.94]),
                   aspectmode="manual", aspectratio=dict(x=1.15, y=0.85, z=0.7)),
    )
    return fig


def surface3d(cur, earlier, color, title, zlabel, z_range, label, best=None, height=430, zticks=None, fmt=",.4f",
              axes=None):
    """3D view of a function of the state (e, s), given as a grid z[e, s] (NaN where undefined).
    `earlier` are grids of earlier iterations, drawn as light dashed wireframes; `cur` is the current iteration,
    drawn as a solid surface with a bold wireframe. `best` marks the states where this action is the arg max.
    Parts outside z_range are clipped, so a narrow range zooms in on the shape of the current iteration."""
    fig = go.Figure()
    if earlier:
        xs, ys, zs = [], [], []
        for z in earlier:
            x, y, w = _wire(z)
            xs += x
            ys += y
            zs += w
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", hoverinfo="skip",
                                   line=dict(color=_rgba(color, 0.35), width=2, dash="dash")))
    if cur is not None:
        fig.add_trace(go.Surface(x=np.arange(cur.shape[1]), y=np.arange(cur.shape[0]), z=cur, opacity=0.35,
                                 colorscale=[[0, color], [1, color]], showscale=False, hoverinfo="skip"))
        x, y, w = _wire(cur)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=w, mode="lines", hoverinfo="skip", line=dict(color=color, width=6)))
        e, s = np.nonzero(np.isfinite(cur))
        fig.add_trace(go.Scatter3d(x=s, y=e, z=cur[e, s], mode="markers", marker=dict(size=3.5, color=color),
                                   hovertemplate=f"{(axes or ICE_AXES)['hover']}<br>{label} = %{{z:{fmt}}}<extra></extra>"))
        if best is not None:
            e, s = np.nonzero(best & np.isfinite(cur))
            fig.add_trace(go.Scatter3d(x=s, y=e, z=cur[e, s], mode="markers", hoverinfo="skip",
                                       marker=dict(size=7, color=C["txt"], symbol="diamond",
                                                   line=dict(color=color, width=2))))
    else:
        _empty3d(fig)
    shape = cur.shape if cur is not None else ((earlier[0].shape if earlier else (3, 5)))
    return _style3d(fig, title, zlabel, z_range, height, zticks, axes, shape)


def closeness(dv, dq, tol, beta, changes, height=430):
    """Sup-norm distance between consecutive iterates, on a log scale: ||v_{k+1} - v_k|| and ||Q_{k+1} - Q_k||
    (over feasible actions). The dotted line is the contraction rate beta^k from the first distance; the dashed
    line is the tolerance; green verticals mark the iterations at which the greedy policy changed."""
    fig = go.Figure()
    n = len(dv)
    if n:
        it = np.arange(1, n + 1)
        y = np.where(np.asarray(dv) > 0, dv, np.nan)
        if beta < 1:
            fig.add_trace(go.Scatter(x=it, y=dv[0] * beta ** (it - 1), mode="lines", name="β<sup>k</sup> · ‖v<sub>1</sub> − v<sub>0</sub>‖∞",
                                     line=dict(color=C["dim"], width=1, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=it, y=y, mode="lines+markers" if n <= 60 else "lines", name="‖v<sub>k+1</sub> − v<sub>k</sub>‖∞",
                                 line=dict(color=C["amber"], width=2.4), marker=dict(size=4),
                                 hovertemplate="k + 1 = %{x}<br>‖v<sub>k+1</sub> − v<sub>k</sub>‖∞ = %{y:.3e}<extra></extra>"))
        if len(dq):
            q = np.where(np.asarray(dq) > 0, dq, np.nan)
            fig.add_trace(go.Scatter(x=np.arange(2, n + 1), y=q, mode="lines", name="‖Q<sub>k+1</sub> − Q<sub>k</sub>‖∞",
                                     line=dict(color=C["cyan"], width=1.8, dash="dash"),
                                     hovertemplate="k + 1 = %{x}<br>‖Q<sub>k+1</sub> − Q<sub>k</sub>‖∞ = %{y:.3e}<extra></extra>"))
        for c in changes:
            fig.add_vline(x=c, line=dict(color=C["green"], width=1, dash="dot"))
    else:
        _waiting(fig, "NO ITERATIONS YET  ▸  PRESS ▶ RUN OR +1 ABOVE")
    fig.add_hline(y=tol, line=dict(color=C["red"], width=1.2, dash="dash"))
    fig.update_yaxes(type="log", exponentformat="e", title_text="sup-norm distance (log scale)",
                     title_font=dict(size=10, color=C["dim"]))
    fig.update_xaxes(title_text="iteration k + 1", title_font=dict(size=10, color=C["dim"]))
    fig.update_layout(title=dict(text="CONVERGENCE  ▸  ‖v<sub>k+1</sub> − v<sub>k</sub>‖∞  AND  ‖Q<sub>k+1</sub> − Q<sub>k</sub>‖∞", x=0.02,
                                 font=dict(family=FONT, size=11, color=C["amber"])))
    fig = style_figure(fig, height=height)
    fig.update_layout(margin=dict(t=56), showlegend=True, legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom",
                                                   font=dict(size=10)))
    return fig


def policy_history(pis, colors, labels, height=430):
    """The greedy policy over the iterations in 3D: one step line per state, x = iteration k + 1, y = state,
    z = the action pi_{k+1}(e, s), each segment colored by its action. `pis` is an (iterations, states) array."""
    fig = go.Figure()
    k = len(colors)
    n_states = len(labels)
    if len(pis):
        z = np.asarray(pis, dtype=int).T                    # (states, iterations)
        n = z.shape[1]
        for a in range(k):                                  # one trace per action, so each has a legend entry
            xs, ys, zs = [], [], []
            for i in range(n_states):
                for j in np.nonzero(z[i] == a)[0]:
                    xs += [j + 0.5, j + 1.5, None]
                    ys += [i, i, None]
                    zs += [a, a, None]
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=f"store {a}",
                                       line=dict(color=colors[a], width=7), hoverinfo="skip"))
        it = np.tile(np.arange(1, n + 1), n_states)
        st_ = np.repeat(np.arange(n_states), n)
        fig.add_trace(go.Scatter3d(x=it, y=st_, z=z.ravel(), mode="markers", marker=dict(size=2, color=C["dim"], opacity=0),
                                   customdata=np.repeat(labels, n), showlegend=False,
                                   hovertemplate="k + 1 = %{x}<br>(e, s) = %{customdata}<br>π = %{z}<extra></extra>"))
    else:
        _empty3d(fig)
    ax = dict(backgroundcolor=C["panel"], gridcolor=C["grid"], zerolinecolor=C["line"], linecolor=C["line"],
              color=C["dim"], showbackground=True, title_font=dict(size=10, color=C["dim"]), tickfont=dict(size=9))
    fig.update_layout(
        paper_bgcolor=C["panel"], height=height, showlegend=bool(len(pis)),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="top", font=dict(size=10)),
        font=dict(family=FONT, size=11, color=C["txt"]), margin=dict(l=0, r=0, t=34, b=0),
        hoverlabel=dict(bgcolor="#111", font_family=FONT),
        title=dict(text="POLICY OVER THE ITERATIONS  ▸  π<sub>k+1</sub>(e, s)", x=0.02, y=0.98, yanchor="top",
                   font=dict(family=FONT, size=11, color=C["amber"])),
        scene=dict(xaxis=dict(ax, title="iteration k + 1"),
                   yaxis=dict(ax, title="state (e, s)", tickvals=list(range(0, n_states, 2)),
                              ticktext=labels[::2]),
                   zaxis=dict(ax, title="a", range=[-0.3, k - 0.7], tickvals=list(range(k))),
                   camera=dict(eye=dict(x=1.2, y=-1.4, z=0.65), center=dict(x=0, y=0, z=-0.12)),
                   domain=dict(x=[0, 1], y=[0, 0.9]),
                   aspectmode="manual", aspectratio=dict(x=1.25, y=1.0, z=0.5)),
    )
    return fig


def utility(c_max=4, height=360):
    """u(c) = log(c + 1) on [0, c_max] as a line, with markers and values at the whole scoops c = 0, ..., c_max."""
    c = np.linspace(0, c_max, 200)
    k = np.arange(c_max + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c, y=np.log(c + 1), mode="lines", line=dict(color=C["amber"], width=2.4),
                             hovertemplate="c = %{x:.2f}<br>u(c) = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=k, y=np.log(k + 1), mode="markers+text", text=[f"{v:.4f}" for v in np.log(k + 1)],
                             textposition="top left", textfont=dict(size=10, color=C["txt"]),
                             marker=dict(size=8, color=C["amber"], line=dict(color=C["bg"], width=1)),
                             hovertemplate="c = %{x}<br>u(c) = %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="c  scoops eaten", title_font=dict(size=10, color=C["dim"]), tickvals=list(k),
                     range=[-0.2, c_max + 0.2])
    fig.update_yaxes(title_text="u(c)", title_font=dict(size=10, color=C["dim"]), range=[-0.1, np.log(c_max + 1) + 0.25])
    fig.update_layout(title=dict(text="UTILITY  ▸  u(c) = log(c + 1)", x=0.02,
                                 font=dict(family=FONT, size=11, color=C["amber"])))
    return style_figure(fig, height=height)


def heatmap(z, color, title, z_range=None, fmt=".4f", best=None, discrete=None, height=280, axes=None, labels=None):
    """A function of the state as a heatmap over the grid z[e, s] (x = s, y = e), with the value written in each
    cell and NaN (infeasible) cells shown as −∞. `discrete` is a list of colors, one per integer value (the policy);
    otherwise the cells shade from dark to `color` over z_range. `best` marks with ◆ the cells where this action is
    the arg max. `axes` names the axes of another grid (default: the ice cream grid); `labels` replaces the text of
    each cell (a grid of strings, e.g. the policy arrows)."""
    fig = go.Figure()
    ax = axes or ICE_AXES
    if z is None:
        _waiting(fig, "NO ITERATIONS YET  ▸  PRESS ▶ RUN OR +1 ABOVE")
    else:
        z = np.asarray(z, dtype=float)
        text = [[("−∞" if np.isnan(x) else (f"{x:.0f}" if discrete else f"{x:{fmt}}"))
                 + (" ◆" if best is not None and best[i, j] and not np.isnan(x) else "")
                 for j, x in enumerate(row)] for i, row in enumerate(z)]
        if labels is not None:
            text = labels
        if discrete:
            k = len(discrete)
            scale = [x for a, c in enumerate(discrete) for x in ([a / k, c], [(a + 1) / k, c])]
            lo, hi = -0.5, k - 0.5
        else:
            # capped at 0.55 so the white value text stays readable on the lightest cell, even for a light color
            scale = [[0, _rgba(color, 0.06)], [1, _rgba(color, 0.55)]]
            lo, hi = z_range if z_range else (np.nanmin(z), np.nanmax(z)) if np.isfinite(z).any() else (0, 1)
        fig.add_trace(go.Heatmap(z=z, x=np.arange(z.shape[1]), y=np.arange(z.shape[0]), zmin=lo, zmax=hi,
                                 colorscale=scale, showscale=False, xgap=2, ygap=2, text=text,
                                 texttemplate="%{text}", textfont=dict(size=12, color="#000000" if discrete else "#ffffff"),
                                 hovertemplate=ax["hover"] + "<br>%{text}<extra></extra>"))
    rows, cols = (np.asarray(z).shape if z is not None else (3, 5))
    fig.update_xaxes(title_text=ax["x"], title_font=dict(size=10, color=C["dim"]),
                     tickvals=list(range(cols)), showgrid=False, zeroline=False)
    fig.update_yaxes(title_text=ax["y"], title_font=dict(size=10, color=C["dim"]),
                     tickvals=list(range(rows)) if ax["y"] else [], showgrid=False, zeroline=False)
    fig.update_layout(title=dict(text=title, x=0.02, font=dict(family=FONT, size=11, color=C["amber"])))
    return style_figure(fig, height=height)


def policy_heatmap(pis, colors, labels, height=280):
    """The greedy policy over the iterations as a heatmap: one row per state, one column per iteration k + 1,
    colored by the action pi_{k+1}(e, s). `pis` is an (iterations, states) array of actions."""
    fig = go.Figure()
    if len(pis):
        k = len(colors)
        scale = [x for a, c in enumerate(colors) for x in ([a / k, c], [(a + 1) / k, c])]
        z = np.asarray(pis).T
        fig.add_trace(go.Heatmap(
            z=z, x=np.arange(1, z.shape[1] + 1), y=labels, zmin=-0.5, zmax=k - 0.5, colorscale=scale,
            xgap=1 if z.shape[1] <= 60 else 0, ygap=1, showscale=False,
            hovertemplate="k + 1 = %{x}<br>(e, s) = %{y}<br>π = %{z}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
    else:
        _waiting(fig, "NO ITERATIONS YET  ▸  PRESS ▶ RUN OR +1 ABOVE")
    fig.update_xaxes(title_text="iteration k + 1", title_font=dict(size=10, color=C["dim"]))
    fig.update_yaxes(title_text="state (e, s)", title_font=dict(size=10, color=C["dim"]), tickfont=dict(size=9))
    fig.update_layout(title=dict(text="POLICY OVER THE ITERATIONS  ▸  HEATMAP", x=0.02,
                                 font=dict(family=FONT, size=11, color=C["amber"])))
    fig = style_figure(fig, height=height)
    fig.update_layout(margin=dict(l=70, t=40))
    return fig




def deploy_series(df, n_total, series, height=520):
    """Deployment of a policy as stacked time series, drawn up to the current step: `df` holds the steps so far,
    `series` is a list of (column, label, color); the x axis covers all n_total steps so the chart fills in."""
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for r, (col, label, color) in enumerate(series, start=1):
        if len(df):
            fig.add_trace(go.Scatter(x=df["t"], y=df[col], mode="lines+markers", line=dict(color=color, width=2,
                                     shape="hv"), marker=dict(size=6, color=color), name=label,
                                     hovertemplate=f"t = %{{x}}<br>{label} = %{{y}}<extra></extra>"), row=r, col=1)
            last = df.iloc[-1]
            fig.add_trace(go.Scatter(x=[last["t"]], y=[last[col]], mode="markers", hoverinfo="skip",
                                     marker=dict(size=12, color=color, line=dict(color=C["amber"], width=2))),
                          row=r, col=1)
        fig.update_yaxes(title_text=label, title_font=dict(size=10, color=C["dim"]), row=r, col=1, dtick=1)
    fig.update_xaxes(range=[0.5, n_total + 0.5], row=len(series), col=1, title_text="day t",
                     title_font=dict(size=10, color=C["dim"]))
    if not len(df):
        _waiting(fig, "PRESS ▶ PLAY OR +1")
    return style_figure(fig, height=height)
