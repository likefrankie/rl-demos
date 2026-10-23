"""The list of demos: the one place that says which demos exist, how the menu groups them (environment, then
method), what the landing page writes about them, which data each one reads, and which are hidden.

Home.py builds the menu from it, demo_pages/0000_All_Demos.py the cards, and streamlit_deploy.py the copy that is
deployed to Streamlit Community Cloud (only the shown demos and their data).
"""

# Each demo:
#   env, kind     the menu group (first level) and the subtitle of its section on the landing page
#   method, url   the menu entry (second level) and its URL path
#   page          the page file
#   code_base     the manual/<code_base> whose output the demo reads (None if it reads no data files)
#   data          the files of that output it reads, relative to repos/output/rl-exercises/manual/<code_base>/
#   lab, text     the landing page card
#   hidden        True: the page file stays, but it is left out of the menu, the landing page, and the deployment
VI_DATA = ["run/value_iteration_run.npz", "deployment/policy_rollout.csv"]
GRID_DATA = VI_DATA + ["deployment/frames"]

DEMOS = [
    dict(env="Ice Cream Scoop", kind="MARKOV DECISION PROCESS", method="Value Iteration", url="Value_Iteration",
         page="demo_pages/0002_Value_Iteration.py", code_base="ice-cream-scoop_value-iteration", data=VI_DATA,
         lab="0002-01-02 lab exercise 11",
         text="Watch every Bellman backup: the terms p × [r + β v_k] of one state, the Q table with the policy "
              "(arg max) and value (max), v and Q in 3D, convergence, and the optimal policy deployed day by day."),
    dict(env="1D Gridworld", kind="MARKOV DECISION PROCESS", method="Value Iteration",
         url="Gridworld_1D_Value_Iteration", page="demo_pages/0002_Gridworld1D_Value_Iteration.py",
         code_base="gridworld1D_value-iteration", data=GRID_DATA, lab="0002-01-02 roover 1D gridworld labs",
         text="A robot on a 21-segment path with a hole and a prize: every backup, the Q table, v and π along the "
              "segment, and episodes of the optimal policy."),
    dict(env="2D Gridworld", kind="MARKOV DECISION PROCESS", method="Value Iteration",
         url="Gridworld_2D_Value_Iteration", page="demo_pages/0002_Gridworld2D_Value_Iteration.py",
         code_base="gridworld2D_value-iteration", data=GRID_DATA, lab="0002-01-02 roover 2D gridworld labs",
         text="A robot in a 4 × 4 grid with slips, blocked cells, a cheese, and a trap: every backup, Q and v on the "
              "grid, the policy arrows, and episodes of the optimal policy."),
    dict(env="3D Gridworld", kind="MARKOV DECISION PROCESS", method="Value Iteration",
         url="Gridworld_3D_Value_Iteration", page="demo_pages/0002_Gridworld3D_Value_Iteration.py",
         code_base="gridworld3D_value-iteration", data=GRID_DATA, lab="0002-01-02 roover 3D gridworld labs",
         text="The 2D gridworld through four layers: every backup, Q and v layer by layer, the policy arrows, and "
              "episodes of the optimal policy in 3D."),
    dict(env="Optimal Execution", kind="LINEAR QUADRATIC OPTIMAL CONTROL", method="Linear Quadratic Dynamic Programming",
         url="Optimal_Execution", page="demo_pages/0003_Optimal_Execution.py", code_base=None, data=[],
         lab="0003-01-02 optimal execution lab", hidden=True,
         text="Buy S̄ shares over T periods with price impact and an information variable. Compare the optimal policy "
              "with an equal split and buying all at once, run after run."),
]

LANDING = "demo_pages/0000_All_Demos.py"


def shown_demos():
    return [d for d in DEMOS if not d.get("hidden")]


def by_env(demos=None):
    """The demos grouped by environment, in the order of DEMOS."""
    groups = {}
    for d in demos if demos is not None else shown_demos():
        groups.setdefault(d["env"], []).append(d)
    return groups


def shown(path):
    return any(d["page"] == path for d in shown_demos()) or path == LANDING
