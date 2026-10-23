"""Precomputed runs that the demos replay.

A demo reads only these data files, never the code that produced them (manual/<env>_<agent>/ in the repo), so it
can be shared without that code. In the course repo the files live outside the repo, in
repos/output/rl-exercises/manual/<env>_<agent>/; a deployed copy (made by streamlit_deploy.py) carries them in its own
data/ folder. The environment variable RL_DATA_DIR overrides both.
"""

import json
import os

import numpy as np
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DATA = os.path.normpath(os.path.join(HERE, "..", "data"))   # the data folder of a deployed copy (streamlit_deploy.py)
REPO_DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "output", "rl-exercises", "manual"))
DATA_ROOT = os.environ.get("RL_DATA_DIR", APP_DATA if os.path.isdir(APP_DATA) else REPO_DATA)


def path(code_base, file):
    return os.path.join(DATA_ROOT, code_base, file)


@st.cache_data(show_spinner=False)
def load(code_base, file):
    """The arrays of one run as a dict, with its settings under "meta"; None if the file does not exist."""
    p = path(code_base, file)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        run = {k: z[k] for k in z.files if k != "meta"}
        run["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return run


@st.cache_data(show_spinner=False)
def load_csv(code_base, file):
    """A CSV file of a code base's output (for example the deployment episodes) as a DataFrame; None if missing."""
    import pandas as pd
    p = path(code_base, file)
    return pd.read_csv(p) if os.path.exists(p) else None
