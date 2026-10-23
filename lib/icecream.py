"""Labels and grid layout of the ice cream scoop states, for the demo page. It holds no model and no solver: the
transition probabilities, rewards, and every value iteration iterate come from the precomputed run of
manual/ice-cream-scoop_value-iteration.

State (e, s): the scoops the parents added last night, e in {0, 1, 2}, and the scoops available this morning,
s in {0, ..., 4}. State i = 5 e + s, which is "State i + 1" in the lab. Action a in {0, 1, 2}: the scoops stored.
"""

import numpy as np

E = np.arange(3)            # parents' scoops e
SC = np.arange(5)           # scoops available s
A = np.arange(3)            # scoops stored a
NE, NS, K = E.size, SC.size, A.size
N = NE * NS
STATES = [(e, s) for e in E for s in SC]


def idx(e, s):
    return NS * e + s


def grid(x):
    """A vector over the 15 states as a (3, 5) grid z[e, s], with NaN where it is -inf (infeasible)."""
    z = np.array(x, dtype=float).reshape(NE, NS)
    z[~np.isfinite(z)] = np.nan
    return z
