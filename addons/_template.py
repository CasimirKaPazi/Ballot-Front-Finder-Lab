"""<Name> (<abbreviation>), an add-on method: what it does, in a few sentences.

Copy this file to addons/<something>.py (files starting with _ are not loaded), fill it in, and the
method is available everywhere by its name (run.py, sweep.py, analyze.py, groups). Delete the file
to remove it. To share a method, share the file.

What a method is (see methods.py, Method):
    Method(ballot, decide, feats)
      ballot  the ballot type: "fav", "rank", "approval", "score", "fav_run", "app_run", "veto",
              "weak", "trunc", "trunc3", "grade3" (methods.py docstring). The ballot type decides
              the sincere ballots, laziness, the strategic moves and the measures; reuse one.
      feats   names of per-voter ballot features the decision needs; they are summed over voters
              (and over poll samples), which keeps tallies fast. Built-in ones are in
              methods.FEATURES ("sum", "gt" = pairwise wins, "first", "firsts", ...); new ones go
              into FEATURES below.
      decide  decide(s) -> winner probabilities (length C), from s = {feature name: summed array}.
              Ties: split the win, so results are exact lotteries.
    Method(ballot, full=tally) instead, for rules that need the whole ballot matrix B (n x C):
      tally(B) -> winner probabilities. Slower: no feature sums.
Score ranges: read methods.MAXS when called (it can change with --max-score), not at import.
"""
import numpy as np
import methods
from methods import Method


def my_feature(B):
    """Per voter (rows of the ballot matrix B, n x C): any array; it is reshaped to (n, -1) and summed."""
    return B


def decide(s):
    tot = s["my_feature"]                                   # summed over voters: shape (C,)
    win = np.flatnonzero(tot >= tot.max() - methods.EPS)
    P = np.zeros(len(tot))
    P[win] = 1 / len(win)
    return P


FEATURES = {"my_feature": (my_feature, lambda C: (C,))}   # name: (per-voter function, shape of the sum given C)
METHODS = {"My method": Method("score", decide, ("my_feature",))}
GROUP = "addons"                                           # an existing group (e.g. "rated") or a new one


def test():
    """Optional checks, run by test_methods.py: a small profile with a known winner."""
    B = np.array([[5, 3, 0], [0, 5, 3], [3, 5, 0]], dtype=float)
    assert np.allclose(METHODS["My method"].tally(B), [0, 1, 0])
