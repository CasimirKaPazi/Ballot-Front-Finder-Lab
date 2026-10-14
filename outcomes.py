"""Outcomes and what they are worth.

Every method returns its outcome as winner probabilities P: one number per candidate, summing to 1
(a lottery; ties are split). What an outcome is worth to someone who values the candidates at v
(a voter's utilities, a group's summed utilities, the electorate's total utility, each candidate's
worst margin, ...) is its expected value: value(P, v).

All code that judges an outcome goes through value():
    strategy.py      what a group gains from a ballot, over the polls
    metrics.py       VSE and WME (total utility and worst margin of the winner, and of random ballot)
    nomination.py    what a candidate gains from exiting, re-entering or fielding a clone
    participation.py what absent voters gain (no-show paradox)
A different kind of outcome (e.g. a committee, for proportional methods) needs a different value()
and is the place to start, together with the methods that produce it and the measures in metrics.py.
"""
import numpy as np


def value(P, v):
    """Expected value of the outcome P for the values v.
    P: (C,) one outcome, or (k, C) several (e.g. one per poll); also a difference of two outcomes.
    v: (C,) one evaluator, or (n, C) n evaluators (one row each).
    Returns P @ v: a number, or (k,); with n evaluators (n,) or (k, n)."""
    v = np.asarray(v)
    return P @ (v.T if v.ndim == 2 else v)
