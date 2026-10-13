"""Ballot types: everything about a ballot format in one place.

A ballot matrix B has one row per voter and one column per candidate. Each ballot type is one
Ballot entry in BALLOTS (name -> Ballot), and a method names its type by that name
(Method("score", ...)). The entry says:
    sincere   the honest ballots from utilities: function (u, lazy) -> B (lazy: laziness 0..1;
              types without laziness ignore it)
    lazy      True if laziness (--lazy) applies to this type
    ordered   True if the ballot records an order of the candidates (used for the Condorcet winner
              in the ballots, metrics.ballot_condorcet)

The encodings (see also methods.py):
  "fav"      one-hot: 1 for the named candidate
  "rank"     rank points: best C-1 ... worst 0. Truncated rankings: the k ranked candidates get
             C-1 ... C-k, the unranked ones 0 (tied last; a full ranking is k = C-1)
  "strict"   as rank, but always a full ranking (the Inverse-* methods on rankings)
  "approval" 0/1
  "score"    0..MAXS (default 0..5; methods.set_max_score changes the range)
  "fav_run"  two-round election, plurality first round: the sincere ranking as rank points
             (C-1 ... 0) for the separate runoff, plus C for the one candidate marked in round 1
  "app_run"  two-round election, approval first round: as fav_run, plus C for every approved candidate
  "veto"     -1 for the one candidate voted against, 0 for all others
  "weak"     ranking with ties allowed: levels 0 ... C-1, higher is better, equal = tied
  "trunc"    truncated ranking: the k ranked candidates get k ... 1, unranked ones 0 (tied last)
  "trunc3"   as trunc, at most TRUNC3 (3) ranked (IRV-3)
  "grade3"   three grades: 2 = good, 1 = OK, 0 = bad

The settings these depend on (score range MAXS, laziness LAZY, TRUNC3) are still kept in methods.py
and read from there when a function is called.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


# ---------- laziness ----------
def cares(u, lazy):
    """Per voter and candidate: True if a lazy voter (laziness `lazy`) ranks the candidate."""
    mn, mx = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    p = (u - mn) / np.where(mx - mn > 1e-12, mx - mn, 1)
    return (p >= lazy - 1e-12) | (u >= mx)


def truncate(R, keep):
    """Rank points R (C-1 ... 0, strict) truncated to the candidates in `keep` (a top segment of the
    ranking): those keep their points C-1 ... C-k, the others get 0 (unranked, tied last)."""
    C = R.shape[1]
    k = keep.sum(axis=1, keepdims=True)
    full = k >= C - 1                                        # ranking all but one = the full ranking
    return np.where(keep | full, R, 0.0)


# ---------- sincere ballots ----------
def fav_ballots(u, lazy=0.0):
    """Mark the favourite."""
    B = np.zeros_like(u)
    B[np.arange(u.shape[0]), u.argmax(axis=1)] = 1
    return B


def rank_ballots(u, lazy=0.0):
    """Order by utility, always complete."""
    order = np.argsort(-u, axis=1, kind="stable")
    B = np.empty_like(u)
    C = u.shape[1]
    B[np.arange(u.shape[0])[:, None], order] = np.arange(C - 1, -1, -1)
    return B


def approval_ballots(u, lazy=0.0):
    """Every candidate above the voter's mean utility (at least the favourite); with laziness only
    those the voter also cares about (cares): at 1 a bullet vote for the favourite."""
    a = u > u.mean(axis=1, keepdims=True) + 1e-12
    if lazy > 0:
        a &= cares(u, lazy)
    none = ~a.any(axis=1)
    a[none, u[none].argmax(axis=1)] = True
    return a.astype(float)


def score_ballots(u, lazy=0.0):
    """0..MAXS stretched to the voter's own utility range, rounded; with laziness the candidates the
    voter does not care about (cares) are left blank, which counts as 0: at 1 only the favourite
    gets a score."""
    mn, mx = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    d = np.where(mx - mn > 1e-12, mx - mn, 1.0)
    B = np.rint(methods.MAXS * (u - mn) / d)
    return np.where(cares(u, lazy), B, 0.0) if lazy > 0 else B


def fav_run_ballots(u, lazy=0.0):
    """Mark the favourite; ranking by utility."""
    return rank_ballots(u) + u.shape[1] * fav_ballots(u)


def app_run_ballots(u, lazy=0.0):
    """Approve every candidate above the voter's mean utility; ranking by utility. Laziness: the
    round-1 approvals only."""
    return rank_ballots(u) + u.shape[1] * approval_ballots(u, lazy)


def veto_ballots(u, lazy=0.0):
    """Against the least liked candidate."""
    B = np.zeros_like(u)
    B[np.arange(u.shape[0]), u.argmin(axis=1)] = -1
    return B


def lazy_rank_ballots(u, lazy=None):
    """Honest rankings with laziness: the candidates the voter cares about (cares) keep their rank
    points, the rest are unranked (0). lazy=None: the current laziness (methods.LAZY)."""
    lazy = methods.LAZY if lazy is None else lazy
    R = rank_ballots(u)
    return R if lazy <= 0 else truncate(R, cares(u, lazy))


def trunc_ballots(u, lazy=None):
    """Rank every candidate above the voter's mean utility, leave the rest unranked; with laziness
    only those of them the voter also cares about."""
    lazy = methods.LAZY if lazy is None else lazy
    R, A = rank_ballots(u), approval_ballots(u)
    if lazy > 0:
        A = A * cares(u, lazy)
    k = A.sum(axis=1, keepdims=True)
    return np.where(A > 0, R - (u.shape[1] - k) + 1, 0.0)   # top k get k ... 1


def trunc3_ballots(u, lazy=None):
    """Rank the TRUNC3 (3) best-liked candidates (all, with 3 or fewer), leave the rest unranked;
    with laziness only those of them the voter also cares about."""
    lazy = methods.LAZY if lazy is None else lazy
    C = u.shape[1]
    k = min(methods.TRUNC3, C)
    R = rank_ballots(u)
    keep = R >= C - k
    if lazy > 0:
        keep = keep & cares(u, lazy)
        kk = keep.sum(axis=1, keepdims=True)
        return np.where(keep, R - (C - kk) + 1, 0.0)            # the kept top kk get kk ... 1
    return np.where(keep, R - (C - k) + 1, 0.0)              # top k get k ... 1


def grade3_ballots(u, lazy=0.0):
    """0..2, stretched to the voter's own utility range, rounded (as score, with 3 levels)."""
    mn, mx = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    d = np.where(mx - mn > 1e-12, mx - mn, 1.0)
    B = np.rint(2 * (u - mn) / d)
    return np.where(cares(u, lazy), B, 0.0) if lazy > 0 else B         # blank = bad (0), as for scores


# ---------- the ballot types ----------
@dataclass
class Ballot:
    """One ballot format (see the module docstring)."""
    name: str
    sincere: Callable           # (u, lazy) -> ballot matrix
    lazy: bool = False          # laziness applies
    ordered: bool = False       # records an order of the candidates


BALLOTS = {b.name: b for b in [
    Ballot("fav", fav_ballots),
    Ballot("rank", lazy_rank_ballots, lazy=True, ordered=True),
    Ballot("strict", rank_ballots, ordered=True),
    Ballot("approval", approval_ballots, lazy=True),
    Ballot("score", score_ballots, lazy=True, ordered=True),
    Ballot("fav_run", fav_run_ballots),
    Ballot("app_run", app_run_ballots, lazy=True),
    Ballot("veto", veto_ballots),
    Ballot("weak", lazy_rank_ballots, lazy=True, ordered=True),     # strict when honest; ties only strategic
    Ballot("trunc", trunc_ballots, lazy=True, ordered=True),
    Ballot("trunc3", trunc3_ballots, lazy=True, ordered=True),
    Ballot("grade3", grade3_ballots, lazy=True, ordered=True),
]}


# At the end: methods.py imports from this file, and the functions above read its settings (MAXS,
# LAZY, TRUNC3) only when called, so importing either file first works.
import methods  # noqa: E402
