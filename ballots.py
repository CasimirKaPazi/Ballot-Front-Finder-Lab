"""Ballot types: everything about a ballot format in one place.

A ballot matrix B has one row per voter and one column per candidate. Each ballot type is one
Ballot entry in BALLOTS (name -> Ballot), and a method names its type by that name
(Method("score", ...)). The entry says:
    sincere   the honest ballots from utilities: function (u, lazy) -> B (lazy: laziness 0..1;
              types without laziness ignore it)
    lazy      True if laziness (--lazy) applies to this type
    mutate    one small random change of a single ballot: function (b, rng) -> ballot (the random
              search of the strategy models)
    exaggerate  the naive exaggerations a group tries around the two poll leaders: function
              (Ug, A, Bo) -> list of ballots (Ug: the group's summed utilities, A: the leader it
              prefers, Bo: the other; strategy.exaggerations)
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


# ---------- strategic changes: one small random change to a single ballot ----------
# Used by the random search of the strategy models (strategy.py). Each takes a ballot vector b
# (length C, a copy the function may change) and the random generator, and returns the changed ballot.
def _mutate_fav(b, rng):
    C = len(b)
    cur = int(b.argmax())
    new = (cur + 1 + rng.integers(C - 1)) % C
    b[:] = 0
    b[new] = 1
    return b


def _mutate_strict(b, rng):
    i, j = rng.choice(len(b), 2, replace=False)
    b[i], b[j] = b[j], b[i]
    return b


def _mutate_rank(b, rng):
    """Truncatable: swap two places, or rank one more / fewer."""
    C = len(b)
    ranked = [int(c) for c in np.argsort(-b, kind="stable") if b[c] > 0]
    k = len(ranked)                          # a full ranking has C-1 candidates above 0
    order = ranked + list(rng.permutation([c for c in range(C) if b[c] <= 0]))
    if rng.random() < 0.5:
        i, j = rng.choice(C, 2, replace=False)
        order[i], order[j] = order[j], order[i]
    else:
        k = min(C - 1, max(1, k + (1 if rng.random() < 0.5 else -1)))
    b = np.zeros(C)
    for pos, c in enumerate(order[:k]):
        b[c] = C - 1 - pos
    return b


def _mutate_approval(b, rng):
    C = len(b)
    if C == 2:              # the only valid ballots are A or B: no single flip stays valid, so swap
        return b[::-1].copy()
    while True:
        c = rng.integers(C)
        b[c] = 1 - b[c]
        if 0 < b.sum() < C:
            break
        b[c] = 1 - b[c]
    return b


def _mutate_fav_run(b, rng):
    """Only the round-1 mark changes; the runoff ranking stays."""
    C = len(b)
    pts, mark = np.mod(b, C), b >= C
    cur = int(mark.argmax())
    mark[:] = False
    mark[(cur + 1 + rng.integers(C - 1)) % C] = True
    return pts + C * mark


def _mutate_app_run(b, rng):
    """Only the round-1 approvals change; the runoff ranking stays."""
    C = len(b)
    pts, mark = np.mod(b, C), b >= C
    if C == 2:
        mark = mark[::-1].copy()
    else:
        while True:
            c = rng.integers(C)
            mark[c] = not mark[c]
            if 0 < mark.sum() < C:
                break
            mark[c] = not mark[c]
    return pts + C * mark


def _mutate_veto(b, rng):
    C = len(b)
    cur = int(b.argmin())
    new = (cur + 1 + rng.integers(C - 1)) % C
    b[:] = 0
    b[new] = -1
    return b


def _mutate_weak(b, rng):
    C = len(b)
    c = rng.integers(C)
    b[c] = (b[c] + 1 + rng.integers(C - 1)) % C          # any other level; equal levels are ties
    return b


def _mutate_truncated(b, rng, kmax):
    """trunc / trunc3: swap two places, or rank one more / fewer (at most kmax ranked)."""
    C = len(b)
    ranked = [int(c) for c in np.argsort(-b, kind="stable") if b[c] > 0]
    order = ranked + list(rng.permutation([c for c in range(C) if b[c] == 0]))
    k = len(ranked)
    if rng.random() < 0.5:
        i, j = rng.choice(C, 2, replace=False)
        order[i], order[j] = order[j], order[i]
    else:
        k = min(kmax, max(1, k + (1 if rng.random() < 0.5 else -1)))
    b = np.zeros(C)
    for pos, c in enumerate(order[:k]):
        b[c] = k - pos
    return b


def _mutate_trunc(b, rng):
    return _mutate_truncated(b, rng, len(b) - 1)


def _mutate_trunc3(b, rng):
    return _mutate_truncated(b, rng, min(methods.TRUNC3, len(b)))


def _mutate_score(b, rng):
    """Any other level, uniformly. For ranges above 0..5, half of the proposals instead use the six
    levels of the 0..5 scale stretched to the range (0, 20, 40, ... 100 for 0..100), so the chance
    of trying an extreme or a coarse step does not shrink with the range and learning is not slowed
    artificially. 0..5 and below: unchanged."""
    S = methods.MAXS
    c = rng.integers(len(b))
    if S > 5 and rng.random() < 0.5:
        levels = [x for x in np.rint(np.arange(6) * S / 5) if x != b[c]]
        b[c] = levels[rng.integers(len(levels))]
    else:
        b[c] = (b[c] + 1 + rng.integers(S)) % (S + 1)
    return b


def _mutate_grade3(b, rng):
    c = rng.integers(len(b))
    b[c] = (b[c] + 1 + rng.integers(2)) % 3
    return b


# ---------- naive exaggerations around the two poll leaders ----------
# Used by the poll-round strategy (strategy.exaggerations). Each takes Ug (a group's summed
# utilities), A (the leader the group prefers) and Bo (the other leader), and returns a list of
# ballots to try, most complete first (may be empty).
def _rank_points(o):
    """Rank points C-1 ... 0 for the candidates in order o."""
    r = np.empty(len(o))
    r[o] = np.arange(len(o) - 1, -1, -1)
    return r


def _views(Ug, A, Bo):
    """What the exaggerations are built from: the group's sincere order, who is liked at least as much
    as A (up) and how many (k), and three reorderings: A to the top (comp), Bo to the bottom (bury), both."""
    order = [int(c) for c in np.argsort(-Ug, kind="stable")]      # the group's sincere ranking
    up = Ug >= Ug[A]                                                 # liked at least as much as A
    comp = [A] + [c for c in order if c != A]                       # A moved to the top
    bury = [c for c in order if c != Bo] + [Bo]
    both = [A] + [c for c in bury if c != A]
    return order, up, int(up.sum()), comp, bury, both


def _truncated(order, comp, k):
    """Truncated rankings (rank points C-1 ... for the ranked, 0 for the rest): the sincere order
    down to A (if that leaves two or more unranked), and a bullet vote for A."""
    C = len(order)
    res = []
    for o, kk in ((order, k), (comp, 1)):
        if kk < C - 1:
            t = np.zeros(C)
            t[o[:kk]] = np.arange(C - 1, C - 1 - kk, -1)
            res.append(t)
    return res


def _exag_fav(Ug, A, Bo):
    return [np.eye(len(Ug))[A]]


def _exag_approval(Ug, A, Bo):
    order, up, k, comp, bury, both = _views(Ug, A, Bo)
    return [up.astype(float)] if 0 < k < len(Ug) else []


def _exag_levels(Ug, A, Bo, S):
    """Scores / grades 0..S: top at A and above, 0 at Bo and below, stretched in between; and top for
    everyone liked at least as much as A, 0 for the rest."""
    order, up, k, comp, bury, both = _views(Ug, A, Bo)
    lo = Ug[Bo]
    x = np.clip((Ug - lo) / max(Ug[A] - lo, 1e-12), 0, 1)
    return [np.rint(S * x), S * up.astype(float)]


def _exag_score(Ug, A, Bo):
    return _exag_levels(Ug, A, Bo, methods.MAXS)


def _exag_grade3(Ug, A, Bo):
    return _exag_levels(Ug, A, Bo, 2)


def _exag_strict(Ug, A, Bo):
    order, up, k, comp, bury, both = _views(Ug, A, Bo)
    return [_rank_points(both), _rank_points(comp), _rank_points(bury)]


def _exag_rank(Ug, A, Bo):
    """As strict, and also truncated: rank only down to A, or A alone."""
    order, up, k, comp, bury, both = _views(Ug, A, Bo)
    return [_rank_points(both), _rank_points(comp), _rank_points(bury)] + _truncated(order, comp, k)


def _exag_weak(Ug, A, Bo):
    """Levels, equal = tied: A tied with everyone above it at the top (and Bo alone at the bottom)."""
    C = len(Ug)
    order, up, k, comp, bury, both = _views(Ug, A, Bo)
    tied = np.empty(C)
    tied[up] = C - 1
    rest = [c for c in order if not up[c]]
    tied[rest] = np.arange(C - 2, C - 2 - len(rest), -1)
    tb = tied.copy()
    if not up[Bo]:
        tb[Bo] = tb[rest].min() - 1 if len(rest) > 1 else tb[Bo]
    return [tb - tb.min(), tied - tied.min(), _rank_points(both), _rank_points(bury)] + _truncated(order, comp, k)


def _exag_trunc(Ug, A, Bo):
    """Rank only down to A."""
    C = len(Ug)
    order, up, k, comp, bury, both = _views(Ug, A, Bo)
    if not 0 < k < C:
        return []
    t1 = np.zeros(C); t1[[c for c in order if up[c]]] = np.arange(k, 0, -1)
    t2 = np.zeros(C); t2[[c for c in comp[:k]]] = np.arange(k, 0, -1)
    return [t1, t2]


def _exag_trunc3(Ug, A, Bo):
    """At most TRUNC3 ranked: A first and Bo never ranked; or rank only down to A."""
    C = len(Ug)
    order, up, k, comp, bury, both = _views(Ug, A, Bo)
    K = min(methods.TRUNC3, C)
    t1 = [c for c in order if up[c]][:K]
    t2 = [A] + [c for c in order if c not in (A, Bo)][:K - 1]
    out = []
    for o in (t2, t1):
        t = np.zeros(C); t[o] = np.arange(len(o), 0, -1)
        out.append(t)
    return out


def _exag_fav_run(Ug, A, Bo):
    """Mark A; the runoff ranking stays sincere."""
    C = len(Ug)
    order = _views(Ug, A, Bo)[0]
    return [_rank_points(order) + C * np.eye(C)[A]]


def _exag_app_run(Ug, A, Bo):
    """Approve down to A; the runoff ranking stays sincere."""
    C = len(Ug)
    order, up, k, comp, bury, both = _views(Ug, A, Bo)
    return [_rank_points(order) + C * up] if 0 < k < C else []


def _exag_veto(Ug, A, Bo):
    v = np.zeros(len(Ug)); v[Bo] = -1
    return [v]


# ---------- the ballot types ----------
@dataclass
class Ballot:
    """One ballot format (see the module docstring)."""
    name: str
    sincere: Callable           # (u, lazy) -> ballot matrix
    mutate: Callable            # (b, rng) -> one small random change of ballot vector b (strategy search)
    exaggerate: Callable        # (Ug, A, Bo) -> list of naive exaggerations around the poll leaders
    lazy: bool = False          # laziness applies
    ordered: bool = False       # records an order of the candidates


BALLOTS = {b.name: b for b in [
    Ballot("fav", fav_ballots, _mutate_fav, _exag_fav),
    Ballot("rank", lazy_rank_ballots, _mutate_rank, _exag_rank, lazy=True, ordered=True),
    Ballot("strict", rank_ballots, _mutate_strict, _exag_strict, ordered=True),
    Ballot("approval", approval_ballots, _mutate_approval, _exag_approval, lazy=True),
    Ballot("score", score_ballots, _mutate_score, _exag_score, lazy=True, ordered=True),
    Ballot("fav_run", fav_run_ballots, _mutate_fav_run, _exag_fav_run),
    Ballot("app_run", app_run_ballots, _mutate_app_run, _exag_app_run, lazy=True),
    Ballot("veto", veto_ballots, _mutate_veto, _exag_veto),
    Ballot("weak", lazy_rank_ballots, _mutate_weak, _exag_weak, lazy=True, ordered=True),   # honest: strict
    Ballot("trunc", trunc_ballots, _mutate_trunc, _exag_trunc, lazy=True, ordered=True),
    Ballot("trunc3", trunc3_ballots, _mutate_trunc3, _exag_trunc3, lazy=True, ordered=True),
    Ballot("grade3", grade3_ballots, _mutate_grade3, _exag_grade3, lazy=True, ordered=True),
]}


# At the end: methods.py imports from this file, and the functions above read its settings (MAXS,
# LAZY, TRUNC3) only when called, so importing either file first works.
import methods  # noqa: E402
