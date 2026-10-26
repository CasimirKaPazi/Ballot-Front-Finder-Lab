"""Ballot types: everything about a ballot format in one place.

A ballot matrix B has one row per voter and one column per candidate. Each ballot type is one
Ballot entry in BALLOTS (name -> Ballot), and a method names its type by that name
(Method("score", ...)). The entry says:
    sincere   the honest ballots from utilities: function (u, lazy) -> B (lazy: laziness 0..1;
              types without laziness ignore it)
    lazy      True if laziness (--lazy) applies to this type
    approval  True if the honest ballot uses the approval threshold (approves): then sincere is
              (u, lazy, draw) -> B, with each voter's approval draw (methods.VOTER_APPROVAL)
    mutate    one small random change of a single ballot: function (b, rng) -> ballot (the random
              search of the strategy models)
    exaggerate  the naive exaggerations a group tries around the two poll leaders: function
              (Ug, A, Bo) -> list of ballots (Ug: the group's summed utilities, A: the leader it
              prefers, Bo: the other; strategy.exaggerations)
    implied   what the ballots say about the voters' utilities, on 0..1: function (X) -> values
              (fidelity; X after marks)
    worst     per voter, the reversal mass of the most dishonest ballot the format allows: function
              (P) -> values (the scale of sincerity; P the voters' preferences)
    marks     optional: the part of the ballot the voter chooses, function (X) -> X (two-round
              ballots: the round-1 marks; the runoff ranking is fixed). None: the whole ballot
    ordered   True if the ballot records an order of the candidates (used for the Condorcet winner
              in the ballots, metrics.ballot_condorcet)

A new ballot type (in ballots.py, or in an add-on file as BALLOTS = {name: Ballot(...)}) needs all
of these; see the built-in ones below.

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
  "score_run" two-round election, score first round: the sincere ranking as rank points (C-1 ... 0)
             for the separate runoff, plus C x the round-1 score (0..MAXS)
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
    """Per voter and candidate: True if a lazy voter ranks the candidate. lazy: one cut for every
    voter, or one per voter (an array with one value per row of u)."""
    lazy = np.asarray(lazy, float)
    if lazy.ndim == 1:
        lazy = lazy[:, None]
    mn, mx = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    p = (u - mn) / np.where(mx - mn > 1e-12, mx - mn, 1)
    return (p >= lazy - 1e-12) | (u >= mx)


def lazy_on(lazy):
    """Does laziness `lazy` (one cut or one per voter) leave out anything?"""
    return bool(np.any(np.asarray(lazy) > 0))


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


def approves(u, draw):
    """The honest approval set, per voter and candidate. Each voter has a personal threshold between
    their mean utility and their favourite, at the fraction `draw` (0..1, one per voter or one for all)
    of the way: on utilities stretched to 0..1, mean + draw x (1 - mean). With draws uniform on 0..1, a
    candidate's chance of approval rises linearly from 0 at the voter's mean to 1 at the favourite (a
    gradient, not one cut for everyone); fitted to real approval elections (docs/model.md, Approval).
    The favourite is always approved."""
    mn, mx = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    p = (u - mn) / np.where(mx - mn > 1e-12, mx - mn, 1)
    mu = p.mean(axis=1, keepdims=True)
    d = np.asarray(draw, float)
    d = d[:, None] if d.ndim == 1 else d
    a = p >= mu + d * (1 - mu) - 1e-12
    a[np.arange(len(u)), u.argmax(axis=1)] = True
    return a


def approval_ballots(u, lazy=0.0, draw=0.5):
    """The approval set (approves); with laziness only those the voter also cares about (cares):
    at 1 a bullet vote for the favourite."""
    a = approves(u, draw)
    if lazy_on(lazy):
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
    return np.where(cares(u, lazy), B, 0.0) if lazy_on(lazy) else B


def fav_run_ballots(u, lazy=0.0):
    """Mark the favourite; ranking by utility."""
    return rank_ballots(u) + u.shape[1] * fav_ballots(u)


def app_run_ballots(u, lazy=0.0, draw=0.5):
    """Round 1: the approval set (approves); ranking by utility. Laziness: the round-1 approvals only."""
    return rank_ballots(u) + u.shape[1] * approval_ballots(u, lazy, draw)


def score_run_ballots(u, lazy=0.0):
    """Round 1: the score ballot (score_ballots); ranking by utility. Laziness: the round-1 scores only."""
    return rank_ballots(u) + u.shape[1] * score_ballots(u, lazy)


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
    return truncate(R, cares(u, lazy)) if lazy_on(lazy) else R


def trunc_ballots(u, lazy=None, draw=0.5):
    """Rank the approval set (approves), leave the rest unranked; with laziness only those of them
    the voter also cares about. (MDDA and MAMPO read the ranked candidates as approved.)"""
    lazy = methods.LAZY if lazy is None else lazy
    R, A = rank_ballots(u), approval_ballots(u, 0.0, draw)
    if lazy_on(lazy):
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
    if lazy_on(lazy):
        keep = keep & cares(u, lazy)
        kk = keep.sum(axis=1, keepdims=True)
        return np.where(keep, R - (C - kk) + 1, 0.0)            # the kept top kk get kk ... 1
    return np.where(keep, R - (C - k) + 1, 0.0)              # top k get k ... 1


def grade3_ballots(u, lazy=0.0):
    """0..2, stretched to the voter's own utility range, rounded (as score, with 3 levels)."""
    mn, mx = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    d = np.where(mx - mn > 1e-12, mx - mn, 1.0)
    B = np.rint(2 * (u - mn) / d)
    return np.where(cares(u, lazy), B, 0.0) if lazy_on(lazy) else B         # blank = bad (0), as for scores


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


def _mutate_score_run(b, rng):
    """Only the round-1 scores change (as _mutate_score); the runoff ranking stays."""
    C = len(b)
    pts, sc = np.mod(b, C), np.floor_divide(b, C)
    return pts + C * _mutate_score(sc, rng)


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


def _exag_score_run(Ug, A, Bo):
    """The score exaggerations (_exag_score) in round 1; the runoff ranking stays sincere."""
    C = len(Ug)
    order = _views(Ug, A, Bo)[0]
    return [_rank_points(order) + C * x for x in _exag_score(Ug, A, Bo)]


def _exag_veto(Ug, A, Bo):
    v = np.zeros(len(Ug)); v[Bo] = -1
    return [v]


# ---------- what a ballot says: marks, implied values, the worst ballot ----------
# Used by the sincerity and fidelity measures (strategy.sincerity, strategy.fidelity).
def _marks_run(X):
    """Two-round ballots (fav_run, app_run): only the round-1 marks are the voter's choice (the runoff
    ranking is the sincere one), so sincerity and fidelity look at the marks, as a plurality / approval
    ballot."""
    return (X >= X.shape[1]).astype(float)


def _marks_score_run(X):
    """score_run: only the round-1 scores are the voter's choice; sincerity and fidelity look at them
    as at a score ballot."""
    return np.floor_divide(X, X.shape[1])


def _implied_score(X):
    return X / methods.MAXS


def _implied_grade3(X):
    return X / 2


def _implied_rank(X):
    """Rankings evenly spaced: best 1, worst 0 (unranked candidates at 0)."""
    return X / max(X.shape[1] - 1, 1)


def _implied_veto(X):
    return X + 1


def _implied_trunc(X):
    """Truncated rankings evenly spaced, the unranked at 0."""
    mx = X.max(axis=1, keepdims=True)
    return X / np.where(mx > 0, mx, 1)


def _implied_marks(X):
    """Plurality / approval marks: 1 and 0."""
    return X


def _within_cost(s):
    """cost[i, j] (per voter): sum of differences over all pairs inside the sorted segment s[i..j]."""
    n, C = s.shape
    cost = np.zeros((n, C, C))
    for i in range(C):
        for j in range(i + 1, C):
            seg = s[:, i:j + 1]
            m = j - i + 1
            k = np.arange(m)
            cost[:, i, j] = (seg * (2 * k - m + 1)).sum(axis=1)   # sum_{p<q} (s_q - s_p) for sorted values
    return cost


def _max_separated(P, levels):
    """Per voter: the largest reversal mass a ballot with at most `levels` levels can reach. The worst
    such ballot reverses the order with candidates grouped into contiguous blocks of the sorted
    utilities; reversal mass = all pairs minus the pairs inside blocks, so minimise the mass inside
    blocks (dynamic programme over the sorted candidates)."""
    s = np.sort(P, axis=1)
    n, C = s.shape
    total = _within_cost(s)[:, 0, C - 1] if C > 1 else np.zeros(n)
    if levels >= C:
        return total
    cost = _within_cost(s)
    INF = np.full(n, np.inf)
    best = [cost[:, 0, j] for j in range(C)]                     # one block covering 0..j
    for _ in range(levels - 1):                                  # one more block each time
        best = [np.minimum(best[j], np.min(np.stack([best[i] + cost[:, i + 1, j] if i + 1 <= j else INF
                                                     for i in range(j)] or [INF]), axis=0))
                for j in range(C)]
    return total - best[C - 1]


def _worst_fav(P):
    """A vote for the last choice."""
    return (P - P.min(axis=1, keepdims=True)).sum(axis=1)


def _worst_veto(P):
    """A veto against the favourite."""
    return (P.max(axis=1, keepdims=True) - P).sum(axis=1)


def _worst_approval(P):
    """The most damaging cut."""
    return _max_separated(P, 2)


def _worst_score(P):
    """The reversed order in as many levels as there are."""
    return _max_separated(P, methods.MAXS + 1)


def _worst_grade3(P):
    return _max_separated(P, 3)


def _worst_ranking(P):
    """Rankings (strict, weak, truncated): the reversed ranking."""
    return _max_separated(P, P.shape[1])


def _worst_trunc3(P):
    """At most TRUNC3 ranked: the 3 least liked ranked in reverse, the rest unranked (tied)."""
    C = P.shape[1]
    k = min(methods.TRUNC3, C)
    s = np.sort(P, axis=1)
    return _within_cost(s)[:, 0, C - 1] - (_within_cost(s)[:, k, C - 1] if k < C - 1 else 0)


# ---------- the ballot types ----------
@dataclass
class Ballot:
    """One ballot format (see the module docstring)."""
    name: str
    sincere: Callable           # (u, lazy) -> ballot matrix
    mutate: Callable            # (b, rng) -> one small random change of ballot vector b (strategy search)
    exaggerate: Callable        # (Ug, A, Bo) -> list of naive exaggerations around the poll leaders
    implied: Callable           # (X) -> what the ballots (after marks) say about utilities, on 0..1
    worst: Callable             # (P) -> per voter: reversal mass of the most dishonest ballot allowed
    marks: Callable = None      # (X) -> the part the voter chooses (None: the whole ballot)
    lazy: bool = False          # laziness applies
    approval: bool = False      # the honest ballot uses the approval threshold (approves): sincere(u, lazy, draw)
    ordered: bool = False       # records an order of the candidates
    score_range: bool = False   # uses the score range 0..MAXS (--max-score changes it; the max-score set)

    def chosen(self, X):
        """The part of the ballots the voter chooses (marks), as sincerity and fidelity look at it."""
        return X if self.marks is None else self.marks(X)


BALLOTS = {b.name: b for b in [
    Ballot("fav", fav_ballots, _mutate_fav, _exag_fav, _implied_marks, _worst_fav),
    Ballot("rank", lazy_rank_ballots, _mutate_rank, _exag_rank, _implied_rank, _worst_ranking,
           lazy=True, ordered=True),
    Ballot("strict", rank_ballots, _mutate_strict, _exag_strict, _implied_rank, _worst_ranking, ordered=True),
    Ballot("approval", approval_ballots, _mutate_approval, _exag_approval, _implied_marks, _worst_approval,
           lazy=True, approval=True),
    Ballot("score", score_ballots, _mutate_score, _exag_score, _implied_score, _worst_score,
           lazy=True, ordered=True, score_range=True),
    Ballot("fav_run", fav_run_ballots, _mutate_fav_run, _exag_fav_run, _implied_marks, _worst_fav,
           marks=_marks_run),
    Ballot("app_run", app_run_ballots, _mutate_app_run, _exag_app_run, _implied_marks, _worst_approval,
           marks=_marks_run, lazy=True, approval=True),
    Ballot("score_run", score_run_ballots, _mutate_score_run, _exag_score_run, _implied_score, _worst_score,
           marks=_marks_score_run, lazy=True, score_range=True),
    Ballot("veto", veto_ballots, _mutate_veto, _exag_veto, _implied_veto, _worst_veto),
    Ballot("weak", lazy_rank_ballots, _mutate_weak, _exag_weak, _implied_rank, _worst_ranking,
           lazy=True, ordered=True),                                      # honest: strict; ties only strategic
    Ballot("trunc", trunc_ballots, _mutate_trunc, _exag_trunc, _implied_trunc, _worst_ranking,
           lazy=True, ordered=True, approval=True),
    Ballot("trunc3", trunc3_ballots, _mutate_trunc3, _exag_trunc3, _implied_trunc, _worst_trunc3,
           lazy=True, ordered=True),
    Ballot("grade3", grade3_ballots, _mutate_grade3, _exag_grade3, _implied_grade3,
           _worst_grade3, lazy=True, ordered=True),
]}


# At the end: methods.py imports from this file, and the functions above read its settings (MAXS,
# LAZY, TRUNC3) only when called, so importing either file first works.
import methods  # noqa: E402
