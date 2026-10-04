"""Voting methods: ballot types, sincere ballots, and exact tallies on ballot matrices.

A ballot matrix B has one row per voter and one column per candidate:
  "fav"      one-hot: 1 for the named candidate
  "rank"     rank points: best C-1 ... worst 0
  "approval" 0/1
  "score"    0..5
Every tally returns exact winner probabilities; ties are split uniformly among
the tied candidates and counted exactly.

Sincere ballots from utilities u[v, c]:
  rank     - order by utility
  approval - every candidate above the voter's mean utility
  score    - 0..5, stretched to the voter's own utility range, rounded
"""
from dataclasses import dataclass
from itertools import permutations, product
from typing import Callable
import numpy as np

EPS = 1e-9
MAXS = 5


# ---------- sincere ballots ----------
def fav_ballots(u):
    B = np.zeros_like(u)
    B[np.arange(u.shape[0]), u.argmax(axis=1)] = 1
    return B


def rank_ballots(u):
    order = np.argsort(-u, axis=1, kind="stable")
    B = np.empty_like(u)
    C = u.shape[1]
    B[np.arange(u.shape[0])[:, None], order] = np.arange(C - 1, -1, -1)
    return B


def approval_ballots(u):
    a = u > u.mean(axis=1, keepdims=True) + 1e-12
    none = ~a.any(axis=1)
    a[none, u[none].argmax(axis=1)] = True
    return a.astype(float)


def score_ballots(u):
    mn, mx = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    d = np.where(mx - mn > 1e-12, mx - mn, 1.0)
    return np.rint(MAXS * (u - mn) / d)


SINCERE = {"fav": fav_ballots, "rank": rank_ballots, "approval": approval_ballots, "score": score_ballots}


# ---------- helpers ----------
def margins(B):
    """Pairwise margins: M[a, b] = #(a rated above b) - #(b rated above a)."""
    gt = (B[:, :, None] > B[:, None, :]).sum(axis=0)
    return gt - gt.T


def uniform_max(v):
    top = v >= v.max() - EPS
    return top / top.sum()


def orders_by(values):
    """All orderings high-to-low consistent with values (every order of tied groups)."""
    vals = np.unique(np.round(values, 9))[::-1]
    groups = [list(np.flatnonzero(np.abs(values - x) < 1e-8)) for x in vals]
    return [[c for p in parts for c in p] for parts in product(*[permutations(g) for g in groups])]


def bottom_two_runoff(order_values, match):
    """Order high-to-low; the lowest two meet, the winner climbs. match(a, b) > 0: a beats b."""
    C = len(order_values)
    P = np.zeros(C)
    orders = orders_by(order_values)
    w0 = 1.0 / len(orders)

    def climb(order, i, cur, w):
        if i < 0:
            P[cur] += w
            return
        ch = order[i]
        x = match(cur, ch)
        if x > EPS:
            climb(order, i - 1, cur, w)
        elif x < -EPS:
            climb(order, i - 1, ch, w)
        else:
            climb(order, i - 1, cur, w / 2)
            climb(order, i - 1, ch, w / 2)

    for o in orders:
        climb(o, C - 2, o[C - 1], w0)
    return P


# ---------- tallies ----------
def t_random_ballot(B):
    return B.sum(axis=0) / B.shape[0]


def t_sum(B):
    return uniform_max(B.sum(axis=0))


def t_copeland(B):
    M = margins(B)
    s = (M > 0).sum(axis=1) + 0.5 * ((M == 0).sum(axis=1) - 1)
    return uniform_max(s.astype(float))


def t_irv(B):
    C = B.shape[1]
    P = np.zeros(C)

    def go(alive, w):
        if len(alive) == 1:
            P[alive[0]] += w
            return
        first = np.bincount(B[:, alive].argmax(axis=1), minlength=len(alive))
        low = np.flatnonzero(first == first.min())
        for i in low:
            go([c for j, c in enumerate(alive) if j != i], w / len(low))

    go(list(range(C)), 1.0)
    return P


def t_star(B):
    tot = B.sum(axis=0)
    M = margins(B)
    C = B.shape[1]
    P = np.zeros(C)

    def runoff(a, b, w):
        x = M[a, b] if M[a, b] != 0 else tot[a] - tot[b]
        if x > 0:
            P[a] += w
        elif x < 0:
            P[b] += w
        else:
            P[a] += w / 2
            P[b] += w / 2

    top = np.flatnonzero(tot >= tot.max() - EPS)
    if len(top) >= 2:
        pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
        for a, b in pairs:
            runoff(a, b, 1 / len(pairs))
        return P
    a = top[0]
    rest = np.delete(np.arange(C), a)
    sec = rest[tot[rest] >= tot[rest].max() - EPS]
    for b in sec:
        runoff(a, b, 1 / len(sec))
    return P


def t_btr(B):
    M = margins(B)
    return bottom_two_runoff(B.sum(axis=0), lambda a, b: M[a, b])


def t_mars(B):
    M, tot = margins(B), B.sum(axis=0)
    # share of voters preferring + score as share of maximum, both scaled by n_voters
    return bottom_two_runoff(tot, lambda a, b: M[a, b] + (tot[a] - tot[b]) / MAXS)


def t_mara(B):
    """MARA-3-5: ratings 3-5 count as approval."""
    M, app = margins(B), (B >= 3).sum(axis=0).astype(float)
    return bottom_two_runoff(app, lambda a, b: M[a, b] + (app[a] - app[b]))


@dataclass
class Method:
    ballot: str
    tally: Callable

    def sincere(self, u):
        return SINCERE[self.ballot](u)

    def honest(self, u):
        return self.tally(self.sincere(u))


METHODS = {
    "Random ballot": Method("fav", t_random_ballot),
    "Plurality": Method("fav", t_sum),
    "IRV": Method("rank", t_irv),
    "Borda": Method("rank", t_sum),
    "Copeland": Method("rank", t_copeland),
    "Approval": Method("approval", t_sum),
    "Score": Method("score", t_sum),
    "STAR": Method("score", t_star),
    "BTR-score": Method("score", t_btr),
    "MARS": Method("score", t_mars),
    "MARA": Method("score", t_mara),
}
