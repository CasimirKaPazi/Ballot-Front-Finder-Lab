"""Voting methods: ballot types, sincere ballots, and exact tallies on ballot matrices.

A ballot matrix B has one row per voter and one column per candidate:
  "fav"      one-hot: 1 for the named candidate
  "rank"     rank points: best C-1 ... worst 0
  "approval" 0/1
  "score"    0..5
  "rankapp"  ranking with an approval line: rank points (C-1 ... 0), plus C for approved
             candidates, who are always the top k of the ranking
  "veto"     -1 for the one candidate voted against, 0 for all others
  "weak"     ranking with ties allowed: levels 0 ... C-1, higher is better, equal = tied
  "trunc"    truncated ranking: the k ranked candidates get k ... 1, unranked ones 0 (tied last)
Every tally returns exact winner probabilities; ties are split uniformly among
the tied candidates and counted exactly.

Sincere ballots from utilities u[v, c]:
  rank     - order by utility
  approval - every candidate above the voter's mean utility
  score    - 0..5, stretched to the voter's own utility range, rounded
  rankapp  - ranking by utility, approval line below the last candidate above the mean
  veto     - against the least liked candidate
  weak     - strict ranking by utility (ties are only used strategically)
  trunc    - rank every candidate above the voter's mean utility, leave the rest unranked
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


def rankapp_ballots(u):
    return rank_ballots(u) + u.shape[1] * approval_ballots(u)


def veto_ballots(u):
    B = np.zeros_like(u)
    B[np.arange(u.shape[0]), u.argmin(axis=1)] = -1
    return B


def trunc_ballots(u):
    R, A = rank_ballots(u), approval_ballots(u)
    k = A.sum(axis=1, keepdims=True)
    return np.where(A > 0, R - (u.shape[1] - k) + 1, 0.0)   # top k get k ... 1


SINCERE = {"fav": fav_ballots, "rank": rank_ballots, "approval": approval_ballots, "score": score_ballots,
           "rankapp": rankapp_ballots, "veto": veto_ballots, "weak": rank_ballots, "trunc": trunc_ballots}


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
    order = np.argsort(-values, kind="stable")
    if np.all(np.diff(values[order]) < -1e-8):   # no ties: exactly one order
        return [list(order)]
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


def top_two(values):
    """Finalist pairs (a, b, weight) from first-stage values, ties broken uniformly at random."""
    C = len(values)
    top = np.flatnonzero(values >= values.max() - EPS)
    if len(top) >= 2:
        pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
        return [(a, b, 1 / len(pairs)) for a, b in pairs]
    a = top[0]
    rest = np.delete(np.arange(C), a)
    sec = rest[values[rest] >= values[rest].max() - EPS]
    return [(a, b, 1 / len(sec)) for b in sec]


def runoff(P, M, a, b, w, tiebreak=None):
    """Pairwise majority between a and b; tie goes to the higher tiebreak value, else split."""
    x = M[a, b]
    if x == 0 and tiebreak is not None:
        x = tiebreak[a] - tiebreak[b]
    if x > 0:
        P[a] += w
    elif x < 0:
        P[b] += w
    else:
        P[a] += w / 2
        P[b] += w / 2


_MJ_ORDER = {}


def mj_order(n):
    """Positions in a sorted grade list, in the order majority judgment reads them:
    take the (lower) median, remove it, repeat."""
    if n not in _MJ_ORDER:
        idx, seq = list(range(n)), []
        while idx:
            seq.append(idx.pop((len(idx) - 1) // 2))
        _MJ_ORDER[n] = np.array(seq)
    return _MJ_ORDER[n]


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


def t_random_pair(B):
    """Draw two distinct ballots; same first choice wins, else majority runoff of the two."""
    n, C = B.shape
    first = np.bincount(B.argmax(axis=1), minlength=C).astype(float)
    M = margins(B)
    tot = n * (n - 1)
    P = first * (first - 1) / tot
    for a in range(C):
        for b in range(a + 1, C):
            if first[a] and first[b]:
                runoff(P, M, a, b, 2 * first[a] * first[b] / tot)
    return P


def t_top_two(B):
    """Plurality on first choices, then majority runoff between the top two."""
    C = B.shape[1]
    first = np.bincount(B.argmax(axis=1), minlength=C).astype(float)
    M, P = margins(B), np.zeros(C)
    for a, b, w in top_two(first):
        runoff(P, M, a, b, w)
    return P


def t_approval_runoff(B):
    """Top two by approvals, then majority runoff by ranking; a tie goes to more approvals."""
    C = B.shape[1]
    app = (B >= C).sum(axis=0).astype(float)
    M, P = margins(B), np.zeros(C)
    for a, b, w in top_two(app):
        runoff(P, M, a, b, w, tiebreak=app)
    return P


def t_mj(B):
    """Majority judgment on 0-5 grades: compare median grades, removing medians to break ties."""
    order = mj_order(B.shape[0])
    seqs = [tuple(np.sort(B[:, c])[order]) for c in range(B.shape[1])]
    best = max(seqs)
    top = np.array([s == best for s in seqs], dtype=float)
    return top / top.sum()


def t_minimax(B):
    """Winner has the smallest worst pairwise defeat (by margin)."""
    M = margins(B)
    worst = np.clip(M, 0, None).max(axis=0)   # largest margin by which anyone beats c
    return uniform_max(-worst.astype(float))


def t_baldwin(B):
    """Repeatedly eliminate the lowest Borda score, recounted among remaining candidates."""
    C = B.shape[1]
    P = np.zeros(C)

    def go(alive, w):
        if len(alive) == 1:
            P[alive[0]] += w
            return
        sub = B[:, alive]
        pts = (sub[:, :, None] > sub[:, None, :]).sum(axis=2).sum(axis=0)
        low = np.flatnonzero(pts == pts.min())
        for i in low:
            go([c for j, c in enumerate(alive) if j != i], w / len(low))

    go(list(range(C)), 1.0)
    return P


def t_btr_borda(B):
    M = margins(B)
    return bottom_two_runoff(B.sum(axis=0), lambda a, b: M[a, b])


def t_btr(B):
    M = margins(B)
    return bottom_two_runoff(B.sum(axis=0), lambda a, b: M[a, b])


def t_mars(B):
    M, tot = margins(B), B.sum(axis=0)
    # share of voters preferring + score as share of maximum, both scaled by n_voters
    return bottom_two_runoff(tot, lambda a, b: M[a, b] + (tot[a] - tot[b]) / MAXS)


def mara(B, threshold):
    M, app = margins(B), (B >= threshold).sum(axis=0).astype(float)
    return bottom_two_runoff(app, lambda a, b: M[a, b] + (app[a] - app[b]))


def t_mara(B):
    """MARA-3-5: ratings 3-5 count as approval."""
    return mara(B, 3)


def t_mara_top(B):
    """MARA-Top: only a 5 counts as approval."""
    return mara(B, MAXS)


def t_mara_bottom(B):
    """MARA-Bottom: only a 0 counts as rejection; ratings 1-5 are approvals."""
    return mara(B, 1)


def t_trash(B):
    """TRASH (This Random Acronym Sounds Honest): the plurality winner and the anti-plurality winner
    (fewest last places) meet in a runoff, and the one ranked lower by more voters wins.
    If both are the same candidate, it wins. Ties at each stage are split uniformly."""
    C = B.shape[1]
    first = np.bincount(B.argmax(axis=1), minlength=C).astype(float)
    last = np.bincount(B.argmin(axis=1), minlength=C).astype(float)
    pw, aw = uniform_max(first), uniform_max(-last)
    M = margins(B)
    P = np.zeros(C)
    for a in np.flatnonzero(pw):
        for b in np.flatnonzero(aw):
            w = pw[a] * aw[b]
            if a == b:
                P[a] += w
            else:
                runoff(P, -M, a, b, w)        # reversed majority: the less preferred one wins
    return P


def has_cw(M, alive):
    """Index of the Condorcet winner among alive candidates (strict majority over each), or None."""
    for c in alive:
        if all(M[c, d] > 0 for d in alive if d != c):
            return c
    return None


def t_btr_irv(B):
    """Each round the two candidates with the fewest first choices meet pairwise; the loser is out."""
    C = B.shape[1]
    M = margins(B)
    P = np.zeros(C)

    def go(alive, w):
        if len(alive) == 1:
            P[alive[0]] += w
            return
        first = np.bincount(B[:, alive].argmax(axis=1), minlength=len(alive)).astype(float)
        for a, b, wp in top_two(-first):                  # the two lowest first-choice counts
            ca, cb = alive[a], alive[b]
            x = M[ca, cb]
            for loser, wl in ([(cb, 1.0)] if x > 0 else [(ca, 1.0)] if x < 0 else [(ca, .5), (cb, .5)]):
                go([c for c in alive if c != loser], w * wp * wl)

    go(list(range(C)), 1.0)
    return P


def t_benham(B):
    """IRV, but a Condorcet winner among the remaining candidates wins at once."""
    C = B.shape[1]
    M = margins(B)
    P = np.zeros(C)

    def go(alive, w):
        cw = has_cw(M, alive)
        if cw is not None or len(alive) == 1:
            P[cw if cw is not None else alive[0]] += w
            return
        first = np.bincount(B[:, alive].argmax(axis=1), minlength=len(alive))
        low = np.flatnonzero(first == first.min())
        for i in low:
            go([c for j, c in enumerate(alive) if j != i], w / len(low))

    go(list(range(C)), 1.0)
    return P


DROP_ORDERS = 32
_DROP = {}


def t_condorcet_drop(B):
    """Condorcet winner if there is one. Otherwise voters are dropped one at a time in random order
    until the remaining ballots have a Condorcet winner. Averaged over DROP_ORDERS fixed random
    orders (the same orders on every call, so comparisons during learning carry no extra noise)."""
    n, C = B.shape
    if n not in _DROP:
        _DROP[n] = np.random.default_rng(12345).permuted(np.tile(np.arange(n), (DROP_ORDERS, 1)), axis=1)
    D = np.sign(B[:, :, None] - B[:, None, :])          # each voter's contribution to the margins
    M0 = D.sum(axis=0)
    alive = list(range(C))
    cw = has_cw(M0, alive)
    P = np.zeros(C)
    if cw is not None:
        P[cw] = 1.0
        return P
    for order in _DROP[n]:
        M = M0.copy()
        for v in order:
            M -= D[v]
            cw = has_cw(M, alive)
            if cw is not None:
                P[cw] += 1
                break
        else:
            P += 1 / C
    return P / DROP_ORDERS


def t_sict(B):
    """Symmetrical ICT on rankings with ties. p(X,Y): (X>Y) + (X=Y at bottom) > (Y>X) + (X=Y at top).
    X beats Y if p(X,Y) and not p(Y,X), or both and (X>Y) > (Y>X). One unbeaten candidate wins;
    if none or all are unbeaten, most first places wins; otherwise most first places among the unbeaten."""
    gt = (B[:, :, None] > B[:, None, :]).sum(axis=0)
    top = B == B.max(axis=1, keepdims=True)
    bot = B == B.min(axis=1, keepdims=True)
    eq_top = (top[:, :, None] & top[:, None, :]).sum(axis=0)
    eq_bot = (bot[:, :, None] & bot[:, None, :]).sum(axis=0)
    p = gt + eq_bot > gt.T + eq_top
    beats = (p & ~p.T) | (p & p.T & (gt > gt.T))
    np.fill_diagonal(beats, False)
    unbeaten = ~beats.any(axis=0)
    first = top.sum(axis=0).astype(float)
    if unbeaten.sum() == 1:
        return unbeaten.astype(float)
    if unbeaten.sum() in (0, B.shape[1]):
        return uniform_max(first)
    return uniform_max(np.where(unbeaten, first, -1.0))


def t_mdda(B):
    """Majority Defeat Disqualification Approval: ranked = approved. A candidate is disqualified if
    more than half the voters rank some other candidate strictly above it (unless that would
    disqualify everyone); the most approved remaining candidate wins."""
    n = B.shape[0]
    gt = (B[:, :, None] > B[:, None, :]).sum(axis=0)     # gt[a, b]: voters ranking a above b
    dominated = (gt > n / 2).any(axis=0)
    keep = ~dominated if not dominated.all() else np.ones_like(dominated)
    app = (B > 0).sum(axis=0).astype(float)
    return uniform_max(np.where(keep, app, -1.0))


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
    "Random pair runoff": Method("rank", t_random_pair),
    "Plurality": Method("fav", t_sum),
    "Top-two runoff": Method("rank", t_top_two),
    "IRV": Method("rank", t_irv),
    "Borda": Method("rank", t_sum),
    "Baldwin": Method("rank", t_baldwin),
    "BTR-Borda": Method("rank", t_btr_borda),
    "Copeland": Method("rank", t_copeland),
    "Minimax": Method("rank", t_minimax),
    "Approval": Method("approval", t_sum),
    "Approval runoff": Method("rankapp", t_approval_runoff),
    "Score": Method("score", t_sum),
    "MJ": Method("score", t_mj),
    "STAR": Method("score", t_star),
    "BTR-score": Method("score", t_btr),
    "MARS": Method("score", t_mars),
    "MARA": Method("score", t_mara),
    "MARA-Top": Method("score", t_mara_top),
    "MARA-Bottom": Method("score", t_mara_bottom),
    "Anti-plurality": Method("veto", t_sum),
    "TRASH": Method("rank", t_trash),
    "Condorcet-random-drop": Method("rank", t_condorcet_drop),
    "SICT": Method("weak", t_sict),
    "MDDA": Method("trunc", t_mdda),
    "BTR-IRV": Method("rank", t_btr_irv),
    "Benham": Method("rank", t_benham),
}


def select(include=None, exclude=None):
    """Method names to run: `include` starts from none, `exclude` from all. Matching ignores case."""
    lookup = {k.lower(): k for k in METHODS}

    def resolve(names):
        bad = [x for x in names if x.lower() not in lookup]
        if bad:
            raise SystemExit(f"unknown method(s): {', '.join(bad)}. Available: {', '.join(METHODS)}")
        return {lookup[x.lower()] for x in names}

    chosen = resolve(include) if include else set(METHODS)
    chosen -= resolve(exclude or [])
    return [k for k in METHODS if k in chosen]
