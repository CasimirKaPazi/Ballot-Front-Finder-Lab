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


# ---------- per-voter features ----------
# Every tally except Condorcet-random-drop depends on the ballots only through sums over voters
# of per-voter features (score totals, first places, the pairwise matrix, ...). A method lists the
# features it needs and decides from their sums. During learning (strategy.py) these sums are kept
# per poll sample and updated only for the group whose ballots change, instead of recounting every
# ballot on every sample. All features are integer valued, so the sums are exact.

def _onehot(idx, k):
    z = np.zeros((len(idx), k))
    z[np.arange(len(idx)), idx] = 1
    return z


_MASKS = {}


def masks(C):
    """All candidate subsets as boolean rows; row i is the subset with bit c set for candidate c."""
    if C not in _MASKS:
        _MASKS[C] = (np.arange(2 ** C)[:, None] >> np.arange(C)) & 1 == 1
    return _MASKS[C]


def _subset_id(top):
    return (top * (1 << np.arange(top.shape[1]))).sum(axis=1).astype(int)


def _firsts(B):
    """For every subset A of candidates: one-hot of the voter's highest-rated candidate in A
    (first index on ties). Used for elimination methods: first choices among the remaining."""
    n, C = B.shape
    m = masks(C)
    Bm = np.where(m[None], B[:, None, :], -np.inf)             # (n, 2^C, C)
    out = np.zeros((n, 2 ** C, C))
    idx = Bm[:, 1:].argmax(axis=2)
    out[np.arange(n)[:, None], np.arange(1, 2 ** C)[None, :], idx] = 1
    return out.reshape(n, -1)


def _keepnorm(B):
    """For every subset K: the voter's scores among K stretched to 0..300 (= 60 x 0..5, integers),
    0 outside K and for voters rating all of K equally."""
    n, C = B.shape
    m = masks(C)[None]                                          # (1, 2^C, C)
    lo = np.where(m, B[:, None, :], np.inf).min(axis=2, keepdims=True)
    hi = np.where(m, B[:, None, :], -np.inf).max(axis=2, keepdims=True)
    d = hi - lo
    ok = m & (d > 0)
    out = np.where(ok, 60 * MAXS * (B[:, None, :] - np.where(ok, lo, 0)) / np.where(ok, d, 1), 0.0)
    return out.reshape(n, -1)


def _top(B):
    return B >= B.max(axis=1, keepdims=True)


FEATURES = {   # name: (per-voter feature (n, ...), shape of the sum given C)
    "n": (lambda B: np.ones((len(B), 1)), lambda C: ()),
    "sum": (lambda B: B, lambda C: (C,)),
    "first": (lambda B: _onehot(B.argmax(axis=1), B.shape[1]), lambda C: (C,)),
    "last": (lambda B: _onehot(B.argmin(axis=1), B.shape[1]), lambda C: (C,)),
    "gt": (lambda B: (B[:, :, None] > B[:, None, :]).reshape(len(B), -1), lambda C: (C, C)),
    "app": (lambda B: B >= B.shape[1], lambda C: (C,)),                       # rankapp approvals
    "ge3": (lambda B: B >= 3, lambda C: (C,)),
    "ge5": (lambda B: B >= MAXS, lambda C: (C,)),
    "pos": (lambda B: B > 0, lambda C: (C,)),
    "top": (lambda B: B == B.max(axis=1, keepdims=True), lambda C: (C,)),
    "eqtop": (lambda B: (lambda t: (t[:, :, None] & t[:, None, :]).reshape(len(B), -1))(
        B == B.max(axis=1, keepdims=True)), lambda C: (C, C)),
    "eqbot": (lambda B: (lambda t: (t[:, :, None] & t[:, None, :]).reshape(len(B), -1))(
        B == B.min(axis=1, keepdims=True)), lambda C: (C, C)),
    "grades": (lambda B: (B[:, :, None] == np.arange(MAXS + 1)).reshape(len(B), -1), lambda C: (C, MAXS + 1)),
    "firsts": (_firsts, lambda C: (2 ** C, C)),
    "fset": (lambda B: _onehot(_subset_id(_top(B)), 2 ** B.shape[1]), lambda C: (2 ** C,)),
    "fsetB": (lambda B: (_onehot(_subset_id(_top(B)), 2 ** B.shape[1])[:, :, None] * B[:, None, :]).reshape(
        len(B), -1), lambda C: (2 ** C, C)),
    "excess": (lambda B: np.clip(B[:, :, None] - B[:, None, :], 0, None).reshape(len(B), -1), lambda C: (C, C)),
    "keepnorm": (_keepnorm, lambda C: (2 ** C, C)),
}


def margins_of(s):
    return s["gt"] - s["gt"].T


def first_among(s, alive):
    """First-choice counts among the alive candidates (in the order of alive)."""
    return s["firsts"][int(sum(1 << c for c in alive))][alive]


# ---------- decisions from summed features ----------
def d_sum(s):
    return uniform_max(s["sum"])


def d_copeland(s):
    M = margins_of(s)
    sc = (M > 0).sum(axis=1) + 0.5 * ((M == 0).sum(axis=1) - 1)
    return uniform_max(sc.astype(float))


def d_irv(s):
    C = len(s["sum"]) if "sum" in s else s["firsts"].shape[1]
    P = np.zeros(C)

    def go(alive, w):
        if len(alive) == 1:
            P[alive[0]] += w
            return
        first = first_among(s, alive)
        low = np.flatnonzero(first == first.min())
        for i in low:
            go([c for j, c in enumerate(alive) if j != i], w / len(low))

    go(list(range(C)), 1.0)
    return P


def d_star(s):
    tot, M = s["sum"], margins_of(s)
    P = np.zeros(len(tot))
    for a, b, w in top_two(tot):
        runoff(P, M, a, b, w, tiebreak=tot)
    return P


def d_random_pair(s):
    """Draw two distinct ballots; same first choice wins, else majority runoff of the two."""
    n, first, M = s["n"], s["first"], margins_of(s)
    C = len(first)
    tot = n * (n - 1)
    P = first * (first - 1) / tot
    for a in range(C):
        for b in range(a + 1, C):
            if first[a] and first[b]:
                runoff(P, M, a, b, 2 * first[a] * first[b] / tot)
    return P


def d_top_two(s):
    """Plurality on first choices, then majority runoff between the top two."""
    first, M = s["first"], margins_of(s)
    P = np.zeros(len(first))
    for a, b, w in top_two(first):
        runoff(P, M, a, b, w)
    return P


def d_approval_runoff(s):
    """Top two by approvals, then majority runoff by ranking; a tie goes to more approvals."""
    app, M = s["app"], margins_of(s)
    P = np.zeros(len(app))
    for a, b, w in top_two(app):
        runoff(P, M, a, b, w, tiebreak=app)
    return P


def d_mj(s):
    """Majority judgment on 0-5 grades: compare median grades, removing medians to break ties."""
    h = np.rint(s["grades"]).astype(int)                  # h[c, g]: voters giving c grade g
    order = mj_order(int(h[0].sum()))
    seqs = [tuple(np.repeat(np.arange(MAXS + 1), h[c])[order]) for c in range(len(h))]
    best = max(seqs)
    top = np.array([q == best for q in seqs], dtype=float)
    return top / top.sum()


def d_minimax(s):
    """Winner has the smallest worst pairwise defeat (by margin)."""
    worst = np.clip(margins_of(s), 0, None).max(axis=0)   # largest margin by which anyone beats c
    return uniform_max(-worst.astype(float))


def d_baldwin(s):
    """Repeatedly eliminate the lowest Borda score, recounted among remaining candidates."""
    gt = s["gt"]
    C = len(gt)
    P = np.zeros(C)

    def go(alive, w):
        if len(alive) == 1:
            P[alive[0]] += w
            return
        pts = gt[np.ix_(alive, alive)].sum(axis=1)        # Borda among the remaining (strict rankings)
        low = np.flatnonzero(pts == pts.min())
        for i in low:
            go([c for j, c in enumerate(alive) if j != i], w / len(low))

    go(list(range(C)), 1.0)
    return P


def d_btr(s):
    """Order by total (Borda points or score), bottom-two pairwise runoffs climbing up."""
    M = margins_of(s)
    return bottom_two_runoff(s["sum"], lambda a, b: M[a, b])


def d_mars(s):
    M, tot = margins_of(s), s["sum"]
    # share of voters preferring + score as share of maximum, both scaled by n_voters
    return bottom_two_runoff(tot, lambda a, b: M[a, b] + (tot[a] - tot[b]) / MAXS)


def ara(key):
    def d(s):
        M, app = margins_of(s), s[key]
        return bottom_two_runoff(app, lambda a, b: M[a, b] + (app[a] - app[b]))
    return d


def d_trash(s):
    """TRASH (This Random Acronym Sounds Honest): the plurality runner-up and the anti-plurality
    runner-up (second fewest last places) meet in a runoff, and the one ranked lower by more voters
    wins. If both are the same candidate, it wins. Ties at each stage are split uniformly.
    A deliberately bad reference."""
    return _trash(s, place_dist(s["first"], 1), place_dist(-s["last"], 1))


def _trash(s, pw, aw):
    M = margins_of(s)
    P = np.zeros(len(pw))
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


def d_btr_irv(s):
    """Each round the two candidates with the fewest first choices meet pairwise; the loser is out."""
    M = margins_of(s)
    C = len(M)
    P = np.zeros(C)

    def go(alive, w):
        if len(alive) == 1:
            P[alive[0]] += w
            return
        first = first_among(s, alive)
        for a, b, wp in top_two(-first):                  # the two lowest first-choice counts
            ca, cb = alive[a], alive[b]
            x = M[ca, cb]
            for loser, wl in ([(cb, 1.0)] if x > 0 else [(ca, 1.0)] if x < 0 else [(ca, .5), (cb, .5)]):
                go([c for c in alive if c != loser], w * wp * wl)

    go(list(range(C)), 1.0)
    return P


def d_benham(s):
    """IRV, but a Condorcet winner among the remaining candidates wins at once."""
    M = margins_of(s)
    C = len(M)
    P = np.zeros(C)

    def go(alive, w):
        cw = has_cw(M, alive)
        if cw is not None or len(alive) == 1:
            P[cw if cw is not None else alive[0]] += w
            return
        first = first_among(s, alive)
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
    # all orders at once, in growing chunks of dropped voters (a winner usually appears early)
    order, todo, start, step = _DROP[n], np.arange(DROP_ORDERS), 0, 8
    M = np.broadcast_to(M0, (DROP_ORDERS, C, C))           # margins so far, one row per order in todo
    while len(todo) and start < n:
        cum = M[:, None] - np.cumsum(D[order[todo, start:start + step]], axis=1)   # (orders, k, C, C)
        win = (cum > 0).sum(axis=3) == C - 1               # win[o, k, c]: c is the Condorcet winner
        found = win.any(axis=2)
        hit = found.any(axis=1)
        for o in np.flatnonzero(hit):
            P[win[o, found[o].argmax()].argmax()] += 1
        M = cum[:, -1][~hit]
        todo, start, step = todo[~hit], start + step, step * 2
    P += len(todo) / C
    return P / DROP_ORDERS


def d_sict(s):
    """Symmetrical ICT on rankings with ties. p(X,Y): (X>Y) + (X=Y at bottom) > (Y>X) + (X=Y at top).
    X beats Y if p(X,Y) and not p(Y,X), or both and (X>Y) > (Y>X). One unbeaten candidate wins;
    if none or all are unbeaten, most first places wins; otherwise most first places among the unbeaten."""
    gt, eq_top, eq_bot = s["gt"], s["eqtop"], s["eqbot"]
    p = gt + eq_bot > gt.T + eq_top
    beats = (p & ~p.T) | (p & p.T & (gt > gt.T))
    np.fill_diagonal(beats, False)
    unbeaten = ~beats.any(axis=0)
    first = s["top"]
    if unbeaten.sum() == 1:
        return unbeaten.astype(float)
    if unbeaten.sum() in (0, len(first)):
        return uniform_max(first)
    return uniform_max(np.where(unbeaten, first, -1.0))


def d_mdda(s):
    """Majority Defeat Disqualification Approval: ranked = approved. A candidate is disqualified if
    more than half the voters rank some other candidate strictly above it (unless that would
    disqualify everyone); the most approved remaining candidate wins."""
    dominated = (s["gt"] > s["n"] / 2).any(axis=0)
    keep = ~dominated if not dominated.all() else np.ones_like(dominated)
    return uniform_max(np.where(keep, s["pos"], -1.0))


def place_dist(values, k):
    """Probability that each candidate ends up in place k (0 = first) when ranking by values,
    high to low, with ties broken uniformly at random."""
    order = np.unique(np.round(values, 9))[::-1]
    pos, P = 0, np.zeros(len(values))
    for x in order:
        grp = np.flatnonzero(np.abs(values - x) < 1e-8)
        if pos <= k < pos + len(grp):
            P[grp] = 1 / len(grp)
            return P
        pos += len(grp)
    return P


RP_MAX_ORDERS = 240


def d_ranked_pairs(s):
    """Ranked Pairs (margins): lock pairwise wins from the largest margin down, skipping any that
    would close a cycle; the candidate with no locked defeat wins. Pairs with equal margins are
    locked in every order and the results averaged (up to RP_MAX_ORDERS fixed random orders)."""
    M = margins_of(s)
    C = len(M)
    cw = (M > 0).sum(axis=1) == C - 1                  # a Condorcet winner is never locked out
    if cw.any():
        return cw.astype(float)
    pairs = [(a, b) for a in range(C) for b in range(C) if M[a, b] > 0]
    vals = sorted({M[a, b] for a, b in pairs}, reverse=True)
    groups = [[p for p in pairs if M[p] == v] for v in vals]
    n_orders = 1
    for g in groups:
        n_orders *= int(np.prod(range(1, len(g) + 1)))
    if n_orders <= RP_MAX_ORDERS:
        orders = [[p for part in parts for p in part] for parts in product(*[permutations(g) for g in groups])]
    else:
        r = np.random.default_rng(0)
        orders = [[p for g in groups for p in (g[i] for i in r.permutation(len(g)))] for _ in range(RP_MAX_ORDERS)]
    P = np.zeros(C)
    for order in orders:
        reach = np.eye(C, dtype=bool)                 # reach[x, y]: locked path from x to y
        locked = np.zeros((C, C), dtype=bool)
        for a, b in order:
            if not reach[b, a]:
                locked[a, b] = True
                reach = reach | (reach[:, [a]] & reach[[b], :])
        src = np.flatnonzero(~locked.any(axis=0))
        P[src] += 1 / (len(src) * len(orders))
    return P


def d_schulze(s):
    """Schulze (margins): strongest path strengths; winners beat or tie everyone by path strength."""
    M = margins_of(s)
    C = len(M)
    p = np.where(M > 0, M, 0).astype(float)
    for k in range(C):
        p = np.maximum(p, np.minimum(p[:, [k]], p[[k], :]))
    np.fill_diagonal(p, 0)
    win = np.array([all(p[a, b] >= p[b, a] for b in range(C) if b != a) for a in range(C)], dtype=float)
    return win / win.sum()


def d_approval_runoff_if_needed(s):
    """Ranked approval ballot. The most approved candidate wins outright if more than half the
    voters approve it; otherwise top two by approvals, majority runoff, a tie to more approvals."""
    if s["app"].max() > s["n"] / 2:
        return uniform_max(s["app"])
    return d_approval_runoff(s)


def d_reciprocal_score(s):
    """Reciprocal Score Voting (interpretation): a voter's faction is their top-rated candidate(s).
    Each score s[v, c] is multiplied by min(1, R(c -> F_v) / R(F_v -> c)), where R(X -> Y) is the mean
    normalised rating faction X gives Y (averaged over the members of a set). A candidate nobody
    top-rates has no faction, so scores for it are left unchanged.
    Voters with the same faction set F share the multipliers, so sums per F suffice."""
    cnt, SB = s["fset"], s["fsetB"]                     # voters and their score sums, per faction set
    C = SB.shape[1]
    used = np.flatnonzero(cnt)
    m = masks(C)[used]                                  # (sets, C) membership
    members = m.T.astype(float) @ cnt[used]             # voters top-rating each candidate
    has = members > 0
    R = np.zeros((C, C))                                # R[x, y]: mean rating faction x gives y, in [0, 1]
    R[has] = (m[:, has].T.astype(float) @ SB[used]) / members[has][:, None] / MAXS
    k = m.sum(axis=1, keepdims=True)
    give = m @ R / k                                    # give[F, c]: support faction F gives c
    back = m @ R.T / k                                  # back[F, c]: support c's faction gives back
    mult = np.where(give > 0, np.minimum(1.0, back / np.where(give > 0, give, 1)), 1.0)
    mult = np.where(m | ~has[None, :], 1.0, mult)
    return uniform_max((SB[used] * mult).sum(axis=0))


def d_score_plus(s):
    """Score+ (simplified variant from the votingtheory.org thread): A defeats B robustly if A's total
    minus the scores both share (per voter the lower of the two) still exceeds B's total.
    Candidates with a robust defeat are eliminated; scores are renormalised per voter among the
    rest (lowest to 0, highest to 5); the highest total wins."""
    tot, excess = s["sum"], s["excess"]                 # excess[a, b] = sum max(0, s_a - s_b)
    C = len(tot)
    defeated = (excess > tot[None, :] + EPS).any(axis=0)
    keep = np.flatnonzero(~defeated)
    P = np.zeros(C)
    P[keep] = uniform_max(s["keepnorm"][int(sum(1 << c for c in keep))][keep])
    return P


@dataclass
class Method:
    ballot: str
    decide: Callable = None          # decision from summed features (a dict name -> sum)
    feats: tuple = ()                # features the decision needs
    full: Callable = None            # tally on the full ballot matrix, for methods without features

    def features(self, B):
        """Per-voter features, concatenated: (n, d)."""
        return np.concatenate([FEATURES[f][0](B).reshape(len(B), -1).astype(float) for f in self.feats], axis=1)

    def unpack(self, S):
        """Summed features (d,) -> dict name -> array of its proper shape."""
        C = self._C
        out, i = {}, 0
        for f in self.feats:
            shape = FEATURES[f][1](C)
            k = int(np.prod(shape))
            out[f] = S[i:i + k].reshape(shape) if shape else S[i]
            i += k
        return out

    def decide_many(self, S, C):
        """Winner probabilities for each row of summed features S (samples, d)."""
        self._C = C
        return np.array([self.decide(self.unpack(x)) for x in S])

    def tally(self, B):
        if self.full is not None:
            return self.full(B)
        return self.decide_many(self.features(B).sum(axis=0, keepdims=True), B.shape[1])[0]

    def sincere(self, u):
        return SINCERE[self.ballot](u)

    def honest(self, u):
        return self.tally(self.sincere(u))


M_ = Method
METHODS = {
    "Random pair runoff": M_("rank", d_random_pair, ("n", "first", "gt")),
    "Plurality": M_("fav", d_sum, ("sum",)),
    "Top-two runoff": M_("rank", d_top_two, ("first", "gt")),
    "IRV": M_("rank", d_irv, ("firsts",)),
    "Borda": M_("rank", d_sum, ("sum",)),
    "Baldwin": M_("rank", d_baldwin, ("gt",)),
    "BTR-Borda": M_("rank", d_btr, ("sum", "gt")),
    "Copeland": M_("rank", d_copeland, ("gt",)),
    "Minimax": M_("rank", d_minimax, ("gt",)),
    "Approval": M_("approval", d_sum, ("sum",)),
    "Approval runoff": M_("rankapp", d_approval_runoff, ("app", "gt")),
    "Score": M_("score", d_sum, ("sum",)),
    "MJ": M_("score", d_mj, ("grades",)),
    "STAR": M_("score", d_star, ("sum", "gt")),
    "BTR-score": M_("score", d_btr, ("sum", "gt")),
    "MARS": M_("score", d_mars, ("sum", "gt")),
    "ARA": M_("score", ara("ge3"), ("ge3", "gt")),              # ratings 3-5 count as approval
    "ARA-Top": M_("score", ara("ge5"), ("ge5", "gt")),          # only a 5 counts as approval
    "Anti-plurality": M_("veto", d_sum, ("sum",)),
    "TRASH": M_("rank", d_trash, ("first", "last", "gt")),
    "Ranked pairs": M_("rank", d_ranked_pairs, ("gt",)),
    "Schulze": M_("rank", d_schulze, ("gt",)),
    "Approval runoff if needed": M_("rankapp", d_approval_runoff_if_needed, ("n", "app", "gt")),
    "Reciprocal score": M_("score", d_reciprocal_score, ("fset", "fsetB")),
    "Score+": M_("score", d_score_plus, ("sum", "excess", "keepnorm")),
    "Condorcet-random-drop": M_("rank", full=t_condorcet_drop),
    "SICT": M_("weak", d_sict, ("gt", "top", "eqtop", "eqbot")),
    "MDDA": M_("trunc", d_mdda, ("n", "gt", "pos")),
    "BTR-IRV": M_("rank", d_btr_irv, ("firsts", "gt")),
    "Benham": M_("rank", d_benham, ("firsts", "gt")),
}


# Method groups for --methods / --exclude (tags: a method can be in several groups)
GROUPS = {
    "condorcet": ["Baldwin", "Copeland", "Minimax", "Ranked pairs", "Schulze", "SICT", "Condorcet-random-drop",
                  "BTR-Borda", "BTR-IRV", "Benham"],
    "plurality-based": ["Plurality", "Top-two runoff", "IRV", "BTR-IRV", "Benham"],
    "rated": ["Score", "MJ", "STAR", "BTR-score", "MARS", "ARA", "ARA-Top", "Reciprocal score", "Score+"],
    "approval-based": ["Approval", "Approval runoff", "Approval runoff if needed", "MDDA"],
    "references": ["Random pair runoff", "Anti-plurality", "TRASH"],
}


def select(include=None, exclude=None):
    """Method names to run: `include` starts from none, `exclude` from all. Names are methods or
    groups (see GROUPS); matching ignores case."""
    lookup = {k.lower(): [k] for k in METHODS}
    lookup.update({g: ms for g, ms in GROUPS.items()})

    def resolve(names):
        bad = [x for x in names if x.lower() not in lookup]
        if bad:
            raise SystemExit(f"unknown method(s) or group(s): {', '.join(bad)}. Methods: {', '.join(METHODS)}. "
                             f"Groups: {', '.join(GROUPS)}")
        return {m for x in names for m in lookup[x.lower()]}

    chosen = resolve(include) if include else set(METHODS)
    chosen -= resolve(exclude or [])
    return [k for k in METHODS if k in chosen]
