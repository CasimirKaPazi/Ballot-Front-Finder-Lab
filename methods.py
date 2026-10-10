"""Voting methods: ballot types, sincere ballots, and exact tallies on ballot matrices.

A ballot matrix B has one row per voter and one column per candidate:
  "fav"      one-hot: 1 for the named candidate
  "rank"     rank points: best C-1 ... worst 0
  "approval" 0/1
  "score"    0..MAXS (default 0..5; set_max_score changes the range)
  "fav_run"  two-round election, plurality first round: the sincere ranking as rank points
             (C-1 ... 0) for the separate runoff, plus C for the one candidate marked in round 1
  "app_run"  two-round election, approval first round: as fav_run, plus C for every approved
             candidate (any set). The ranking part is fixed: strategy changes only the round-1 marks,
             since in a separate two-candidate runoff voting for the preferred one is always best
  "veto"     -1 for the one candidate voted against, 0 for all others
  "weak"     ranking with ties allowed: levels 0 ... C-1, higher is better, equal = tied
  "trunc"    truncated ranking: the k ranked candidates get k ... 1, unranked ones 0 (tied last)
  "trunc3"   as trunc, at most 3 ranked (IRV-3)
  "grade3"   three grades: 2 = good, 1 = OK, 0 = bad
Every tally returns exact winner probabilities; ties are split uniformly among
the tied candidates and counted exactly.

Sincere ballots from utilities u[v, c]:
  rank     - order by utility
  approval - every candidate above the voter's mean utility
  score    - 0..MAXS, stretched to the voter's own utility range, rounded
  fav_run  - mark the favourite; ranking by utility
  app_run  - approve every candidate above the voter's mean utility; ranking by utility
  veto     - against the least liked candidate
  weak     - strict ranking by utility (ties are only used strategically)
  trunc    - rank every candidate above the voter's mean utility, leave the rest unranked
  trunc3   - rank the 3 best-liked candidates (all, with 3 or fewer), leave the rest unranked
  grade3   - 0..2, stretched to the voter's own utility range, rounded (as score, with 3 levels)
"""
from dataclasses import dataclass
from itertools import permutations, product
from typing import Callable
import numpy as np

EPS = 1e-9
MAXS = 5          # top of the score range; change it with set_max_score, not directly


def set_max_score(m):
    """Set the score range 0..m for every method on "score" ballots (Score, STAR, MJ, ARA, ...).
    grade3 methods (MCA, 321) keep their three grades. Call it in every process before tallying."""
    global MAXS
    if int(m) < 1:
        raise ValueError("max score must be at least 1")
    MAXS = int(m)


def _keepnorm_scale():
    """Integer scale for _keepnorm: a common multiple of every possible range 1..MAXS, so the
    stretched scores stay integers and sums are exact; for large ranges (where that multiple
    would be huge) plain 0..1 floats, compared with the EPS tolerance."""
    from math import lcm
    L = lcm(*range(1, MAXS + 1))
    return float(L * MAXS) if L <= 10 ** 6 else 1.0


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


def fav_run_ballots(u):
    return rank_ballots(u) + u.shape[1] * fav_ballots(u)


def app_run_ballots(u):
    return rank_ballots(u) + u.shape[1] * approval_ballots(u)


def veto_ballots(u):
    B = np.zeros_like(u)
    B[np.arange(u.shape[0]), u.argmin(axis=1)] = -1
    return B


def trunc_ballots(u):
    R, A = rank_ballots(u), approval_ballots(u)
    k = A.sum(axis=1, keepdims=True)
    return np.where(A > 0, R - (u.shape[1] - k) + 1, 0.0)   # top k get k ... 1


TRUNC3 = 3       # most candidates ranked on a "trunc3" ballot


def trunc3_ballots(u):
    C = u.shape[1]
    k = min(TRUNC3, C)
    R = rank_ballots(u)
    return np.where(R >= C - k, R - (C - k) + 1, 0.0)        # top k get k ... 1


def grade3_ballots(u):
    mn, mx = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    d = np.where(mx - mn > 1e-12, mx - mn, 1.0)
    return np.rint(2 * (u - mn) / d)


SINCERE = {"fav": fav_ballots, "rank": rank_ballots, "approval": approval_ballots, "score": score_ballots,
           "fav_run": fav_run_ballots, "app_run": app_run_ballots, "veto": veto_ballots, "weak": rank_ballots, "trunc": trunc_ballots, "trunc3": trunc3_ballots,
           "grade3": grade3_ballots}


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
# features it needs and decides from their sums. In convergent learning (strategy.learn) these sums
# are kept per poll sample and updated only for the group whose ballots change; in poll rounds
# (strategy.respond) each group's alternatives are evaluated against the polled sums. Neither
# recounts every ballot on every sample. All features are integer valued, so the sums are exact (except keepnorm
# for score ranges above 0..16, see _keepnorm_scale, where ties are found within EPS).

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


def _firsts_ex(B):
    """As _firsts, for truncated rankings: a ballot counts for its highest-ranked candidate in A only
    if it ranks one there (value > 0); otherwise it is exhausted and counts for nobody."""
    n, C = B.shape
    m = masks(C)
    Bm = np.where(m[None], B[:, None, :], -np.inf)
    out = np.zeros((n, 2 ** C, C))
    idx = Bm[:, 1:].argmax(axis=2)
    live = np.take_along_axis(Bm[:, 1:], idx[:, :, None], axis=2)[:, :, 0] > 0
    v, s_ = np.nonzero(live)
    out[v, s_ + 1, idx[v, s_]] = 1
    return out.reshape(n, -1)


def _lasts(B):
    """For every subset A: one-hot of the voter's lowest-ranked candidate in A (Coombs)."""
    n, C = B.shape
    m = masks(C)
    Bm = np.where(m[None], B[:, None, :], np.inf)
    out = np.zeros((n, 2 ** C, C))
    idx = Bm[:, 1:].argmin(axis=2)
    out[np.arange(n)[:, None], np.arange(1, 2 ** C)[None, :], idx] = 1
    return out.reshape(n, -1)


def _firsts_eq(B):
    """As _firsts, but every candidate tied at the top among A gets a whole vote (rankings with ties)."""
    n, C = B.shape
    m = masks(C)[None]
    Bm = np.where(m, B[:, None, :], -np.inf)
    return ((Bm == Bm.max(axis=2, keepdims=True)) & m).reshape(n, -1)


def _lcm_upto(C):
    out = 1
    for k in range(2, C + 1):
        out = out * k // np.gcd(out, k)
    return out


def _dowdall(B):
    """Dowdall points on a strict ranking, 1, 1/2, 1/3, ..., times lcm(1..C) so they are integers."""
    C = B.shape[1]
    pos = (C - 1) - B                                   # 0 = first place
    return _lcm_upto(C) // (pos.astype(int) + 1)


def _firstsecond(B):
    """One-hot of the (first, second) pair of a strict ranking, as a C x C matrix."""
    n, C = B.shape
    order = np.argsort(-B, axis=1, kind="stable")
    return _onehot(order[:, 0] * C + order[:, 1], C * C)


def _keepnorm(B):
    """For every subset K: the voter's scores among K stretched to 0..L·MAXS (integers; for 0..5 that is
    0..300 = 60 x 0..5), 0 outside K and for voters rating all of K equally. See _keepnorm_scale."""
    n, C = B.shape
    m = masks(C)[None]                                          # (1, 2^C, C)
    lo = np.where(m, B[:, None, :], np.inf).min(axis=2, keepdims=True)
    hi = np.where(m, B[:, None, :], -np.inf).max(axis=2, keepdims=True)
    d = hi - lo
    ok = m & (d > 0)
    out = np.where(ok, _keepnorm_scale() * (B[:, None, :] - np.where(ok, lo, 0)) / np.where(ok, d, 1), 0.0)
    return out.reshape(n, -1)


def _top(B):
    return B >= B.max(axis=1, keepdims=True)


FEATURES = {   # name: (per-voter feature (n, ...), shape of the sum given C)
    "n": (lambda B: np.ones((len(B), 1)), lambda C: ()),
    "sum": (lambda B: B, lambda C: (C,)),
    "first": (lambda B: _onehot(B.argmax(axis=1), B.shape[1]), lambda C: (C,)),
    "last": (lambda B: _onehot(B.argmin(axis=1), B.shape[1]), lambda C: (C,)),
    "gt": (lambda B: (B[:, :, None] > B[:, None, :]).reshape(len(B), -1), lambda C: (C, C)),
    "app": (lambda B: B >= B.shape[1], lambda C: (C,)),                       # fav_run / app_run: round-1 marks
    "runoff_gt": (lambda B: (lambda R: (R[:, :, None] > R[:, None, :]).reshape(len(B), -1))(np.mod(B, B.shape[1])),
                  lambda C: (C, C)),                                           # fav_run / app_run: runoff rankings
    "upper": (lambda B: B > MAXS / 2, lambda C: (C,)),                         # above the middle (0..5: 3-5)
    "maxs": (lambda B: B >= MAXS, lambda C: (C,)),                            # top score only
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
    "firsts_eq": (_firsts_eq, lambda C: (2 ** C, C)),
    "dowdall": (_dowdall, lambda C: (C,)),
    "firsts_ex": (_firsts_ex, lambda C: (2 ** C, C)),
    "lasts": (_lasts, lambda C: (2 ** C, C)),
    "firstsecond": (_firstsecond, lambda C: (C, C)),
    "ge1": (lambda B: B >= 1, lambda C: (C,)),
    "ge2": (lambda B: B >= 2, lambda C: (C,)),
    "eq0": (lambda B: B == 0, lambda C: (C,)),
}


def margins_of(s):
    return s["gt"] - s["gt"].T


def first_among(s, alive, key="firsts"):
    """First-choice counts among the alive candidates (in the order of alive)."""
    return s[key][int(sum(1 << c for c in alive))][alive]


# ---------- decisions from summed features ----------
def d_sum(s):
    return uniform_max(s["sum"])


def d_copeland(s):
    M = margins_of(s)
    sc = (M > 0).sum(axis=1) + 0.5 * ((M == 0).sum(axis=1) - 1)
    return uniform_max(sc.astype(float))


def d_irv(s, key="firsts"):
    C = s[key].shape[1]
    P = np.zeros(C)

    def go(alive, w):
        if len(alive) == 1:
            P[alive[0]] += w
            return
        first = first_among(s, alive, key)
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


def _common_total(g):
    """Grade counts as integers with the same total for every candidate. Real counts already are;
    estimated poll counts (strategy.respond) can be fractional, slightly negative, and differ in
    total, so they are clipped at 0 and scaled to the mean total (largest remainders)."""
    g = np.clip(np.asarray(g, float), 0, None)
    if np.allclose(g, np.rint(g)) and len(set(np.rint(g).sum(axis=1))) == 1:
        return np.rint(g).astype(int)
    N = max(1, int(round(g.sum(axis=1).mean())))
    out = np.zeros(g.shape, int)
    for c, row in enumerate(g):
        x = row * N / row.sum() if row.sum() > 0 else np.full(len(row), N / len(row))
        f = np.floor(x).astype(int)
        f[np.argsort(-(x - f), kind="stable")[:N - f.sum()]] += 1
        out[c] = f
    return out


def d_mj(s):
    """Majority judgment on 0..MAXS grades: compare median grades, removing medians to break ties."""
    h = _common_total(s["grades"])                        # h[c, g]: voters giving c grade g
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
    """The most approved candidate wins outright if more than half the voters approve it; otherwise
    top two by approvals, majority runoff, a tie to more approvals."""
    if s["app"].max() > s["n"] / 2:
        return uniform_max(s["app"])
    return d_approval_runoff(s)


def d_two_round(s):
    """Two-round election (fav_run ballots): a candidate marked by more than half the voters wins in
    round 1; otherwise the top two meet in a separate runoff, decided by the runoff rankings."""
    if s["app"].max() > s["n"] / 2:
        return uniform_max(s["app"])
    return d_top_two({"first": s["app"], "gt": s["runoff_gt"]})


def d_inverse_two_round(s):
    """Inverse-TTR: round 1 as TTR (most marks; the honest ballot marks the least liked). The runoff
    is a separate two-candidate vote, where voters vote by their real preference whatever the
    ballot means; the fixed runoff ranking of the Inverse-TTR ballot is reversed, so the count reads
    it back (transposed pairwise counts): the real majority decides the runoff."""
    return d_two_round({"n": s["n"], "app": s["app"], "runoff_gt": s["runoff_gt"].T})


def d_approval_two_round(s):
    """Approval primary, then a separate runoff of the top two (app_run ballots); a runoff tie goes
    to more approvals."""
    return d_approval_runoff({"app": s["app"], "gt": s["runoff_gt"]})


def d_approval_two_round_if_needed(s):
    """As d_approval_two_round, but a candidate approved by more than half the voters wins outright."""
    return d_approval_runoff_if_needed({"n": s["n"], "app": s["app"], "gt": s["runoff_gt"]})


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
    rest (lowest to 0, highest to the top score); the highest total wins."""
    tot, excess = s["sum"], s["excess"]                 # excess[a, b] = sum max(0, s_a - s_b)
    C = len(tot)
    defeated = (excess > tot[None, :] + EPS).any(axis=0)
    keep = np.flatnonzero(~defeated)
    if len(keep) == 0:              # impossible for real counts; can happen for estimated poll counts
        keep = np.arange(C)
    P = np.zeros(C)
    P[keep] = uniform_max(s["keepnorm"][int(sum(1 << c for c in keep))][keep])
    return P


# ---------- decisions (continued) ----------
def d_er_irv(s):
    """IRV on rankings with ties: every candidate tied at the top of a ballot (among the remaining)
    gets a whole vote, as in approval. Sincere ballots are strict; ties are learned."""
    return d_irv(s, "firsts_eq")


def d_black(s):
    """Condorcet winner if there is one, otherwise the Borda winner."""
    M = margins_of(s)
    cw = (M > 0).sum(axis=1) == len(M) - 1
    return cw.astype(float) if cw.any() else uniform_max(s["sum"])


def d_ranked_robin(s):
    """Ranked Robin (Equal Vote): most pairwise wins. Ties: greatest sum of margins against the other
    tied candidates, then against all candidates; further ties (the 3rd and 4th tiebreakers) are split."""
    M = margins_of(s)
    C = len(M)
    wins = (M > 0).sum(axis=1).astype(float)
    tied = np.flatnonzero(wins >= wins.max() - EPS)
    for score in (lambda t: M[np.ix_(t, t)].sum(axis=1), lambda t: M[t].sum(axis=1)):
        if len(tied) > 1:
            v = score(tied).astype(float)
            tied = tied[v >= v.max() - EPS]
    P = np.zeros(C)
    P[tied] = 1 / len(tied)
    return P


def d_nanson(s):
    """Nanson: repeatedly eliminate every candidate whose Borda score (among the remaining) is at or
    below the average; if that would eliminate all, they tie."""
    gt = s["gt"]
    alive = list(range(len(gt)))
    while len(alive) > 1:
        pts = gt[np.ix_(alive, alive)].sum(axis=1).astype(float)
        out = pts <= pts.mean() + EPS
        if out.all():
            break
        alive = [c for c, o in zip(alive, out) if not o]
    P = np.zeros(len(gt))
    P[alive] = 1 / len(alive)
    return P


def d_borda_er(s):
    """Borda on rankings with ties: a point for every candidate ranked strictly below (ties score
    nothing). On strict rankings the same as Borda."""
    return uniform_max(s["gt"].sum(axis=1).astype(float))





def _worst_wv(s):
    """Each candidate's worst defeat by winning votes, as a share of voters (0 if undefeated)."""
    gt = s["gt"].astype(float)
    np.fill_diagonal(gt, 0)
    return np.where(gt > gt.T, gt, 0.0).max(axis=0) / s["n"]


def d_mmwv_plus_score(s):
    """Score share (total / (n x top score)) minus worst winning-votes defeat share; highest wins."""
    return uniform_max(s["sum"] / (s["n"] * MAXS) - _worst_wv(s))


def d_mmwv_times_score(s):
    """Score share times (1 - worst winning-votes defeat share), as in MMPOWV; highest wins."""
    return uniform_max(s["sum"] / (s["n"] * MAXS) * (1 - _worst_wv(s)))


def d_ic_mars(s):
    """IC-MARS: as MARS (order by score total, bottom-two runoffs, each match decided by margin + score
    difference / top score), but a pairwise margin counts only beyond the votes rating both candidates
    at the top (tied at the top, as in Improved Condorcet Approval): a margin those votes could
    reverse counts 0. Less weight for votes, more for score."""
    M, tot, eq = margins_of(s), s["sum"], s["eqtop"]
    V = np.sign(M) * np.clip(np.abs(M) - eq, 0, None)
    return bottom_two_runoff(tot, lambda a, b: V[a, b] + (tot[a] - tot[b]) / MAXS)


def d_coombs(s):
    """Coombs: a candidate with more than half of the first choices among the remaining wins;
    otherwise the candidate with the most last places among the remaining is eliminated (ties split)."""
    C = s["firsts"].shape[1]
    P = np.zeros(C)

    def go(alive, w):
        first = first_among(s, alive)
        if len(alive) == 1 or first.max() > first.sum() / 2:
            win = np.flatnonzero(first == first.max()) if len(alive) > 1 else [0]
            for i in win:
                P[alive[i]] += w / len(win)
            return
        last = first_among(s, alive, "lasts")
        top = np.flatnonzero(last == last.max())
        for i in top:
            go([c for j, c in enumerate(alive) if j != i], w / len(top))

    go(list(range(C)), 1.0)
    return P


def d_kemeny(s):
    """Kemeny-Young: the ranking that agrees with the most pairwise preferences (sum over ordered pairs
    of voters preferring the earlier candidate); its top candidate wins. Dynamic programming over
    subsets: best(R) = max over c in R of [votes for c over the rest of R] + best(R without c).
    Candidates that head some optimal ranking split the win."""
    gt = s["gt"].astype(float)
    np.fill_diagonal(gt, 0)
    C = len(gt)
    m, layers = _kemeny_layers(C)
    G = gt @ m.T                                    # G[c, R]: votes for c over the members of R
    bits = 1 << np.arange(C)
    best = np.zeros(1 << C)
    for Rs in layers[:-1]:                          # subsets by size, smallest first
        inR = m[Rs].T                               # (C, len(Rs))
        sub = Rs[None, :] & ~bits[:, None]          # R without c
        best[Rs] = np.where(inR, G[:, Rs] + best[sub], -np.inf).max(axis=0)
    full = (1 << C) - 1
    return uniform_max(G[:, full] + best[full & ~bits])


_KEMENY = {}


def _kemeny_layers(C):
    """Subset masks and the subsets grouped by size (cached per number of candidates)."""
    if C not in _KEMENY:
        m = masks(C).astype(bool)
        size = m.sum(axis=1)
        _KEMENY[C] = (m, [np.flatnonzero(size == k) for k in range(1, C + 1)])
    return _KEMENY[C]


def d_dowdall(s):
    """Dowdall (Nauru): positional, 1, 1/2, 1/3, ... points by rank."""
    return uniform_max(s["dowdall"])


def smith_set(M):
    """Candidates in the Smith set: those that reach every candidate through 'not beaten by' steps."""
    R = M >= 0
    for k in range(len(M)):
        R = R | (R[:, [k]] & R[[k], :])
    return np.flatnonzero(R.all(axis=1))


def d_smith_score(s):
    """Smith//Score: the highest score total among the Smith set (pairwise from the ratings)."""
    sm = smith_set(margins_of(s))
    P = np.zeros(len(s["sum"]))
    P[sm] = uniform_max(s["sum"][sm])
    return P


def d_supplementary(s):
    """Supplementary vote: first and second preference count. The top two by first preferences go on;
    a ballot whose first preference is out counts for its second preference if that is a finalist."""
    first, FS = s["first"], s["firstsecond"]
    C = len(first)
    P = np.zeros(C)
    for a, b, w in top_two(first):
        others = [x for x in range(C) if x not in (a, b)]
        va = first[a] + FS[others, a].sum()
        vb = first[b] + FS[others, b].sum()
        x = va - vb
        if x > 0:
            P[a] += w
        elif x < 0:
            P[b] += w
        else:
            P[a] += w / 2
            P[b] += w / 2
    return P


def d_mca(s):
    """MCA (majority choice approval), MCA-A variant, on three grades: find the first grade level
    (good, then good or OK, then all) at which some candidate is rated that high by more than half
    the voters. One such candidate wins; among several, the most approved (good or OK) wins."""
    n, good, appr = s["n"], s["ge2"], s["ge1"]
    for level in (good, appr, np.full(len(good), n)):
        maj = level > n / 2
        if maj.sum() == 1:
            return maj.astype(float)
        if maj.sum() > 1:
            return uniform_max(np.where(maj, appr, -1.0))
    return uniform_max(appr)


def d_321(s):
    """3-2-1 voting on three grades: 3 semifinalists with the most 'good' (the third only if it has
    at least half as many as the first); of those, 2 finalists with the fewest 'bad'; the finalist
    rated above the other on more ballots wins. Ties at each step: higher score (good 2, OK 1),
    then split. (The rule against three semifinalists of one party does not apply here.)"""
    n, good, bad, score, M = s["n"], s["ge2"], s["eq0"], s["sum"], margins_of(s)
    C = len(good)
    span = 2 * n + 1                                    # score < span, so good dominates
    P = np.zeros(C)
    orders = orders_by(good * span + score)
    for order in orders:
        semis = list(order[:3])
        if len(semis) == 3 and good[semis[2]] < good[semis[0]] / 2:
            semis = semis[:2]
        v = np.full(C, -np.inf)
        v[semis] = -bad[semis] * span + score[semis]
        for a, b, w in top_two(v):
            runoff(P, M, a, b, w / len(orders), tiebreak=score)
    return P


def d_mmpo(s):
    """MMPO (minimax pairwise opposition): the winner's largest opposition, the most voters ranking
    some rival above it (whether or not that rival wins the pair), is smallest."""
    gt = s["gt"].astype(float)
    np.fill_diagonal(gt, 0)
    return uniform_max(-gt.max(axis=0))


def d_minimax_wv(s):
    """Minimax (winning votes): the winner's worst defeat, counted as the votes for the rival in that
    defeat, is smallest."""
    gt = s["gt"].astype(float)
    defeat = np.where(gt > gt.T, gt, 0.0)               # defeat[y, x]: votes of y over x if y beats x
    return uniform_max(-defeat.max(axis=0))


def d_mmpowv(s):
    """MMPOWV (proposed in this lab): pairwise opposition and winning votes multiplied. Each turned into
    a share between 0 (worst) and 1 (best): 1 - largest opposition / n (the most voters ranking some
    rival above the candidate) and 1 - worst defeat / n (votes for the rival in the candidate's worst
    defeat, 0 if undefeated); the highest product wins. Multiplying makes a weakness on either costly
    and is smooth, without thresholds for strategy to aim at. On strict rankings it orders candidates
    like Minimax, so it differs only once equal rankings are learned."""
    n = s["n"]
    gt = s["gt"].astype(float)
    np.fill_diagonal(gt, 0)
    wv = np.where(gt > gt.T, gt, 0.0).max(axis=0)
    po = gt.max(axis=0)
    return uniform_max((1 - wv / n) * (1 - po / n))


@dataclass
class Method:
    ballot: str
    decide: Callable = None          # decision from summed features (a dict name -> sum)
    feats: tuple = ()                # features the decision needs
    full: Callable = None            # tally on the full ballot matrix, for methods without features
    inverse: bool = False            # Inverse-* test methods: the sincere ballot is that of reversed preferences

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

    def pref(self, u):
        """The preferences a sincere ballot expresses: u, or reversed for the Inverse-* methods.
        Sincerity and strategy types are measured against these."""
        return -u if self.inverse else u

    def sincere(self, u):
        return SINCERE[self.ballot](self.pref(u))

    def honest(self, u):
        return self.tally(self.sincere(u))


M_ = Method
METHODS = {
    # condorcet
    "Ranked pairs": M_("rank", d_ranked_pairs, ("gt",)),
    "Schulze": M_("rank", d_schulze, ("gt",)),
    "Schulze-ER": M_("weak", d_schulze, ("gt",)),
    "Kemeny-Young": M_("rank", d_kemeny, ("gt",)),
    "Copeland": M_("rank", d_copeland, ("gt",)),
    "Ranked Robin": M_("weak", d_ranked_robin, ("gt",)),
    "Ranked Robin-SR": M_("rank", d_ranked_robin, ("gt",)),
    "SICT": M_("weak", d_sict, ("gt", "top", "eqtop", "eqbot")),
    "Condorcet-random-drop": M_("rank", full=t_condorcet_drop),
    # minimax
    "Minimax": M_("rank", d_minimax, ("gt",)),
    "Minimax-ER": M_("weak", d_minimax, ("gt",)),
    "MMWV": M_("weak", d_minimax_wv, ("gt",)),
    "MMPO": M_("weak", d_mmpo, ("gt",)),
    "MMPOWV": M_("weak", d_mmpowv, ("n", "gt")),
    # condorcet-hybrid
    "Black": M_("rank", d_black, ("sum", "gt")),
    "Baldwin": M_("rank", d_baldwin, ("gt",)),
    "Baldwin-ER": M_("weak", d_baldwin, ("gt",)),
    "Nanson": M_("rank", d_nanson, ("gt",)),
    "BTR-Borda": M_("rank", d_btr, ("sum", "gt")),
    "BTR-IRV": M_("rank", d_btr_irv, ("firsts", "gt")),
    "Benham": M_("rank", d_benham, ("firsts", "gt")),
    "Smith//Score": M_("score", d_smith_score, ("sum", "gt")),
    # plurality-based
    "Plurality": M_("fav", d_sum, ("sum",)),
    "TTR": M_("fav_run", d_two_round, ("n", "app", "runoff_gt")),
    "Supplementary vote": M_("rank", d_supplementary, ("first", "firstsecond")),
    "IRV": M_("rank", d_irv, ("firsts",)),
    "ER-IRV": M_("weak", d_er_irv, ("firsts_eq",)),
    "IRV-3": M_("trunc3", lambda s: d_irv(s, "firsts_ex"), ("firsts_ex",)),
    "Coombs": M_("rank", d_coombs, ("firsts", "lasts")),
    # approval-based
    "Approval": M_("approval", d_sum, ("sum",)),
    "Approval runoff": M_("app_run", d_approval_two_round, ("app", "runoff_gt")),
    "Approval cond. runoff": M_("app_run", d_approval_two_round_if_needed, ("n", "app", "runoff_gt")),
    "MDDA": M_("trunc", d_mdda, ("n", "gt", "pos")),
    # rated
    "Score": M_("score", d_sum, ("sum",)),
    "MJ": M_("score", d_mj, ("grades",)),
    "MCA": M_("grade3", d_mca, ("n", "ge2", "ge1")),
    "Reciprocal score": M_("score", d_reciprocal_score, ("fset", "fsetB")),
    # hybrid
    "STAR": M_("score", d_star, ("sum", "gt")),
    "BTR-score": M_("score", d_btr, ("sum", "gt")),
    "MARS": M_("score", d_mars, ("sum", "gt")),
    "IC-MARS": M_("score", d_ic_mars, ("sum", "gt", "eqtop")),
    "ARA": M_("score", ara("upper"), ("upper", "gt")),          # ratings above the middle (0..5: 3-5) count as approval
    "ARA-Top": M_("score", ara("maxs"), ("maxs", "gt")),        # only the top score counts as approval
    "Score+": M_("score", d_score_plus, ("sum", "excess", "keepnorm")),
    "321": M_("grade3", d_321, ("n", "ge2", "eq0", "sum", "gt")),
    "MMWV+Score": M_("score", d_mmwv_plus_score, ("n", "sum", "gt")),
    "MMWVxScore": M_("score", d_mmwv_times_score, ("n", "sum", "gt")),
    # positional
    "Borda": M_("rank", d_sum, ("sum",)),
    "Borda-ER": M_("weak", d_borda_er, ("gt",)),
    "Dowdall": M_("rank", d_dowdall, ("dowdall",)),
    # references
    "Random pair runoff": M_("rank", d_random_pair, ("n", "first", "gt")),
    "Anti-plurality": M_("veto", d_sum, ("sum",)),
    "TRASH": M_("rank", d_trash, ("first", "last", "gt")),
    # inverse: tests of the strategy model. Same ballot and count as the original, but the sincere
    # ballot expresses reversed preferences (name the least liked; approve those below the mean; rank
    # upside down), so honest voting elects (nearly) the worst candidate. Voters who learn to reverse
    # their ballot get the original method back.
    "Inverse-Plurality": M_("fav", d_sum, ("sum",), inverse=True),
    "Inverse-Approval": M_("approval", d_sum, ("sum",), inverse=True),
    "Inverse-Minimax": M_("rank", d_minimax, ("gt",), inverse=True),
    "Inverse-IRV": M_("rank", d_irv, ("firsts",), inverse=True),
    "Inverse-ER-IRV": M_("weak", d_er_irv, ("firsts_eq",), inverse=True),
    "Inverse-TTR": M_("fav_run", d_inverse_two_round, ("n", "app", "runoff_gt"), inverse=True),
    "Inverse-Borda": M_("rank", d_sum, ("sum",), inverse=True),
    "Inverse-Score": M_("score", d_sum, ("sum",), inverse=True),
}


# Method groups: every method belongs to exactly one (checked below). They are the panels of the
# chain plots and can be used as names wherever methods are selected (--methods, --exclude, --show,
# --hide, --track).
GROUPS = {
    "condorcet": ["Ranked pairs", "Schulze", "Schulze-ER", "Kemeny-Young", "Copeland", "Ranked Robin",
                  "Ranked Robin-SR", "SICT", "Condorcet-random-drop"],
    "minimax-family": ["Minimax", "Minimax-ER", "MMWV", "MMPO", "MMPOWV"],
    "condorcet-hybrid": ["Black", "Baldwin", "Baldwin-ER", "Nanson", "BTR-Borda", "BTR-IRV", "Benham", "Smith//Score"],
    "plurality-based": ["Plurality", "TTR", "Supplementary vote", "IRV", "ER-IRV", "IRV-3", "Coombs"],
    "approval-based": ["Approval", "Approval runoff", "Approval cond. runoff", "MDDA"],
    "rated": ["Score", "MJ", "MCA", "Reciprocal score"],
    "hybrid": ["STAR", "BTR-score", "MARS", "IC-MARS", "ARA", "ARA-Top", "Score+", "321", "MMWV+Score", "MMWVxScore"],
    "positional": ["Borda", "Borda-ER", "Dowdall"],
    "references": ["Random pair runoff", "Anti-plurality", "TRASH"],
    "inverse": ["Inverse-Plurality", "Inverse-Approval", "Inverse-Minimax", "Inverse-IRV", "Inverse-ER-IRV", "Inverse-TTR",
                "Inverse-Borda", "Inverse-Score"],
}
_grouped = [m for ms in GROUPS.values() for m in ms]
assert sorted(_grouped) == sorted(METHODS), "every method must be in exactly one group"
assert not {g.lower() for g in GROUPS} & {m.lower() for m in METHODS}, "group names must differ from method names"

# Comparison sets: selection names like the groups, across groups (not plot panels)
COMPARISONS = {
    "er-pairs": ["IRV", "ER-IRV", "Borda", "Borda-ER", "Baldwin", "Baldwin-ER", "Minimax",
                 "Minimax-ER", "MMWV", "Schulze", "Schulze-ER", "Ranked Robin-SR", "Ranked Robin"],
}


def group_of(method):
    """The group a method belongs to ("other" for names not in METHODS, e.g. from older CSVs)."""
    return next((g for g, ms in GROUPS.items() if method in ms), "other")


def select(include=None, exclude=None):
    """Method names: `include` starts from none, `exclude` from all. Names are methods, groups (see
    GROUPS) or comparison sets (COMPARISONS); matching ignores case."""
    lookup = {k.lower(): [k] for k in METHODS}
    lookup.update({g: ms for g, ms in {**GROUPS, **COMPARISONS}.items()})

    def resolve(names):
        bad = [x for x in names if x.lower() not in lookup]
        if bad:
            raise SystemExit(f"unknown method(s) or group(s): {', '.join(bad)}. Methods: {', '.join(METHODS)}. "
                             f"Groups: {', '.join(GROUPS)}. Comparisons: {', '.join(COMPARISONS)}")
        return {m for x in names for m in lookup[x.lower()]}

    chosen = resolve(include) if include else set(METHODS)
    chosen -= resolve(exclude or [])
    return [k for k in METHODS if k in chosen]
