"""Voting methods: exact tallies on ballot matrices, method groups, add-ons.

The ballot types (their encodings, sincere ballots, laziness) are defined in ballots.py; in short,
a ballot matrix B has one row per voter and one column per candidate:
  "fav"      one-hot: 1 for the named candidate
  "rank"     rank points: best C-1 ... worst 0. Truncated rankings: the k ranked candidates get
             C-1 ... C-k, the unranked ones 0 (tied last; a full ranking is k = C-1). Voters may
             truncate strategically, and lazy voters truncate their honest ballot (set_lazy)
  "strict"   as rank, but always a full ranking (the Inverse-* methods on rankings)
  "approval" 0/1
  "score"    0..MAXS (default 0..5; set_max_score changes the range)
  "fav_run"  two-round election, plurality first round: the sincere ranking as rank points
             (C-1 ... 0) for the separate runoff, plus C for the one candidate marked in round 1
  "app_run"  two-round election, approval first round: as fav_run, plus C for every approved
             candidate (any set). The ranking part is fixed: strategy changes only the round-1 marks,
             since in a separate two-candidate runoff voting for the preferred one is always best
  "score_run" two-round election, score first round: the runoff ranking as rank points (C-1 ... 0),
             plus C x the round-1 score (0..MAXS); as app_run, only the scores are the voter's choice
  "veto"     -1 for the one candidate voted against, 0 for all others
  "weak"     ranking with ties allowed: levels 0 ... C-1, higher is better, equal = tied
  "trunc"    truncated ranking: the k ranked candidates get k ... 1, unranked ones 0 (tied last)
  "trunc3"   as trunc, at most 3 ranked (IRV-3)
  "grade3"   three grades: 2 = good, 1 = OK, 0 = bad
Every tally returns exact winner probabilities; ties are split uniformly among
the tied candidates and counted exactly.

Sincere ballots from utilities u[v, c]:
  rank     - order by utility; with laziness (set_lazy) only the candidates the voter cares about
  strict   - order by utility, always complete
  approval - the candidates above the voter's personal threshold between mean and favourite (ballots.approves)
  score    - 0..MAXS, stretched to the voter's own utility range, rounded
  fav_run  - mark the favourite; ranking by utility
  app_run  - approvals as for approval; ranking by utility
  score_run - scores as for score; ranking by utility
  veto     - against the least liked candidate
  weak     - strict ranking by utility (ties are only used strategically); laziness as rank, the
             unranked at level 0
  trunc    - rank the candidates the voter would approve (ballots.approves), leave the rest unranked
  trunc3   - rank the 3 best-liked candidates (all, with 3 or fewer), leave the rest unranked
  (trunc and trunc3 with laziness: only those of them the voter also cares about)
Laziness: a voter ranks (approves, scores) a candidate only if its utility, stretched to 0..1
between the voter's least and most liked candidate, is at least the voter's own cut; scores and
grades of the others are left blank = 0. Cut 0 = full ballot, 1 = only the favourite. In a run each
voter's cut is drawn uniformly between 0 and LAZY (--lazy) per election (VOTER_LAZY, set_voter_lazy);
without per-voter cuts, LAZY is everyone's cut. Not for the Inverse-* methods.
  grade3   - 0..2, stretched to the voter's own utility range, rounded (as score, with 3 levels)
"""
import importlib.util
import math
from contextlib import contextmanager
import os
import sys
from dataclasses import dataclass
from itertools import permutations, product
from typing import Callable
import numpy as np

EPS = 1e-9
MAXS = 5          # top of the score range; change it with set_max_score, not directly
LAZY = 0.0        # laziness of honest rankings, 0..1; change it with set_lazy, not directly
VOTER_LAZY = None # per-voter laziness cuts of the current election (set_voter_lazy), or None
VOTER_APPROVAL = None  # per-voter approval draws of the current election (set_voter_approval), or None
APPROVAL_DEFAULT = 0.5  # the draw used without per-voter draws: the threshold halfway from mean to favourite


def set_lazy(x):
    """Laziness of honest ballots (rankings, approval, scores, grades: the ballot types with lazy=True in ballots.py): a voter ranks a candidate only
    if its utility, stretched to 0..1 between the voter's least and most liked candidate, is at least x
    (0 = full rankings, 1 = only the favourite; the favourite is always ranked). It is only the honest
    starting point: strategic voters may rank more or less. Call it in every process."""
    global LAZY
    if not 0 <= float(x) <= 1:
        raise ValueError("laziness must be between 0 and 1")
    LAZY = float(x)


def set_voter_lazy(cuts):
    """Per-voter laziness cuts for the current election (one per voter, 0..1), or None: everyone
    uses LAZY. run.py draws them uniformly between 0 and --lazy. Sincere ballots of the whole
    electorate then use each voter's own cut; call with None when the election is done."""
    global VOTER_LAZY
    VOTER_LAZY = None if cuts is None else np.asarray(cuts, float)


def lazy_of(voters):
    """The laziness cut for a group of voters (indices): their mean cut, or LAZY without per-voter cuts."""
    return LAZY if VOTER_LAZY is None else float(VOTER_LAZY[voters].mean())


def set_voter_approval(draws):
    """Per-voter approval draws for the current election (one per voter, 0..1), or None. A voter's
    honest approval threshold lies this far from their mean utility to their favourite (ballots.approves);
    run.py draws them uniformly. Call with None when the election is done."""
    global VOTER_APPROVAL
    VOTER_APPROVAL = None if draws is None else np.asarray(draws, float)


def approval_of(voters):
    """The approval draw for a group of voters (indices): their mean draw, or APPROVAL_DEFAULT."""
    return APPROVAL_DEFAULT if VOTER_APPROVAL is None else float(VOTER_APPROVAL[voters].mean())


def set_max_score(m):
    """Set the score range 0..m for every method on "score" ballots (Score, STAR, MJ, ARA, ...).
    grade3 methods (MCA, 321) keep their three grades. Call it in every process before tallying."""
    global MAXS
    if int(m) < 1:
        raise ValueError("max score must be at least 1")
    MAXS = int(m)


# Settings a method can fix for itself, instead of following the run's value:
#   Method("score", d_sum, ("sum",), settings={"max_score": 10})
# name -> (read the current value, set it). Convention: a method with a fixed setting is named
# Name-X after its value (Score-10), and the base version of a method does not fix it, so it follows
# --max-score and stays comparable across runs (and the max-score sweep keeps working).
SETTINGS = {"max_score": (lambda: MAXS, set_max_score)}


@contextmanager
def applied(settings):
    """Within the block, the given settings (dict name -> value) are in force; afterwards the previous
    values come back. Each process runs one method at a time, so this is safe in the worker processes."""
    if not settings:
        yield
        return
    old = {k: SETTINGS[k][0]() for k in settings}
    try:
        for k, v in settings.items():
            SETTINGS[k][1](v)
        yield
    finally:
        for k, v in old.items():
            SETTINGS[k][1](v)


def _keepnorm_scale():
    """Integer scale for _keepnorm: a common multiple of every possible range 1..MAXS, so the
    stretched scores stay integers and sums are exact; for large ranges (where that multiple
    would be huge) plain 0..1 floats, compared with the EPS tolerance."""
    from math import lcm
    L = lcm(*range(1, MAXS + 1))
    return float(L * MAXS) if L <= 10 ** 6 else 1.0


# ---------- sincere ballots: in ballots.py ----------
TRUNC3 = 3       # most candidates ranked on a "trunc3" ballot

from ballots import BALLOTS  # noqa: E402


# ---------- helpers ----------
def margins(B):
    """Pairwise margins: M[a, b] = #(a rated above b) - #(b rated above a)."""
    gt = (B[:, :, None] > B[:, None, :]).sum(axis=0)
    return gt - gt.T


def uniform_max(v):
    top = v >= v.max() - EPS
    return top / top.sum()


TIE_MAX_ORDERS = 720


def orders_by(values):
    """All orderings high-to-low consistent with values (every order of tied groups); with more than
    TIE_MAX_ORDERS of them (from 7 candidates tied), that many fixed random ones."""
    order = np.argsort(-values, kind="stable")
    if np.all(np.diff(values[order]) < -1e-8):   # no ties: exactly one order
        return [list(order)]
    vals = np.unique(np.round(values, 9))[::-1]
    groups = [list(np.flatnonzero(np.abs(values - x) < 1e-8)) for x in vals]
    n = 1
    for g in groups:
        n *= math.factorial(len(g))
        if n > TIE_MAX_ORDERS:
            r = np.random.default_rng(0)
            return [[g[i] for g in groups for i in r.permutation(len(g))] for _ in range(TIE_MAX_ORDERS)]
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
    """For every subset A of candidates: one-hot of the voter's highest-ranked candidate in A. Used for
    elimination methods: first choices among the remaining. Truncated rankings: a ballot whose
    candidates left in A are all unranked (0, two or more tied) is exhausted and counts for nobody."""
    n, C = B.shape
    m = masks(C)
    Bm = np.where(m[None], B[:, None, :], -np.inf)             # (n, 2^C, C)
    out = np.zeros((n, 2 ** C, C))
    top = Bm[:, 1:].max(axis=2, keepdims=True)
    idx = Bm[:, 1:].argmax(axis=2)
    ntop = (Bm[:, 1:] == top).sum(axis=2)
    live = (top[:, :, 0] > 0) | (ntop == 1)
    v, s_ = np.nonzero(live)
    out[v, s_ + 1, idx[v, s_]] = 1
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
    """For every subset A: the voter's lowest-ranked candidate in A (Coombs). Candidates tied at the
    bottom (unranked on a truncated ranking) share the last place equally."""
    n, C = B.shape
    m = masks(C)[None]
    Bm = np.where(m, B[:, None, :], np.inf)
    low = (Bm == Bm.min(axis=2, keepdims=True)) & m
    low[:, 0] = False
    return (low / np.maximum(low.sum(axis=2, keepdims=True), 1)).reshape(n, -1)


def _firsts_eq(B):
    """As _firsts, but every candidate tied at the top among A gets a whole vote (rankings with ties).
    Candidates at level 0 tied with others are unranked (truncation): a ballot with only those left
    in A is exhausted, as on a truncated ranking, rather than giving each a whole vote."""
    n, C = B.shape
    m = masks(C)[None]
    Bm = np.where(m, B[:, None, :], -np.inf)
    top = Bm.max(axis=2, keepdims=True)
    at = (Bm == top) & m
    live = (top > 0) | (at.sum(axis=2, keepdims=True) == 1)
    return (at & live).reshape(n, -1)


def _coalitions(B, strict):
    """For every subset S of candidates: 1 if the voter is solidly committed to S (strict: every
    candidate in S ranked strictly above every candidate outside, DSC) or acquiesces to it (not
    strict: no candidate outside ranked strictly above one inside, DAC). The empty set counts 0."""
    n, C = B.shape
    m = masks(C)[None]
    lo_in = np.where(m, B[:, None, :], np.inf).min(axis=2)
    hi_out = np.where(~m, B[:, None, :], -np.inf).max(axis=2)
    out = (lo_in > hi_out) if strict else (lo_in >= hi_out)
    out[:, 0] = False
    return out


def _levelled(B):
    """STLR runoff scores: L[a, b] = the voter's score for a, rescaled so that the higher of a and b
    gets the top score (a's score x MAXS / max(a, b)); 0 if the voter gives both 0."""
    x = np.maximum(B[:, :, None], B[:, None, :])
    return np.where(x > 0, B[:, :, None] * MAXS / np.where(x > 0, x, 1), 0.0).reshape(len(B), -1)


def _nscore_sub(B):
    """For every subset S: the voter's scores among S stretched to the full range 0..MAXS (lowest in S
    gets 0, highest MAXS; all 0 if the voter rates S all the same), 0 outside S."""
    n, C = B.shape
    m = masks(C)[None]
    lo = np.where(m, B[:, None, :], np.inf).min(axis=2, keepdims=True)
    hi = np.where(m, B[:, None, :], -np.inf).max(axis=2, keepdims=True)
    span = hi - lo
    out = np.where(m & (span > 0), (B[:, None, :] - lo) * MAXS / np.where(span > 0, span, 1), 0.0)
    return out.reshape(n, -1)


def _soft_margins(B, frac):
    """Soft pairwise margins: per voter, the score difference a - b clipped to +-frac x MAXS and divided
    by that, so a difference of frac of the range or more counts as a full vote for one side and smaller
    ones count in proportion. frac -> 0 gives majority margins, frac = 1 score differences."""
    k = frac * MAXS
    return (np.clip(B[:, :, None] - B[:, None, :], -k, k) / k).reshape(len(B), -1)


def _diff_hist(B):
    """Per pair (a, b): one-hot of the integer score difference a - b, shifted to 0 .. 2 MAXS
    (score ballots are integers), so clipped sums can be formed for any clip after counting."""
    d = np.rint(B[:, :, None] - B[:, None, :]).astype(int) + MAXS
    return (d[:, :, :, None] == np.arange(2 * MAXS + 1)).reshape(len(B), -1)


def _lcm_upto(C):
    out = 1
    for k in range(2, C + 1):
        out = out * k // np.gcd(out, k)
    return out


def _dowdall(B):
    """Dowdall points, 1, 1/2, 1/3, ..., times lcm(1..C) so they are integers; unranked candidates on a
    truncated ranking (0, two or more tied) get 0."""
    C = B.shape[1]
    pos = (C - 1) - B                                   # 0 = first place
    pts = _lcm_upto(C) // (pos.astype(int) + 1)
    return np.where(_unranked(B), 0, pts)


def _unranked(B):
    """Truncated rankings: the candidates at 0 when two or more share it (a lone 0 is the last place of
    a full ranking)."""
    z = B == 0
    return z & (z.sum(axis=1, keepdims=True) > 1)


def _firstsecond(B):
    """One-hot of the (first, second) pair of a ranking, as a C x C matrix. A truncated ranking with
    only one candidate ranked has no second preference: (first, first)."""
    n, C = B.shape
    order = np.argsort(-B, axis=1, kind="stable")
    second = np.where(_unranked(B)[np.arange(n), order[:, 1]], order[:, 0], order[:, 1])
    return _onehot(order[:, 0] * C + second, C * C)


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


def _levels(B):
    """Score ballots read as levels: per voter, for each level j = 0 .. MAXS-1 (score MAXS - j), 1
    where the candidate has that score (score 0 is no level). Shape (n, MAXS, C)."""
    return np.stack([np.isclose(B, MAXS - j) for j in range(MAXS)], axis=1).astype(float)


FEATURES = {   # name: (per-voter feature (n, ...), shape of the sum given C)
    "n": (lambda B: np.ones((len(B), 1)), lambda C: ()),
    "sum": (lambda B: B, lambda C: (C,)),
    "first": (lambda B: _onehot(B.argmax(axis=1), B.shape[1]), lambda C: (C,)),
    "last": (lambda B: (lambda lo: lo / lo.sum(axis=1, keepdims=True))(B == B.min(axis=1, keepdims=True)),
             lambda C: (C,)),                                                 # ties at the bottom share
    "gt": (lambda B: (B[:, :, None] > B[:, None, :]).reshape(len(B), -1), lambda C: (C, C)),
    "app": (lambda B: B >= B.shape[1], lambda C: (C,)),                       # fav_run / app_run: round-1 marks
    "run_sum": (lambda B: np.floor_divide(B, B.shape[1]), lambda C: (C,)),   # score_run: round-1 scores
    "runoff_gt": (lambda B: (lambda R: (R[:, :, None] > R[:, None, :]).reshape(len(B), -1))(np.mod(B, B.shape[1])),
                  lambda C: (C, C)),                                           # fav_run / app_run: runoff rankings
    "levels": (_levels, lambda C: (MAXS, C)),                                  # DMS: voters per score level
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
    "solid": (lambda B: _coalitions(B, True), lambda C: (2 ** C,)),
    "acquiesce": (lambda B: _coalitions(B, False), lambda C: (2 ** C,)),
    "levelled": (_levelled, lambda C: (C, C)),
    "nscore_sub": (_nscore_sub, lambda C: (2 ** C, C)),
    "diffhist": (_diff_hist, lambda C: (C, C, 2 * MAXS + 1)),
    **{f"soft{int(f * 100)}": ((lambda f: lambda B: _soft_margins(B, f))(f), lambda C: (C, C)) for f in (0.2, 0.4)},
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
    r = np.rint(g)
    tot = r.sum(axis=1)
    if np.all(np.abs(g - r) <= 1e-8 + 1e-5 * np.abs(r)) and np.all(tot == tot[0]):   # as np.allclose(g, r)
        return r.astype(int)
    rs = g.sum(axis=1)
    N = max(1, int(round(rs.mean())))
    pos = rs > 0
    x = np.where(pos[:, None], g * N / np.where(pos, rs, 1)[:, None], N / g.shape[1])
    f = np.floor(x).astype(int)
    rank = np.argsort(-(x - f), axis=1, kind="stable")              # largest remainders first, per row
    k = N - f.sum(axis=1)                                             # how many get one more
    f[np.arange(len(f))[:, None], rank] += np.arange(g.shape[1])[None, :] < k[:, None]
    return f


def d_mj(s):
    """Majority judgment on 0..MAXS grades: compare median grades, removing medians to break ties."""
    h = _common_total(s["grades"])                        # h[c, g]: voters giving c grade g
    order = mj_order(int(h[0].sum()))
    # seq[c, j]: c's grade at sorted position order[j] (the number of grades whose cumulative count is
    # at most that position); the best sequence in lexicographic order wins
    seq = (np.cumsum(h, axis=1)[:, None, :] <= order[None, :, None]).sum(axis=2)
    alive = np.arange(len(h))
    for j in range(seq.shape[1]):
        col = seq[alive, j]
        alive = alive[col == col.max()]
        if len(alive) == 1:
            break
    top = np.zeros(len(h))
    top[alive] = 1.0
    return top / top.sum()


def d_mav(s):
    """Majority Approval Voting (graded Bucklin) on 0..MAXS grades: votes are counted grade by grade
    from the top. At the first grade where some candidate is graded at least that high by a majority,
    the candidate with a majority and the most votes at the grade above wins (at the top grade: the
    largest majority); remaining ties by the votes at this grade, then shared."""
    h = _common_total(s["grades"])                        # h[c, g]: voters giving c grade g
    n = h[0].sum()
    at_least = np.cumsum(h[:, ::-1], axis=1)[:, ::-1]     # at_least[c, g]: voters giving c grade g or more
    top = h.shape[1] - 1
    for g in range(top, -1, -1):
        maj = at_least[:, g] > n / 2
        if maj.any():
            above = at_least[:, g + 1] if g < top else at_least[:, g]
            return uniform_max(np.where(maj, above * (n + 1) + at_least[:, g], -1.0))
    return uniform_max(at_least[:, 0].astype(float))


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
    """BTR (BTR-score, BTR-Borda): order by total (score or Borda points), bottom-two pairwise runoffs
    climbing up; a pairwise tie goes to the candidate higher in that order (since 2026-10-26; before, a
    tie was split evenly). With strict pairwise results BTR-score is the same as Smith//Score unless the
    Smith set has 4 or more members: a candidate outside the Smith set loses every match against one
    inside, and in a 3-cycle the top-scored member beats whichever of the other two survives their
    match. The tie rule extends this to pairwise ties, where Smith//Score likewise decides by score."""
    M = margins_of(s)
    # climb(cur, ch): cur comes from below, ch is the next higher in the order
    return bottom_two_runoff(s["sum"], lambda a, b: M[a, b] if M[a, b] != 0 else -1)


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


def d_mampo(s):
    """Majority Approval, Minimum Pairwise Opposition (Kevin Venzke): ranked = approved. If nobody is
    approved by more than half the voters, the most approved wins. Otherwise, among those approved by
    more than half, the one whose largest pairwise opposition (most voters ranking some rival above
    it) is smallest wins; ties share."""
    app = s["pos"]
    maj = app > s["n"] / 2
    if not maj.any():
        return uniform_max(app)
    opp = s["gt"].max(axis=0).astype(float)                # gt[y, x]: voters ranking y above x
    return uniform_max(np.where(maj, -opp, -np.inf))


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
    for g in groups:                                   # exact integers: np.prod overflowed from 21 tied pairs
        n_orders *= math.factorial(len(g))
        if n_orders > RP_MAX_ORDERS:
            break
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


def d_score_two_round(s):
    """Score runoff: the two highest score totals (round 1) meet in a separate runoff, decided by the
    runoff rankings (score_run ballots); a runoff tie goes to the higher score total."""
    tot, M = s["run_sum"], margins_of({"gt": s["runoff_gt"]})
    P = np.zeros(len(tot))
    for a, b, w in top_two(tot):
        runoff(P, M, a, b, w, tiebreak=tot)
    return P


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


# ---------- coalitions, STLR, iterated normalized score ----------
MAX_TIED_ORDERS = 5    # equal-strength coalitions: average over their orders up to this many conflicting ones


def _bits(x):
    return [c for c in range(x.bit_length()) if x >> c & 1]


def descending_coalitions(strength, C):
    """DSC / DAC: consider candidate sets from the strongest down; each makes every candidate outside it
    ineligible, unless that would leave none (then it is skipped); the last eligible candidate wins.
    Sets of equal strength whose order matters (each would change the eligible set, and not all can be
    applied) are applied in every order with equal weight (at most MAX_TIED_ORDERS of them; beyond that,
    in the order of their index). Several still eligible at the end share the win."""
    from itertools import permutations as perms
    full = (1 << C) - 1
    order = [S for S in np.argsort(-strength, kind="stable") if strength[S] > 0 and S != full]
    groups = []
    for S in order:
        if groups and abs(strength[S] - strength[groups[-1][0]]) < EPS:
            groups[-1].append(int(S))
        else:
            groups.append([int(S)])
    P = np.zeros(C)

    def go(elig, gi, w):
        while gi < len(groups) and bin(elig).count("1") > 1:
            live = [S for S in groups[gi] if (S & elig) not in (0, elig)]
            gi += 1
            if not live:
                continue
            if len(live) == 1:
                elig &= live[0]
                continue
            orders = list(perms(live)) if len(live) <= MAX_TIED_ORDERS else [live]
            for o in orders:
                e = elig
                for S in o:
                    if S & e:
                        e &= S
                go(e, gi, w / len(orders))
            return
        for c in _bits(elig):
            P[c] += w / bin(elig).count("1")

    go(full, 0, 1.0)
    return P


def d_dsc(s):
    """Descending Solid Coalitions."""
    st = s["solid"]
    return descending_coalitions(st, int(np.log2(len(st))))


def d_dac(s):
    """Descending Acquiescing Coalitions."""
    st = s["acquiesce"]
    return descending_coalitions(st, int(np.log2(len(st))))


def d_stlr(s):
    """STLR: the two highest score totals go on; in the runoff each ballot's scores for the two are
    rescaled so that the preferred one gets the top score (a ballot giving both 0 counts for neither);
    a runoff tie goes to the higher score total."""
    tot = s["sum"]
    L = s["levelled"]
    M = L - L.T
    P = np.zeros(len(tot))
    for a, b, w in top_two(tot):
        runoff(P, M, a, b, w, tiebreak=tot)
    return P


def depth_values(cnt, scored=True, power=1.0):
    """DMS: cnt (MAXS, C) voters per level (levels feature). Values per depth k = 1 .. MAXS (rows: the
    top k levels) and candidate: scored, the points from those levels divided by the points of the
    levels counted (each level weighted by its points to the power `power`); not scored, the voters in
    those levels divided by k."""
    M = cnt.shape[0]
    k = np.arange(1, M + 1)[:, None]
    if not scored:
        return np.cumsum(cnt, axis=0) / k
    w = ((M - np.arange(M)) ** power)[:, None]             # level weights: points M, M-1, ..., 1
    return np.cumsum(w * cnt, axis=0) / np.cumsum(w, axis=0)


def leximin_winners(V):
    """V: (depths, C). Candidates whose values sorted ascending are lexicographically highest."""
    S = np.sort(V, axis=0)
    alive = np.arange(V.shape[1])
    for row in S:
        alive = alive[row[alive] >= row[alive].max() - EPS]
        if len(alive) == 1:
            break
    return alive


def d_dms(s):
    """DMS (depth maximin, scores): for every depth k (the top k score levels), each candidate's points
    from those levels divided by the points of the levels counted; the highest lowest value over the
    depths wins, ties by the next lowest (leximin), still tied: split."""
    win = leximin_winners(depth_values(s["levels"]))
    P = np.zeros(s["levels"].shape[1])
    P[win] = 1 / len(win)
    return P


def d_esr(s):
    """ESR (expanding support runoff): for every depth k (the top k score levels), the candidates with
    the most score points from those levels (all tied for most at that depth) join the set; at the
    full depth that is the score total, so the Score winner is always in it. A set of one wins
    outright; otherwise every ballot is stretched to the full range over the set (lowest in the set 0,
    highest the top score, not rounded; a ballot rating the whole set alike counts for nobody) and
    the highest total wins (ties split)."""
    cnt, sub = s["levels"], s["nscore_sub"]
    w = (cnt.shape[0] - np.arange(cnt.shape[0]))[:, None]     # level points: MAXS, MAXS-1, ..., 1
    A = np.cumsum(w * cnt, axis=0)                            # points from the top k levels, per depth
    sup = sorted({int(c) for row in A for c in np.flatnonzero(row >= row.max() - EPS)})
    P = np.zeros(cnt.shape[1])
    if len(sup) == 1:
        P[sup[0]] = 1.0
        return P
    pts = sub[int(sum(1 << c for c in sup))][sup]
    win = [sup[i] for i in np.flatnonzero(pts >= pts.max() - EPS)]
    P[win] = 1 / len(win)
    return P


def d_iterated_normalized_score(s, key="nscore_sub"):
    """Score, but each round every ballot is stretched to the full range over the remaining candidates;
    the lowest total is eliminated (ties for last split the win), until one is left."""
    sub = s[key]
    C = sub.shape[1]
    P = np.zeros(C)

    def go(alive, w):
        if len(alive) == 1:
            P[alive[0]] += w
            return
        pts = sub[int(sum(1 << c for c in alive))][alive]
        low = np.flatnonzero(pts <= pts.min() + EPS)
        for i in low:
            go([c for j, c in enumerate(alive) if j != i], w / len(low))

    go(list(range(C)), 1.0)
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


def d_copeland_margins(s):
    """Copeland//Margins: most pairwise wins. Ties: greatest sum of margins against the other tied
    candidates, then against all candidates; further ties are split. Equal Vote's earlier definition
    of Ranked Robin (until 2026-10-19 the lab's name for it); Equal Vote now uses "Ranked Robin" for
    any method electing the Condorcet winner, with a tiebreaker of the jurisdiction's choice."""
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


def d_mars_t(s):
    """MARS-T: as MARS, but each pairwise margin is scaled by the share of ballots that score the two
    candidates differently (1 - t, t = share scoring them equally): pairs that many voters do not tell
    apart are decided more by the score difference."""
    M, tot = margins_of(s), s["sum"]
    t = s["diffhist"][:, :, MAXS].astype(float) / s["n"]
    V = M * (1 - t)
    return bottom_two_runoff(tot, lambda a, b: V[a, b] + (tot[a] - tot[b]) / MAXS)


def soft_minimax(key):
    """Soft Minimax: each candidate's worst soft margin (see _soft_margins) against any rival; the best
    worst margin wins, ties share. Between Minimax (sharp margins) and Score (score differences: the
    worst difference of a total is against the runner-up, so the highest total wins)."""
    def d(s):
        D = s[key].astype(float)
        np.fill_diagonal(D, np.inf)
        return uniform_max(D.min(axis=1))
    return d


def _soft_minimax_clip(H, k):
    """Minimax on soft margins from difference counts H (C, C, 2 MAXS + 1) with a clip k[a, b] (points)
    per pair: sum of clip(difference, -k, k) / k; a pair with k = 0 (both always 0) counts 0."""
    x = np.arange(H.shape[2]) - MAXS
    kk = np.maximum(k, 1e-9)[:, :, None]
    D = (H * np.clip(x[None, None, :], -kk, kk)).sum(axis=2) / kk[:, :, 0]
    D = np.where(k > 0, D, 0.0)
    np.fill_diagonal(D, np.inf)
    return uniform_max(D.min(axis=1))


def d_soft_level(s):
    """Soft Minimax-L: the clip for a pair is the mean of the two candidates' average scores (A averages
    3, B 2: clip 2.5 points). Pairs of well-liked candidates are compared more by score difference,
    pairs of little-liked ones more by majority vote."""
    H = s["diffhist"].astype(float)
    avg = s["sum"] / s["n"]
    return _soft_minimax_clip(H, (avg[:, None] + avg[None, :]) / 2)


GEOMEDIAN_STEPS = 20     # Weiszfeld iterations at most; stop earlier when no coordinate moves by GEOMEDIAN_TOL
GEOMEDIAN_TOL = 1e-3     # (checked: the same winner as 500 steps to 1e-10 in 2000 random poll-sized profiles)


def t_geometric_median(B):
    """Geometric median of the score ballots (the point with the smallest total Euclidean distance to all
    ballots, as vectors over the candidates), by at most GEOMEDIAN_STEPS Weiszfeld iterations from the
    mean; the candidate with the highest coordinate wins. The mean would be Score; the coordinate-wise
    median is a median grade. Each ballot pulls the median toward itself with the same force however
    far away it is, so exaggerating a ballot in the same direction adds no pull."""
    x = B.mean(axis=0)
    for _ in range(GEOMEDIAN_STEPS):
        d = np.linalg.norm(B - x, axis=1)
        w = 1.0 / np.maximum(d, 1e-9)
        nx = (w[:, None] * B).sum(axis=0) / w.sum()
        if np.abs(nx - x).max() < GEOMEDIAN_TOL:
            x = nx
            break
        x = nx
    return uniform_max(np.round(x, 6))


def d_soft_diff(s):
    """Soft Minimax-D: the clip for a pair is 1 point plus the difference of the two candidates' average
    scores: pairs close in average score are compared by majority vote, pairs far apart more by score
    difference."""
    H = s["diffhist"].astype(float)
    avg = s["sum"] / s["n"]
    return _soft_minimax_clip(H, np.abs(avg[:, None] - avg[None, :]) + 1)


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
    settings: dict = None            # settings this method fixes for itself (SETTINGS), e.g. {"max_score": 10}

    def applied(self):
        """Context in which this method's own settings are in force (none: the run's settings).
        run.py runs all of a method's work in it; tally and sincere also use it, for direct calls."""
        return applied(self.settings)

    def features(self, B):
        """Per-voter features, concatenated: (n, d)."""
        return np.concatenate([FEATURES[f][0](B).reshape(len(B), -1).astype(float) for f in self.feats], axis=1)

    def _layout(self, C):
        """(name, start, end, shape) of each feature in the concatenated vector, cached per C and score
        range (some shapes depend on MAXS)."""
        cache = self.__dict__.setdefault("_layouts", {})
        key = (C, MAXS)
        if key not in cache:
            out, i = [], 0
            for f in self.feats:
                shape = FEATURES[f][1](C)
                k = int(np.prod(shape))
                out.append((f, i, i + k, shape))
                i += k
            cache[key] = out
        return cache[key]

    def unpack(self, S):
        """Summed features (d,) -> dict name -> array of its proper shape."""
        return {f: S[a:b].reshape(shape) if shape else S[a] for f, a, b, shape in self._layout(self._C)}

    def decide_many(self, S, C):
        """Winner probabilities for each row of summed features S (samples, d)."""
        self._C = C
        return np.array([self.decide(self.unpack(x)) for x in S])

    def tally(self, B):
        if self.settings:
            with self.applied():
                return self._tally(B)
        return self._tally(B)

    def _tally(self, B):
        if self.full is not None:
            return self.full(B)
        return self.decide_many(self.features(B).sum(axis=0, keepdims=True), B.shape[1])[0]

    def pref(self, u):
        """The preferences a sincere ballot expresses: u, or reversed for the Inverse-* methods.
        Sincerity and strategy types are measured against these."""
        return -u if self.inverse else u

    def sincere(self, u, lazy=None, approval=None):
        """Sincere ballots. lazy: the cut(s) to use; None = each voter's own (VOTER_LAZY, which needs
        one row of u per voter) or LAZY. approval: the approval draw(s) for ballots with an approval
        threshold (ballots.approves); None = each voter's own (VOTER_APPROVAL) or APPROVAL_DEFAULT."""
        b = BALLOTS[self.ballot]
        if lazy is None:
            lazy = LAZY if VOTER_LAZY is None else VOTER_LAZY
            if VOTER_LAZY is not None and len(VOTER_LAZY) != u.shape[0]:
                raise ValueError("per-voter laziness needs one row per voter; pass lazy= for other rows")
        if approval is None:
            approval = APPROVAL_DEFAULT if VOTER_APPROVAL is None else VOTER_APPROVAL
            if VOTER_APPROVAL is not None and len(VOTER_APPROVAL) != u.shape[0]:
                raise ValueError("per-voter approval draws need one row per voter; pass approval= for other rows")
        with self.applied():
            lz = lazy if b.lazy and not self.inverse else 0.0                  # Inverse-*: never lazy
            if b.approval:
                return b.sincere(self.pref(u), lz, approval)
            return b.sincere(self.pref(u), lz)

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
    "Copeland//Margins-ER": M_("weak", d_copeland_margins, ("gt",)),
    "Copeland//Margins": M_("rank", d_copeland_margins, ("gt",)),
    "SICT": M_("weak", d_sict, ("gt", "top", "eqtop", "eqbot")),
    "Condorcet-random-drop": M_("rank", full=t_condorcet_drop),
    # minimax
    "Minimax": M_("rank", d_minimax, ("gt",)),
    "Minimax-ER": M_("weak", d_minimax, ("gt",)),
    "MMWV": M_("weak", d_minimax_wv, ("gt",)),
    "MMPO": M_("weak", d_mmpo, ("gt",)),
    "MMPOWV": M_("weak", d_mmpowv, ("n", "gt")),
    # condorcet-based
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
    "DSC": M_("weak", d_dsc, ("solid",)),
    "DAC": M_("weak", d_dac, ("acquiesce",)),
    "Coombs": M_("rank", d_coombs, ("firsts", "lasts")),
    # approval-based
    "Approval": M_("approval", d_sum, ("sum",)),
    "Approval runoff": M_("app_run", d_approval_two_round, ("app", "runoff_gt")),
    "Approval cond. runoff": M_("app_run", d_approval_two_round_if_needed, ("n", "app", "runoff_gt")),
    "MDDA": M_("trunc", d_mdda, ("n", "gt", "pos")),
    "MAMPO": M_("trunc", d_mampo, ("n", "gt", "pos")),
    # rated
    "Score": M_("score", d_sum, ("sum",)),
    "MJ": M_("score", d_mj, ("grades",)),
    "MAV": M_("score", d_mav, ("grades",)),
    "MCA": M_("grade3", d_mca, ("n", "ge2", "ge1")),
    "Reciprocal score": M_("score", d_reciprocal_score, ("fset", "fsetB")),
    "INS": M_("score", d_iterated_normalized_score, ("nscore_sub",)),
    "DMS": M_("score", d_dms, ("levels",)),
    "ESR": M_("score", d_esr, ("levels", "nscore_sub")),
    "Geometric median": M_("score", full=t_geometric_median),
    # hybrid
    "STAR": M_("score", d_star, ("sum", "gt")),
    "Score runoff": M_("score_run", d_score_two_round, ("run_sum", "runoff_gt")),
    "BTR-score": M_("score", d_btr, ("sum", "gt")),
    "MARS": M_("score", d_mars, ("sum", "gt")),
    "IC-MARS": M_("score", d_ic_mars, ("sum", "gt", "eqtop")),
    "MARS-T": M_("score", d_mars_t, ("n", "sum", "gt", "diffhist")),
    "ARA": M_("score", ara("upper"), ("upper", "gt")),          # ratings above the middle (0..5: 3-5) count as approval
    "ARA-Top": M_("score", ara("maxs"), ("maxs", "gt")),        # only the top score counts as approval
    "Score+": M_("score", d_score_plus, ("sum", "excess", "keepnorm")),
    "321": M_("grade3", d_321, ("n", "ge2", "eq0", "sum", "gt")),
    "MMWV+Score": M_("score", d_mmwv_plus_score, ("n", "sum", "gt")),
    "MMWVxScore": M_("score", d_mmwv_times_score, ("n", "sum", "gt")),
    "STLR": M_("score", d_stlr, ("sum", "levelled")),
    "Soft Minimax": M_("score", soft_minimax("soft20"), ("soft20",)),
    "Soft Minimax-40": M_("score", soft_minimax("soft40"), ("soft40",)),
    "Soft Minimax-L": M_("score", d_soft_level, ("n", "sum", "diffhist")),
    "Soft Minimax-D": M_("score", d_soft_diff, ("n", "sum", "diffhist")),
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
    "Inverse-Minimax": M_("strict", d_minimax, ("gt",), inverse=True),
    "Inverse-IRV": M_("strict", d_irv, ("firsts",), inverse=True),
    "Inverse-ER-IRV": M_("weak", d_er_irv, ("firsts_eq",), inverse=True),
    "Inverse-TTR": M_("fav_run", d_inverse_two_round, ("n", "app", "runoff_gt"), inverse=True),
    "Inverse-Borda": M_("strict", d_sum, ("sum",), inverse=True),
    "Inverse-Score": M_("score", d_sum, ("sum",), inverse=True),
}


# Method groups: every method belongs to exactly one (checked below). They are the panels of the
# chain plots and can be used as names wherever methods are selected (--methods, --exclude, --show,
# --hide, --track).
GROUPS = {
    "condorcet": ["Ranked pairs", "Schulze", "Schulze-ER", "Kemeny-Young", "Copeland", "Copeland//Margins",
                  "Copeland//Margins-ER", "SICT", "Condorcet-random-drop"],
    "minimax-family": ["Minimax", "Minimax-ER", "MMWV", "MMPO", "MMPOWV"],
    "condorcet-based": ["Black", "Baldwin", "Baldwin-ER", "Nanson", "BTR-Borda", "BTR-IRV", "Benham", "Smith//Score"],
    "plurality-based": ["Plurality", "Supplementary vote", "IRV", "ER-IRV", "IRV-3", "Coombs", "DSC", "DAC"],
    "approval-based": ["Approval", "MDDA", "MAMPO"],
    "rated": ["Score", "Reciprocal score", "INS", "DMS", "ESR", "Geometric median"],
    # median grade or Bucklin-like: the grade level at which a candidate reaches a majority
    "median-based": ["MJ", "MAV", "MCA"],
    "hybrid": ["STAR", "BTR-score", "MARS", "IC-MARS", "MARS-T", "ARA", "ARA-Top", "Score+", "321", "MMWV+Score", "MMWVxScore", "STLR", "Soft Minimax",
               "Soft Minimax-40", "Soft Minimax-L", "Soft Minimax-D"],
    "positional": ["Borda", "Borda-ER", "Dowdall"],
    # a real, separate runoff: voters come back and vote between the finalists (model.md, Two-round
    # elections); a class of its own, as the runoff is assumed sincere and turnout unchanged
    "two-round": ["TTR", "Approval runoff", "Approval cond. runoff", "Score runoff"],
    "references": ["Random pair runoff", "Anti-plurality", "TRASH"],
    "inverse": ["Inverse-Plurality", "Inverse-Approval", "Inverse-Minimax", "Inverse-IRV", "Inverse-ER-IRV", "Inverse-TTR",
                "Inverse-Borda", "Inverse-Score"],
}
# Renamed methods and groups: old name -> current name. Results files store method names, so a run
# from before a rename would otherwise no longer combine with newer runs. analyze.py reads old names
# as the current ones, and old names work wherever methods are selected (--methods, --hide, ...).
# Only for a pure rename (the same rule, the same results); a method whose rule changed gets a new
# name and no entry here. A removed method gets no entry either: its old rows stay as they are.
RENAMED = {
    "Top-two runoff": "TTR",                                 # 2026-10-07
    "Approval runoff if needed": "Approval cond. runoff",    # 2026-10-07
    "Minimax-wv": "MMWV",                                    # 2026-10-07
    "MARA": "ARA",
    "Ranked Robin": "Copeland//Margins-ER",                       # 2026-10-19: Equal Vote's name now means any Condorcet method
    "Ranked Robin-SR": "Copeland//Margins",                    # 2026-10-19
    "BSR-S": "ESR",                                          # 2026-10-22: the test add-on's name
}
RENAMED_GROUPS = {
    "condorcet-hybrid": "condorcet-based",                   # 2026-10-12
}


def current_name(name, table=None):
    """The current name of a method (or, with table=RENAMED_GROUPS, a group): follows RENAMED,
    also over several renames; a name that was never renamed comes back unchanged."""
    table = RENAMED if table is None else table
    seen = set()
    while name in table and name not in seen:
        seen.add(name)
        name = table[name]
    return name


# ---------- add-ons: methods in separate files ----------
ADDON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "addons")
ADDONS = {}          # method name -> add-on file it comes from


def _load_addons(folder=ADDON_DIR):
    """Load every add-on file in `folder` (addons/*.py, except names starting with _). An add-on file
    defines METHODS = {name: Method(...)}, and may define FEATURES = {name: (per-voter function,
    shape)} for new ballot features, BALLOTS = {name: ballots.Ballot(...)} for new ballot types, GROUP = "a group name" (default "addons"; an existing group or
    a new one; or a dict {method: group} for methods of one file in different groups, the rest in "addons") and test() for its own checks (run by test_methods.py). Its methods are then used like
    the built-in ones. Deleting the file removes them. See addons/_template.py and docs/methods.md, Add-ons."""
    if not os.path.isdir(folder):
        return
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        name = "bffl_addon_" + fn[:-3]
        spec = importlib.util.spec_from_file_location(name, os.path.join(folder, fn))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod                     # so worker processes and pickling find it
        spec.loader.exec_module(mod)
        for k, v in getattr(mod, "BALLOTS", {}).items():     # new ballot types (ballots.Ballot entries)
            if k in BALLOTS:
                raise SystemExit(f"add-on {fn}: ballot type {k!r} exists already")
            BALLOTS[k] = v
        for k, v in getattr(mod, "FEATURES", {}).items():
            if k in FEATURES:
                raise SystemExit(f"add-on {fn}: feature {k!r} exists already")
            FEATURES[k] = v
        new = getattr(mod, "METHODS", {})
        if not new:
            raise SystemExit(f"add-on {fn} defines no METHODS")
        for k, m in new.items():
            if k in METHODS:
                raise SystemExit(f"add-on {fn}: method {k!r} exists already (built in or in another add-on)")
            if set(m.settings or ()) - set(SETTINGS):
                raise SystemExit(f"add-on {fn}: method {k!r} fixes an unknown setting; known: {', '.join(SETTINGS)}")
            if m.ballot not in BALLOTS:
                raise SystemExit(f"add-on {fn}: method {k!r} uses an unknown ballot type {m.ballot!r} "
                                 f"(the types are in ballots.py: {', '.join(BALLOTS)})")
            if k in RENAMED:
                raise SystemExit(f"add-on {fn}: {k!r} is the old name of {current_name(k)!r}; choose another name")
            METHODS[k] = m
            ADDONS[k] = fn
        group = getattr(mod, "GROUP", "addons")
        for k in new:
            GROUPS.setdefault(group.get(k, "addons") if isinstance(group, dict) else group, []).append(k)


_load_addons()
_grouped = [m for ms in GROUPS.values() for m in ms]
assert sorted(_grouped) == sorted(METHODS), "every method must be in exactly one group"
assert not {g.lower() for g in GROUPS} & {m.lower() for m in METHODS}, "group names must differ from method names"
assert all(m.ballot in BALLOTS for m in METHODS.values()), "every method's ballot type must be in ballots.BALLOTS"
assert all(set(m.settings or ()) <= set(SETTINGS) for m in METHODS.values()), "unknown setting in a method"
assert not set(RENAMED) & set(METHODS), "an old name (RENAMED) must not be a current method"
assert not set(RENAMED_GROUPS) & set(GROUPS), "an old group name (RENAMED_GROUPS) must not be a current group"

# Comparison sets: selection names like the groups, across groups (not plot panels)
COMPARISONS = {
    # the default selection of run.py and sweep.py: the methods most discussed (those of Quinn's vse-sim
    # and the common single-winner proposals) and the strongest found here; overlaps the groups
    "standard": ["Plurality", "TTR", "Supplementary vote", "IRV", "ER-IRV", "IRV-3", "Borda", "Approval",
                 "Approval runoff", "Approval cond. runoff", "Score", "MJ", "MAV", "STAR", "Score runoff", "321", "MARS", "ARA",
                 "BTR-score", "Smith//Score", "BTR-IRV", "Soft Minimax-D", "Ranked pairs", "Schulze", "Kemeny-Young",
                 "Copeland", "Minimax", "MMWV", "MMPO"],
    # the standard set and the strongest newer methods not in it: the selection of the reference runs
    # (docs/reference-runs.md), so every test shows them across all conditions
    "extended": None,
    "er-pairs": ["IRV", "ER-IRV", "Borda", "Borda-ER", "Baldwin", "Baldwin-ER", "Minimax",
                 "Minimax-ER", "MMWV", "Schulze", "Schulze-ER", "Copeland//Margins", "Copeland//Margins-ER"],
}

COMPARISONS["extended"] = COMPARISONS["standard"] + ["ESR", "DMS", "Geometric median", "Soft Minimax",
                                                     "Soft Minimax-40", "MARS-T"]


def follows_score_range(method):
    """True for a method whose results depend on --max-score: its ballot type uses the score range
    (score_range in ballots.Ballot) and it does not fix its own range (settings)."""
    m = METHODS[method]
    return BALLOTS[m.ballot].score_range and (m.settings or {}).get("max_score") is None


# every method that follows --max-score, add-ons included (built from the ballot types, so a new
# method on score ballots joins by itself): the selection for a max-score sweep
COMPARISONS["max-score"] = [k for k in METHODS if follows_score_range(k)]


def sincerity_defined(method):
    """Sincerity and fidelity are left undefined (blank) for methods whose honest ballot works against
    the voter's own interest: TRASH (the candidate ranked lower wins) and the Inverse-* methods (honest
    voting elects the worst candidate). There, what a ballot says and whom it helps point opposite
    ways, so moving away from the honest ballot is not a cost of strategy."""
    return method != "TRASH" and not (method in METHODS and METHODS[method].inverse) \
        and not method.startswith("Inverse-")


def group_of(method):
    """The group a method belongs to ("other" for names not in METHODS, e.g. from older CSVs)."""
    return next((g for g, ms in GROUPS.items() if method in ms), "other")




def select(include=None, exclude=None):
    """Method names: `include` starts from none, `exclude` from all. Names are methods, groups (see
    GROUPS), comparison sets (COMPARISONS) or "all"; matching ignores case."""
    lookup = {k.lower(): [current_name(k)] for k in RENAMED if current_name(k) in METHODS}   # old names
    lookup.update({g.lower(): GROUPS[current_name(g, RENAMED_GROUPS)] for g in RENAMED_GROUPS
                   if current_name(g, RENAMED_GROUPS) in GROUPS})
    lookup.update({k.lower(): [k] for k in METHODS})
    lookup.update({g: ms for g, ms in {**GROUPS, **COMPARISONS}.items()})
    lookup["all"] = list(METHODS)

    def resolve(names):
        bad = [x for x in names if x.lower() not in lookup]
        if bad:
            raise SystemExit(f"unknown method(s) or group(s): {', '.join(bad)}. Methods: {', '.join(METHODS)}. "
                             f"Groups: {', '.join(GROUPS)}. Comparisons: {', '.join(COMPARISONS)}")
        return {m for x in names for m in lookup[x.lower()]}

    chosen = resolve(include) if include else set(METHODS)
    chosen -= resolve(exclude or [])
    return [k for k in METHODS if k in chosen]


def run_selection(include=None, exclude=None):
    """The methods a run uses: as select, but without --methods the default is the "standard" set
    (COMPARISONS): the most discussed and the strongest methods. Name others, groups or "all" in
    --methods to run them (add-ons too); --exclude then takes methods out of that."""
    if not include:
        include = ["standard"]
    return select(include, exclude)
