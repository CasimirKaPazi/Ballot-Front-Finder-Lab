"""Moves: how a voter changes their ballot, as a sequence of steps naming candidates, and the ballot
that follows (render). Used by the individual voters (individual.py, pooled.py, inertia.py); the naive
moves and push-over (_naive, _push) are the same for every format and include the naive exaggerations
and push-over of the ranking-group model (ballots.py, used by --engine groups), which the tests check. (Until
2026-10-31 also an experimental engine of moves shared within audiences, run.py --moves 1.)

- A voter applies a move to their own utilities and casts the ballot that follows from them in the
  method's format. Two voters following the same move can cast different ballots: rated ballots keep
  their own intensities.
  Formats with a "nothing" level (approval, truncated rankings) give nothing to the candidates a move
  put at the bottom; after a levelling step (upto, downfrom, below) they support exactly the others
  (after an upto: those at or above its level).
  A ballot with everything at one level gives the top grade to all (score formats).
  Scores with an approval cutoff take their scores as score ballots do and their approvals as approval
  ballots do (the cutoff then at the lowest score approved).
      ("top", X)          X above everyone
      ("bottom", Y)       Y below everyone
      ("above", X, Y)     X just above Y, where the voter prefers Y (the campaign move)
      ("upto", X)         everyone liked at least as much as X tied with X, at the top
      ("downfrom", Y)     everyone liked no more than Y tied with Y, at the bottom
      ("below", X)        everyone liked less than X tied at the bottom: nothing below X
"""
import numpy as np

import methods
from ballots import BALLOTS, cares, cut_join, cut_parts, lazy_on, rank_ballots

NUDGE = 1e-6     # top, bottom, above: this share of the voter's utility range beyond the others
TIE = 1e-9       # ordinal formats: ties a move leaves are broken by the voter's own order, by this much
TOL = 1e-9
STEPS = ("top", "bottom", "above", "upto", "downfrom", "below")
ORDINAL = {"fav", "rank", "strict", "trunc", "trunc3", "veto", "fav_run"}   # formats without ties
NOTHING = {"approval", "app_run", "rank", "trunc", "trunc3"}  # bottom level -> not approved / unranked


def _naive(ballot, A, B):
    """The naive moves for A, the leader preferred, against B, the other: the same for every ballot format
    (since 2026-11-22). Which of them a format can express is up to render: approval collapses them to
    who is approved, a strict ranking breaks the ties they make, scores keep them, a one-name ballot
    keeps the top. Before, each format had its own list (scores only the levelling moves, rankings
    only the reversals), so that score voters could not try compromise or burial and an ordinal method
    gave different results on score ballots than on rankings with ties, though both express the same.
        compromise and burial    top A; bottom B; both; below A (nothing below A); top A and below A
                                 (in either order: truncated rankings render them differently)
        levelling                upto A (tied with everyone liked at least as much); upto A and bottom
                                 B; upto A and downfrom B; upto A and below A (A and above, else nothing)
    (ballot: kept for the callers; the list does not depend on it.)"""
    return [(("top", A), ("bottom", B)), (("top", A),), (("bottom", B),), (("below", A),),
            (("top", A), ("below", A)), (("below", A), ("top", A)), (("upto", A),), (("upto", A), ("bottom", B)),
            (("upto", A), ("downfrom", B)), (("upto", A), ("below", A))]


def _push(ballot, A, B, X):
    """Push-over: the ballot for A against B, with X, the next place, supported too; in both forms, the
    ranking one (A top, B bottom, X above all) and the levelling one (A and above tied at the top,
    nothing below A, X at the top), for every format (render expresses what the format can).
    (ballot: kept for the callers.)"""
    return [(("top", A), ("bottom", B), ("top", X)), (("upto", A), ("below", A), ("top", X))]


def apply(U, move):
    """Utilities (rows) after the move; the candidates it put at the bottom (bool, rows x C: by bottom,
    downfrom or below, and not raised again since); whether a step levelled anything (upto, downfrom,
    below); and per row the utility of the top level an upto made (-inf without one)."""
    U = np.array(U, dtype=float)
    span = U.max(axis=1) - U.min(axis=1)
    span = np.where(span > 0, span, 1.0)
    low = np.zeros(U.shape, bool)
    levelled = False
    up = np.full((len(U), 1), -np.inf)
    for st in move:
        kind, X = st[0], st[1]
        if kind == "top":
            U[:, X] = U.max(axis=1) + NUDGE * span
            low[:, X] = False
        elif kind == "bottom":
            U[:, X] = U.min(axis=1) - NUDGE * span
            low[:, X] = True
        elif kind == "above":
            Y = st[2]
            need = U[:, Y] >= U[:, X]
            U[need, X] = U[need, Y] + NUDGE * span[need]
            low[need, X] = False
        elif kind == "upto":
            m = U >= U[:, X:X + 1]
            up = U[:, X:X + 1].copy()
            U = np.where(m, up, U)
            low &= ~m
            levelled = True
        elif kind == "downfrom":
            m = U <= U[:, X:X + 1]
            U = np.where(m, U[:, X:X + 1], U)
            low |= m
            levelled = True
        elif kind == "below":
            m = U < U[:, X:X + 1]
            floor = np.where(m, U, np.inf).min(axis=1, keepdims=True)
            U = np.where(m, floor, U)
            low |= m
            levelled = True
        else:
            raise ValueError(f"unknown step {kind}")
    return U, low, levelled, up


def _levels(U):
    """Dense levels per row: equal utilities share a level (ranking with ties), 0 = lowest."""
    mn = U.min(axis=1, keepdims=True)
    span = np.maximum(U.max(axis=1, keepdims=True) - mn, 1e-12)
    R = np.round((U - mn) / span, 9)
    order = np.argsort(R, axis=1, kind="stable")
    Rs = np.take_along_axis(R, order, axis=1)
    dense = np.concatenate([np.zeros((len(R), 1)), np.cumsum(np.diff(Rs, axis=1) > 0, axis=1)], axis=1)
    out = np.empty_like(U, dtype=float)
    np.put_along_axis(out, order, dense, axis=1)
    return out


def own_params(M):
    """The voters' own laziness cuts and approval draws, for the rows M (Method.sincere)."""
    lazy = methods.LAZY if methods.VOTER_LAZY is None else methods.VOTER_LAZY[M]
    appr = methods.APPROVAL_DEFAULT if methods.VOTER_APPROVAL is None else methods.VOTER_APPROVAL[M]
    return lazy, appr


def render(method, U0, move, sincere_rows, lazy, appr):
    """The ballots of the voters with utilities U0 (rows) following the move. The sincere ballot of the
    changed utilities, in the method's format, with the voter's own laziness and approval draw; a tie
    the format cannot express is broken by the voter's own order; formats with a "nothing" level
    (approval, truncated rankings) give the bottom level a move made nothing; rankings with ties keep
    the ties a move made. Two-round ballots keep the voter's own runoff part (sincere_rows). For the
    Inverse-* methods the ballot is read the ordinary way (as the naive exaggerations are)."""
    name = method.ballot
    C = U0.shape[1]
    U, low, levelled, up = apply(U0, move)
    sup = ~low & (U >= up - 1e-12)                         # supported after a levelling step
    flat = (U.max(axis=1) - U.min(axis=1) <= 0)[:, None]  # everything at one level
    if name == "weak":
        L = _levels(U)
        lz = lazy if not method.inverse else 0.0
        X = np.where(cares(U, lz), L + 1, 0.0) if lazy_on(lz) else L
        return X
    if name in ORDINAL:                                    # ties broken by the voter's own order
        mn, mx = U0.min(axis=1, keepdims=True), U0.max(axis=1, keepdims=True)
        U = U + TIE * (mx - mn) * (U0 - mn) / np.where(mx - mn > 0, mx - mn, 1.0)
    X = method.sincere(-U if method.inverse else U, lazy, appr)
    marks = BALLOTS[name].marks is not None
    if name in NOTHING:
        if name in ("approval", "app_run"):
            part = np.floor_divide(X, C) if marks else X
            part = sup.astype(float) if levelled else np.where(low, 0.0, part)
            X = part * C if marks else part
        elif name == "rank":
            X = np.where(low, 0.0, X)
        else:                                              # trunc, trunc3: rank the supported, top first
            R = rank_ballots(U)
            keep = (sup if levelled else (X > 0) & ~low) & (R >= C - (min(methods.TRUNC3, C) if name == "trunc3" else C))
            kk = keep.sum(axis=1, keepdims=True)
            X = np.where(keep, R - (C - kk) + 1, 0.0)
    if name == "scorecut":                                 # the cutoff as for approval, the scores as for score
        Sc, a = cut_parts(X)
        a = sup if levelled else a & ~low
        Sc = np.where(flat, methods.MAXS, Sc)
        return cut_join(Sc, a | flat)
    if name in ("score", "grade3"):                        # a reversal the move makes stays one on the scale
        X = _strict(X, move, 2 if name == "grade3" else methods.MAXS)
    if name in ("score", "grade3", "score_run"):          # everything at one level: the top grade to all
        top = 2 if name == "grade3" else methods.MAXS
        X = np.where(flat, top * C if marks else top, X)
    if marks:                                              # two-round: the voter's own runoff part
        X = np.floor_divide(X, C) * C + np.mod(sincere_rows, C)
    return X


def _strict(X, move, top):
    """Score ballots: the steps that put a candidate strictly above or below others (top, bottom, above)
    keep doing so after the scores are rounded. The move places the candidate only a hair beyond the
    others (NUDGE), which rounding to the scale would turn into a tie: then the candidate takes the
    end of the scale and the others it should pass give up one level (as far as the scale allows).
    Applied step by step, so a later step wins, as in apply."""
    X = np.array(X, dtype=float)
    for st in move:
        kind = st[0]
        if kind == "top":
            c = st[1]
            others = np.arange(X.shape[1]) != c
            X[:, others] = np.minimum(X[:, others], top - 1)
            X[:, c] = top
        elif kind == "bottom":
            c = st[1]
            others = np.arange(X.shape[1]) != c
            X[:, others] = np.maximum(X[:, others], 1)
            X[:, c] = 0
        elif kind == "above":
            a, b = st[1], st[2]
            tie = X[:, a] <= X[:, b]
            X[:, b] = np.where(tie & (X[:, b] >= top), top - 1, X[:, b])
            X[:, a] = np.where(tie, np.minimum(X[:, b] + 1, top), X[:, a])
    return X


def random_move(base, C, rng):
    """One random step after the move `base` (a type and its candidates drawn uniformly)."""
    kind = STEPS[rng.integers(len(STEPS))]
    if kind == "above":
        X, Y = (int(x) for x in rng.choice(C, 2, replace=False))
        return tuple(base) + ((kind, X, Y),)
    return tuple(base) + ((kind, int(rng.integers(C))),)
