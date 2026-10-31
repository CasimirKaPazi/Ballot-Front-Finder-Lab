"""Moves: how a voter changes their ballot, as a sequence of steps naming candidates, and the ballot
that follows (render). Used by the individual voters (individual.py) and by the tests that check the
moves against the naive exaggerations and push-over of the ranking-group model (_naive, _push). (Until
2026-10-31 also an experimental engine of moves shared within audiences, run.py --moves 1.)

- A voter applies a move to their own utilities and casts the ballot that follows from them in the
  method's format. Two voters following the same move can cast different ballots: rated ballots keep
  their own intensities.
  Formats with a "nothing" level (approval, truncated rankings) give nothing to the candidates a move
  put at the bottom; after a levelling step (upto, downfrom, below) they support exactly the others
  (after an upto: those at or above its level).
  A ballot with everything at one level gives the top grade to all (score formats).
      ("top", X)          X above everyone
      ("bottom", Y)       Y below everyone
      ("above", X, Y)     X just above Y, where the voter prefers Y (the campaign move)
      ("upto", X)         everyone liked at least as much as X tied with X, at the top
      ("downfrom", Y)     everyone liked no more than Y tied with Y, at the bottom
      ("below", X)        everyone liked less than X tied at the bottom: nothing below X
"""
import numpy as np

import methods
from ballots import BALLOTS, cares, lazy_on, rank_ballots

NUDGE = 1e-6     # top, bottom, above: this share of the voter's utility range beyond the others
TIE = 1e-9       # ordinal formats: ties a move leaves are broken by the voter's own order, by this much
TOL = 1e-9
STEPS = ("top", "bottom", "above", "upto", "downfrom", "below")
ORDINAL = {"fav", "rank", "strict", "trunc", "trunc3", "veto", "fav_run"}   # formats without ties
NOTHING = {"approval", "app_run", "rank", "trunc", "trunc3"}  # bottom level -> not approved / unranked


def _naive(ballot, A, B):
    """Today's naive exaggerations (ballots.py, _exag_*), as moves: A the leader preferred, B the other."""
    split = (("upto", A), ("below", A))
    ranking = [(("top", A), ("bottom", B)), (("top", A),), (("bottom", B),)]
    known = {
        "fav": [(("top", A),)], "fav_run": [(("top", A),)],
        "approval": [split], "app_run": [split],
        "score": [(("upto", A), ("downfrom", B)), split], "grade3": [(("upto", A), ("downfrom", B)), split],
        "score_run": [(("upto", A), ("downfrom", B)), split],
        "strict": ranking,
        "rank": ranking + [(("below", A),), (("top", A), ("below", A))],
        "weak": [(("upto", A), ("bottom", B)), (("upto", A),), (("top", A), ("bottom", B)), (("bottom", B),),
                 (("below", A),), (("top", A), ("below", A))],
        "trunc": [(("below", A),), (("below", A), ("top", A))],
        "trunc3": [(("top", A), ("bottom", B)), (("below", A),)],
        "veto": [(("bottom", B),)],
    }
    # formats of add-ons: the ranking moves, and the levels as for scores
    return known.get(ballot, ranking + [(("upto", A), ("downfrom", B)), split])


def _push(ballot, A, B, X):
    """Today's push-over (ballots.py, _push_*), as a move: the ballot for A against B, X supported too."""
    if ballot == "veto":
        return None
    if ballot in ("fav", "fav_run"):
        return (("top", X),)
    if ballot in ("approval", "app_run", "score", "grade3", "score_run"):
        return (("upto", A), ("below", A), ("top", X))
    if ballot == "trunc":
        return (("below", A), ("top", X))
    return (("top", A), ("bottom", B), ("top", X))       # rankings, with ties, at most 3


def apply(U, move):
    """Utilities (rows) after the move; the candidates it put at the bottom (bool, rows x C: by bottom,
    downfrom or below, and not raised again since); whether a step levelled anything (upto, downfrom,
    below); and per row the utility of the top level an upto made (-inf without one)."""
    U = np.array(U, dtype=float)
    k = np.arange(len(U))
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
    if name in ("score", "grade3", "score_run"):          # everything at one level: the top grade to all
        top = 2 if name == "grade3" else methods.MAXS
        X = np.where(flat, top * C if marks else top, X)
    if marks:                                              # two-round: the voter's own runoff part
        X = np.floor_divide(X, C) * C + np.mod(sincere_rows, C)
    return X


def random_move(base, C, rng):
    """One random step after the move `base` (a type and its candidates drawn uniformly)."""
    kind = STEPS[rng.integers(len(STEPS))]
    if kind == "above":
        X, Y = (int(x) for x in rng.choice(C, 2, replace=False))
        return tuple(base) + ((kind, X, Y),)
    return tuple(base) + ((kind, int(rng.integers(C))),)
