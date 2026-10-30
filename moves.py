"""Shared moves: an experimental answer to the poll rounds (run.py --moves 1), in place of the ranking
groups of strategy.py. It is called by strategy.respond once per round, after the polls, the leaders
and the strategic voters of the round are known.

- A move is a sequence of steps, each naming candidates; a voter applies it to their own utilities
  and casts the ballot that follows from them in the method's format (render). Two voters following
  the same move can cast different ballots: rated ballots keep their own intensities.
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
- Audiences share moves. One per place in the polls (the two leaders and the next place); every
  strategic voter belongs to those of their first and second choice among the places, and with
  campaigns also to the audience of the campaign they listen to (the candidate they like most among
  those campaigning).
- An audience tries: the moves its members followed in the last round; the naive exaggerations and
  the push-over of today's model, written as moves (NAIVE, PUSH), for both orders of the leaders; for
  a campaign audience the recommended move instead; and `search` random moves, each one random step
  after the audience's most followed move of the last round.
- Every voter judges every move of their audiences by their own expected utility over the polls, as
  if a share s of the audience's strategic members switched from sincere to it (each applying it to
  their own utilities), the rest of the counts as polled. s = 0: one voter's own impact, as the change
  of an average member of the audience (a stand-in for the voter's own ballot, which would cost one
  count per voter). A move that strictly gains is taken (the
  best one). Otherwise aggressive (bold) voters take the best of the naive moves (exaggerations,
  push-over for the leader they prefer, their own move of the last round) if it does not lose (bold
  ones with --risk up to their tolerance): they exaggerate when it cannot hurt, as in today's model;
  the naive moves for the other leader, campaign recommendations, random moves and the moves of others
  are taken only if they gain. Kind voters take only gains.
  Otherwise they vote sincerely. Ties keep the move followed in the last round.
"""
from collections import Counter

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
    out = np.empty_like(U)
    for i, row in enumerate(U):
        span = max(row.max() - row.min(), 1e-12)
        _, inv = np.unique(np.round((row - row.min()) / span, 9), return_inverse=True)
        out[i] = inv
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


def answer(method, count, u, B, sincere, F_sin, polls, tot_dev, F, rng, *, active, eager, tol_v, places, recs,
           share, search, last):
    """One round of shared moves. count: the method as voters count a poll (Method.poll_view). B: the ballots the polls were drawn from; F: their features (None:
    a method without feature sums, polled by tallying, see strategy.poll_tallies); tot_dev: the polled
    counts (samples x d); F_sin: features of the sincere ballots; places: the poll places (leaders and
    next place, None where not found); recs: campaign recommendations [(c, (X, Y), P)]; last: each
    voter's move in the last round (dict). Returns the new ballots, the moves followed (dict voter ->
    move), and the followers per campaign candidate (array C)."""
    from strategy import poll_tallies
    n, C = u.shape
    strat = np.flatnonzero(active)
    pl = [c for c in places if c is not None]
    a, b = places[0], places[1]
    third = places[2] if len(places) > 2 else None
    # audiences: key -> members (voter order)
    aud = {}
    for v in strat:
        first_two = sorted(pl, key=lambda c: -u[v, c])[:2]
        for c in first_two:
            aud.setdefault(("place", c), []).append(int(v))
    rec_of = {}
    if recs:
        cs = [c for c, _, _ in recs]
        rec_of = {c: (("above",) + tuple(mv),) for c, mv, _ in recs}
        for v in strat:
            c = max(cs, key=lambda x: u[v, x])
            aud.setdefault(("camp", c), []).append(int(v))
    best_all, best_naive = {}, {}                         # voter -> (gain, move, row, from campaign c or -1)
    for key in sorted(aud, key=lambda k: (k[0], k[1])):
        M = np.array(aud[key])
        prev = Counter(last[v] for v in M if last.get(v) is not None)
        moves = [mv for mv, _ in sorted(prev.items(), key=lambda kv: (-kv[1], kv[0]))]
        naive = {}                                        # naive move -> the leader it is for
        if key[0] == "place":
            if b is not None:
                for A, Bo in ((a, b), (b, a)):
                    for mv in _naive(method.ballot, A, Bo) + (
                            [_push(method.ballot, A, Bo, third)] if third is not None else []):
                        if mv is not None:
                            moves.append(mv)
                            naive.setdefault(mv, (A, Bo))
        base = moves[0] if prev else ()
        for _ in range(search):
            moves.append(random_move(base, C, rng))
        seen, uniq = set(), []
        for mv in moves:
            if mv not in seen:
                seen.add(mv)
                uniq.append(mv)
        lazy, appr = own_params(M)
        s_eff = share if share > 0 else 1.0 / len(M)        # 0: one voter's impact (an average member)
        if F is not None:
            cur = s_eff * (F_sin[M].sum(axis=0) - F[M].sum(axis=0))
            base_rows = tot_dev + cur                     # the share of the audience voting sincerely
            P0 = count.decide_many(base_rows, C)
        else:
            take = M[rng.permutation(len(M))[:int(np.ceil(s_eff * len(M)))]]
            B0 = B.copy()
            B0[take] = sincere[take]
            P0 = poll_tallies(count.tally, B0, polls)
        uM = u[M]
        for mv in uniq:
            X = render(method, u[M], mv, sincere[M], lazy, appr)
            if F is not None:
                D = s_eff * (count.features(X).sum(axis=0) - F_sin[M].sum(axis=0))
                P = count.decide_many(base_rows + D, C)
            else:
                B2 = B0.copy()
                pos = {v: i for i, v in enumerate(M)}
                B2[take] = X[[pos[v] for v in take]]
                P = poll_tallies(count.tally, B2, polls)
            g = (P - P0).mean(axis=0) @ uM.T               # each member's own expected gain
            camp_c = key[1] if key[0] == "camp" and mv == rec_of.get(key[1]) else -1
            side = naive.get(mv)
            for i, v in enumerate(M):
                gi = float(g[i])
                entry = (gi, mv, X[i], camp_c)
                # without a gain, only the naive moves for the leader the voter prefers (or their own last move)
                own = side is not None and u[v, side[0]] >= u[v, side[1]]
                for best, ok in ((best_all, True), (best_naive, own or mv == last.get(v))):
                    if not ok:
                        continue
                    cb = best.get(v)
                    if (cb is None or gi > cb[0] + TOL
                            or (abs(gi - cb[0]) <= TOL and mv == last.get(v) and cb[1] != mv)):
                        best[v] = entry
    newB = sincere.copy()
    followed = {}
    follow = np.zeros(C)
    for v in strat:
        v = int(v)
        pick = None
        if v in best_all and best_all[v][0] > TOL:
            pick = best_all[v]
        elif eager[v] and v in best_naive and best_naive[v][0] >= -TOL - tol_v[v]:
            pick = best_naive[v]
        if pick is not None:
            gi, mv, row, cc = pick
            newB[v] = row
            followed[v] = mv
            if cc >= 0:
                follow[cc] += 1
    return newB, followed, follow
