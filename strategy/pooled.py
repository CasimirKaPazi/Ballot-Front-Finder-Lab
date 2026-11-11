"""Pooled discovery (run.py --engine pooled, the default since 2026-11-06): individual voters whose
moves are found and valued per type of voter, by their order of the poll's top three, counting the
round's polls again (answer_pooled); each voter then decides alone. strategy.rounds calls the engine
(PooledEngine) once per round."""
import numpy as np

from . import moves as mv
from .common import IndividualVoters, judge, aims, _PollChances
from .polls import poll_totals, _widened, _leaders


POOL_LIBRARY = 8     # pooled discovery: moves a type keeps from its search (the most recent)


POOL_STEPS = 6       # pooled discovery: the longest move the search builds


def answer_pooled(method, count, u, B, sincere, polls_T, places, *, threshold, share, memory, search=0, rng=None):
    """One round with pooled discovery (run.py --engine pooled). Discovery is shared, the
    decision individual:
    - Types: the voters by their order of the poll's places (up to three: the leaders and the next place,
      as the ranking groups find them), six types for three places.
    - Discovery: for each type the moves of the ranking groups (moves._naive: the naive exaggerations
      for the leader the type prefers against the other; moves._push: the push-over for the next place),
      each rendered by every voter of the type on their own utilities. A share `share` (--coordination)
      of the type's change of ballots (from what they voted in the last round) is added to each of the
      round's polls (polls_T: one row of totals per poll), and the polls are counted again: the winner
      chances the move promises. The type returning to its sincere ballots is counted the same way.
      Nobody sees individual ballots: discovery counts published totals, as anyone knowing the method can.
    - Search (search > 0, with rng): each type also tries `search` random proposals per round, each one
      random step (moves.random_move) after its best move so far (its sincere ballots if none gains),
      as the ranking groups search. A proposal that raises the type's mean gain (each voter's gain over
      the type's sincere ballots as a share of their utility range, all voters of the type) is kept in
      the type's library (memory["library"], per type: its order of the places; the POOL_LIBRARY most
      recent, moves of up to POOL_STEPS steps) and offered again in later rounds, published like the
      rest.
    - Decision: each strategic voter values every published move of their type by their own utilities
      against their type's sincere ballots (as a share of their utility range), its promise over the
      polls as they are weighted by their trust (judge, trust), and makes the most valuable move if
      that exceeds their cost (threshold), else votes sincerely.
    Every voter of a type counts in its change, strategic or not: discovery does not know who follows.
    Returns the new ballots and the voters who vote strategically this round."""
    n, C = u.shape
    P0 = count.decide_many(polls_T, C).mean(axis=0)
    judge(u, _PollChances(P0), memory)
    newB = sincere.copy()
    memory["follow"] = np.zeros(C)
    active = np.zeros(n, bool)
    able = np.isfinite(threshold)
    for v in [v for v in memory.get("held", {}) if not able[v]]:
        memory["held"].pop(v)
    places = [c for c in places if c is not None]
    if len(places) < 2 or not able.any():
        return newB, active
    lo_u, hi_u = u.min(axis=1), u.max(axis=1)
    span = np.where(hi_u > lo_u, hi_u - lo_u, 1.0)
    F_cur = count.features(B)
    order = np.argsort(-u[:, places], axis=1, kind="stable")
    types = {}
    for v in range(n):
        types.setdefault(tuple(places[i] for i in order[v]), []).append(v)

    memo = {}

    def chances(D):                                # the same change counted once per round (exactly equal)
        key = np.round(D, 9).tobytes()
        if key not in memo:
            memo[key] = count.decide_many(polls_T + D[None], C).mean(axis=0)
        return memo[key]

    for key, mem in types.items():
        mem = np.array(mem)
        V = mem[able[mem]]
        if not len(V):
            continue
        A = min(places[:2], key=key.index)                   # the leader the type prefers
        Bo = places[1] if A == places[0] else places[0]
        moves = list(mv._naive(method.ballot, A, Bo))
        if len(places) > 2:
            p = mv._push(method.ballot, A, Bo, places[2])
            if p is not None:
                moves.append(p)
        lib = memory.setdefault("library", {}).setdefault(key, [])
        moves = list(dict.fromkeys(moves + lib))
        lazy, appr = mv.own_params(mem)
        D_s = share * (count.features(sincere[mem]) - F_cur[mem]).sum(axis=0)
        P_s = chances(D_s)
        Us = u[mem] @ P_s
        Xs, Ps, Ds = [], [], []

        def tryout(m):
            X = mv.render(method, u[mem], m, sincere[mem], lazy, appr)
            D = share * (count.features(X) - F_cur[mem]).sum(axis=0)
            P = chances(D)
            return X, P, float(np.mean((u[mem] @ P - Us) / span[mem])), D    # the type's mean gain

        best_m, best_g = (), 0.0
        for m in moves:
            X, P, g, D = tryout(m)
            Xs.append(X)
            Ps.append(P)
            Ds.append(D)
            if g > best_g + 1e-12:
                best_m, best_g = m, g
        if search > 0 and rng is not None:
            for _ in range(search):
                m = mv.random_move(best_m if len(best_m) < POOL_STEPS else best_m[:POOL_STEPS - 1], C, rng)
                if m in moves:
                    continue
                X, P, g, D = tryout(m)
                if g > best_g + 1e-12:
                    best_m, best_g = m, g
                    moves.append(m)
                    Xs.append(X)
                    Ps.append(P)
                    Ds.append(D)
                    lib.append(m)
            del lib[:-POOL_LIBRARY]
        sel = able[mem]                                     # the strategic voters of the type
        U = u[V]
        before = U @ P0
        EU = np.stack([U @ P_s] + [U @ P for P in Ps])     # sincere first
        tr = memory.get("trust", {})
        for j in range(1, len(EU)):
            gain = EU[j] - before
            if not tr or not (gain > 0).any():          # nobody's trust shaken yet, or no gain to discount
                continue
            ks = aims(moves[j - 1])                 # once per move, not per voter
            rho = np.array([min((tr.get((int(v), k), (1.0, 1))[0] for k in ks), default=1.0) for v in V])
            EU[j] = before + np.where(gain > 0, rho * gain, gain)
        best = np.argmax(EU, axis=0)
        promised = {j: Ps[j - 1] for j in range(1, len(EU))}
        value = (EU[best, np.arange(len(V))] - EU[0]) / span[V]
        best = np.where(value > threshold[V], best, 0)
        for i, v in enumerate(V):
            v = int(v)
            j = int(best[i])
            if j > 0:
                newB[v] = Xs[j - 1][np.flatnonzero(sel)[i]]
                memory["last"][v] = (moves[j - 1], float(U[i] @ promised[j] - before[i]), float(before[i]))
                memory.setdefault("held", {})[v] = moves[j - 1]
                active[v] = True
            else:
                memory.setdefault("held", {}).pop(v, None)
    return newB, active


class PooledEngine(IndividualVoters):
    """Pooled discovery, one round at a time: the round's polls as the ranking groups see them, their
    three leading places (polls._leaders), and answer_pooled."""

    def round(self, r, polls):
        """One round: the new ballots (no campaigns: 0 followers, no candidate)."""
        ctx = self.ctx
        pm, C = ctx.pm, ctx.C
        F, tot, WF = poll_totals(pm, ctx.B, polls, ctx.n, ctx.m)
        dev = ctx.poll_error * (WF - tot)           # the polls as the ranking groups see them
        places = _leaders(lambda j: pm.decide_many(_widened(tot, dev, j), C), C, 3)
        if ctx.leaders is not None:
            ctx.leaders[r] = (places[0], places[1])
        fol = self.coordination
        B, ctx.active = answer_pooled(ctx.method, pm, ctx.u, ctx.B, ctx.sincere, tot + dev, places,
                                      threshold=self.threshold, share=float(fol) if np.ndim(fol) == 0 else 0.25,
                                      memory=self.memory, search=ctx.search, rng=self.rng)
        return B, 0.0, -1
