"""Inertia (run.py --engine inertia; docs/strategy-design.md): voters reason from their honest ballot,
their home. Each round a voter weighs the moves of the races they care about, guessing what the move
would do if people like them made it, and casts the best one if it beats their honest ballot by more
than their inertia; otherwise they vote honestly. strategy.rounds calls the engine (InertiaEngine) once
per round.

- Inertia: one trait per voter, theta, a share of their utility range: their unwillingness to leave
  their honest ballot. Always strategic voters have 0, adaptive ones their personal draw (uniform
  0..1) times --inertia, the others never leave home (inf). Going back home costs nothing.
- Sides: the poll's places (the leaders and the next place, as the ranking groups find them); each
  pair of them is a race, and in each race a voter is on the side of the candidate of the pair they
  prefer (none if they like both alike). Three places: three races, six sides, every voter on three.
- Moves: for a side, the moves for its candidate against the other (moves._naive: the naive
  exaggerations; moves._push: the push-over for the third place), each cast by every member from their
  own utilities.
- A voter's guess: they do not know how many on their side will make a move. They value it against
  their side at home: the round's polls counted again with the side's members back on their honest
  ballots, and with a share s of them casting the move instead, everyone else voting as the poll
  shows. The share is unknown, so the value is averaged over shares (SHARES: the midpoints of four
  equal parts of 0..1). After a round in which the move was weighed, the voters see what share of their
  side cast it, and guess around that (a Beta distribution, its evidence EVIDENCE per round, older
  rounds weighing MEMORY as much as the next; SHARES then are its quartile midpoints).
- Memory: a move's value to a voter is a weighted average over the rounds it was weighed in (each
  earlier round weighing MEMORY as much as the next), so counter-moves seen before stay in mind. A move
  of a race no longer among the top three counts as gaining nothing in that round: its value fades.
- Decision: a voter casts the move of greatest remembered value, if it exceeds their inertia (both as
  shares of their utility range); otherwise their honest ballot.
"""
import numpy as np
from scipy.special import betaincinv

from . import moves as mv
from .polls import poll_totals, _widened, _leaders

SHARES = (0.125, 0.375, 0.625, 0.875)   # the shares of their side a voter imagines making a move (not knowing)
MEMORY = 0.8            # the weight of each earlier round against the round after it (values, evidence)
EVIDENCE = 4.0          # the weight of one round's observed share against the uniform guess (Beta(1, 1))


class InertiaEngine:
    """The inertia engine, one round at a time. ctx: the run's shared state (rounds.Context); inertia:
    the largest inertia of an adaptive voter, a share of their utility range (--inertia)."""
    OPTIONS = ("inertia",)

    def __init__(self, ctx, inertia=1.0):
        self.ctx = ctx
        self.theta = np.where(ctx.fixed, 0.0, np.where(ctx.adapt, ctx.q_adapt * inertia, np.inf))
        u = ctx.u
        lo, hi = u.min(axis=1), u.max(axis=1)
        self.span = np.where(hi > lo, hi - lo, 1.0)
        self.Fs = None                          # the honest ballots as polls count them (on first use)
        self.memory = {}                        # (side, move) -> (weighted sums of values, of weights, voters, ballots)
        self.evidence = {}                      # (side, move) -> (cast, not cast): evidence for the share
        self.weighed = set()                    # the (side, move) weighed in the round before
        self.stats = []                         # per round: (share of all voters changing, moves weighed)

    def bold(self):
        return float("nan")

    def _shares(self, key):
        """The shares of their side a voter imagines making a move: the midpoints of four equal parts of
        0..1 (not knowing), or, once the move was weighed, the quartile midpoints of a Beta distribution
        with the shares seen as evidence."""
        if key not in self.evidence:
            return SHARES
        a, b = self.evidence[key]
        return tuple(float(x) for x in betaincinv(1.0 + a, 1.0 + b, np.array(SHARES)))

    def round(self, r, polls):
        """One round: the new ballots (no campaigns: 0 followers, no candidate)."""
        ctx = self.ctx
        pm, C, n, u, method = ctx.pm, ctx.C, ctx.n, ctx.u, ctx.method
        if self.Fs is None:
            self.Fs = pm.features(ctx.sincere)          # each voter's honest ballot, as polls count it
        F, tot, WF = poll_totals(pm, ctx.B, polls, n, ctx.m)
        dev = ctx.poll_error * (WF - tot)
        places = [c for c in _leaders(lambda j: pm.decide_many(_widened(tot, dev, j), C), C, 3) if c is not None]
        if ctx.leaders is not None:
            ctx.leaders[r] = (places[0], places[1] if len(places) > 1 else None)
        B, sincere, theta, span, Fs = ctx.B, ctx.sincere, self.theta, self.span, self.Fs
        able = np.isfinite(theta)
        polls_T = tot + dev
        memo = {}

        def chances(D):
            key = np.round(D, 9).tobytes()
            if key not in memo:
                memo[key] = pm.decide_many(polls_T + D[None], C).mean(axis=0)
            return memo[key]

        best_gain, best_x = np.zeros(n), {}
        weighed = set()
        if len(places) >= 2 and able.any():
            for i, X in enumerate(places):
                for Y in places[i + 1:]:
                    third = next((c for c in places if c not in (X, Y)), None)
                    for A, Bo in ((X, Y), (Y, X)):
                        mem = np.flatnonzero(u[:, A] > u[:, Bo])
                        if not able[mem].any():
                            continue
                        H = (Fs[mem] - F[mem]).sum(axis=0)          # the side at home
                        vH = u[mem] @ chances(H)
                        moves = list(mv._naive(method.ballot, A, Bo))
                        if third is not None:
                            p = mv._push(method.ballot, A, Bo, third)
                            if p is not None:
                                moves.append(p)
                        lazy, appr = mv.own_params(mem)
                        for m in dict.fromkeys(moves):
                            Xm = mv.render(method, u[mem], m, sincere[mem], lazy, appr)
                            away = ~np.all(np.isclose(Xm, sincere[mem]), axis=1)
                            if not away.any():
                                continue
                            key = (A, Bo, m)
                            if key in self.weighed:                 # the share seen casting it after last round
                                cast = float(np.all(np.isclose(B[mem], Xm), axis=1)[away].mean())
                                a, b = self.evidence.get(key, (0.0, 0.0))
                                self.evidence[key] = (MEMORY * a + EVIDENCE * cast, MEMORY * b + EVIDENCE * (1 - cast))
                            weighed.add(key)
                            dX = (pm.features(Xm[away]) - Fs[mem[away]]).sum(axis=0)
                            P = np.mean([chances(H + t * dX) for t in self._shares(key)], axis=0)
                            g = (u[mem] @ P - vH) / span[mem]          # against the side at home
                            Sw, W, _, _ = self.memory.get(key, (np.zeros(n), np.zeros(n), None, None))
                            Sw, W = MEMORY * Sw, MEMORY * W
                            Sw[mem] += g
                            W[mem] += 1.0
                            self.memory[key] = (Sw, W, mem[away & able[mem]], Xm[away & able[mem]])
        # moves not weighed this round (their race is not among the poll's top three) are remembered as
        # gaining nothing now: their value fades
        for key, (Sw, W, who, Xw) in self.memory.items():
            if key not in weighed:
                W = MEMORY * W
                W[who] += 1.0
                self.memory[key] = (MEMORY * Sw, W, who, Xw)
        for key, (Sw, W, who, Xw) in self.memory.items():
            g = Sw[who] / np.maximum(W[who], 1e-12)                    # remembered over the rounds
            for k in np.flatnonzero(g > best_gain[who]):
                v = int(who[k])
                best_gain[v], best_x[v] = g[k], Xw[k]
        self.weighed = weighed
        newB = sincere.copy()                           # home, unless a move gains more than their inertia
        for v, x in best_x.items():
            if best_gain[v] > theta[v]:
                newB[v] = x
        ctx.active = ~np.all(np.isclose(newB, sincere), axis=1)
        self.stats.append((float(np.mean(np.any(newB != B, axis=1))), len(weighed)))
        return newB, 0.0, -1
