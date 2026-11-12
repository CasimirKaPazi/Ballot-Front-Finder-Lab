"""Inertia (run.py --engine inertia; docs/strategy-design.md): voters reason from their honest ballot,
their home. Each round a voter weighs the moves of the races they care about, guessing what the move
would do if people like them made it, and casts the best one if it beats their honest ballot by more
than their inertia; otherwise they vote honestly. strategy.rounds calls the engine (InertiaEngine) once
per round.

- Inertia: one trait per voter, theta, a share of their utility range: their unwillingness to leave
  their honest ballot. Always strategic voters have 0, adaptive ones their personal draw (uniform
  0..1) times --inertia, the others never leave home (inf). Going back toward home costs nothing.
- Sides: the poll's places (the leaders and the next place, as the ranking groups find them); each
  pair of them is a race, and in each race a voter is on the side of the candidate of the pair they
  prefer (none if they like both alike). Three places: three races, six sides, every voter on three.
- Moves: for a side, the moves for its candidate against the other (moves._naive: the naive
  exaggerations; moves._push: the push-over for the third place), each cast by every member from their
  own utilities.
- A voter's guess: they do not know how many on their side will make a move. They value it against
  their side at home: the round's polls counted again with the side's members back on their honest
  ballots, and with a share s of them casting the move instead, everyone else voting as the poll
  shows. The share is unknown, so the value is averaged over shares (--shares: the midpoints of four
  equal parts of 0..1). After a round in which the move was weighed, the voters see what share of their
  side cast it (alone or with other moves), and guess around that (a Beta distribution, its evidence --evidence per round, older
  rounds weighing --memory as much as the next; the shares then are its quantile midpoints).
- Memory: a move's value to a voter is a weighted average over the rounds it was weighed in (each
  earlier round weighing --memory as much as the next), so counter-moves seen before stay in mind. A move
  of a race no longer among the top three counts as gaining nothing in that round: its value fades.
- Decision: one step per round. A voter's ballot is their honest one with the moves they cast, one on
  top of the other (their stack). If the latest move's remembered value no longer exceeds their inertia
  (both as shares of their utility range), they take it back: their ballot goes back to the one before
  it, at no cost (a move further down is checked once it is the latest). Otherwise they add their best
  new move whose value exceeds their inertia and that fits on top of the others (InertiaEngine._on_top),
  if there is one. Each move is valued on its own, against the side at home, as above.
"""
import numpy as np
from scipy.special import betaincinv

from . import moves as mv
from .measures import implied_values
from .polls import poll_totals, _widened, _leaders



class InertiaEngine:
    """The inertia engine, one round at a time. ctx: the run's shared state (rounds.Context). The options
    (run.py flags, strategy.ENGINE_PARAMS):
      inertia        the largest inertia of an adaptive voter, a share of their utility range (--inertia)
      inertia_shape  an adaptive voter's inertia is inertia x q ** inertia_shape for their personal draw q,
                     uniform on 0..1: 1 uniform, below 1 more voters near the largest inertia, above 1 more
                     near 0 (--inertia-shape)
      shares         how many shares of their side a voter imagines making a move: the midpoints of that
                     many equal parts of 0..1, or of the Beta distribution once learned (--shares)
      evidence       the weight of one round's observed share against the uniform guess, Beta(1, 1)
                     (--evidence)
      memory         the weight of each earlier round against the round after it, for the values of moves
                     and the evidence (--memory)"""
    OPTIONS = ("inertia", "inertia_shape", "shares", "evidence", "memory")

    def __init__(self, ctx, inertia=1.0, inertia_shape=1.0, shares=4, evidence=4.0, memory=0.8):
        self.ctx = ctx
        self.shares = tuple((np.arange(int(shares)) + 0.5) / int(shares))
        self.evidence_weight, self.memory_weight = float(evidence), float(memory)
        q = np.where(ctx.adapt, ctx.q_adapt, 0.0)
        self.theta = np.where(ctx.fixed, 0.0, np.where(ctx.adapt, q ** float(inertia_shape) * inertia, np.inf))
        u = ctx.u
        lo, hi = u.min(axis=1), u.max(axis=1)
        self.span = np.where(hi > lo, hi - lo, 1.0)
        self.Fs = None                          # the honest ballots as polls count them (on first use)
        self.memory = {}                        # (side, move) -> (weighted sums of values, of weights, voters, ballots)
        self.evidence = {}                      # (side, move) -> (cast, not cast): evidence for the share
        self.weighed = set()                    # the (side, move) weighed in the round before
        self.stacks = {}                        # voter -> their moves cast, oldest first: [((A, B, move), ballot)]
        self.stats = []                         # per round: (share of all voters changing, moves weighed)

    def bold(self):
        return float("nan")

    def _shares(self, key):
        """The shares of their side a voter imagines making a move: the midpoints of four equal parts of
        0..1 (not knowing), or, once the move was weighed, the quartile midpoints of a Beta distribution
        with the shares seen as evidence."""
        if key not in self.evidence:
            return self.shares
        a, b = self.evidence[key]
        return tuple(float(x) for x in betaincinv(1.0 + a, 1.0 + b, np.array(self.shares)))

    def _on_top(self, v, stack, key):
        """Voter v's ballot with the move key on top of the moves of their stack, or None if the two do not
        fit: the ballot must differ from the one before and from the honest one, and in the race of every
        move in it, the side's candidate must stay above the other (by what the ballot says)."""
        ctx = self.ctx
        rows = np.array([v])
        steps = tuple(st for k, _ in stack for st in k[2]) + key[2]
        lazy, appr = mv.own_params(rows)
        x = mv.render(ctx.method, ctx.u[rows], steps, ctx.sincere[rows], lazy, appr)[0]
        if np.allclose(x, stack[-1][1]) or np.allclose(x, ctx.sincere[v]):
            return None
        val = implied_values(ctx.method, x[None])[0]
        if any(not val[A] > val[B] for (A, B, _), _ in stack + [(key, None)]):
            return None
        return x

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
                                cast = float(np.mean([any(k == key for k, _ in self.stacks.get(int(w), ()))
                                                      for w in mem[away]]))
                                a, b = self.evidence.get(key, (0.0, 0.0))
                                mw, ew = self.memory_weight, self.evidence_weight
                                self.evidence[key] = (mw * a + ew * cast, mw * b + ew * (1 - cast))
                            weighed.add(key)
                            dX = (pm.features(Xm[away]) - Fs[mem[away]]).sum(axis=0)
                            P = np.mean([chances(H + t * dX) for t in self._shares(key)], axis=0)
                            g = (u[mem] @ P - vH) / span[mem]          # against the side at home
                            Sw, W, _, _ = self.memory.get(key, (np.zeros(n), np.zeros(n), None, None))
                            Sw, W = self.memory_weight * Sw, self.memory_weight * W
                            Sw[mem] += g
                            W[mem] += 1.0
                            self.memory[key] = (Sw, W, mem[away & able[mem]], Xm[away & able[mem]])
        # moves not weighed this round (their race is not among the poll's top three) are remembered as
        # gaining nothing now: their value fades
        for key, (Sw, W, who, Xw) in self.memory.items():
            if key not in weighed:
                W = self.memory_weight * W
                W[who] += 1.0
                self.memory[key] = (self.memory_weight * Sw, W, who, Xw)
        self.weighed = weighed
        # each voter's moves worth more than their inertia (remembered over the rounds), best first
        value = {key: Sw / np.maximum(W, 1e-12) for key, (Sw, W, _, _) in self.memory.items()}
        offers = {}
        for key, (Sw, W, who, Xw) in self.memory.items():
            g = value[key][who]
            for k in np.flatnonzero(g > theta[who]):
                offers.setdefault(int(who[k]), []).append((g[k], key, Xw[k]))
        # one step per voter: back to the ballot before their latest move if it no longer pays,
        # otherwise their best new move that pays, on top of the moves they cast
        newB = sincere.copy()
        for v in set(self.stacks) | set(offers):
            stack = self.stacks.get(v, [])
            if stack and not value[stack[-1][0]][v] > theta[v]:
                stack = stack[:-1]
            else:
                held = {k for k, _ in stack}
                for g, key, x in sorted(offers.get(v, []), key=lambda o: -o[0]):
                    if key in held:
                        continue
                    if stack:
                        x = self._on_top(v, stack, key)
                        if x is None:
                            continue
                    stack = stack + [(key, x)]
                    break
            if stack:
                self.stacks[v] = stack
                newB[v] = stack[-1][1]
            else:
                self.stacks.pop(v, None)
        self.choice = {v: tuple(k for k, _ in st) for v, st in self.stacks.items()}   # diagnostics
        ctx.active = ~np.all(np.isclose(newB, sincere), axis=1)
        self.stats.append((float(np.mean(np.any(newB != B, axis=1))), len(weighed)))
        return newB, 0.0, -1
