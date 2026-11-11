"""The ranking groups (run.py --engine groups, the engine of version 1.0.0).
  - Leaders: the two candidates that win most often over the round's polls, the same for
    everyone. If one candidate wins every poll, the polls are widened (deviations x 2, 4, ...,
    counts kept on the side of zero of the true count) until another one wins somewhere: the race
    that would matter first. (polls._leaders)
  - Groups are whole ranking groups (strategic voters with the same sincere ranking, no cap):
    voters who share a ranking and see the same poll reach the same answer, so they act as a
    bloc. Factions across rankings are not built in; they emerge when groups answer alike.
  - Answer: a group estimates the count for a ballot b as poll count - its own previous ballots
    + its size x b, and judges b by its expected utility gain over its sincere ballots, over
    the polls. It first tries a few candidate ballots: its previous answer, the naive
    exaggerations of its ballot type around the two leaders (A = the one it prefers, B = the
    other; `exaggerations`): approve everyone liked at least as much as A; top score down to A,
    0 from B down, stretched in between; vote A; A to the top and/or B to the bottom of a
    ranking; a ranking truncated after A, or A alone; and the like; and a push-over: the ballot for A
    against B with the next place in the polls supported too (`pushover` in ballots.py). Aggressive voters take the best if it does not lose in the polls:
    exaggeration wins ties, they exaggerate when it cannot hurt, whatever the gap. Kind voters
    take it only if it strictly gains (honesty wins ties). `aggressive` marks the aggressive
    voters (None = all; drawn in run.py); aggressive and kind voters form separate groups. Then random
    search (`search` proposals, default 1, each a small change of the best ballot so far) takes a
    change only if it strictly gains. Since the previous answer is tried first, the search goes on
    over the rounds from where it stood. A group none of whose candidates is safe and whose search finds no
    gain stays sincere.
  - Adaptive voters decide each round whether to vote strategically, by their stake in the race that
    matters: |u(A) - u(B)| for the round's two leaders A and B, rescaled to 0..1 between their least and
    most liked candidate. A voter is strategic in a round if the stake exceeds their personal threshold,
    else sincere in that round; strategic ones join the group of their ranking. Supporters of both
    leaders have a stake in a close race, so attack and defence come from the same rule.
  - Risk (--risk 1, docs/campaigns.md): instead of the fixed aggressive split, each voter is careful or
    bold by their satisfaction with the polls. Campaigns (--campaign 1): every candidate recommends a
    move each round (_campaigns); bold (aggressive) groups consider it.
"""
import numpy as np

import methods
from outcomes import value
from ballots import BALLOTS
from .polls import poll_tallies, poll_totals, TOL, _widened, _smaller, _leaders


def mutate(b, ballot, rng):
    """One small change to a single ballot vector (length C) of type `ballot` (ballots.py)."""
    return BALLOTS[ballot].mutate(b.copy(), rng)


def _gain(method, T, F_old, sincere_rows, size, C, Ug):
    """For a group: gain(ballot) = expected utility gain over the polls (as voters see them) of the
    group casting this ballot instead of its sincere ballots."""
    F_sin = method.features(sincere_rows).sum(axis=0)
    base = T - F_old + F_sin                       # the polls with the group voting sincerely
    u_sin = value(method.decide_many(base, C), Ug)       # per poll
    cache = {}

    def gain(ballot):
        key = ballot.tobytes()
        if key not in cache:
            D = size * method.features(ballot[None])[0] - F_sin
            cache[key] = float(np.mean(value(method.decide_many(base + D, C), Ug) - u_sin))
        return cache[key]
    return gain


def _gain_full(method, B, mem, sincere_rows, polls, Ug):
    """Methods without feature sums: the group's sampled members changed in each poll."""
    B0 = B.copy()
    B0[mem] = sincere_rows
    u_sin = value(poll_tallies(method.tally, B0, polls), Ug)
    cache = {}

    def gain(ballot):
        key = ballot.tobytes()
        if key not in cache:
            B2 = B.copy()
            B2[mem] = ballot
            cache[key] = float(np.mean(value(poll_tallies(method.tally, B2, polls), Ug) - u_sin))
        return cache[key]
    return gain


CAMPAIGN_NUDGE = 1e-6    # campaign moves: "a little more" = this share of the voter's range above


def _lift(U, X, Y):
    """Utilities with X moved to just above Y in the rows that prefer Y to X (the others unchanged)."""
    U = np.array(U, dtype=float)
    span = U.max(axis=1) - U.min(axis=1)
    need = U[:, Y] >= U[:, X]
    U[need, X] = U[need, Y] + CAMPAIGN_NUDGE * np.where(span[need] > 0, span[need], 1.0)
    return U


def _move_ballots(method, U, X, Y, lazy=None, approval=None):
    """The ballots for the campaign move "X above Y" (docs/campaigns.md, The moves), one row per row of
    U: the sincere ballot of the utilities with X moved to just above Y where the voter prefers Y
    (the others vote sincerely). For the Inverse-* methods the ballot is read the ordinary way, as the
    naive exaggerations are: following a campaign does not bring back the reversed honest ballot.
    Two-round ballots keep their sincere runoff ranking: only the round-1 marks follow the campaign."""
    pref = -_lift(U, X, Y) if method.inverse else _lift(U, X, Y)     # Method.sincere reverses Inverse-*
    M = method.sincere(pref, lazy, approval)
    if BALLOTS[method.ballot].marks is not None:          # two-round: the runoff ranking stays sincere
        C = U.shape[1]
        M = np.floor_divide(M, C) * C + np.mod(method.sincere(U, lazy, approval), C)
    return M


def _campaigns(method, pm, u, B, F, T, polls, P_now, a, b, share, cand_u):
    """Every candidate's recommendation for this round (docs/campaigns.md, The candidate's plan).
    The moves: "X above Y" for every pair with the poll winner a (_move_ballots).
    A candidate's audience: the voters who prefer them to their strongest rival (a, or for a the runner-up
    b). For each move, a share `share` of the audience casts it instead of their current ballot (B, or
    its features F; T: the poll counts), and the candidate's expected utility over the polls (cand_u:
    candidates' utilities, one row per candidate) is compared with P_now (the polls as they are). Each
    candidate recommends their best move if that gains them something.
    pm: the method as voters count a poll (Method.poll_view); the ballots are the method's.
    Returns [(candidate, (X, Y), winner lotteries per poll with the campaign)]."""
    n, C = u.shape
    if b is None:
        return []
    rival = np.full(C, a)
    rival[a] = b
    AUD = (u > u[:, rival]).T.astype(float)              # AUD[c, v] = 1: voter v prefers c to c's rival
    cs = np.flatnonzero(AUD.sum(axis=1) > 0)
    if not len(cs):
        return []
    S = len(polls)
    now = (P_now.mean(axis=0) @ cand_u.T)[cs]           # each candidate's expected utility now
    best = {}
    pairs = [(x, a) for x in range(C) if x != a] + [(a, y) for y in range(C) if y != a]
    for X, Y in pairs:
        M = _move_ballots(method, u, X, Y)
        if F is not None:
            D = pm.features(M) - F                    # per voter: the change in their features
            rows = T[None, :, :] + share * (AUD[cs] @ D)[:, None, :]
            P = pm.decide_many(rows.reshape(-1, T.shape[1]), C).reshape(len(cs), S, C)
        else:                                             # no feature sums: the first share of the audience
            P = np.empty((len(cs), S, C))
            for i, c in enumerate(cs):
                aud = np.flatnonzero(AUD[c])
                take = aud[:int(np.ceil(share * len(aud)))]
                B2 = B.copy()
                B2[take] = M[take]
                P[i] = poll_tallies(pm.tally, B2, polls)
        val = np.einsum("isj,ij->i", P, cand_u[cs]) / S - now
        for i in np.flatnonzero(val > TOL):
            c = int(cs[i])
            if c not in best or val[i] > best[c][0] + TOL:
                best[c] = (float(val[i]), (X, Y), P[i])
    return [(c, mv, P) for c, (v, mv, P) in sorted(best.items())]


def exaggerations(ballot, Ug, a, b):
    """Naive exaggerations of a group's ballot around the two poll leaders a, b (Ug: the group's
    summed utilities; A = the leader it prefers, B = the other). Most complete first. What they are
    depends on the ballot type: its `exaggerate` in ballots.py."""
    if b is None:                                  # no runner-up (exact polls): no race to exaggerate for
        return []
    A, Bo = (a, b) if Ug[a] >= Ug[b] else (b, a)
    return BALLOTS[ballot].exaggerate(Ug, A, Bo)


def _answer(gain, ballot_type, ref, cands, search, rng, eager=True, tol=0.0, promises=None):
    """A group's answer to one poll round. First the candidate ballots (its previous answer, the
    naive exaggerations, campaign recommendations): the best one is taken if it does not lose in the
    polls (gain >= 0) for aggressive voters (eager: exaggeration wins ties, they exaggerate when it
    cannot hurt), or only if it strictly gains for kind voters (honesty wins ties). Then random search
    (`search` proposals, each a small change of the best ballot so far) from that ballot, or from the
    sincere one: a change is taken only if it strictly gains. The previous answer comes first, so the
    search goes on from where it stood in the round before. None = stay sincere.
    tol: --risk, bold voters accept a loss up to this much (in the group's summed utility).
    promises: one per candidate ballot, the gain a campaign recommendation promises if their audience
    follows (None for other ballots; only bold voters get recommendations). Among ballots that gain
    the same alone, the larger promise wins (docs/campaigns.md, The voter's choice)."""
    best, best_v, best_p = None, 0.0, 0.0
    for i, c in enumerate(cands):
        v = gain(c)
        p = promises[i] if promises is not None and promises[i] is not None else None
        if best is None:
            take = (v > TOL or (eager and v >= -TOL - tol)
                    or (eager and p is not None and p > TOL and v >= -TOL - tol))
        else:
            pp = v if p is None else p
            take = v > best_v + TOL or (abs(v - best_v) <= TOL and pp > best_p + TOL)
        if take:
            best, best_v, best_p = c, v, (v if p is None else p)
    cur = best if best is not None else ref
    for _ in range(search):
        prop = mutate(cur, ballot_type, rng)
        v = gain(prop)
        if v > best_v + TOL:
            best, best_v, cur = prop, v, prop
    return best


class GroupsEngine:
    """The ranking groups, one round at a time (strategy.rounds runs the rounds; the module docstring).
    ctx: the run's shared state (rounds.Context). aggressive: per voter, aggressive (True) or kind (None:
    all aggressive); campaign: None (off) or {"share", "cand_u"}; risk: None (off: the fixed aggressive
    split) or {"threshold": one per voter, 0..1, "kappa": tolerance factor}."""
    OPTIONS = ("aggressive", "campaign", "risk")

    def __init__(self, ctx, aggressive=None, campaign=None, risk=None):
        self.ctx, self.aggressive, self.campaign, self.risk = ctx, aggressive, campaign, risk
        n, u, method = ctx.n, ctx.u, ctx.method
        self.risky = risk is not None
        self.bold_v = np.zeros(n, bool)            # --risk: the voters bold this round (by satisfaction)
        self.tol_v = np.zeros(n)                   # --risk: each bold voter's tolerance for a loss
        self.keys = [tuple(r) for r in np.argsort(-u, axis=1, kind="stable")]
        self.lo, hi = u.min(axis=1), u.max(axis=1)
        self.span = np.where(hi > self.lo, hi - self.lo, 1.0)
        self.names, self.groups, self.eager, self.ref, self.Ug = self.setup(self.build())
        self.shared = {}                           # each group's answer in the last round (absent = sincere)
        self.places = 3 if BALLOTS[method.ballot].pushover is not None else 2   # the third: push-over target

    def bold(self):
        """The share of the strategic voters who are bold (--risk), else nan."""
        a = self.ctx.active
        return float((self.bold_v & a).sum()) / max(1, int(a.sum())) if self.risky else float("nan")

    def build(self):
        """Ranking groups of the active voters, aggressive (bold) and kind (careful) voters apart, in voter
        order. With --risk the attitude is the voter's this round, else the fixed split."""
        gs = {}
        for v, k in enumerate(self.keys):
            if self.ctx.active[v]:
                e = bool(self.bold_v[v]) if self.risky else (self.aggressive is None or bool(self.aggressive[v]))
                gs.setdefault((k, e), []).append(v)
        return gs

    def setup(self, gkeys):
        u, method = self.ctx.u, self.ctx.method
        groups = [np.array(x) for x in gkeys.values()]
        return (list(gkeys), groups, [k[1] for k in gkeys],
                [method.sincere(u[g].mean(axis=0, keepdims=True), methods.lazy_of(g), methods.approval_of(g))[0] for g in groups],
                [u[g].sum(axis=0) for g in groups])

    def round(self, r, polls):
        """One round: the new ballots, the share of all voters following a campaign recommendation and
        the candidate whose recommendation has the most followers (-1: none)."""
        ctx = self.ctx
        method, u, rng, n, C, pm, B, sincere = ctx.method, ctx.u, ctx.rng, ctx.n, ctx.C, ctx.pm, ctx.B, ctx.sincere
        inc, search, risky, camp = ctx.inc, ctx.search, self.risky, self.campaign is not None
        F = tot = dev = None
        if inc:
            F, tot, WF = poll_totals(pm, B, polls, n, ctx.m)
            dev = ctx.poll_error * (WF - tot)      # poll deviations from the true count, poll_error x sampling error
            a, b, *rest = _leaders(lambda j: pm.decide_many(_widened(tot, dev, j), C), C, self.places)
        else:
            a, b, *rest = _leaders(lambda j: poll_tallies(pm.tally, B, _smaller(polls, j, rng, n)), C, self.places)
        third = rest[0] if rest else None          # push-over target: the next place after the two leaders
        if ctx.leaders is not None:
            ctx.leaders[r] = (a, b)
        P_now = None                               # winner lotteries of the polls as they are (campaigns, risk)
        if camp or risky:
            P_now = pm.decide_many(tot + dev, C) if inc else poll_tallies(pm.tally, B, polls)
        rebuild = False
        if ctx.n_adaptive:                         # adaptive voters: strategic this round if the race matters to them
            stake = np.abs(u[:, a] - u[:, b]) / self.span if b is not None else np.zeros(n)
            now = ctx.fixed | (stake > ctx.q_adapt)
            if not np.array_equal(now, ctx.active):
                ctx.active, rebuild = now, True
        if risky:                                  # careful or bold, by the voter's satisfaction with the polls
            sat = (u @ P_now.mean(axis=0) - self.lo) / self.span
            now = sat < self.risk["threshold"]
            self.tol_v = np.where(now, self.risk["kappa"] * (self.risk["threshold"] - sat) * self.span, 0.0)
            if not np.array_equal(now & ctx.active, self.bold_v & ctx.active):
                rebuild = True
            self.bold_v = now
        if rebuild:
            self.names, self.groups, self.eager, self.ref, self.Ug = self.setup(self.build())
        names, groups, eager, ref, Ug, tol_v = self.names, self.groups, self.eager, self.ref, self.Ug, self.tol_v
        recs = []
        if camp:                                   # every candidate's recommendation (docs/campaigns.md)
            recs = _campaigns(method, pm, u, B, F if inc else None, tot + dev if inc else None, polls, P_now, a, b,
                              self.campaign["share"], self.campaign["cand_u"])
        answers, follow = [], np.zeros(C)
        for g, mem in enumerate(groups):
            if inc:
                gain = _gain(pm, tot + dev, F[mem].sum(axis=0), sincere[mem], len(mem), C, Ug[g])
            else:
                gain = _gain_full(pm, B, mem, sincere[mem], polls, Ug[g])
            prev = self.shared.get(names[g])
            cands = ([prev] if prev is not None else []) + exaggerations(method.ballot, Ug[g], a, b)
            push = BALLOTS[method.ballot].pushover
            if push is not None and third is not None:         # push-over: support the next place too
                A, Bo = (a, b) if Ug[g][a] >= Ug[g][b] else (b, a)
                cands.append(push(Ug[g], A, Bo, third))
            if BALLOTS[method.ballot].marks is not None:          # two-round: keep the group's own runoff part
                cands = [np.floor_divide(x, C) * C + np.mod(ref[g], C) for x in cands]
            promises = [None] * len(cands)
            camp_moves = []
            if recs and eager[g]:                  # recommendations are for bold (aggressive) voters
                now_v = float(np.mean(value(P_now, Ug[g])))
                # a group listens to one campaign: that of the candidate it likes most among those campaigning
                for c, (X, Y), P in [max(recs, key=lambda rc: Ug[g][rc[0]])]:
                    g_star = float(np.mean(value(P, Ug[g]))) - now_v       # what it promises the group
                    if g_star > TOL:
                        x = _move_ballots(method, u[mem].mean(axis=0, keepdims=True), X, Y,
                                          methods.lazy_of(mem), methods.approval_of(mem))[0]
                        cands.append(x)
                        promises.append(g_star)
                        camp_moves.append((c, x))
            x = _answer(gain, method.ballot, ref[g], cands, search, rng, eager[g],
                        tol=float(tol_v[mem].sum()) if risky and eager[g] else 0.0,
                        promises=promises if camp_moves else None)
            answers.append(x)
            if x is not None:
                for c, y in camp_moves:
                    if np.array_equal(x, y):
                        follow[c] += len(mem)
                        break
        self.shared = {k: x for k, x in zip(names, answers) if x is not None}
        B = sincere.copy()
        for mem, x in zip(groups, answers):       # everyone switches at once
            if x is not None:
                B[mem] = x
        return B, float(follow.sum()) / n, int(np.argmax(follow)) if follow.sum() > 0 else -1
