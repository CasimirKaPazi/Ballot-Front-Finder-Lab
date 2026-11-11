"""Individual voters reading the average of the polls (run.py --engine arithmetic, and exact: the same
with each move's poll counted again; individual voters since 2026-10-31). strategy.rounds calls the engine
(ArithmeticEngine, ExactEngine) once per round.

- The polls of a round, averaged (a poll of polls): the totals of `samples` samples of the voters (as
  voters count them, Method.poll_view), scaled up to the electorate; read as one poll of their
  combined size.
- How close each candidate is to winning, in voters: the smallest number of voters switching from the
  poll winner A to candidate c that makes c the winner (closeness). A switching voter changes one
  ballot that supports A over c into the same ballot supporting c over A (switch_pair), so only A and
  c change on it; the count of the poll with k such switches is the count of its totals plus k times
  that change. The runner-up B is the closest candidate, the next place T the next closest.
- The chance of a candidate c behind: Phi(-k_c / sigma), sigma = poll_error x the sampling error of a
  two-way split, in voters (n x 0.5 / sqrt(poll size)); A has the rest (B and T only).
- A voter compares their sincere ballot with compromise for one of the poll's places (A, B, T;
  supporting it as far as the format allows: obvious), burial of one (it and everyone liked no more
  to the bottom, where the format allows; strict rankings: it last: burial) and both (compromise for
  one, burial of another; for a voter preferring A, compromise for T with burial of B is the
  push-over), by simple arithmetic on the poll: the change of their own ballot (from what they voted
  in the last round) is worth so many voters switching from A to c (its projection on one switching
  voter's change of the totals), for the runner-up and the next place, and how each of these races
  moves with the other (Reading.cross, a scenario table read from the poll: a runoff method can put B
  out when T passes it, which makes a push-over pay; under a Condorcet method T beating A can make a
  cycle that B wins, which makes burial of A pay). They expect f voters to do the same: themselves
  and --coordination of their camp, the voters with the same favourite (bloc); c needs that many
  times fewer (or more) voters, below 0 c leads, and the chances follow.
- A move's value is its gain in expected outcome over the sincere ballot, as a share of the voter's
  utility range (a promise weighted by trust: judge). The voter makes the most valuable move if its
  value exceeds their cost of voting insincerely (threshold: 0 for always strategic voters, their
  draw x --honesty for adaptive ones, never for sincere ones), else votes sincerely. Nobody recounts
  the poll with their ballot changed: what a move does under a method is for the results to show,
  not for voters to foresee. (Until 2026-11-04: a stake gate, a boldness per voter and a test against
  the poll's uncertainty, chained; docs/model.md, Individual voters.)
"""
from math import erf, isfinite, sqrt

import numpy as np

import methods
from ballots import BALLOTS
from . import moves as mv
from .common import IndividualVoters, judge, trust, aims  # noqa: F401 (aims: for the tests)
from .polls import poll_totals


STEPS_FULL = 14      # bisection steps for the closeness of a candidate, over 0 .. n voters


GRID = (-24, -12, -6, -2, 0, 2, 6, 12, 24)   # scenario table: voters switching between A and another


STEPS_TABLE = 10     # bisection steps in the table (0.1 voter)


def _Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def switch_pair(ballot, C, A, c):
    """Two ballots of the format: one supporting A over c, the other the same with A and c swapped.
    Rankings: A first, c second, the rest in candidate order; one name: A; approval: A alone; scores
    and grades: the top for A, 0 for the rest; two-round ballots: round 1 as above, runoff ranking as
    for rankings; veto: against c. Formats of add-ons: the sincere ballot of utilities with A at the
    top, c at the bottom, the rest in between."""
    rest = [x for x in range(C) if x not in (A, c)]

    def rank_points(order, k=None):
        b = np.zeros(C)
        k = len(order) if k is None else k
        for pos, x in enumerate(order[:k]):
            b[x] = C - 1 - pos
        return b

    def swap(b):
        b = b.copy()
        b[[A, c]] = b[[c, A]]
        return b

    S = methods.MAXS
    if ballot in ("fav",):
        b = np.eye(C)[A]
    elif ballot == "approval":
        b = np.eye(C)[A]
    elif ballot == "score":
        b = S * np.eye(C)[A]
    elif ballot == "grade3":
        b = 2 * np.eye(C)[A]
    elif ballot in ("rank", "strict", "weak"):
        b = rank_points([A, c] + rest)
    elif ballot in ("trunc", "trunc3"):
        b = np.zeros(C)
        b[A], b[c] = 2, 1
    elif ballot == "veto":
        b = -np.eye(C)[c]
    elif ballot in ("fav_run", "app_run"):
        b = rank_points([A, c] + rest) + C * np.eye(C)[A]
    elif ballot == "score_run":
        b = rank_points([A, c] + rest) + C * S * np.eye(C)[A]
    else:
        u = np.full((1, C), 0.5)
        u[0, A], u[0, c] = 1.0, 0.0
        b1 = BALLOTS[ballot].sincere(u, 0.0)[0] if not BALLOTS[ballot].approval else BALLOTS[ballot].sincere(u, 0.0, 0.5)[0]
        u2 = u.copy()
        u2[0, [A, c]] = u2[0, [c, A]]
        b2 = BALLOTS[ballot].sincere(u2, 0.0)[0] if not BALLOTS[ballot].approval else BALLOTS[ballot].sincere(u2, 0.0, 0.5)[0]
        return b1, b2
    if ballot == "veto":
        return b, -np.eye(C)[A]
    if ballot in ("fav_run", "app_run", "score_run"):
        mark = np.floor_divide(b, C)
        runoff = np.mod(b, C)
        return b, swap(mark) * C + swap(runoff)
    return b, swap(b)


def _wins(count, rows, c, C):
    """Does c win the counts in rows (k x d)? (at least half of the winner lottery)"""
    return count.decide_many(rows, C)[:, c] >= 0.5 - 1e-9


def _closeness(count, T, c, C, delta, lo, hi, steps):
    """Bisection for one candidate c over rows of totals T (k x d), between lo and hi (arrays)."""
    win_lo = _wins(count, T + lo[:, None] * delta[None], c, C)
    win_hi = _wins(count, T + hi[:, None] * delta[None], c, C)
    a, b = lo.copy(), hi.copy()
    for _ in range(steps):
        mid = (a + b) / 2
        w = _wins(count, T + mid[:, None] * delta[None], c, C)
        b = np.where(w, mid, b)
        a = np.where(w, a, mid)
    out = (a + b) / 2
    out = np.where(win_lo, lo, out)
    return np.where(win_hi | win_lo, out, np.inf)


def _chances(kB, kT, sigma):
    pA, (pB, pT) = _chances_many([kB, kT], sigma)
    return pA, pB, pT


def _chances_many(ks, sigma):
    """Chances of the candidates behind A, from their closeness (one array per race, rows alike):
    Phi(-k / sigma) each, scaled down together if they add up to more than 1; A has the rest."""
    if not ks:
        return np.ones(1), []
    sg = float(sigma)                         # plain floats: the same numbers, much faster than numpy scalars
    ps = [np.array([_Phi(-x / sg) if isfinite(x) else 0.0 for x in np.atleast_1d(k).astype(float).tolist()])
          for k in ks]
    tot = sum(ps)
    scale = np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0)
    ps = [p * scale for p in ps]
    return 1 - sum(ps), ps


class Reading:
    """What one poll says: the winner A, the runner-up B and next place T (by closeness), their
    closeness in voters, and the chances."""

    def __init__(self, count, ballot, T, C, n, m, poll_error, exact=False):
        self.count, self.C, self.n = count, C, n
        self.exact = exact               # --engine exact: moves valued by _value_exact
        self.sigma = max(poll_error, 1e-9) * n * 0.5 / sqrt(m)
        P = count.decide_many(T[None], C)[0]
        self.A = A = int(np.argmax(P))
        self.T0 = T
        self.delta = {}
        ks = {}
        for c in range(C):
            if c == A:
                continue
            b1, b2 = switch_pair(ballot, C, A, c)
            d = count.features(b2[None])[0] - count.features(b1[None])[0]
            self.delta[c] = d
            ks[c] = _closeness(count, T[None], c, C, d, np.zeros(1), np.full(1, float(n)), STEPS_FULL)[0]
        order = sorted(ks, key=lambda c: (ks[c], c))
        self.B = order[0] if order and np.isfinite(ks[order[0]]) else None
        self.T = order[1] if len(order) > 1 and np.isfinite(ks[order[1]]) else None
        self.k = ks
        # the races a move is valued in: A against B and T (against every candidate who can win was a
        # trial of 2026-11-04, journal.md)
        self.races = [c for c in (self.B, self.T) if c is not None]
        pA, ps = _chances_many([ks[c] for c in self.races], self.sigma)
        self.P = np.zeros(C)
        self.P[A] = pA[0]
        for c, p in zip(self.races, ps):
            self.P[c] = p[0]
        # How each race moves with the other (the cross-effects): a scenario table read from the poll, for
        # x voters switching from A to d (x < 0: from d to A), the closeness of c again (c, d = B, T and T,
        # B). Plurality: voters leaving A for T bring B closer as well; a runoff method: T passing B in the
        # first round can put B out of the runoff (push-over); a Condorcet method: T beating A can make a
        # cycle that B wins (burial of A under T). The table covers the scale of the moves voters imagine.
        self.grid = np.array([x for x in GRID if abs(x) <= n], float)
        self.curve = {}
        pairs = [(c, d) for c in (self.B, self.T) if c is not None for d in self.races if d != c]
        if self.B is not None and self.T is not None:
            for c, d in pairs:
                rows = T[None] + self.grid[:, None] * self.delta[d][None]
                kc = _closeness(count, rows, c, C, self.delta[c], np.zeros(len(rows)),
                                np.full(len(rows), float(n)), STEPS_TABLE)
                kc[self.grid == 0] = ks[c]
                self.curve[(c, d)] = np.minimum(kc, float(n))

    def cross(self, c, d, x):
        """c's closeness with x voters switching from A to d (per voter, interpolated in the table)."""
        k = self.curve.get((c, d))
        if k is None:
            return np.full(np.shape(x), min(self.k[c], float(self.n)))
        return np.interp(x, self.grid, k)


def obvious(ballot, A, Bo):
    """The obvious move for the leader A against the other leader Bo: compromise (see the module
    docstring), as a move of moves.py."""
    if ballot == "veto":
        return (("bottom", Bo),)
    if ballot in ("approval", "app_run"):
        return (("upto", A), ("below", A))
    if ballot in ("score", "grade3", "score_run") or BALLOTS[ballot].score_range:
        return (("upto", A),)
    return (("top", A),)


def burial(ballot, A, Bo):
    """Burial of the other leader Bo, as far as the format allows without putting Bo below anyone the
    voter likes less where it can: Bo and everyone liked no more than Bo to the bottom (scores and
    grades: the lowest score; approval: not approved; rankings that can be truncated or tied: unranked
    or tied last). Strict rankings can only put Bo last, below candidates liked less (a reversal: burial
    proper). None where the format has no burial (one name, two-round marks, veto)."""
    if ballot in ("fav", "fav_run", "veto"):
        return None
    if ballot == "strict":
        return (("bottom", Bo),)
    return (("downfrom", Bo),)


def _value(r, S, U, C):
    """Chances and expected outcome after a change S of the poll totals (rows, one per row of U): its
    worth in voters switching from A to each candidate of the races (projection on delta), with the
    cross-effects on B and T (Reading.cross: each other race's switches move their closeness, added
    up). Returns (P, EU): winner chances per row, and the expected outcome of U."""
    races = r.races
    if r.exact:
        return _value_exact(r, S, U, C)
    s = {c: (S @ r.delta[c]) / max(float(r.delta[c] @ r.delta[c]), 1e-12) for c in races}
    k = {}
    for c in races:
        if c in (r.B, r.T):
            k0 = min(r.k[c], float(r.n))
            base = np.full(len(S), k0)
            for d in races:
                if d != c and (c, d) in r.curve:
                    base = base + (r.cross(c, d, s[d]) - k0)
            k[c] = np.where(np.isfinite(r.k[c]) | (base < r.n), base, np.inf) - s[c]
        else:
            k[c] = r.k[c] - s[c]
    pA, ps = _chances_many([k[c] for c in races], r.sigma)
    P = np.zeros((len(S), C))
    P[:, r.A] = pA
    for c, p in zip(races, ps):
        P[:, c] += p
    return P, (P * U).sum(axis=1)


def _value_exact(r, S, U, C):
    """_value with the exact response (run.py --engine exact): the averaged poll counted again with each
    change, and the closeness of the races found by the same bisection as for the poll itself (from 0 to
    n voters switching from A; for a race leading after the change, how far, down to 3 sigma below 0,
    where the counts still mean something). The belief is the arithmetic's: the same races, spread and
    chances. Distinct changes are counted once."""
    Su, inv = np.unique(np.round(S, 9), axis=0, return_inverse=True)
    inv = inv.ravel()
    rows = r.T0[None] + Su
    z, hi, neg = np.zeros(len(Su)), np.full(len(Su), float(r.n)), np.full(len(Su), -3 * r.sigma)
    k = []
    for c in r.races:
        kc = _closeness(r.count, rows, c, C, r.delta[c], z, hi, STEPS_FULL)
        lead = kc <= 0
        if lead.any():
            kc[lead] = _closeness(r.count, rows[lead], c, C, r.delta[c], neg[lead], z[lead], STEPS_FULL)
        k.append(kc[inv])
    pA, ps = _chances_many(k, r.sigma)
    P = np.zeros((len(S), C))
    P[:, r.A] = pA
    for c, p in zip(r.races, ps):
        P[:, c] += p
    return P, (P * U).sum(axis=1)


CAMPAIGN_SEARCH = 1  # random variations of their last recommendation each candidate tries per round


def plan(method, count, u, sincere, F_cur, r, strategies, share, cand_u, memory=None, rng=None):
    """Every candidate's campaign for this round (docs/campaigns.md, Campaigns for individual voters).
    A candidate's audience: the voters who prefer them to their strongest rival in the poll (A; for A,
    the runner-up B). The moves considered: the round's strategies (the moves the voters consider),
    "X above Y" for every pair with the poll winner A, and CAMPAIGN_SEARCH random variations of each
    candidate's last recommendation and that recommendation again (their own search, for them alone;
    memory["camp_held"], rng). For each move, a share `share` of the audience casting it changes the
    totals by that share of the audience's changes (each voter's ballot following the move against
    what they voted in the last round); the candidate values that change with the same arithmetic as
    the voters (_value), by their own preferences (cand_u: one row per candidate), and recommends the
    best move if it gains them anything over no campaign (campaigning costs a candidate nothing;
    whether a recommendation is worth following is for each voter to judge, with the same test of the
    poll's uncertainty as every move). Nobody recounts the poll (a recount per candidate, move and
    poll more than doubled the run time).
    Returns {c: (move, dP)}: the move and the change of the winner chances it promises if followed as
    planned, and the ballots of every voter following each recommended move ({move: ballots})."""
    n, C = u.shape
    if r.B is None:
        return {}, {}
    rival = np.full(C, r.A)
    rival[r.A] = r.B
    AUD = (u > u[:, rival]).T.astype(float)                       # AUD[c, v]: v prefers c to c's rival
    cs = np.flatnonzero(AUD.sum(axis=1) > 0)
    held = memory.setdefault("camp_held", {}) if memory is not None else {}
    if not len(cs):
        held.clear()
        return {}, {}
    pos = {int(c): i for i, c in enumerate(cs)}
    every = np.arange(len(cs))
    moves = {}                                                     # move -> the candidates who consider it
    for m in list(strategies) + [(("above", X, r.A),) for X in range(C) if X != r.A] \
            + [(("above", r.A, Y),) for Y in range(C) if Y != r.A]:
        moves.setdefault(m, set()).update(every.tolist())
    for c in cs:                                                   # their own search: the last recommendation
        own = [held[int(c)]] if int(c) in held else []             # again and random variations of it
        if rng is not None:
            own += [mv.random_move(held.get(int(c), ()), C, rng) for _ in range(CAMPAIGN_SEARCH)]
        for m in own:
            moves.setdefault(m, set()).add(pos[int(c)])
    lazy, appr = mv.own_params(np.arange(n))
    batch, rows, rendered = [], [], {}
    for m, who in moves.items():
        X = mv.render(method, u, m, sincere, lazy, appr)
        D = count.features(X) - F_cur                              # each voter's change, following it
        idx = cs[np.array(sorted(who))]
        batch.append((m, idx))
        rows.append(share * (AUD[idx] @ D))                        # one row per candidate considering it
        rendered[m] = X
    owner = np.concatenate([idx for _, idx in batch])
    P, _ = _value(r, np.concatenate(rows), cand_u[owner], C)       # all moves of the round at once
    dP = P - r.P[None]
    gain = (dP * cand_u[owner]).sum(axis=1)
    best, ballots, at = {}, {}, 0
    for m, idx in batch:
        for k in range(len(idx)):
            c, g = int(idx[k]), gain[at + k]
            if g > 1e-9 and (c not in best or g > best[c][0] + 1e-12):
                best[c] = (float(g), m, dP[at + k])
                ballots[m] = rendered[m]
        at += len(idx)
    held.clear()
    held.update({c: m for c, (_, m, _) in best.items()})
    recs = {c: (m, dP) for c, (_, m, dP) in best.items()}
    return recs, {m: ballots[m] for m, _ in recs.values()}


def bloc(u, share):
    """How many voters each voter expects to vote as they do: themselves and a share `share`
    (--coordination) of the others whose favourite is theirs (a camp acts alike; the poll is all a voter
    knows, but the size of their own camp they can judge)."""
    fav = u.argmax(axis=1)
    camp = np.bincount(fav, minlength=u.shape[1])[fav]
    return 1.0 + share * (camp - 1)


def answer(method, count, u, B, sincere, reading, *, threshold, followers, memory=None, rng=None, search=0,
           campaign=None):
    """One round. threshold: per voter, the cost of voting insincerely, as a share of their own
    utility range (0 = always strategic, inf = never); followers: per voter, how many voters they
    expect to vote as they do, or a number: the share --coordination of their camp (bloc). Returns the
    new ballots and the voters who vote strategically this round (bool).

    A voter compares their sincere ballot with compromise for one of the poll's places (A, B, T),
    burial of one, and both (different places): for each ballot X, the change of their own ballot (from
    what they voted in the last round) is worth w_c = (change . delta_c) / |delta_c|^2 switching voters
    in the race between the poll winner A and c (delta_c: one voter switching from A to c), for the
    runner-up B and the next place T; with f voters making it, c needs k_c - f w_c switching voters
    (below 0: c leads), and the chances follow (_chances, with the poll's uncertainty). With rng and
    search, each voter also tries `search` random variations of the move they hold (discovery). A
    move's value is its gain in the voter's expected outcome over the sincere ballot, as a share of
    their utility range, a promise weighted by their trust in it (judge; memory: {"last": ...,
    "trust": ...}, kept across the rounds). The voter makes the most valuable move if its value exceeds
    their cost, else votes sincerely.
    campaign: None (off) or {"share": --coordination, "cand_u": candidates' utilities}: every candidate
    may recommend a move (plan); each voter considers the recommendation of the candidate they like most
    among those campaigning, valued by what it promises if followed as planned (their expected outcome
    now, plus the change of the winner chances the candidate worked out), provided their own part alone
    (with their followers) does not harm them. The voters following each candidate are counted in
    memory["follow"]."""
    n, C = u.shape
    r = reading
    if memory is not None:
        judge(u, r, memory)
    newB = sincere.copy()
    if memory is not None:
        memory["follow"] = np.zeros(C)
    if r.B is None:
        return newB, np.zeros(n, bool)
    lo_u, hi_u = u.min(axis=1), u.max(axis=1)
    span = np.where(hi_u > lo_u, hi_u - lo_u, 1.0)
    able = np.isfinite(threshold)                                  # sincere voters never move
    if memory is not None:                                         # voters who sit this round out vote
        for v in [v for v in memory.get("held", {}) if not able[v]]:   # sincerely: their move is let go
            memory["held"].pop(v)
    places = [c for c in (r.A, r.B, r.T) if c is not None]
    F_cur = count.features(B)
    V = np.flatnonzero(able)
    if not len(V):
        return newB, np.zeros(n, bool)
    lazy, appr = mv.own_params(V)
    lazy = np.atleast_1d(lazy) if np.ndim(lazy) else lazy
    appr = np.atleast_1d(appr) if np.ndim(appr) else appr
    # the strategies: compromise for one of the places, burial of one, or both (different places)
    strategies = []
    for c in places:
        strategies.append(obvious(method.ballot, c, next(x for x in places if x != c)))
    for c in places:
        b = burial(method.ballot, None, c)
        if b is not None:
            strategies.append(b)
            for d in places:
                if d != c:
                    strategies.append(obvious(method.ballot, d, c) + b)
    seen, uniq = set(), []
    for m in strategies:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    options = [sincere[V]] + [mv.render(method, u[V], m, sincere[V], lazy, appr) for m in uniq]
    keys = [[None] * len(V)] + [[m] * len(V) for m in uniq]      # the move behind each option, per voter
    # Discovery: `search` random variations per voter, each one random step after the move they hold (or
    # after their sincere ballot), valued like every other move: taken only if the poll arithmetic
    # promises more, and dropped again when the next poll shows it did not deliver (judge).
    if memory is not None and rng is not None:
        held = memory.setdefault("held", {})
        for _ in range(search):
            ks = [mv.random_move(held.get(int(v), ()), C, rng) for v in V]
            rows = [mv.render(method, u[v:v + 1], ks[i], sincere[v:v + 1],
                              np.atleast_1d(lazy[i] if np.ndim(lazy) else lazy),
                              np.atleast_1d(appr[i] if np.ndim(appr) else appr))[0] for i, v in enumerate(V)]
            options.append(np.array(rows))
            keys.append(ks)
    f = (bloc(u, float(followers)) if np.ndim(followers) == 0 else np.asarray(followers, float))[V]
    eu = []
    for X in options:
        D = count.features(X) - F_cur[V]                           # the change of each voter's own ballot,
        eu.append(_value(r, f[:, None] * D, u[V], C)[1])           # made by f voters (worth and cross-effects)
    # Campaigns: each voter considers the recommendation of the candidate they like most among those
    # campaigning, valued by what it promises if followed as planned (the candidate's chances with the
    # campaign), if their own part, made by f voters as any move, does not harm them.
    camp_of = None
    if campaign is not None:
        recs, rec_ballots = plan(method, count, u, sincere, F_cur, r, uniq, campaign["share"], campaign["cand_u"],
                                 memory, rng)
        if recs:
            cs = np.array(sorted(recs))
            camp_of = cs[np.argmax(u[V][:, cs], axis=1)]           # ties: the first campaigning candidate
            X = np.stack([rec_ballots[recs[int(c)][0]][v] for c, v in zip(camp_of, V)])
            own = _value(r, f[:, None] * (count.features(X) - F_cur[V]), u[V], C)[1]
            promise = u[V] @ r.P + np.array([float(recs[int(c)][1] @ u[v]) for c, v in zip(camp_of, V)])
            options.append(X)
            keys.append([recs[int(c)][0] + (("campaign", int(c)),) for c in camp_of])
            eu.append(np.where(own >= eu[0] - 1e-12, promise, -np.inf))
    EU = np.stack(eu)
    before = u[V] @ r.P                                            # expected outcome keeping one's ballot
    if memory is not None:                                         # promises weighted by trust in the move
        for j in range(1, len(options)):
            rho = np.array([trust(memory, int(v), keys[j][i]) for i, v in enumerate(V)])
            gain = EU[j] - before
            EU[j] = before + np.where(gain > 0, rho * gain, gain)  # trust discounts promises, never makes
                                                                   # a harmful move look good
    # The most valuable move, made if its value (gain over the sincere ballot, as a share of the voter's
    # utility range) exceeds the voter's cost of voting insincerely
    best = np.argmax(EU, axis=0)                                   # ties: the first (sincere first)
    value = (EU[best, np.arange(len(V))] - EU[0]) / span[V]
    best = np.where(value > threshold[V], best, 0)
    for j, X in enumerate(options):
        sel = best == j
        newB[V[sel]] = X[sel]
        if memory is not None:
            for i in np.flatnonzero(sel):
                v = int(V[i])
                if j > 0:
                    memory["last"][v] = (keys[j][i], float(eu[j][i] - before[i]), float(before[i]))
                    move = tuple(st for st in keys[j][i] if st[0] != "campaign")
                    memory.setdefault("held", {})[v] = move
                    if move != keys[j][i]:                         # followed a campaign
                        memory["follow"][keys[j][i][-1][1]] += 1
                else:
                    memory.setdefault("held", {}).pop(v, None)
    active = np.zeros(n, bool)
    active[V[best > 0]] = True
    return newB, active


class ArithmeticEngine(IndividualVoters):
    """Individual voters (--engine arithmetic), one round at a time: the average of the round's polls is
    read (Reading) and each voter answers it alone (answer), a move valued by arithmetic (_value).
    campaign: None (off) or {"share", "cand_u"} (plan)."""
    OPTIONS = IndividualVoters.OPTIONS + ("campaign",)
    exact = False

    def __init__(self, ctx, coordination=0.25, honesty=0.3, campaign=None):
        super().__init__(ctx, coordination, honesty)
        self.campaign = campaign

    def round(self, r, polls):
        """One round: the new ballots, the share of all voters following a campaign recommendation and
        the candidate whose recommendation has the most followers (-1: none)."""
        ctx = self.ctx
        F, tot, WF = poll_totals(ctx.pm, ctx.B, polls, ctx.n, ctx.m)
        T0 = tot + ctx.poll_error * (WF.mean(axis=0) - tot)      # the average of the polls
        reading = Reading(ctx.pm, ctx.method.ballot, T0, ctx.C, ctx.n, ctx.m * ctx.samples, ctx.poll_error,
                          exact=self.exact)
        if ctx.leaders is not None:
            ctx.leaders[r] = (reading.A, reading.B)
        # (answer is looked up when called, so that tools/response_check.py can put its own in)
        B, ctx.active = answer(ctx.method, ctx.pm, ctx.u, ctx.B, ctx.sincere, reading, threshold=self.threshold,
                               followers=self.coordination, memory=self.memory, rng=self.rng, search=ctx.search,
                               campaign=self.campaign)
        fw = self.memory["follow"]
        return B, float(fw.sum()) / ctx.n, int(np.argmax(fw)) if fw.sum() > 0 else -1


class ExactEngine(ArithmeticEngine):
    """--engine exact: as arithmetic, each move's poll counted again (_value_exact; a variant for analysis)."""
    exact = True
