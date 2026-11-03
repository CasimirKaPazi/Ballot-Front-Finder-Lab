"""Strategic voting in poll rounds (respond): a single election with a few polls before it.
  Round 0: every voter votes sincerely ("naive", uninformed).
  Round r (1 ... rounds): a poll of round r-1 is published, and every strategic voter answers it
  at the same time; nobody sees anyone else's answer before the next poll.
  Two engines answer: individual voters (individual.py; run.py --individual 1, the default since
  2026-10-31), each deciding alone from the averaged polls, or the ranking groups below
  (--individual 0, the engine of version 1.0.0). The polls are the same for both.
  - A poll publishes only the method's own counts (the summed ballot features the method decides
    from: first preferences for Plurality, approvals for Approval, the pairwise matrix for
    Condorcet methods, ...), from a random sample of `poll_size` voters (without replacement),
    scaled up to the electorate. `samples` independent polls (5) stand for the uncertainty.
    Nobody sees anyone's preferences or individual ballots. Only totals a real poll could show are
    published (per candidate, per pair, per grade): methods counted from totals over every subset of
    candidates are counted by the voters with a stand-in (Method.poll_view, POLL in methods.py).
  - Uncertainty: voters treat a poll as `poll_error` times less precise than its sampling error
    (default 2: real polls miss results by about twice their stated sampling error, Shirani-Mehr,
    Rothschild, Goel & Gelman 2018). The polls are spread around the true count by that factor.
  The rest of this docstring describes the ranking groups.
  - Leaders: the two candidates that win most often over the round's polls, the same for
    everyone. If one candidate wins every poll, the polls are widened (deviations x 2, 4, ...,
    counts kept on the side of zero of the true count) until another one wins somewhere: the race
    that would matter first. (_leaders)
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
  - Settled: round `rounds` gives the same ballots as the round before (a fixed point).
  rounds = 0 gives the sincere ballots.

Partial strategy: `strategic` marks the voters who take part (a random share of the
electorate, drawn in run.py, the same voters for every method). The others always vote sincerely.
Adaptive voters decide each round whether to vote strategically, by their stake
in the race that matters: |u(A) - u(B)| for the round's two leaders A and B, rescaled to 0..1
between their least and most liked candidate. A voter is strategic in a round if the stake exceeds
their personal threshold (`adaptive`: one value per voter, drawn once in run.py, uniform on 0..1,
nan = not adaptive), else sincere in that round; strategic ones join the group of their ranking.
Supporters of both leaders have a stake in a close race, so attack and defence come from the same
rule. The shares strategic per round go into `activated`.
"""
import numpy as np
import methods
from outcomes import value
from ballots import BALLOTS, _max_separated  # noqa: F401 (_max_separated: for the tests)

def mutate(b, ballot, rng):
    """One small change to a single ballot vector (length C) of type `ballot` (ballots.py)."""
    return BALLOTS[ballot].mutate(b.copy(), rng)


def _marks(method, X):
    """The part of the ballots the voter chooses (ballots.py, Ballot.marks): the round-1 marks of
    two-round ballots, else the whole ballot."""
    return BALLOTS[method.ballot].chosen(X)


def _reversal_mass(X, P):
    """Per voter: sum of P[a] - P[b] over the pairs with P[a] > P[b] that ballot X puts the other way."""
    w = np.clip(P[:, :, None] - P[:, None, :], 0, None)            # w[a, b] = P[a] - P[b] where positive
    return (w * (X[:, None, :] > X[:, :, None])).sum(axis=(1, 2))


def worst_reversal(method, P):
    """Per voter: the reversal mass of the most dishonest ballot the method's format allows (the scale
    of sincerity; its `worst` in ballots.py). Plurality / two-round with plurality: a vote for the last
    choice; veto: against the favourite; rankings (strict, weak, truncated): the reversed ranking;
    approval: the most damaging cut; score / three grades: the reversed order in as many levels as
    there are; at most 3 ranked (trunc3): the 3 least liked ranked in reverse, the rest unranked."""
    return BALLOTS[method.ballot].worst(P)


def sincerity(method, B, u):
    """Per voter, how far the ballot goes against the voter's own preferences, within its format:
    1 - R(B) / R_max. R is the reversal mass: the sum of u[a] - u[b] over the pairs the voter prefers
    a > b but the ballot puts b above a. Ties (exaggeration, levelling, unranked candidates) are not
    reversals and cost nothing: they say less, not something false (fidelity measures how much is
    said). R_max is the reversal mass of the most dishonest ballot the format allows (worst_reversal),
    so 0 means "as much against your own preferences as this ballot format permits", in every format.
    Preferences are method.pref(u) (reversed for the Inverse-* methods)."""
    P = method.pref(u)
    worst = worst_reversal(method, P)
    return 1 - _reversal_mass(_marks(method, B), P) / np.where(worst > 1e-12, worst, 1)


def implied_values(method, X):
    """What a ballot says about the voter's utilities, on 0..1 (its `implied` in ballots.py): scores
    and grades by their value, rankings evenly spaced (best 1, worst 0), approval / plurality marks
    1 and 0, veto 0 for the vetoed candidate and 1 for the rest, truncated rankings evenly spaced with
    the unranked at 0. Two-round ballots by their round-1 marks (see _marks)."""
    return BALLOTS[method.ballot].implied(_marks(method, X))


def fidelity(method, B, u):
    """Per voter, how well the ballot reproduces the voter's utility differences: utilities stretched
    to 0..1 (favourite 1, least liked 0) against the ballot's implied values (implied_values). Over all
    pairs: 1 - sum |du - dv| / sum 2 |du|, with du, dv the differences in utility and in implied
    value. 1 = the ballot shows the differences exactly; a tie loses half of a pair (0.5 if
    everything were tied); a mirrored pair loses all of it; below 0 is possible. It includes what the
    format cannot express (a ranking spaces candidates evenly, approval has two values): the sincere
    ballot's fidelity is the format's own loss, the drop from there what strategy adds."""
    P = method.pref(u)
    mn, mx = P.min(axis=1, keepdims=True), P.max(axis=1, keepdims=True)
    p = (P - mn) / np.where(mx - mn > 1e-12, mx - mn, 1)
    v = implied_values(method, B)
    du = p[:, :, None] - p[:, None, :]
    dv = v[:, :, None] - v[:, None, :]
    den = 2 * np.abs(du).sum(axis=(1, 2))
    return 1 - np.abs(du - dv).sum(axis=(1, 2)) / np.where(den > 1e-12, den, 1)


def strategy_types(B, u, leaders=None, sincere=None):
    """Shares of voters whose ballot reverses at least one real preference, by kind:
      compromise - the favourite is not at the top, and a top candidate is liked above the voter's
                   mean or is one of the poll leaders (a viable second choice, or the lesser evil
                   of the two leaders, raised)
      push-over  - the favourite is not at the top, and every top candidate is liked below the
                   mean and is no poll leader (a weak candidate raised, to steer eliminations or
                   runoffs)
      burial     - the favourite stays at the top, but a preference further down is reversed
      exaggeration - the ballot differs from the sincere one but reverses no preference: candidates
                   tied, left unranked or given the lowest or highest score together with others
                   (a candidate buried or raised as far as the format allows without a reversal);
                   only with `sincere`, the voters' sincere ballots (since 2026-10-30)
    The four are exclusive; sincere voters count in none. leaders: the last poll round's two
    leaders (since 2026-10-28; None: by the mean alone, as before). With `sincere`, a reversal the
    sincere ballot already makes is not counted (delegated methods: the order filled in from the
    favourite is no strategy; other methods' sincere ballots never reverse a preference)."""
    R = (B[:, :, None] > B[:, None, :]) & (u[:, :, None] < u[:, None, :])
    if sincere is not None:
        R &= ~(sincere[:, :, None] > sincere[:, None, :])
    rev = R.any(axis=(1, 2))
    top = B >= B.max(axis=1, keepdims=True)
    fav_top = top[np.arange(len(B)), u.argmax(axis=1)]
    best_top = np.where(top, u, -np.inf).max(axis=1)
    above_mean = best_top >= u.mean(axis=1)
    if leaders is not None:
        for c in leaders:
            if c is not None:
                above_mean = above_mean | top[:, c]
    out = {"compromise": float(np.mean(rev & ~fav_top & above_mean)),
           "pushover": float(np.mean(rev & ~fav_top & ~above_mean)),
           "burial": float(np.mean(rev & fav_top))}
    if sincere is not None:
        out["exaggeration"] = float(np.mean(~rev & (B != sincere).any(axis=1)))
    return out


def poll_tallies(tally, B, polls):
    """Winner probabilities on each poll sample, shape (samples, C)."""
    return np.array([tally(B[idx]) for idx in polls])


WIDEN = 6       # widen the polls up to 2**WIDEN times to find a runner-up
TOL = 1e-9


def default_poll_size(n):
    """Half the electorate, rounded to an odd number (100 -> 51, 99 -> 49, 200 -> 101)."""
    m = n // 2
    return m if m % 2 else m + 1


def respond(method, u, rng, rounds=1, samples=10, poll_size=0, search=1, strategic=None, poll_error=2.0,
            report=None, aggressive=None, start=None, adaptive=None, activated=None, campaign=None, risk=None,
            cstats=None, leaders=None, individual=False):
    """Poll rounds (see the module docstring). Returns ballots of the last round, whether it
    settled (same as the round before), and mean sincerity (all voters).
    report: a list of round numbers; then `rounds` is ignored, the rounds run up to the highest,
    and the result is a list of such triples, one per listed round. Round r here is exactly what
    rounds = r gives (each round depends only on the ones before it, and draws its random numbers
    in sequence), so one run serves a sweep over the number of rounds.
    start: ballots (n x C) the first poll is drawn from instead of the sincere ones (run.py --start
    previous: a previous election's result); only the information of round 1 changes.
    adaptive: personal thresholds of the adaptive voters (one per voter, nan = not adaptive; None =
    none): in each round a voter is strategic if their stake in the race between the two leaders,
    rescaled to 0..1, exceeds it (see the module docstring). activated: a dict, filled with
    {round: share of the adaptive voters strategic in that round} (0 for round 0).
    campaign: None (off) or {"share": phi, "cand_u": candidates' utilities, C x C}: every candidate
    recommends a move each round (docs/campaigns.md). risk: None (off: the fixed `aggressive` split) or
    {"threshold": one per voter, 0..1, "kappa": tolerance factor}: each voter is careful or bold by their
    satisfaction with the polls. cstats: a dict, filled with {round: {"follow": share of all voters
    following a recommendation, "top": candidate whose recommendation has the most followers (-1:
    none), "bold": share of the strategic voters who are bold}}. leaders: a dict, filled with {round: the
    round's two poll leaders (a, b)}, as the polls the voters answered showed them. individual: voters
    read the averaged polls and each decides alone (individual.py): None/False (the ranking groups), True
    (each voter expects only themselves to vote as they do), or per voter the number of voters they
    expect to (their boldness)."""
    if report is not None:
        want = sorted(set(int(r) for r in report))
        if want[0] < 0:
            raise ValueError("poll rounds must be 0 or more")
        snaps = {}
        _respond(method, u, rng, max(want), samples, poll_size, search, strategic, poll_error, aggressive,
                 lambda r, res: snaps.__setitem__(r, res) if r in want else None, start, adaptive, activated,
                 campaign, risk, cstats, leaders, individual)
        return [snaps[int(r)] for r in report]
    return _respond(method, u, rng, rounds, samples, poll_size, search, strategic, poll_error, aggressive,
                    start=start, adaptive=adaptive, activated=activated, campaign=campaign, risk=risk, cstats=cstats,
                    leaders=leaders, individual=individual)


def _respond(method, u, rng, rounds, samples, poll_size, search, strategic, poll_error, aggressive=None,
             snapshot=None, start=None, adaptive=None, activated=None, campaign=None, risk=None, cstats=None,
             leaders=None, individual=False):
    n, C = u.shape
    sincere = method.sincere(u)
    B = sincere.copy()
    fixed = np.ones(n, bool) if strategic is None else np.asarray(strategic, bool).copy()
    q = None if adaptive is None else np.asarray(adaptive, float)
    adapt = np.zeros(n, bool) if q is None else ~np.isnan(q) & ~fixed
    q_adapt = np.where(adapt, np.nan_to_num(q, nan=np.inf), np.inf)
    n_adaptive = int(adapt.sum())
    active = fixed.copy()

    risky = risk is not None
    bold = np.zeros(n, bool)                   # --risk: the voters bold this round (by satisfaction)
    tol_v = np.zeros(n)                        # --risk: each bold voter's tolerance for a loss

    def record(r, follow=0.0, top=-1):
        if activated is not None:
            activated[r] = float((active & adapt).sum()) / n_adaptive if n_adaptive else 0.0
        if cstats is not None:
            cstats[r] = {"follow": follow, "top": top,
                         "bold": float((bold & active).sum()) / max(1, int(active.sum())) if risky else float("nan")}
    record(0)
    if snapshot:
        snapshot(0, (B, True, 1.0))
    if rounds <= 0:
        return B, True, 1.0
    keys = [tuple(r) for r in np.argsort(-u, axis=1, kind="stable")]
    lo, hi = u.min(axis=1), u.max(axis=1)
    span = np.where(hi > lo, hi - lo, 1.0)

    def build():
        """Ranking groups of the active voters, aggressive (bold) and kind (careful) voters apart, in voter
        order. With --risk the attitude is the voter's this round, else the fixed split."""
        gs = {}
        for v, k in enumerate(keys):
            if active[v]:
                e = bool(bold[v]) if risky else (aggressive is None or bool(aggressive[v]))
                gs.setdefault((k, e), []).append(v)
        return gs

    gkeys = build()
    if not gkeys and not adapt.any():
        for r in range(1, rounds + 1):
            record(r)
            if snapshot:
                snapshot(r, (B, True, 1.0))
        return B, True, 1.0
    m = min(n, poll_size if poll_size > 0 else default_poll_size(n))
    # methods without feature sums can't scale counts: a smaller poll has about poll_error times the error
    # (only approximates poll_error >= 1; smaller values give polls of poll_size voters)
    m_full = max(1, min(m, int(round(m / poll_error ** 2)))) if poll_error > 0 else m

    def setup(gkeys):
        groups = [np.array(x) for x in gkeys.values()]
        return (list(gkeys), groups, [k[1] for k in gkeys],
                [method.sincere(u[g].mean(axis=0, keepdims=True), methods.lazy_of(g), methods.approval_of(g))[0] for g in groups],
                [u[g].sum(axis=0) for g in groups])

    names, groups, eager, ref, Ug = setup(gkeys)
    camp = campaign is not None
    shared = {}                                # each group's answer in the last round (absent = sincere)
    if start is not None:                      # the first poll shows these ballots (a previous result)
        B = np.array(start, dtype=float)
    pm = method.poll_view()                    # how voters count a poll (docs/model.md, What a poll shows)
    inc = pm.full is None
    settled = False
    places = 3 if BALLOTS[method.ballot].pushover is not None else 2   # the third is the push-over target
    individual = None if individual is False else individual
    if individual is not None:                 # the average of the polls, each voter alone (individual.py)
        import individual as _ind
        thr = np.where(fixed, 0.0, np.where(adapt, q_adapt, np.inf))
        fol = np.ones(n) if individual is True else np.asarray(individual, float)   # followers each voter expects
        ind_memory = {"last": {}, "trust": {}}  # what each voter's move promised, and their trust in moves
        # discovery draws from its own stream (spawned, so the polls stay those of the run without it)
        ind_rng = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])
    for r in range(1, rounds + 1):
        polls = np.stack([rng.choice(n, m if inc else m_full, replace=False) for _ in range(samples)])
        if individual is not None:
            F = pm.features(B)
            W = np.zeros((samples, n))
            W[np.arange(samples)[:, None], polls] = n / m
            tot = F.sum(axis=0)
            T0 = tot + poll_error * ((W @ F).mean(axis=0) - tot)      # the average of the polls
            reading = _ind.Reading(pm, method.ballot, T0, C, n, m * samples, poll_error)
            if leaders is not None:
                leaders[r] = (reading.A, reading.B)
            B_prev = B
            B, active = _ind.answer(method, pm, u, B, sincere, reading, threshold=thr, followers=fol,
                                    memory=ind_memory, rng=ind_rng, search=search, campaign=campaign)
            fw = ind_memory["follow"]
            record(r, float(fw.sum()) / n, int(np.argmax(fw)) if fw.sum() > 0 else -1)
            settled = np.array_equal(B, B_prev)
            if snapshot:
                snapshot(r, (B, settled, float(sincerity(method, B, u).mean())))
            continue
        if inc:
            F = pm.features(B)
            W = np.zeros((samples, n))
            W[np.arange(samples)[:, None], polls] = n / m
            tot = F.sum(axis=0)
            dev = poll_error * (W @ F - tot)       # poll deviations from the true count, poll_error x sampling error
            a, b, *rest = _leaders(lambda j: pm.decide_many(_widened(tot, dev, j), C), C, places)
        else:
            a, b, *rest = _leaders(lambda j: poll_tallies(pm.tally, B, _smaller(polls, j, rng, n)), C, places)
        third = rest[0] if rest else None          # push-over target: the next place after the two leaders
        if leaders is not None:
            leaders[r] = (a, b)
        P_now = None                               # winner lotteries of the polls as they are (campaigns, risk)
        if camp or risky:
            P_now = pm.decide_many(tot + dev, C) if inc else poll_tallies(pm.tally, B, polls)
        rebuild = False
        if n_adaptive:                             # adaptive voters: strategic this round if the race matters to them
            stake = np.abs(u[:, a] - u[:, b]) / span if b is not None else np.zeros(n)
            now = fixed | (stake > q_adapt)
            if not np.array_equal(now, active):
                active, rebuild = now, True
        if risky:                                  # careful or bold, by the voter's satisfaction with the polls
            sat = (u @ P_now.mean(axis=0) - lo) / span
            now = sat < risk["threshold"]
            tol_v = np.where(now, risk["kappa"] * (risk["threshold"] - sat) * span, 0.0)
            if not np.array_equal(now & active, bold & active):
                rebuild = True
            bold = now
        if rebuild:
            names, groups, eager, ref, Ug = setup(build())
        recs = []
        if camp:                                   # every candidate's recommendation (docs/campaigns.md)
            recs = _campaigns(method, pm, u, B, F if inc else None, tot + dev if inc else None, polls, P_now, a, b,
                              campaign["share"], campaign["cand_u"])
        answers, follow = [], np.zeros(C)
        for g, mem in enumerate(groups):
            if inc:
                gain = _gain(pm, tot + dev, F[mem].sum(axis=0), sincere[mem], len(mem), C, Ug[g])
            else:
                gain = _gain_full(pm, B, mem, sincere[mem], polls, Ug[g])
            prev = shared.get(names[g])
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
        record(r, float(follow.sum()) / n, int(np.argmax(follow)) if follow.sum() > 0 else -1)
        shared, B_prev = {k: x for k, x in zip(names, answers) if x is not None}, B
        B = sincere.copy()
        for mem, x in zip(groups, answers):       # everyone switches at once
            if x is not None:
                B[mem] = x
        settled = np.array_equal(B, B_prev)       # the ballots repeat the round before
        if snapshot:
            snapshot(r, (B, settled, float(sincerity(method, B, u).mean())))
    return B, settled, float(sincerity(method, B, u).mean())


def _widened(tot, dev, j):
    """Polls with deviations x 2**j. For j > 0 each count is kept on the same side of zero as the
    true count (counts are >= 0, veto counts <= 0): far-widened polls can otherwise produce
    impossible negative counts and pick a runner-up from a meaningless tally."""
    T = tot + (2 ** j) * dev
    if j > 0:
        T = np.where(tot >= 0, np.maximum(T, 0), np.minimum(T, 0))
    return T


def _smaller(polls, j, rng, n):
    """Methods without feature sums: wider polls by sampling fewer voters (variance ~ 4**j)."""
    if j == 0:
        return polls
    k = max(1, polls.shape[1] // 4 ** j)
    return np.stack([rng.choice(n, k, replace=False) for _ in range(len(polls))])


def _leaders(winners_at, C, k=2):
    """The k leading candidates of a poll round (winners_at(j): winner probabilities per poll, with the
    deviations of the polls x 2**j). One after the other, the next is the candidate that wins most often
    among the rest at the smallest widening at which any of the rest wins somewhere: first the poll
    leader, then the runner-up of the race that would matter first, then the next place. Each widening is
    computed once. Returns a list of k candidates, None where none appears."""
    found, at = [], {}
    for j in range(0, WIDEN + 1):
        while len(found) < k:
            if j not in at:
                at[j] = winners_at(j).mean(axis=0)
            Q = at[j].copy()
            Q[found] = 0
            if Q.max() <= 0:
                break
            found.append(int(np.argsort(-Q, kind="stable")[0]))
        if len(found) == k:
            break
    return found + [None] * (k - len(found))


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
