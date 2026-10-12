"""Strategic voting. Two models; run.py uses poll rounds unless --converge is set.

POLL ROUNDS (default; respond): a single election with a few polls before it.
  Round 0: every voter votes sincerely ("naive", uninformed).
  Round r (1 ... rounds): a poll of round r-1 is published, and every strategic group answers it
  at the same time; nobody sees anyone else's answer before the next poll.
  - A poll publishes only the method's own counts (the summed ballot features the method decides
    from: first preferences for Plurality, approvals for Approval, the pairwise matrix for
    Condorcet methods, ...), from a random sample of `poll_size` voters (without replacement),
    scaled up to the electorate. `samples` independent polls (30) stand for the uncertainty.
    Nobody sees anyone's preferences or individual ballots.
  - Uncertainty: voters treat a poll as `poll_error` times less precise than its sampling error
    (default 2: real polls miss results by about twice their stated sampling error, Shirani-Mehr,
    Rothschild, Goel & Gelman 2018). The polls are spread around the true count by that factor.
  - Leaders: the two candidates that win most often over the round's polls, the same for
    everyone. If one candidate wins every poll, the polls are widened (deviations x 2, 4, ...,
    counts kept on the side of zero of the true count) until another one wins somewhere: the race
    that would matter first. (_leaders)
  - Groups are whole ranking groups (strategic voters with the same sincere ranking, no cap):
    voters who share a ranking and see the same poll reach the same answer, so they act as a
    bloc. Factions across rankings are not built in; they emerge when groups answer alike.
  - Answer: a group estimates the count for a ballot b as poll count - its own previous ballots
    + its size x b, and judges b by its expected utility gain over its sincere ballots, over
    the polls. It first tries a few candidate ballots: its previous answer and the naive
    exaggerations of its ballot type around the two leaders (A = the one it prefers, B = the
    other; `exaggerations`): approve everyone liked at least as much as A; top score down to A,
    0 from B down, stretched in between; vote A; A to the top and/or B to the bottom of a
    ranking; and the like. Aggressive voters take the best if it does not lose in the polls:
    exaggeration wins ties, they exaggerate when it cannot hurt, whatever the gap. Kind voters
    take it only if it strictly gains (honesty wins ties). `aggressive` marks the aggressive
    voters (None = all; drawn in run.py); aggressive and kind voters form separate groups. Then random
    search (`search` proposals at most, stopping after PATIENCE = 3 in a row) takes a change only
    if it strictly gains. A group none of whose candidates is safe and whose search finds no
    gain stays sincere.
  - Settled: round `rounds` gives the same ballots as the round before (a fixed point).
  rounds = 0 gives the sincere ballots.

CONVERGENT LEARNING (--converge N; learn): the earlier model. Groups take turns over N rounds,
each seeing the others' latest strategic ballots through a fresh full-electorate poll, and
keep a change if it pays. It finds where an electorate ends up after long adjustment, with
far more information and coordination than one election offers. Kept for comparison.
  - Coordination: `group_size` caps how many voters move together. Each ranking group is
    split at random into subgroups of at most that size; 0 means no cap.
  - Loop (rounds): a fresh "poll" is drawn each round: `samples` bootstrap resamples of the
    electorate. Groups take turns in random order. A group proposes a ballot for all its
    members: usually a small change to its current shared ballot, sometimes a return to
    sincere. It keeps the proposal if its members' total expected utility, averaged over the
    poll samples, goes up; a return to sincere is also kept when it is not worse (honesty wins
    ties), so deviations that stopped paying off fade out instead of accumulating.

Partial strategy (both): `strategic` marks the voters who take part (a random share of the
electorate, drawn in run.py, the same voters for every method). The others always vote sincerely.
"""
import numpy as np
import methods

BACK_TO_SINCERE = 0.15  # share of proposals that return a deviating group to sincere


def mutate(b, ballot, rng):
    """One small change to a single ballot vector (length C)."""
    b = b.copy()
    C = len(b)
    if ballot == "fav":
        cur = int(b.argmax())
        new = (cur + 1 + rng.integers(C - 1)) % C
        b[:] = 0
        b[new] = 1
    elif ballot == "rank":
        i, j = rng.choice(C, 2, replace=False)
        b[i], b[j] = b[j], b[i]
    elif ballot == "approval":
        if C == 2:              # the only valid ballots are A or B: no single flip stays valid, so swap
            return b[::-1].copy()
        while True:
            c = rng.integers(C)
            b[c] = 1 - b[c]
            if 0 < b.sum() < C:
                break
            b[c] = 1 - b[c]
    elif ballot in ("fav_run", "app_run"):          # only the round-1 marks change; the runoff ranking stays
        pts, mark = np.mod(b, C), b >= C
        if ballot == "fav_run":
            cur = int(mark.argmax())
            mark[:] = False
            mark[(cur + 1 + rng.integers(C - 1)) % C] = True
        elif C == 2:
            mark = mark[::-1].copy()
        else:
            while True:
                c = rng.integers(C)
                mark[c] = not mark[c]
                if 0 < mark.sum() < C:
                    break
                mark[c] = not mark[c]
        b = pts + C * mark
    elif ballot == "veto":
        cur = int(b.argmin())
        new = (cur + 1 + rng.integers(C - 1)) % C
        b[:] = 0
        b[new] = -1
    elif ballot == "weak":
        c = rng.integers(C)
        b[c] = (b[c] + 1 + rng.integers(C - 1)) % C          # any other level; equal levels are ties
    elif ballot in ("trunc", "trunc3"):
        kmax = C - 1 if ballot == "trunc" else min(methods.TRUNC3, C)
        ranked = [int(c) for c in np.argsort(-b, kind="stable") if b[c] > 0]
        order = ranked + list(rng.permutation([c for c in range(C) if b[c] == 0]))
        k = len(ranked)
        if rng.random() < 0.5:
            i, j = rng.choice(C, 2, replace=False)
            order[i], order[j] = order[j], order[i]
        else:
            k = min(kmax, max(1, k + (1 if rng.random() < 0.5 else -1)))
        b = np.zeros(C)
        for pos, c in enumerate(order[:k]):
            b[c] = k - pos
    elif ballot == "score":
        # Any other level, uniformly. For ranges above 0..5, half of the proposals instead use the
        # six levels of the 0..5 scale stretched to the range (0, 20, 40, ... 100 for 0..100), so
        # the chance of trying an extreme or a coarse step does not shrink with the range and
        # learning is not slowed artificially. 0..5 and below: unchanged.
        S = methods.MAXS
        c = rng.integers(C)
        if S > 5 and rng.random() < 0.5:
            levels = [x for x in np.rint(np.arange(6) * S / 5) if x != b[c]]
            b[c] = levels[rng.integers(len(levels))]
        else:
            b[c] = (b[c] + 1 + rng.integers(S)) % (S + 1)
    elif ballot == "grade3":
        c = rng.integers(C)
        b[c] = (b[c] + 1 + rng.integers(2)) % 3
    return b


def _marks(method, X):
    """Two-round ballots (fav_run, app_run): only the round-1 marks are the voter's choice (the runoff
    ranking is the sincere one), so sincerity and fidelity look at the marks, as a plurality / approval
    ballot."""
    if method.ballot in ("fav_run", "app_run"):
        return (X >= X.shape[1]).astype(float)
    return X


def _reversal_mass(X, P):
    """Per voter: sum of P[a] - P[b] over the pairs with P[a] > P[b] that ballot X puts the other way."""
    w = np.clip(P[:, :, None] - P[:, None, :], 0, None)            # w[a, b] = P[a] - P[b] where positive
    return (w * (X[:, None, :] > X[:, :, None])).sum(axis=(1, 2))


def _within_cost(s):
    """cost[i, j] (per voter): sum of differences over all pairs inside the sorted segment s[i..j]."""
    n, C = s.shape
    cost = np.zeros((n, C, C))
    for i in range(C):
        for j in range(i + 1, C):
            seg = s[:, i:j + 1]
            m = j - i + 1
            k = np.arange(m)
            cost[:, i, j] = (seg * (2 * k - m + 1)).sum(axis=1)   # sum_{p<q} (s_q - s_p) for sorted values
    return cost


def _max_separated(P, levels):
    """Per voter: the largest reversal mass a ballot with at most `levels` levels can reach. The worst
    such ballot reverses the order with candidates grouped into contiguous blocks of the sorted
    utilities; reversal mass = all pairs minus the pairs inside blocks, so minimise the mass inside
    blocks (dynamic programme over the sorted candidates)."""
    s = np.sort(P, axis=1)
    n, C = s.shape
    total = _within_cost(s)[:, 0, C - 1] if C > 1 else np.zeros(n)
    if levels >= C:
        return total
    cost = _within_cost(s)
    INF = np.full(n, np.inf)
    best = [cost[:, 0, j] for j in range(C)]                     # one block covering 0..j
    for _ in range(levels - 1):                                  # one more block each time
        best = [np.minimum(best[j], np.min(np.stack([best[i] + cost[:, i + 1, j] if i + 1 <= j else INF
                                                     for i in range(j)] or [INF]), axis=0))
                for j in range(C)]
    return total - best[C - 1]


def worst_reversal(method, P):
    """Per voter: the reversal mass of the most dishonest ballot the method's format allows (the scale
    of sincerity). Plurality / two-round with plurality: a vote for the last choice; veto: against the
    favourite; rankings (strict, weak, truncated): the reversed ranking; approval: the most damaging
    cut; score / three grades: the reversed order in as many levels as there are; at most 3 ranked
    (trunc3): the 3 least liked ranked in reverse, the rest unranked."""
    b = method.ballot
    C = P.shape[1]
    if b in ("fav", "fav_run"):
        return (P - P.min(axis=1, keepdims=True)).sum(axis=1)
    if b == "veto":
        return (P.max(axis=1, keepdims=True) - P).sum(axis=1)
    if b in ("approval", "app_run"):
        return _max_separated(P, 2)
    if b == "score":
        return _max_separated(P, methods.MAXS + 1)
    if b == "grade3":
        return _max_separated(P, 3)
    if b == "trunc3":
        k = min(methods.TRUNC3, C)
        s = np.sort(P, axis=1)                                  # ranked: the k least liked; tied: the rest
        return _within_cost(s)[:, 0, C - 1] - (_within_cost(s)[:, k, C - 1] if k < C - 1 else 0)
    return _max_separated(P, C)                                  # rank, weak, trunc: full reversal


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
    """What a ballot says about the voter's utilities, on 0..1: scores and grades by their value,
    rankings evenly spaced (best 1, worst 0), approval / plurality marks 1 and 0, veto 0 for the
    vetoed candidate and 1 for the rest, truncated rankings evenly spaced with the unranked at 0.
    Two-round ballots by their round-1 marks (see _marks)."""
    b = method.ballot
    C = X.shape[1]
    X = _marks(method, X)
    if b == "score":
        return X / methods.MAXS
    if b == "grade3":
        return X / 2
    if b in ("rank", "weak"):
        return X / max(C - 1, 1)
    if b == "veto":
        return X + 1
    if b in ("trunc", "trunc3"):
        mx = X.max(axis=1, keepdims=True)
        return X / np.where(mx > 0, mx, 1)
    return X                                                     # fav, approval, fav_run / app_run marks


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


def strategy_types(B, u):
    """Shares of voters whose ballot reverses at least one real preference, by kind:
      compromise - the favourite is not at the top, and the best-liked top candidate is liked
                   above the voter's mean (a viable second choice raised)
      push-over  - the favourite is not at the top, and every top candidate is liked below the
                   mean (a weak candidate raised, to steer eliminations or runoffs)
      burial     - the favourite stays at the top, but a preference further down is reversed
    The three are exclusive; sincere voters count in none."""
    rev = ((B[:, :, None] > B[:, None, :]) & (u[:, :, None] < u[:, None, :])).any(axis=(1, 2))
    top = B >= B.max(axis=1, keepdims=True)
    fav_top = top[np.arange(len(B)), u.argmax(axis=1)]
    best_top = np.where(top, u, -np.inf).max(axis=1)
    above_mean = best_top >= u.mean(axis=1)
    return {"compromise": float(np.mean(rev & ~fav_top & above_mean)),
            "pushover": float(np.mean(rev & ~fav_top & ~above_mean)),
            "burial": float(np.mean(rev & fav_top))}


def poll_tallies(tally, B, polls):
    """Winner probabilities on each poll sample, shape (samples, C)."""
    return np.array([tally(B[idx]) for idx in polls])


def group_value(P, Ug):
    """Expected total utility of a group, averaged over poll samples."""
    return float(np.mean(P @ Ug))


def learn(method, u, rng, rounds=20, samples=10, group_size=0, strategic=None):
    """Convergent learning (see the module docstring). Return learned ballots, whether learning
    settled, and mean sincerity (all voters). strategic: boolean mask of voters who take part
    (None = all). run.py passes --samples (default 10, chosen for poll rounds)."""
    n, C = u.shape
    sincere = method.sincere(u)
    B = sincere.copy()
    keys = [tuple(r) for r in np.argsort(-u, axis=1, kind="stable")]
    groups = {}
    for v, k in enumerate(keys):
        if strategic is None or strategic[v]:
            groups.setdefault(k, []).append(v)
    if not groups:
        return B, True, 1.0
    groups = [np.array(m) for m in groups.values()]
    if group_size > 0:   # split each ranking group at random into subgroups of at most group_size
        groups = [part for m in groups for part in np.array_split(rng.permutation(m), -(-len(m) // group_size))]
    shared = [None] * len(groups)              # None = members vote sincerely
    ref = [method.sincere(u[m].mean(axis=0, keepdims=True))[0] for m in groups]
    Ug = [u[m].sum(axis=0) for m in groups]
    # Methods with features (all but Condorcet-random-drop) keep the summed features per poll sample
    # and update them only for the group that changes; the result is the same as recounting.
    inc = method.full is None
    F = method.features(B) if inc else None
    changed_last = True
    for _ in range(rounds):
        polls = rng.integers(0, n, (samples, n))
        if inc:
            W = np.stack([np.bincount(p, minlength=n) for p in polls]).astype(float)   # (samples, n)
            S_cur = W @ F
            P_cur = method.decide_many(S_cur, C)
        else:
            P_cur = poll_tallies(method.tally, B, polls)   # current ballots, tallied once per change
        changed_last = False
        for g in rng.permutation(len(groups)):
            mem = groups[g]
            to_sincere = shared[g] is not None and rng.random() < BACK_TO_SINCERE
            if to_sincere:
                prop_rows = sincere[mem]
            else:
                base = shared[g] if shared[g] is not None else ref[g]
                prop = mutate(base, method.ballot, rng)
                prop_rows = np.broadcast_to(prop, (len(mem), C))
            if inc:
                F_mem = method.features(np.ascontiguousarray(prop_rows))
                S_new = S_cur + W[:, mem] @ (F_mem - F[mem])
                P_new = method.decide_many(S_new, C)
            else:
                B2 = B.copy()
                B2[mem] = prop_rows
                P_new = poll_tallies(method.tally, B2, polls)
            gain = group_value(P_new, Ug[g]) - group_value(P_cur, Ug[g])
            if gain > 1e-12 or (to_sincere and gain >= -1e-12):   # honesty wins ties
                if inc:
                    F[mem], S_cur = F_mem, S_new
                    B = B.copy()
                    B[mem] = prop_rows
                else:
                    B = B2
                P_cur = P_new
                shared[g] = None if to_sincere else prop
                changed_last = True
    return B, not changed_last, float(sincerity(method, B, u).mean())


PATIENCE = 3    # poll rounds: random search stops after this many proposals in a row without gain
WIDEN = 6       # poll rounds: widen the polls up to 2**WIDEN times to find a runner-up
TOL = 1e-9


def default_poll_size(n):
    """Half the electorate, rounded to an odd number (100 -> 51, 99 -> 49, 200 -> 101)."""
    m = n // 2
    return m if m % 2 else m + 1


def respond(method, u, rng, rounds=1, samples=10, poll_size=0, search=10, strategic=None, poll_error=2.0,
            report=None, aggressive=None, start=None):
    """Poll rounds (see the module docstring). Returns ballots of the last round, whether it
    settled (same as the round before), and mean sincerity (all voters).
    report: a list of round numbers; then `rounds` is ignored, the rounds run up to the highest,
    and the result is a list of such triples, one per listed round. Round r here is exactly what
    rounds = r gives (each round depends only on the ones before it, and draws its random numbers
    in sequence), so one run serves a sweep over the number of rounds.
    start: ballots (n x C) the first poll is drawn from instead of the sincere ones (run.py --start
    previous: a previous election's result); only the information of round 1 changes."""
    if report is not None:
        want = sorted(set(int(r) for r in report))
        if want[0] < 0:
            raise ValueError("poll rounds must be 0 or more")
        snaps = {}
        _respond(method, u, rng, max(want), samples, poll_size, search, strategic, poll_error, aggressive,
                 lambda r, res: snaps.__setitem__(r, res) if r in want else None, start)
        return [snaps[int(r)] for r in report]
    return _respond(method, u, rng, rounds, samples, poll_size, search, strategic, poll_error, aggressive,
                    start=start)


def _respond(method, u, rng, rounds, samples, poll_size, search, strategic, poll_error, aggressive=None,
             snapshot=None, start=None):
    n, C = u.shape
    sincere = method.sincere(u)
    B = sincere.copy()
    if snapshot:
        snapshot(0, (B, True, 1.0))
    if rounds <= 0:
        return B, True, 1.0
    keys = [tuple(r) for r in np.argsort(-u, axis=1, kind="stable")]
    groups = {}
    for v, k in enumerate(keys):      # ranking groups, aggressive and kind voters apart
        if strategic is None or strategic[v]:
            groups.setdefault((k, aggressive is None or bool(aggressive[v])), []).append(v)
    eager = [k[1] for k in groups]
    if not groups:
        if snapshot:
            for r in range(1, rounds + 1):
                snapshot(r, (B, True, 1.0))
        return B, True, 1.0
    groups = [np.array(x) for x in groups.values()]
    m = min(n, poll_size if poll_size > 0 else default_poll_size(n))
    # methods without feature sums can't scale counts: a smaller poll has about poll_error times the error
    # (only approximates poll_error >= 1; smaller values give polls of poll_size voters)
    m_full = max(1, min(m, int(round(m / poll_error ** 2)))) if poll_error > 0 else m
    ref = [method.sincere(u[g].mean(axis=0, keepdims=True))[0] for g in groups]
    Ug = [u[g].sum(axis=0) for g in groups]
    shared = [None] * len(groups)              # each group's answer in the last round (None = sincere)
    if start is not None:                      # the first poll shows these ballots (a previous result)
        B = np.array(start, dtype=float)
    inc = method.full is None
    settled = False
    for r in range(1, rounds + 1):
        polls = np.stack([rng.choice(n, m if inc else m_full, replace=False) for _ in range(samples)])
        if inc:
            F = method.features(B)
            W = np.zeros((samples, n))
            W[np.arange(samples)[:, None], polls] = n / m
            tot = F.sum(axis=0)
            dev = poll_error * (W @ F - tot)       # poll deviations from the true count, poll_error x sampling error
            a, b = _leaders(lambda j: method.decide_many(_widened(tot, dev, j), C), C)
        else:
            a, b = _leaders(lambda j: poll_tallies(method.tally, B, _smaller(polls, j, rng, n)), C)
        answers = []
        for g, mem in enumerate(groups):
            if inc:
                gain = _gain(method, tot + dev, F[mem].sum(axis=0), sincere[mem], len(mem), C, Ug[g])
            else:
                gain = _gain_full(method, B, mem, sincere[mem], polls, Ug[g])
            cands = ([shared[g]] if shared[g] is not None else []) + exaggerations(method.ballot, Ug[g], a, b)
            answers.append(_answer(gain, method.ballot, ref[g], cands, search, rng, eager[g]))
        shared, B_prev = answers, B
        B = sincere.copy()
        for mem, x in zip(groups, shared):        # everyone switches at once
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


def _leaders(winners_at, C):
    """The two leading candidates of a poll round: the two that win most often over the polls
    (winners_at(0): winner probabilities per poll). If one candidate wins every poll, the polls
    are widened (winners_at(j): deviations x 2**j) until another one wins somewhere: the race that
    would matter first. Returns (leader, runner-up); the runner-up is None if none appears."""
    P = winners_at(0).mean(axis=0)
    order = np.argsort(-P, kind="stable")
    a = int(order[0])
    if P[order[1]] > 0:
        return a, int(order[1])
    for j in range(1, WIDEN + 1):
        Q = winners_at(j).mean(axis=0)
        Q[a] = 0
        if Q.max() > 0:
            return a, int(Q.argmax())
    return a, None


def _gain(method, T, F_old, sincere_rows, size, C, Ug):
    """For a group: gain(ballot) = expected utility gain over the polls (as voters see them) of the
    group casting this ballot instead of its sincere ballots."""
    F_sin = method.features(sincere_rows).sum(axis=0)
    base = T - F_old + F_sin                       # the polls with the group voting sincerely
    u_sin = method.decide_many(base, C) @ Ug       # per poll
    cache = {}

    def gain(ballot):
        key = ballot.tobytes()
        if key not in cache:
            D = size * method.features(ballot[None])[0] - F_sin
            cache[key] = float(np.mean(method.decide_many(base + D, C) @ Ug - u_sin))
        return cache[key]
    return gain


def _gain_full(method, B, mem, sincere_rows, polls, Ug):
    """Methods without feature sums: the group's sampled members changed in each poll."""
    B0 = B.copy()
    B0[mem] = sincere_rows
    u_sin = poll_tallies(method.tally, B0, polls) @ Ug
    cache = {}

    def gain(ballot):
        key = ballot.tobytes()
        if key not in cache:
            B2 = B.copy()
            B2[mem] = ballot
            cache[key] = float(np.mean(poll_tallies(method.tally, B2, polls) @ Ug - u_sin))
        return cache[key]
    return gain


def exaggerations(ballot, Ug, a, b):
    """Naive exaggerations of a group's ballot around the two poll leaders a, b (Ug: the group's
    summed utilities; A = the leader it prefers, B = the other). Most complete first."""
    C = len(Ug)
    if b is None:                                  # no runner-up (exact polls): no race to exaggerate for
        return []
    A, Bo = (a, b) if Ug[a] >= Ug[b] else (b, a)
    order = [int(c) for c in np.argsort(-Ug, kind="stable")]      # the group's sincere ranking
    up = Ug >= Ug[A]                                                 # liked at least as much as A
    k = int(up.sum())
    out = []

    def rank_points(o):
        r = np.empty(C)
        r[o] = np.arange(C - 1, -1, -1)
        return r

    comp = [A] + [c for c in order if c != A]                       # A moved to the top
    bury = [c for c in order if c != Bo] + [Bo]
    both = [A] + [c for c in bury if c != A]
    if ballot == "fav":
        out = [np.eye(C)[A]]
    elif ballot == "approval":
        if 0 < k < C:
            out = [up.astype(float)]
    elif ballot in ("score", "grade3"):
        S = methods.MAXS if ballot == "score" else 2
        lo = Ug[Bo]
        x = np.clip((Ug - lo) / max(Ug[A] - lo, 1e-12), 0, 1)       # top at A and above, 0 at B and below
        out = [np.rint(S * x), S * up.astype(float)]
    elif ballot == "rank":
        out = [rank_points(both), rank_points(comp), rank_points(bury)]
    elif ballot == "weak":                                           # levels, equal = tied
        tied = np.empty(C)
        tied[up] = C - 1                                             # A tied with everyone above it, at the top
        rest = [c for c in order if not up[c]]
        tied[rest] = np.arange(C - 2, C - 2 - len(rest), -1)
        tb = tied.copy()
        if not up[Bo]:
            tb[Bo] = tb[rest].min() - 1 if len(rest) > 1 else tb[Bo]
        out = [tb - tb.min(), tied - tied.min(), rank_points(both), rank_points(bury)]
    elif ballot == "trunc":                                          # rank only down to A
        if 0 < k < C:
            t1 = np.zeros(C); t1[[c for c in order if up[c]]] = np.arange(k, 0, -1)
            t2 = np.zeros(C); t2[[c for c in comp[:k]]] = np.arange(k, 0, -1)
            out = [t1, t2]
    elif ballot == "trunc3":                                         # at most 3 ranked
        K = min(methods.TRUNC3, C)
        t1 = [c for c in order if up[c]][:K]                          # rank only down to A
        t2 = [A] + [c for c in order if c not in (A, Bo)][:K - 1]     # A first, B never ranked
        for o in (t2, t1):
            t = np.zeros(C); t[o] = np.arange(len(o), 0, -1)
            out.append(t)
    elif ballot == "fav_run":                                        # mark A; the runoff ranking stays sincere
        out = [rank_points(order) + C * np.eye(C)[A]]
    elif ballot == "app_run":                                        # approve down to A; runoff ranking sincere
        if 0 < k < C:
            out = [rank_points(order) + C * up]
    elif ballot == "veto":
        v = np.zeros(C); v[Bo] = -1
        out = [v]
    return out


def _answer(gain, ballot_type, ref, cands, search, rng, eager=True):
    """A group's answer to one poll round. First the candidate ballots (its previous answer, then
    the naive exaggerations): the best one is taken if it does not lose in the polls (gain >= 0)
    for aggressive voters (eager: exaggeration wins ties, they exaggerate when it cannot hurt), or
    only if it strictly gains for kind voters (honesty wins ties). Then random search (at most
    `search` proposals, stopping after PATIENCE in a row) from that ballot, or from the sincere
    one: a change is taken only if it strictly gains. None = stay sincere."""
    best, best_v = None, 0.0
    for c in cands:
        v = gain(c)
        if v > best_v + TOL or (eager and best is None and v >= -TOL):
            best, best_v = c, v
    cur, misses = (best if best is not None else ref), 0
    for _ in range(search):
        prop = mutate(cur, ballot_type, rng)
        v = gain(prop)
        if v > best_v + TOL:
            best, best_v, cur, misses = prop, v, prop, 0
        else:
            misses += 1
            if misses >= PATIENCE:
                break
    return best
