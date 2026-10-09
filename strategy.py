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
    everyone. If one candidate wins every poll, the polls are widened (deviations x 2, 4, ...)
    until another one wins somewhere: the race that would matter first. (_leaders)
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
from methods import SINCERE

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
    elif ballot == "rankapp":
        pts = np.where(b >= C, b - C, b)
        k = int((b >= C).sum())
        if rng.random() < 0.5:
            i, j = rng.choice(C, 2, replace=False)
            pts[i], pts[j] = pts[j], pts[i]
        else:
            k = min(C - 1, max(1, k + (1 if rng.random() < 0.5 else -1)))
        b = pts + C * (pts >= C - k)
    elif ballot == "veto":
        cur = int(b.argmin())
        new = (cur + 1 + rng.integers(C - 1)) % C
        b[:] = 0
        b[new] = -1
    elif ballot == "weak":
        c = rng.integers(C)
        b[c] = (b[c] + 1 + rng.integers(C - 1)) % C          # any other level; equal levels are ties
    elif ballot == "trunc":
        ranked = [int(c) for c in np.argsort(-b, kind="stable") if b[c] > 0]
        order = ranked + list(rng.permutation([c for c in range(C) if b[c] == 0]))
        k = len(ranked)
        if rng.random() < 0.5:
            i, j = rng.choice(C, 2, replace=False)
            order[i], order[j] = order[j], order[i]
        else:
            k = min(C - 1, max(1, k + (1 if rng.random() < 0.5 else -1)))
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


def sincerity(B, u):
    """Per voter: 1 minus the utility-weighted share of preferences the ballot reverses.

    For every pair the ballot orders (a rated above b) while the voter likes b more,
    add u[b] - u[a]; divide by the sum of |u[a] - u[b]| over all pairs; subtract from 1.
    1 = no reversed preference (every sincere ballot), 0 = every preference reversed.
    """
    above = B[:, :, None] > B[:, None, :]                  # ballot says a > b
    loss = np.clip(u[:, None, :] - u[:, :, None], 0, None)  # u[b] - u[a] where positive
    total = np.abs(u[:, :, None] - u[:, None, :]).sum(axis=(1, 2)) / 2
    return 1 - (above * loss).sum(axis=(1, 2)) / np.where(total > 0, total, 1)


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
    """Return learned ballots, whether learning settled, and mean sincerity (all voters).
    strategic: boolean mask of voters who take part (None = all)."""
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
    ref = [SINCERE[method.ballot](u[m].mean(axis=0, keepdims=True))[0] for m in groups]
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
    return B, not changed_last, float(sincerity(B, u).mean())


PATIENCE = 3    # poll rounds: random search stops after this many proposals in a row without gain
WIDEN = 6       # poll rounds: widen the polls up to 2**WIDEN times to find a runner-up
TOL = 1e-9


def default_poll_size(n):
    """Half the electorate, rounded to an odd number (100 -> 51, 99 -> 49, 200 -> 101)."""
    m = n // 2
    return m if m % 2 else m + 1


def respond(method, u, rng, rounds=1, samples=30, poll_size=0, search=10, strategic=None, poll_error=2.0,
            report=None, aggressive=None):
    """Poll rounds (see the module docstring). Returns ballots of the last round, whether it
    settled (same as the round before), and mean sincerity (all voters).
    report: a list of round numbers; then `rounds` is ignored, the rounds run up to the highest,
    and the result is a list of such triples, one per listed round. Round r here is exactly what
    rounds = r gives (each round depends only on the ones before it, and draws its random numbers
    in sequence), so one run serves a sweep over the number of rounds."""
    if report is not None:
        want = sorted(set(int(r) for r in report))
        snaps = {}
        _respond(method, u, rng, max(want), samples, poll_size, search, strategic, poll_error, aggressive,
                 lambda r, res: snaps.__setitem__(r, res) if r in want else None)
        return [snaps[int(r)] for r in report]
    return _respond(method, u, rng, rounds, samples, poll_size, search, strategic, poll_error, aggressive)


def _respond(method, u, rng, rounds, samples, poll_size, search, strategic, poll_error, aggressive=None,
             snapshot=None):
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
    m_full = max(1, min(m, int(round(m / poll_error ** 2)))) if poll_error > 0 else m
    ref = [SINCERE[method.ballot](u[g].mean(axis=0, keepdims=True))[0] for g in groups]
    Ug = [u[g].sum(axis=0) for g in groups]
    shared = [None] * len(groups)              # each group's answer in the last round (None = sincere)
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
            a, b = _leaders(lambda j: method.decide_many(tot + (2 ** j) * dev, C), C)
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
        settled = all((x is None and y is None) or (x is not None and y is not None and np.array_equal(x, y))
                      for x, y in zip(answers, shared))
        shared = answers
        B = sincere.copy()
        for mem, x in zip(groups, shared):        # everyone switches at once
            if x is not None:
                B[mem] = x
        if snapshot:
            snapshot(r, (B, settled, float(sincerity(B, u).mean())))
    return B, settled, float(sincerity(B, u).mean())


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
    u_sin = np.array([method.tally(B0[idx]) for idx in polls]) @ Ug

    def gain(ballot):
        B2 = B.copy()
        B2[mem] = ballot
        return float(np.mean(np.array([method.tally(B2[idx]) for idx in polls]) @ Ug - u_sin))
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
    bury = [c for c in order if c != Bo] + ([Bo] if Bo is not None else [])
    both = [A] + [c for c in bury if c != A]
    if ballot == "fav":
        out = [np.eye(C)[A]]
    elif ballot == "approval":
        if 0 < k < C:
            out = [up.astype(float)]
    elif ballot in ("score", "grade3"):
        S = methods.MAXS if ballot == "score" else 2
        lo = Ug[Bo] if Bo is not None else Ug.min()
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
        if Bo is not None and not up[Bo]:
            tb[Bo] = tb[rest].min() - 1 if len(rest) > 1 else tb[Bo]
        out = [tb - tb.min(), tied - tied.min(), rank_points(both), rank_points(bury)]
    elif ballot == "trunc":                                          # rank only down to A
        if 0 < k < C:
            t1 = np.zeros(C); t1[[c for c in order if up[c]]] = np.arange(k, 0, -1)
            t2 = np.zeros(C); t2[[c for c in comp[:k]]] = np.arange(k, 0, -1)
            out = [t1, t2]
    elif ballot == "rankapp":                                        # ranking + approval line at A
        if 0 < k < C:
            for o in (bury, both, order):          # approved: everyone liked at least as much as A
                o2 = [c for c in o if up[c]] + [c for c in o if not up[c]]   # the approved on top
                out.append(rank_points(o2) + C * up)
    elif ballot == "veto":
        if Bo is not None:
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
