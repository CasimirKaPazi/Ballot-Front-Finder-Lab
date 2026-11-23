"""False positives of strategy: whether a move pays the voters who make it, judged on the ballots as
they were cast (a diagnostic, the same for every engine, format and move; docs/model.md, Checking
strategy: false positives; tools/falsepos.py runs it)."""
import numpy as np

from .measures import implied_values


FP_TOL = 1e-9        # false_positives: a gain within this share of the voter's utility range is none


def weak_order(values, labels):
    """A ballot's order of a few candidates as text: labels from best to worst, '>' between levels, '='
    within one (values within 1e-9 tie): weak_order([3, 1, 3], "123") == "1=3>2"."""
    idx = sorted(range(len(values)), key=lambda i: -values[i])
    out = labels[idx[0]] if idx else ""
    for a, b in zip(idx, idx[1:]):
        out += ("=" if abs(values[a] - values[b]) <= 1e-9 else ">") + labels[b]
    return out


def false_positives(method, B, sincere, u, rng, samples=500, poll_size=0, poll_error=2.0, share=1.0, view="count", base=None):
    """Whether strategy pays the voters who use it, judged on the ballots as they were cast (a diagnostic,
    the same for every engine, format and move; docs/model.md, Checking strategy: false positives).
    Who acts together: the voters whose ballot differs from their sincere one are grouped into
    coalitions by two things only the ballots tell: their sincere order of the three leading places
    (as the polls of the final ballots show them, polls._leaders), and what their ballot does among
    these three (the order or tie it puts each pair of them in). So a coalition is the voters of one
    type making one kind of move in the race that matters (burial of a rival, compromise, levelling,
    push-over, or a change outside the three, which all fall in one coalition per type), whatever
    engine or move produced it.
    The counterfactual: the coalition goes home (its members cast their sincere ballots), everyone
    else keeps theirs. A member's gain is their expected utility with the cast ballots minus that with
    the coalition at home, as a share of their utility range, counted two ways:
      exact   by the true count of all ballots (most moves change nothing there: no single result hangs
              on them)
      polls   averaged over `samples` polls of the cast ballots (the same with and without the
              coalition): random samples of poll_size voters (0: the run's default) / poll_error**2,
              so that they deviate from the true count as much as voters assume (--poll-error; as
              strategy.rounds polls the methods without feature sums), the uncertainty strategy
              answers. A move whose gain is none or negative here is a false positive: it buys
              nothing, or costs, against what the others actually cast, even with the uncertainty
              the voters assume.
    Returns a dict: changed (share of all voters), and over the voters who changed: fp (share with a
    polls gain <= FP_TOL: false positives), fp_none (gain within +-FP_TOL: the move changes nothing),
    fp_hurt (gain < -FP_TOL: it backfires), gain (mean polls gain), exact_help and exact_hurt (shares
    with an exact gain above or below the tolerance), exact_gain (mean), coalitions (their number), and
    patterns: per kind of move, {pattern: (voters, false positives, sum of their polls gains)}, the
    pattern being what the ballot does to the voter's own three (weak_order, 1 = the one of the three
    they like most: "1>3>2" buries their second, "2>1>3" compromises, "1=2>3" levels, "1>2>3" changes
    something else). Shares are nan when nobody changed. The share with no gain depends on the
    number of polls (a small gain shows in few of them): about 500 are needed for it to settle.
    share: the share of a coalition that goes home, each member at random with that chance, drawn
    afresh for every poll (and for each of up to 100 true counts): 1 (the default) sends the whole
    coalition home; the engine's --coordination (0.25) asks what the pooled engine assumes when it
    values a move, that a quarter of a type follows it. A smaller share gives smaller gains (the
    sign is what tells a false positive).
    view: how the polls are counted. "count" (the default): the method's own count of each sampled
    poll (for every method alike; the samples are made smaller by poll_error**2 to widen them).
    "engine": as the engines see a poll: the method's poll view (Method.poll_view: the method itself
    where it is counted from totals a poll publishes, else its stand-in, e.g. IRV estimated from first
    preferences and transfers), on polls of poll_size voters scaled up, deviations x poll_error. The
    difference between the two views is the part of the false positives that comes from voters
    reading polls through the poll view, not from the ballots having moved.
    base: the others' ballots (default: B, the ballots as cast). Given the ballots of the round
    before (the ones the voters' polls were taken of), each coalition's move is judged against what
    the voters saw when they made it: the others keep their ballots of that round, the coalition
    casts B or goes home, and the leading places are those of that round's polls. The difference
    from the default is the part of the false positives that comes from everyone moving at once.
    Not covered: whether a more honest ballot than the sincere one would have done as well
    (levelling instead of a reversal), and coalitions across types."""
    from .polls import default_poll_size, poll_totals, _leaders, _smaller, _widened
    B, sincere, u = np.asarray(B, float), np.asarray(sincere, float), np.asarray(u, float)
    O = B if base is None else np.asarray(base, float)          # the others' ballots
    n, C = u.shape
    changed = np.any(B != sincere, axis=1)
    nan = float("nan")
    out = {"changed": float(changed.mean()), "fp": nan, "fp_none": nan, "fp_hurt": nan, "gain": nan,
           "exact_help": nan, "exact_hurt": nan, "exact_gain": nan, "coalitions": 0, "patterns": {}}
    if not changed.any():
        return out
    m = min(n, poll_size if poll_size > 0 else default_poll_size(n))
    if view == "count":
        m = max(1, min(m, int(round(m / poll_error ** 2)))) if poll_error > 0 else m
    polls = np.stack([rng.choice(n, m, replace=False) for _ in range(samples)])
    widened = {}
    if view == "engine":
        pm = method.poll_view()
        F, tot, WF = poll_totals(pm, O, polls, n, m)
        dev = poll_error * (WF - tot)

        def winners_at(j):
            if j not in widened:
                widened[j] = pm.decide_many(_widened(tot, dev, j), C)
            return widened[j]

        def mean_polls(X):
            rows = np.flatnonzero(np.any(X != O, axis=1))
            D = (pm.features(X[rows]) - F[rows]).sum(axis=0) if len(rows) else 0.0
            return pm.decide_many(tot + dev + D, C).mean(axis=0)
    elif view == "count":
        def winners_at(j):
            if j not in widened:
                widened[j] = np.array([method.tally(O[idx]) for idx in _smaller(polls, j, rng, n)])
            return widened[j]

        def mean_polls(X):
            return np.mean([method.tally(X[idx]) for idx in polls], axis=0)
    else:
        raise ValueError(f"view must be 'count' or 'engine', not {view!r}")
    places = [c for c in _leaders(winners_at, C, 3) if c is not None]
    span = u.max(axis=1) - u.min(axis=1)
    span = np.where(span > 1e-12, span, 1.0)
    vB = implied_values(method, B)[:, places]
    order = np.argsort(-u[:, places], axis=1, kind="stable")
    pairs = [(i, j) for i in range(len(places)) for j in range(i + 1, len(places))]
    coal = {}
    for v in np.flatnonzero(changed):
        kind = tuple(int(np.sign(round(vB[v, i] - vB[v, j], 9))) for i, j in pairs)
        coal.setdefault((tuple(order[v]), kind), []).append(v)

    if base is None:
        P_exact, P_polls = method.tally(B), mean_polls(B)
    g_exact, g_polls = np.zeros(n), np.zeros(n)

    def home(G):                                   # the ballots with (a share of) coalition G at home
        X = O.copy()
        X[G] = B[G]
        H = G if share >= 1 else G[rng.random(len(G)) < share]
        X[H] = sincere[H]
        return X
    for G in coal.values():
        G = np.array(G)
        if base is not None:                       # the others as they were: the coalition's move alone
            X = O.copy()
            X[G] = B[G]
            P_exact, P_polls = method.tally(X), mean_polls(X)
        if share >= 1:
            X = home(G)
            P_e, P_p = method.tally(X), mean_polls(X)
        else:
            P_e = np.mean([method.tally(home(G)) for _ in range(min(samples, 100))], axis=0)
            P_p = (np.mean([method.tally(home(G)[idx]) for idx in polls], axis=0) if view == "count" else
                   np.mean([mean_polls(home(G)) for _ in range(min(samples, 100))], axis=0))
        g_exact[G] = u[G] @ (P_exact - P_e) / span[G]
        g_polls[G] = u[G] @ (P_polls - P_p) / span[G]
    labels = "123"[:len(places)]
    pat = {}
    for v in np.flatnonzero(changed):
        key = weak_order(list(vB[v, order[v]]), labels)
        c, f, g = pat.get(key, (0, 0, 0.0))
        pat[key] = (c + 1, f + int(g_polls[v] <= FP_TOL), g + float(g_polls[v]))
    out["patterns"] = pat
    gp, ge = g_polls[changed], g_exact[changed]
    out.update(fp=float(np.mean(gp <= FP_TOL)), fp_none=float(np.mean(np.abs(gp) <= FP_TOL)),
               fp_hurt=float(np.mean(gp < -FP_TOL)), gain=float(gp.mean()),
               exact_help=float(np.mean(ge > FP_TOL)), exact_hurt=float(np.mean(ge < -FP_TOL)),
               exact_gain=float(ge.mean()), coalitions=len(coal))
    return out
