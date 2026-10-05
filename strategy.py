"""Learned strategy: groups of voters with the same sincere ranking learn a shared ballot.

Partial strategy: `strategic` marks the voters who take part (a random share of the electorate,
drawn in run.py, the same voters for every method). The others always vote sincerely.

Coordination: `group_size` caps how many voters move together. Each ranking group (strategic
voters with the same sincere ranking) is split at random into subgroups of at most that size;
0 means no cap (whole ranking groups).

Loop (rounds):
  - A fresh "poll" is drawn each round: `samples` bootstrap resamples of the electorate
    (voters drawn with replacement). This stands in for uncertainty about how others vote,
    so a group gains from a ballot change in proportion to how often it would be pivotal.
  - Groups take turns in random order. A group proposes a ballot for all its members:
    usually a small change to its current shared ballot, sometimes a return to sincere.
  - The group keeps the proposal if its members' total expected utility, averaged over
    the poll samples, goes up. A return to sincere is also kept when it is not worse:
    voters prefer honesty when indifferent, so deviations that stopped paying off fade out
    instead of accumulating.
Voters start sincere. A group that has never deviated proposes changes to the sincere
ballot of its average member.
"""
import numpy as np
from methods import SINCERE, MAXS

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
        c = rng.integers(C)
        b[c] = (b[c] + 1 + rng.integers(MAXS)) % (MAXS + 1)
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
