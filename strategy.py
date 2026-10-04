"""Learned strategy: groups of voters with the same sincere ranking learn a shared ballot.

Loop (rounds):
  - A fresh "poll" is drawn each round: `samples` bootstrap resamples of the electorate
    (voters drawn with replacement). This stands in for uncertainty about how others vote,
    so a group gains from a ballot change in proportion to how often it would be pivotal.
  - Groups take turns in random order. A group proposes a ballot for all its members:
    usually a small change to its current shared ballot, sometimes a return to sincere.
  - The group keeps the proposal if its members' total expected utility, averaged over
    the poll samples, goes up.
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
    elif ballot == "score":
        c = rng.integers(C)
        b[c] = (b[c] + 1 + rng.integers(MAXS)) % (MAXS + 1)
    return b


def dissonance(B, u):
    """Per voter: utility-weighted share of preferences the ballot reverses.

    For every pair the ballot orders (a rated above b) while the voter likes b more,
    add u[b] - u[a]; divide by the sum of |u[a] - u[b]| over all pairs.
    0 = no reversed preference (every sincere ballot), 1 = every preference reversed.
    """
    above = B[:, :, None] > B[:, None, :]                  # ballot says a > b
    loss = np.clip(u[:, None, :] - u[:, :, None], 0, None)  # u[b] - u[a] where positive
    total = np.abs(u[:, :, None] - u[:, None, :]).sum(axis=(1, 2)) / 2
    return (above * loss).sum(axis=(1, 2)) / np.where(total > 0, total, 1)


def group_value(tally, B, polls, Ug):
    """Expected total utility of a group, averaged over poll samples."""
    return float(np.mean([tally(B[idx]) @ Ug for idx in polls]))


def learn(method, u, rng, rounds=15, samples=10):
    """Return learned ballots, whether learning settled, and mean cognitive dissonance."""
    n, C = u.shape
    sincere = method.sincere(u)
    B = sincere.copy()
    keys = [tuple(r) for r in np.argsort(-u, axis=1, kind="stable")]
    groups = {}
    for v, k in enumerate(keys):
        groups.setdefault(k, []).append(v)
    groups = [np.array(m) for m in groups.values()]
    shared = [None] * len(groups)              # None = members vote sincerely
    ref = [SINCERE[method.ballot](u[m].mean(axis=0, keepdims=True))[0] for m in groups]
    Ug = [u[m].sum(axis=0) for m in groups]
    changed_last = True
    for _ in range(rounds):
        polls = rng.integers(0, n, (samples, n))
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
            B2 = B.copy()
            B2[mem] = prop_rows
            if group_value(method.tally, B2, polls, Ug[g]) > group_value(method.tally, B, polls, Ug[g]) + 1e-12:
                B = B2
                shared[g] = None if to_sincere else prop
                changed_last = True
    return B, not changed_last, float(dissonance(B, u).mean())
