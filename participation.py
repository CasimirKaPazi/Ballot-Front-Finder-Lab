"""Participation check (no-show paradox): could some voters get a better result by staying home?

For one ballot set B (sincere or learned) and one method: voters with the same sincere ranking and
the same ballot form a group. For each group, 1, half (rounded up) or all of its members leave out
their ballots while everyone else votes as before. If the winner lottery that results is better for
those who stayed home (higher expected utility, summed over them) than the one with their ballots,
the method failed participation for them. Leaving out a ballot means leaving out the whole ballot,
also the runoff part of a two-round election (staying home in both rounds).

This is a diagnostic: nobody actually abstains, and it does not change the strategic results.
Strategic voters in the poll rounds only change their ballot; staying home is not among their options.

Returns per election:
    noshow         1 if some group gains by staying home, else 0
    noshow_voters  share of voters in such groups (members of a group where 1, half or all could gain)
    noshow_gain    largest gain per absent voter, as a share of that voter's utility range
                   (favourite minus least liked), 0 when nobody gains
"""
import numpy as np
from outcomes import value

TOL = 1e-9


def _groups(B, u):
    """Indices of voters with the same sincere ranking and the same ballot, as lists."""
    key = np.concatenate([np.argsort(-u, axis=1, kind="stable"), B], axis=1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    order = np.argsort(inv, kind="stable")
    splits = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, splits)


def no_show(method, B, u, value_u, back):
    """B: ballots (n x C) in the field the voters face; u: their utilities in that field (for the
    groups); value_u: utilities on the original field (n x C0) and back: winner lottery in the
    field -> lottery on the original field, as in run.py. See the module docstring."""
    n, C = B.shape
    P0 = back(method.tally(B))
    groups = _groups(B, u)
    cases = []                                     # (group, k)
    for g in groups:
        for k in sorted({1, (len(g) + 1) // 2, len(g)}):
            if k < n:                              # never leave out everyone
                cases.append((g, k))
    if method.full is None:                        # features are sums over voters: subtract the absent ones
        F = method.features(B)
        tot = F.sum(axis=0)
        S = np.array([tot - k * F[g[0]] for g, k in cases])
        Ps = method.decide_many(S, C)
    else:
        keep = np.ones(n, bool)
        Ps = []
        for g, k in cases:
            keep[:] = True
            keep[g[:k]] = False
            Ps.append(method.tally(B[keep]))
    rng_u = value_u.max(axis=1) - value_u.min(axis=1)
    gaining, best = set(), 0.0
    for (g, k), P in zip(cases, Ps):
        absent = g[:k]
        d = value(back(P) - P0, value_u[absent])     # gain of each absent voter
        if d.sum() > TOL:
            gaining.add(int(g[0]))
            best = max(best, float(np.max(d / np.where(rng_u[absent] > 0, rng_u[absent], 1))))
    members = sum(len(g) for g in groups if int(g[0]) in gaining)
    return {"noshow": int(bool(gaining)), "noshow_voters": members / n, "noshow_gain": best}


NAN = {"noshow": float("nan"), "noshow_voters": float("nan"), "noshow_gain": float("nan")}
