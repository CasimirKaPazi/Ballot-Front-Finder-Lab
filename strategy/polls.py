"""Polls: what the voters of every engine see of a round (docs/model.md, What a poll shows). A poll
counts a random sample of the voters' ballots by the method's own counts (Method.poll_view), scaled up
to the electorate, with its deviations from the true count widened by --poll-error."""
import numpy as np


def poll_tallies(tally, B, polls):
    """Winner probabilities on each poll sample, shape (samples, C)."""
    return np.array([tally(B[idx]) for idx in polls])


WIDEN = 6       # widen the polls up to 2**WIDEN times to find a runner-up


TOL = 1e-9


def default_poll_size(n):
    """Half the electorate, rounded to an odd number (100 -> 51, 99 -> 49, 200 -> 101)."""
    m = n // 2
    return m if m % 2 else m + 1


def poll_totals(pm, B, polls, n, m):
    """The features of the ballots B as voters count them (pm: Method.poll_view, with feature sums),
    their true totals, and the totals of each poll (one row per poll: its sample, scaled up from m
    voters to the electorate of n). Returns (F, tot, polls' totals)."""
    F = pm.features(B)
    W = np.zeros((len(polls), n))
    W[np.arange(len(polls))[:, None], polls] = n / m
    return F, F.sum(axis=0), W @ F


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
