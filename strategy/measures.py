"""How strategic a ballot is, measured against the voter's preferences: sincerity, fidelity, the
values a ballot implies, and the kinds of strategy (compromise, push-over, burial, exaggeration).
The same for every engine."""
import numpy as np

from ballots import BALLOTS


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


def fidelity(method, B, u, honest=None):
    """Per voter, how much of what the honest ballot says about the voter's preferences the ballot B
    keeps (since 2026-11-09). Over the pairs the voter prefers a > b, weighted by u[a] - u[b]: a pair
    the honest ballot puts in order loses half its weight if B ties it and all of it if B reverses it;
    a pair the honest ballot ties loses half if B reverses it. Fidelity is 1 - loss / (the weight of
    the pairs the honest ballot puts in order): 1 = B says everything the honest ballot says (the
    honest ballot itself, so the format's own limits do not count), 0.5 = B ties all of it, 0 = B
    reverses all of it; below 0 only if B also reverses pairs the honest ballot ties. Ballots are
    compared by their implied values (implied_values: two-round ballots by their round-1 marks).
    honest: the voters' honest ballots (default: the method's sincere ballots for u)."""
    P = method.pref(u)
    H = method.sincere(u) if honest is None else honest
    vh, vl = implied_values(method, H), implied_values(method, B)
    du = P[:, :, None] - P[:, None, :]
    w = np.where(du > 1e-12, du, 0.0)                  # each preferred pair once, weighted
    sh = np.sign(np.round(vh[:, :, None] - vh[:, None, :], 9))
    sl = np.sign(np.round(vl[:, :, None] - vl[:, None, :], 9))
    loss = (w * np.maximum(sh - sl, 0) / 2).sum(axis=(1, 2))
    den = (w * (sh > 0)).sum(axis=(1, 2))
    return np.where(den > 1e-12, 1 - loss / np.where(den > 1e-12, den, 1), 1.0)


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
