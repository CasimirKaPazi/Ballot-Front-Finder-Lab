"""Manipulation test: can a coalition of voters, starting from honest ballots, make their candidate win?

After John Huang's votesim (one- and two-sided strategy) and the vulnerability measure of
Green-Armytage, Tideman & Cosman (2016). For one method and one election, from the sincere ballots:

- T, the "topdog": the honest winner (the most likely one if the honest result is a lottery).
- For every other candidate X (the "underdog"): the coalition is every voter who prefers X to T.
  It tries each naive exaggeration of its ballot type around the pair, with X as the candidate it
  prefers and T as the one it opposes (BALLOTS[...].exaggerate: e.g. X to the top and T to the
  bottom of a ranking, approve everyone liked at least as much as X, top score down to X and 0
  from T down). Every member uses the same tactic, applied to their own utilities, so their ballots
  can still differ. Everyone else votes sincerely.
- One-sided: a tactic succeeds if the coalition gains (expected utility summed over its members,
  higher than with the honest result). Over all X and tactics, the successful one with the lowest
  total utility is the recorded outcome: the worst a rational, informed coalition can do to the
  electorate. Without any success, the honest result.
- Two-sided: for each successful one-sided tactic, the topdog's supporters (everyone else) answer
  with an exaggeration for T against X: of their ballot type's naive exaggerations, the one that
  leaves the coalition the least gain, so a manipulation stands only if it pays against every answer.
  Recorded: the lowest total utility among the manipulations that stand (each with that answer);
  without one, the honest result. (votesim keeps all two-sided outcomes, backfires included, and
  lets the defence bullet-vote; here the defence tries every exaggeration of its ballot type, and
  only manipulations that still pay count.)

This is a diagnostic beside the poll rounds: it asks how bad deliberate, informed manipulation from
honest ballots can get, not what typical strategy does. It does not change any other result.

Returns per election:
    manip1     1 if some coalition gains one-sided, else 0
    manip2     1 if some coalition still gains against the defence, else 0
    U_manip1   total utility (original field) of the recorded one-sided outcome
    U_manip2   the same, two-sided
"""
import numpy as np
from ballots import BALLOTS
from outcomes import value

TOL = 1e-9


def _tactics(ballot, u, voters, a, b, sincere):
    """Per voter (rows of u): the naive exaggerations for a against b. Returns one ballot matrix per
    tactic (the i-th exaggeration of every voter); a voter whose ballot type offers fewer keeps their
    last one, or their sincere ballot if it offers none (approval when they would approve everyone)."""
    per = [BALLOTS[ballot].exaggerate(u[v], a, b) for v in voters]
    k = max((len(x) for x in per), default=0)
    return [np.array([x[min(i, len(x) - 1)] if x else sincere[j] for j, x in enumerate(per)]) for i in range(k)]


def manipulation(method, B0, uf, value_u, back):
    """B0: sincere ballots in the field the voters face; uf: their utilities in that field;
    value_u: utilities on the original field and back: lottery in the field -> original field
    (as in run.py). See the module docstring."""
    n, C = uf.shape
    U = value_u.sum(axis=0)
    P0f = method.tally(B0)
    P0 = back(P0f)
    U0 = float(value(P0, U))
    T = int(np.argmax(P0f))
    best1, best2, m1, m2 = U0, U0, 0, 0
    for X in range(C):
        if X == T:
            continue
        coal = np.flatnonzero(uf[:, X] > uf[:, T])
        if len(coal) == 0:
            continue
        rest = np.setdiff1d(np.arange(n), coal)
        with method.applied():                     # real utilities: ranking X first helps X under every count
            attacks = _tactics(method.ballot, uf, coal, X, T, B0[coal])
            defence = _tactics(method.ballot, uf, rest, T, X, B0[rest]) if len(rest) else []
        for A in attacks:
            B = B0.copy()
            B[coal] = A
            P = back(method.tally(B))
            if value(P - P0, value_u[coal]).sum() <= TOL:
                continue                           # backfires or changes nothing for the coalition
            m1 = 1
            best1 = min(best1, float(value(P, U)))
            answer, gain = None, np.inf            # the answer that leaves the coalition the least gain
            for D in defence:
                B2 = B.copy()
                B2[rest] = D
                P2 = back(method.tally(B2))
                g = float(value(P2 - P0, value_u[coal]).sum())
                if g < gain - TOL:
                    answer, gain = P2, g
                if gain <= TOL:
                    break                          # countered: no need to try further answers
            if answer is not None and gain > TOL:
                m2 = 1
                best2 = min(best2, float(value(answer, U)))
            if not defence:                        # nobody left to defend: the manipulation stands
                m2 = 1
                best2 = min(best2, float(value(P, U)))
    return {"manip1": m1, "manip2": m2, "U_manip1": best1, "U_manip2": best2}


NAN = {"manip1": "", "manip2": "", "U_manip1": "", "U_manip2": ""}
