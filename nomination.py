"""Strategic nomination: candidates of the initial field withdraw, re-enter, or field clones, as long
as that changes the (sincere) result in their favour. After Green-Armytage (2014), "Strategic voting
and nomination", Social Choice and Welfare 42, with clones added.

Field. Only the candidates of the initial draw take part; nobody new enters. A clone is an almost
identical copy of an original: its position is the original's plus a small offset (CLONE_SD per
dimension, drawn once per election, the same for every method), so voters rank the two next to
each other and split between them at random.

Candidates' preferences. A candidate sits in the issue space like a voter and values a winner by
the same utility, exp(-distance / width): themselves 1, others by distance. A clone counts exactly as
their original (the same platform; otherwise the clone's small random offset would make candidates
field clones of the winner just because the copy happens to sit a little closer to them). A lottery
of winners is valued by its expected utility. These are policy preferences: a candidate gives way
when someone close to them can win instead of someone far away.

Moves, by any original candidate (in the race or out of it):
    exit       leave the race (exits=1); at least two originals stay in
    re-entry   come back after an exit (exits=1)
    clone      field a clone of any original in the race, themselves included (clones = most clones in
               the race at once, 0 = none; at most one clone per original)
    unclone    withdraw a clone they fielded
A move is made only if it strictly improves the mover's expected utility, judged by the method's
result with SINCERE ballots on the new field (candidates foresee honest voting, as in
Green-Armytage; they do not foresee learned strategy).

Dynamics: in each round the candidates take turns in random order; each makes its best improving
move, if any. When a whole round passes without a move, the field is an equilibrium (settled);
otherwise the run stops after ROUNDS rounds (cycles are possible) and the last field is used.

In the race (--race-cost, run.py): after each poll round but the last, the candidates take turns
again in random order, one move each at most, without re-entry. They judge by the round's polls:
the poll ballots of the round just cast, carried to the new field (carry), counted by the method,
averaged over the polls. A move is made only if it gains more than a cost, a share of the mover's
utility range (their own win against their worst candidate's). The voters then go on from the
carried ballots on the new field, with a fresh engine (what they had learned about the old field
is gone). See Race.

Evaluation (run.py): the result on the final field is judged on the ORIGINAL field. A clone's
chance of winning counts as its original's; withdrawn candidates stay in the comparison. So
VSE, WME, the random ballot and the best candidate are the same as without nomination: the
change in VSE and WME is the effect of nomination on the outcome.
"""
import numpy as np
import methods as M
from models import distances, utility
from outcomes import value as outcome_value

CLONE_SD = 0.01   # offset of a clone from its original, per dimension (positions are O(1), width about 1)
ROUNDS = 20
TOL = 1e-12


class Field:
    """In-race originals (bool per original) and clones (original -> sponsor)."""

    def __init__(self, C):
        self.inn = np.ones(C, bool)
        self.clones = {}

    def copy(self):
        f = Field(len(self.inn))
        f.inn, f.clones = self.inn.copy(), dict(self.clones)
        return f

    def entries(self):
        """Candidates on the ballot: (original, is_clone), originals first, in index order."""
        return ([(int(d), 0) for d in np.flatnonzero(self.inn)]
                + [(d, 1) for d in sorted(self.clones)])

    def key(self):
        return (tuple(self.inn), tuple(sorted(self.clones.items())))


def utilities(entries, u, u_clone):
    """Voter utilities for the candidates on the ballot."""
    return np.stack([u_clone[:, d] if k else u[:, d] for d, k in entries], axis=1)


def origins(entries):
    return np.array([d for d, _ in entries])


def to_original(P, entries, C):
    """Winner probabilities on the field -> on the original candidates (a clone counts as its original)."""
    return np.bincount(origins(entries), weights=P, minlength=C)


def moves(f, i, exits, clones, reentry=True):
    """All fields candidate i can move to (reentry=False: in the race, no coming back after an exit)."""
    out = []
    if exits:
        if f.inn[i] and f.inn.sum() > 2:
            g = f.copy(); g.inn[i] = False
            g.clones = {d: s for d, s in g.clones.items() if g.inn[d]}   # a clone needs its original in the race
            out.append(g)
        elif not f.inn[i] and reentry:
            g = f.copy(); g.inn[i] = True; out.append(g)
    if clones:
        if len(f.clones) < clones:
            for d in np.flatnonzero(f.inn):
                if int(d) not in f.clones:
                    g = f.copy(); g.clones[int(d)] = i; out.append(g)
        for d, s in f.clones.items():
            if s == i:
                g = f.copy(); del g.clones[d]; out.append(g)
    return out


def nominate(method, u, u_clone, cu, rng, exits=0, clones=0, rounds=ROUNDS):
    """Run the nomination dynamics for one method.
    u, u_clone: voters' utilities for originals and their clones (n x C each);
    cu: candidates' utilities, cu[i, d] = candidate i's utility if original d (or its clone) wins.
    Returns the final Field and stats (exits, clones, settled, moves)."""
    C = u.shape[1]
    f = Field(C)
    if not exits and not clones:
        return f, {"nom_exits": 0, "nom_clones": 0, "nom_settled": 1, "nom_moves": 0}
    cache = {}

    def result(g):
        k = g.key()
        if k not in cache:
            e = g.entries()
            with M.candidates(M.candidate_utilities(cu, e)):      # for methods where candidates vote
                cache[k] = (e, method.honest(utilities(e, u, u_clone)))
        return cache[k]

    def value(i, g):
        e, P = result(g)
        return float(outcome_value(to_original(P, e, C), cu[i]))

    settled, n_moves = False, 0
    for _ in range(rounds):
        moved = False
        for i in rng.permutation(C):
            i = int(i)
            now = value(i, f)
            best, best_v = None, now + TOL
            for g in moves(f, i, exits, clones):
                v = value(i, g)
                if v > best_v:
                    best, best_v = g, v
            if best is not None:
                f, moved, n_moves = best, True, n_moves + 1
        if not moved:
            settled = True
            break
    return f, {"nom_exits": int((~f.inn).sum()), "nom_clones": len(f.clones), "nom_settled": int(settled),
               "nom_moves": n_moves}


def carry(method, B, ents_from, ents_to, u, u_clone):
    """Ballots B cast on the field ents_from, carried to the field ents_to: read as the values they imply
    (strategy.implied_values) and cast again in the method's format (Method.sincere; Inverse-* methods
    read the ordinary way). A candidate who left is dropped. A new clone gets the value the ballot gives
    its original, and each voter puts the one of the two they like better (by their utilities u,
    u_clone) just above the other: the ballot otherwise stays as it was, and the two split the
    original's support as the voters' sincere preferences split it."""
    from strategy import implied_values         # strategy does not import nomination; kept local anyway
    V = implied_values(method, B)
    at = {e: j for j, e in enumerate(ents_from)}
    W = np.empty((len(B), len(ents_to)))
    for i, (d, k) in enumerate(ents_to):
        j = at.get((d, k))
        if j is not None:
            W[:, i] = V[:, j]
        else:                                     # a new clone (the original is in the race)
            W[:, i] = V[:, at[(d, 0)]] + 1e-9 * np.sign(u_clone[:, d] - u[:, d])
    return method.sincere(-W if method.inverse else W)


class Race:
    """Strategic nomination between poll rounds (see the module text). One per method and election.
    field: the field after nomination before the race; cost: the smallest gain worth a move, as a share
    of the mover's utility range; rng: the order of the candidates (own stream)."""

    def __init__(self, method, u, u_clone, cu, field, exits, clones, cost, rng):
        self.m, self.u, self.u_clone, self.cu = method, u, u_clone, cu
        self.f, self.exits, self.clones, self.cost, self.rng = field, exits, clones, cost, rng
        self.C = u.shape[1]
        self.moves = 0
        self.span = cu.max(axis=1) - cu.min(axis=1)

    def step(self, B, polls):
        """Ballots B were cast on the current field; polls: the round's poll samples (voter indices).
        Each candidate in turn makes at most one move. Returns True if the field changed."""
        start = self.f.entries()
        cache = {}

        def result(g):
            k = g.key()
            if k not in cache:
                e = g.entries()
                Bg = B if e == start else carry(self.m, B, start, e, self.u, self.u_clone)
                with M.candidates(M.candidate_utilities(self.cu, e)):
                    P = np.mean([self.m.tally(Bg[p]) for p in polls], axis=0)
                cache[k] = to_original(P, e, self.C)
            return cache[k]

        changed = False
        for i in self.rng.permutation(self.C):
            i = int(i)
            now = float(outcome_value(result(self.f), self.cu[i]))
            best, best_v = None, now + self.cost * self.span[i]
            for g in moves(self.f, i, self.exits, self.clones, reentry=False):
                v = float(outcome_value(result(g), self.cu[i]))
                if v > best_v:
                    best, best_v = g, v
            if best is not None:
                self.f, changed, self.moves = best, True, self.moves + 1
        return changed


def clone_setup(voters, cands, width, rng, voter_weights=None, cand_weights=None, form="exp"):
    """Per election: voters' utilities for each original's clone, and candidates' utilities for
    every original (cu[i, d]). rng: the election's own clone stream (the same for every method).
    Distances and utilities as in models.py (with the model's dimension weights, if any)."""
    cpos = cands + rng.normal(0.0, CLONE_SD, cands.shape)
    u_clone = utility(distances(voters, cpos, voter_weights), width, form)
    return u_clone, cand_prefs(cands, width, cand_weights, form)


def cand_prefs(cands, width, cand_weights=None, form="exp"):
    """Candidates' utilities for every original: cu[i, d], candidate i's utility if d wins (by the same
    utility as voters'; no random draw). Used by nomination, campaigns and the methods where candidates
    cast or fill in votes (methods.candidate_utilities)."""
    return utility(distances(cands, cands, cand_weights), width, form)
