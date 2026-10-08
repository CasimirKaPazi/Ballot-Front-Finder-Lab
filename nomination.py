"""Strategic nomination: candidates of the initial field withdraw, re-enter, or field clones, as long
as that changes the (sincere) result in their favour. After Green-Armytage (2014), "Strategic voting
and nomination", Social Choice and Welfare 42, with clones added.

Field. Only the candidates of the initial draw take part; nobody new enters. A clone is an almost
identical copy of an original: its position is the original's plus a small offset (CLONE_SD per
dimension, drawn once per election, the same for every method), so voters rank the two next to
each other and split between them at random.

Candidates' preferences. A candidate sits in the issue space like a voter and values a winner by
the same utility, exp(-distance / width): itself 1, others by distance. A clone counts exactly as
its original (the same platform; otherwise the clone's small random offset would make candidates
field clones of the winner just because the copy happens to sit a little closer to them). A lottery
of winners is valued by its expected utility. These are policy preferences: a candidate gives way
when someone close to it can win instead of someone far away.

Moves, by any original candidate (in the race or out of it):
    exit       leave the race (exits=1); at least two originals stay in
    re-entry   come back after an exit (exits=1)
    clone      field a clone of any original in the race, itself included (clones = most clones in
               the race at once, 0 = none; at most one clone per original)
    unclone    withdraw a clone it fielded
A move is made only if it strictly improves the mover's expected utility, judged by the method's
result with SINCERE ballots on the new field (candidates foresee honest voting, as in
Green-Armytage; they do not foresee learned strategy).

Dynamics: in each round the candidates take turns in random order; each makes its best improving
move, if any. When a whole round passes without a move, the field is an equilibrium (settled);
otherwise the run stops after ROUNDS rounds (cycles are possible) and the last field is used.

Evaluation (run.py): the result on the final field is judged on the ORIGINAL field. A clone's
chance of winning counts as its original's; withdrawn candidates stay in the comparison. So
VSE, WME, the random ballot and the best candidate are the same as without nomination: the
change in VSE and WME is the effect of nomination on the outcome.
"""
import numpy as np

CLONE_SD = 0.01   # offset of a clone from its original, per dimension (positions are O(1), width 1.5)
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


def moves(f, i, exits, clones):
    """All fields candidate i can move to."""
    out = []
    if exits:
        if f.inn[i] and f.inn.sum() > 2:
            g = f.copy(); g.inn[i] = False
            g.clones = {d: s for d, s in g.clones.items() if g.inn[d]}   # a clone needs its original in the race
            out.append(g)
        elif not f.inn[i]:
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
            cache[k] = (e, method.honest(utilities(e, u, u_clone)))
        return cache[k]

    def value(i, g):
        e, P = result(g)
        return float(to_original(P, e, C) @ cu[i])

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


def clone_setup(voters, cands, width, rng):
    """Per election: voters' utilities for each original's clone, and candidates' utilities for
    every original (cu[i, d]). rng: the election's own clone stream (the same for every method)."""
    cpos = cands + rng.normal(0.0, CLONE_SD, cands.shape)
    u_clone = np.exp(-np.linalg.norm(voters[:, None, :] - cpos[None, :, :], axis=2) / width)
    cu = np.exp(-np.linalg.norm(cands[:, None, :] - cands[None, :, :], axis=2) / width)
    return u_clone, cu
