"""What a method's ballots and count can carry (a diagnostic, not part of the model).

For every method and number of candidates this lists, from the method's own definition (its ballot
format and the features it is counted from), with no electorate and no strategy:

    ballots    how many ballots the format has
    counts     how many of them the count tells apart
    vertices   the vertices of the ballot polytope, the convex hull of what single ballots add to the
               count. In a large electorate a best reply is one of them, so this is what a strategic
               ballot can say
    rank       the number of independent numbers about the electorate the count keeps, not counting
               the size of the electorate
    sym        whether the ballot polytope is centrally symmetric (opposite voters then add the same
               to the strategic count, whoever they are)
    EC         whether a ballot together with its reverse never changes the outcome (the Equality
               Criterion); "-" where the format has no reverses

The count is the coarsest additive summary of ballots that determines the outcome. A method's features
are an additive summary that determines it, but they can hold more than the outcome ever depends on
(a number added to all totals; both directions of a pairwise count where only the margin matters). So
the tool finds what the outcome depends on by looking:

1. Search. It draws large random profiles, finds points between two of them where the outcome changes,
   and measures there how the boundary lies (its normal: by how much each of a set of ballots moves the
   outcome across). A normal is kept only if it predicts the boundary in other directions too. The
   normals found span what the outcome depends on; a ballot's count vector is what they see of it.
2. Test. Changes of ballots that none of the normals sees should never change the outcome. Many such
   changes are tried on random profiles; where the outcome does differ, the boundary in between is
   measured and taken in, and the test starts again.

Both steps are random, so the result is evidence, not proof: every normal kept is real, but a boundary
that random profiles rarely come near can be missed by both steps, and the count found is then too
small. A rank without a mark means the test found nothing more. A "+" on the rank (and "?" on vertices,
sym and EC) means the test ended unclear: it saw differences whose boundary it could not measure, or ran out of
time. That happens for rules with boundaries that are not flat (products of totals) or that move in
whole steps (medians counted position by position). Other seeds (--seed) are a further check.

A "+" on the vertices means there are at least so many: with more than 4000 different count vectors
(Score runoff with four candidates) only the vertices that random directions find are counted, and sym
is left open.

Not covered: methods without features (counted from the ballots directly), methods whose outcome is a
lottery, methods that need the candidates' own ballots (asset voting, delegation), and add-on ballot
formats without an entry in ENUMERATE below.

Usage (from the lab's folder):
    python tools/information.py                                  # the standard set, 3 candidates
    python tools/information.py --cands 3 4 --methods Plurality Approval Score Borda IRV Schulze
    python tools/information.py --methods Approval --show       # also list one ballot per vertex
    python tools/information.py --methods 'smith >> "Score"'    # a composition
Writes nothing; prints the table. Three candidates take a few seconds per method, four up to a minute
for score ballots (0..5: 1296 ballots).
"""
import argparse
import os
import sys
import time
from itertools import permutations, product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the lab's modules
import methods  # noqa: E402
from methods import METHODS  # noqa: E402
import ballots as ballots_mod  # noqa: E402

VOTERS = 100000   # the size of the profiles in which boundaries are looked for
DELTA = 1e-3      # how far from a boundary the normal is measured (in ballots; far above the tallies' EPS)
REACH = 10.0      # how far a single direction is followed to find the boundary again (in ballots)
CHECK = 1e-5      # a normal is kept if it predicts the boundary in other directions this closely
TOL = 1e-4        # normals are measured to about 1e-6; count vectors closer than this are the same


# ---------- every ballot of a format ----------
def _rankings(C):
    return np.array(list(permutations(range(C))), float)


def _all(levels, C):
    return np.array(list(product(range(levels), repeat=C)), float)


def _marks(C):
    """Approval marks: any set but the empty and the full one (as the lab's approval ballots)."""
    X = _all(2, C)
    s = X.sum(axis=1)
    return X[(s > 0) & (s < C)]


def _truncated(C, kmax, points):
    out = []
    for k in range(1, kmax + 1):
        for order in permutations(range(C), k):
            b = np.zeros(C)
            for pos, c in enumerate(order):
                b[c] = points(k, pos)
            out.append(b)
    return np.unique(np.array(out), axis=0)


def _scorecut(C):
    S = _all(methods.MAXS + 1, C)
    out = [ballots_mod.cut_join(S, S >= t) for t in range(methods.MAXS + 1)]
    return np.unique(np.concatenate(out), axis=0)


def _two_round(C, marks):
    R = _rankings(C)
    return (R[:, None, :] + C * marks[None, :, :]).reshape(-1, C)


ENUMERATE = {   # ballot type -> every ballot for C candidates, in the encoding of ballots.py
    "fav": lambda C: np.eye(C),
    "veto": lambda C: -np.eye(C),
    "strict": _rankings,
    "rank": lambda C: _truncated(C, C - 1, lambda k, pos: C - 1 - pos),
    "weak": lambda C: _all(C, C),
    "trunc": lambda C: _truncated(C, C - 1, lambda k, pos: k - pos),
    "trunc3": lambda C: _truncated(C, min(methods.TRUNC3, C), lambda k, pos: k - pos),
    "approval": _marks,
    "score": lambda C: _all(methods.MAXS + 1, C),
    "grade3": lambda C: _all(3, C),
    "scorecut": _scorecut,
    "fav_run": lambda C: _two_round(C, np.eye(C)),
    "app_run": lambda C: _two_round(C, _marks(C)),
    "score_run": lambda C: _two_round(C, _all(methods.MAXS + 1, C)),
}


def _rev_scorecut(X):
    S, a = ballots_mod.cut_parts(X)
    return (methods.MAXS - S) + (methods.MAXS + 1) * ~a


REVERSE = {     # ballot type -> the reversed ballots (those not in the format are left out afterwards)
    "strict": lambda X: X.shape[1] - 1 - X,
    "rank": lambda X: X.shape[1] - 1 - X,
    "weak": lambda X: X.shape[1] - 1 - X,
    "approval": lambda X: 1 - X,
    "score": lambda X: methods.MAXS - X,
    "grade3": lambda X: 2 - X,
    "scorecut": _rev_scorecut,
    "app_run": lambda X: (X.shape[1] - 1 - np.mod(X, X.shape[1])) + X.shape[1] * (X < X.shape[1]),
    "score_run": lambda X: (X.shape[1] - 1 - np.mod(X, X.shape[1]))
    + X.shape[1] * (methods.MAXS - np.floor_divide(X, X.shape[1])),
}


# ---------- the count ----------
class Count:
    """A method's count for C candidates: the ballots, what each adds to the features (with the size of
    the electorate as one more number), and a basis of the space these span."""

    def __init__(self, method, C):
        self.method, self.C = method, C
        self.X = ENUMERATE[method.ballot](C)
        F = method.features(self.X)
        self.F = np.concatenate([F, np.ones((len(F), 1))], axis=1)       # last column: one voter
        self.basis = _independent_rows(self.F)                           # ballots whose vectors span the rest
        self.E = self.F[self.basis]                                      # (D, d + 1)
        self.D = len(self.basis)
        self.coords = np.linalg.lstsq(self.E.T, self.F.T, rcond=None)[0].T   # each ballot in the basis: (k, D)
        self.size = np.ones(self.D)                                      # the size of the electorate on the basis
        self.normals = np.zeros((0, self.D))                             # an orthonormal basis of the normals found
        self.raw = []                                                    # every normal measured
        self._vertices = None

    def outcome(self, S):
        """The outcome at summed features S (with the size as last entry), as something to compare."""
        self.method._C = self.C
        return tuple(np.round(np.asarray(self.method.decide(self.method.unpack(S[:-1])), float), 6))

    def profile(self, rng, n=VOTERS):
        """Summed features of a random profile of n voters. The shares of the ballots are drawn in three
        ways, so that mixed, lopsided and nearly balanced profiles all occur: over all ballots, from
        even to uneven; over a few ballots only; and close to equal shares of every ballot, where
        everything is nearly tied and the boundaries of most rules meet."""
        k = len(self.F)
        w = 0.02 * rng.dirichlet(np.ones(k))                  # a little of everything: no exact ties
        mode = rng.integers(3)
        if mode == 0:
            w += 0.98 * rng.dirichlet(np.full(k, 10 ** rng.uniform(-2, 0.5)))
        elif mode == 1:
            some = rng.choice(k, size=min(k, int(rng.integers(2, 9))), replace=False)
            w[some] += 0.98 * rng.dirichlet(np.ones(len(some)))
        else:
            w = np.ones(k) + 10 ** rng.uniform(-3, 0) * rng.uniform(-1, 1, size=k)
            w /= w.sum()
        return n * w @ self.F

    # ----- what the outcome depends on -----
    def normal(self, rng, S0=None, S1=None):
        """One boundary between two outcomes and how it lies: the normal as its values on the basis
        (a row of self.normals), or None if no clean boundary was found. Between two random profiles,
        or between the two given."""
        if S0 is None:
            S0, S1 = self.profile(rng), self.profile(rng)
        a, b = self.outcome(S0), self.outcome(S1)
        if a == b:
            return None
        lo, hi = 0.0, 1.0
        for _ in range(60):                                   # two outcomes that meet on the way from S0 to S1
            mid = (lo + hi) / 2
            o = self.outcome(S0 + mid * (S1 - S0))
            if o == a:
                lo = mid
            else:
                hi = mid
        w = (S1 - S0) / np.linalg.norm(S1 - S0)
        star = S0 + (lo + hi) / 2 * (S1 - S0)
        base = star - DELTA * w
        b = self.outcome(star + DELTA * w)                    # (at the boundary itself the tally sees a tie)
        if self.outcome(base) != a or b == a:
            return None
        nv = np.zeros(self.D)
        for k in range(self.D):                               # follow each basis ballot until the boundary
            t = self._crossing(base, self.E[k], a, b)
            if t is None:
                return None
            nv[k] = 0.0 if t == 0 else DELTA / t
        for _ in range(6):                                    # the boundary is flat here: other directions agree
            v = rng.normal(size=self.D)
            slope = nv @ v
            if abs(slope) * REACH < 2 * DELTA:
                continue
            t = DELTA / slope
            d = v @ self.E
            if self.outcome(base + (1 - CHECK) * t * d) != a or self.outcome(base + (1 + CHECK) * t * d) != b:
                return None
        if not nv.any():
            return None
        return nv / np.linalg.norm(nv)

    def _crossing(self, base, d, a, b):
        """Where base + t d passes from outcome a to b within REACH: t, 0 for nowhere, None for unclear."""
        up, down = self.outcome(base + REACH * d), self.outcome(base - REACH * d)
        if up == a and down == a:
            return 0
        if (up == b) == (down == b) or {up, down} - {a, b}:
            return None
        sign = 1.0 if up == b else -1.0
        lo, hi = 0.0, REACH
        for _ in range(40):
            mid = (lo + hi) / 2
            if self.outcome(base + sign * mid * d) == a:
                lo = mid
            else:
                hi = mid
        return sign * (lo + hi) / 2

    def search(self, rng, points=300, patience=150, until=np.inf):
        """Collect normals until the rank reaches what the features allow, or `patience` boundaries in a
        row add nothing, or `points` boundaries were looked at, or the time `until` has come."""
        quiet = found = 0
        for _ in range(20 * points):
            if found >= points or quiet >= patience or self.dim() >= self.D or time.time() > until:
                break
            nv = self.normal(rng)
            if nv is None:
                continue
            found += 1
            quiet = 0 if self.add(nv) else quiet + 1
        return found

    def add(self, nv):
        """Take in a measured normal; True if it lies outside the span of those found before."""
        self.raw.append(nv)
        rest = nv - self.normals.T @ (self.normals @ nv)
        new = np.linalg.norm(rest) > 1e-3
        if new:
            self.normals = np.vstack([self.normals, rest / np.linalg.norm(rest)])
        return new

    def settle(self):
        """The basis of the normals from all those measured together (their errors average out)."""
        if self.raw:
            self.normals = np.linalg.svd(np.array(self.raw), full_matrices=False)[2][:self.dim()]

    def dim(self):
        return len(self.normals)

    def refute(self, rng, trials=2000, until=np.inf):
        """Test that nothing else matters: a change of ballots on which every normal found vanishes should
        never change the outcome. Such changes are drawn at random (so many voters casting some ballots
        instead of others) and applied to random profiles, small and large. Where the outcome differs,
        the boundary in between is measured: a normal outside those found is taken in ("new"); one
        inside them means the change was not quite blind to them (rounding), which is no finding.
        Returns "new", "clean" (nothing found), or "unclear" (differences whose boundary could not be
        measured, or out of time)."""
        if self.dim() >= self.D:
            return "clean"
        null = np.linalg.svd(self.normals, full_matrices=True)[2][self.dim():] if self.dim() else np.eye(self.D)
        unclear = 0
        for _ in range(trials):
            if time.time() > until:
                return "unclear"
            c = rng.normal(size=len(null)) @ null             # a change the normals found do not see
            plus, minus = np.clip(c, 0, None) @ self.E, np.clip(-c, 0, None) @ self.E
            S = self.profile(rng)
            for share in (0.01, 0.3, 3.0):
                t = share * VOTERS / np.abs(c).sum()
                a, b = self.outcome(S + t * plus), self.outcome(S + t * minus)
                if a == b or max(a) < 1 or max(b) < 1:        # (a tie is no difference to measure)
                    continue
                nv = self.normal(rng, S + t * plus, S + t * minus)
                if nv is None:
                    unclear += 1
                elif self.add(nv):
                    return "new"
        return "unclear" if unclear > 2 else "clean"

    def complete(self, rng, trials=2000, until=np.inf):
        """Test and take in what the test finds, in turn, until it finds nothing more (True); False if
        it ends unclear (the count found may then be too small)."""
        for _ in range(4 * self.D):
            self.settle()
            result = self.refute(rng, trials, until)
            if result != "new":
                return result == "clean"
        return False

    # ----- results -----
    def rank(self):
        """The rank of the count found, not counting the size of the electorate, and the most the
        features allow."""
        with_size = _row_space(np.vstack([self.normals, self.size / np.sqrt(self.D)]))
        return len(with_size) - 1, self.D - 1

    def points(self):
        """What each ballot adds to the count found: (k, rank found)."""
        return self.coords @ self.normals.T

    def vertices(self, limit=4000):
        """Indices (into self.X) of one ballot per vertex of the ballot polytope, the number of different
        count vectors, and whether the vertices are all of them (with more than `limit` count vectors
        only those found as a clear maximum in some random direction are given: a lower bound)."""
        if self._vertices is None or self._vertices[0] != (self.dim(), limit):
            uniq, first = _unique_rows(self.points())
            hull, complete = _hull(uniq, limit=limit)
            self._vertices = ((self.dim(), limit), first[hull], len(uniq), complete)
        return self._vertices[1:]

    def symmetric(self):
        """Is the ballot polytope centrally symmetric? None if not all its vertices are known."""
        vert, _, complete = self.vertices()
        if not complete:
            return None
        V = self.points()[vert]
        if V.shape[1] == 0:
            return True
        mirror = 2 * V.mean(axis=0) - V
        return all(np.abs(V - m).max(axis=1).min() < TOL for m in mirror)

    def equality(self):
        """True if every ballot with its reverse adds nothing to the count found, False if some pair
        does, None if the format has no reverses."""
        rev = REVERSE.get(self.method.ballot)
        if rev is None:
            return None
        index = {tuple(x): i for i, x in enumerate(self.X)}
        pairs = [(i, index[tuple(r)]) for i, r in enumerate(rev(self.X)) if tuple(r) in index]
        if not pairs:
            return None
        P = self.points()
        return all(np.abs(P[i] + P[j]).max() < TOL for i, j in pairs) if P.shape[1] else True


def _independent_rows(F):
    """Indices of rows of F that are linearly independent and span all rows (greedy, in order)."""
    out, Q = [], np.zeros((0, F.shape[1]))
    for i, f in enumerate(F):
        r = f - Q.T @ (Q @ f)
        if np.linalg.norm(r) > 1e-8 * max(1.0, np.linalg.norm(f)):
            Q = np.vstack([Q, r / np.linalg.norm(r)])
            out.append(i)
            if len(out) == F.shape[1]:
                break
    return out


def _row_space(M, tol=1e-3):
    """An orthonormal basis of the rows of M (rows are unit vectors measured to about 1e-5)."""
    if not len(M):
        return M
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[s > tol]


def _unique_rows(P, tol=TOL):
    """The different rows of P (rows closer than tol are the same) and the index of the first of each."""
    if P.shape[1] == 0:
        return P[:1], np.array([0])
    from scipy.spatial import cKDTree
    label = np.arange(len(P))
    for i, j in sorted(cKDTree(P).query_pairs(tol)):          # i < j: every row takes the lowest index near it
        label[j] = min(label[j], label[i])
    first = np.unique(label)
    return P[first], first


def _hull(P, tol=TOL, limit=np.inf):
    """Indices of the rows of P (all different) that are vertices of their convex hull, and True. The
    rows are measured, so a row counts as a vertex only if it lies more than tol outside the hull of
    the others. With more than `limit` rows: only the rows that are a clear maximum in one of many
    random directions (all of them vertices, but not all vertices), and False."""
    from scipy.optimize import linprog
    k = len(P)
    if k <= 2:
        return np.arange(k), True
    Z = P - P.mean(axis=0)
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    d = int((s > 1e-6 * max(s[0], 1e-12)).sum())
    Z = Z @ Vt[:d].T                                          # the points in their affine hull
    Z = Z / np.abs(Z).max()
    if k == d + 1:
        return np.arange(k), True
    rng = np.random.default_rng(0)
    sure = set()
    for _ in range(1 if k <= limit else 40):
        proj = Z @ rng.normal(size=(d, 500 + 50 * d))
        top2 = np.partition(proj, -2, axis=0)[-2:]
        clear = (top2[1] - top2[0]) > 1e-3                    # a clear maximum in some direction: a vertex
        sure |= set(np.unique(proj.argmax(axis=0)[clear]).tolist())
    if k > limit:
        return np.array(sorted(sure)), False

    def gap(i, others):
        """How far row i is from the mixtures of the rows `others` (largest coordinate), and the way out."""
        n = len(others)
        A = np.block([[Z[others].T, -np.ones((d, 1))], [-Z[others].T, -np.ones((d, 1))]])
        res = linprog(np.append(np.zeros(n), 1.0), A_ub=A, b_ub=np.concatenate([Z[i], -Z[i]]),
                      A_eq=np.append(np.ones(n), 0.0)[None], b_eq=[1.0], bounds=(0, None), method="highs")
        return res.fun, Z[i] - res.x[:n] @ Z[others]

    for i in range(k):
        while i not in sure:
            if not sure:
                sure.add(int(Z[:, 0].argmax()))
                continue
            far, way = gap(i, sorted(sure))
            if far <= tol:
                break                                         # a mixture of vertices found: no vertex
            val = Z @ way                                     # the rows furthest that way hold a vertex
            tied = np.flatnonzero(val >= val.max() - tol * np.abs(way).sum())
            if i in tied:                                     # row i is among them: a vertex unless the
                rest = [j for j in tied if j != i]            # others there mix to it
                if not rest or gap(i, rest)[0] > tol:
                    sure.add(i)
                    break
                tied = np.array(rest)
            best = int(tied[(Z[tied] @ (way + 1e-3 * np.abs(way).max() * rng.normal(size=d))).argmax()])
            if best in sure:                                  # (nothing new that way: decide by all rows)
                if gap(i, [j for j in range(k) if j != i])[0] > tol:
                    sure.add(i)
                break
            sure.add(best)
    return np.array(sorted(sure)), True


# ---------- one method ----------
def lottery(count, rng, tries=20):
    """True if the method's outcome is a lottery at ordinary profiles (no boundaries to look for)."""
    outs = [count.outcome(count.profile(rng)) for _ in range(tries)]
    return sum(max(o) < 1 - 1e-6 for o in outs) > tries / 2


def analyse(name, C, seed=0, points=400, trials=2000, seconds=120):
    """The row of one method for C candidates: a dict, or a string saying why there is none."""
    M = METHODS[name]
    if M.ballot not in ENUMERATE:
        return f"ballot format {M.ballot!r} has no entry in ENUMERATE"
    if M.full is not None or not M.feats or M.decide is None:
        return "counted from the ballots directly, no features"
    if M.delegate:
        return "delegation needs the candidates' own ballots"
    with M.applied():
        try:
            count = Count(M, C)
            rng = np.random.default_rng(seed)
            if lottery(count, rng):
                return "the outcome is a lottery"
            until = time.time() + seconds
            found = count.search(rng, points, until=until)
            exact = count.complete(rng, trials, until)
            count.settle()
            if not count.dim():
                return "no clean boundary between outcomes found (a count in whole steps, or too slow)"
        except Exception as e:                                # a count this tool cannot drive
            return f"not analysed ({type(e).__name__}: {e})"
        lo, hi = count.rank()
        vert, counts, complete = count.vertices()
        return dict(method=name, cands=C, ballot=M.ballot, ballots=len(count.X), counts=counts,
                    vertices=len(vert), vertices_complete=complete, rank=lo, rank_features=hi, exact=exact, symmetric=count.symmetric(),
                    equality=count.equality(), boundaries=found, vertex_ballots=count.X[vert])


def _yes(x, exact):
    return "-" if x is None else ("yes" if x else "no") + ("" if exact else "?")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--methods", nargs="+", default=None, help="methods, groups or sets (default: the standard set)")
    ap.add_argument("--exclude", nargs="+", default=None)
    ap.add_argument("--cands", type=int, nargs="+", default=[3])
    ap.add_argument("--max-score", type=int, default=None, help="the score range 0..max (run.py --max-score)")
    ap.add_argument("--points", type=int, default=400, help="boundaries looked at per method, at most")
    ap.add_argument("--trials", type=int, default=2000, help="random changes tried per test that nothing else matters")
    ap.add_argument("--seconds", type=float, default=120, help="time for the search per method and number of candidates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="list one ballot per vertex")
    a = ap.parse_args()
    if a.max_score is not None:
        methods.set_max_score(a.max_score)
    names = methods.run_selection(a.methods, a.exclude)
    width = max(len(n) for n in names)
    for C in a.cands:
        print(f"\n{C} candidates" + (f", scores 0..{methods.MAXS}" if any(methods.follows_score_range(n) for n in names) else ""))
        print(f"{'method':{width}}  {'ballot':9} {'ballots':>7} {'counts':>7} {'vertices':>8} {'rank':>7}  {'sym':4} {'EC':4}")
        left = []
        for n in names:
            t0 = time.time()
            r = analyse(n, C, a.seed, a.points, a.trials, a.seconds)
            if isinstance(r, str):
                left.append((n, r))
                continue
            ex = r["exact"]
            rank = str(r["rank"]) + ("" if ex else "+")
            print(f"{n:{width}}  {r['ballot']:9} {r['ballots']:7d} {r['counts']:7d} "
                  f"{str(r['vertices']) + ('' if r['vertices_complete'] else '+') + ('' if ex else '?'):>8} {rank:>7}  {_yes(r['symmetric'], ex):4} "
                  f"{_yes(r['equality'], ex):4}" + (f"   ({time.time() - t0:.0f} s)" if time.time() - t0 > 20 else ""),
                  flush=True)
            if a.show:
                for b in r["vertex_ballots"]:
                    print(" " * width + "    " + " ".join(f"{int(x)}" for x in b))
        for n, why in left:
            print(f"{n:{width}}  - {why}")


if __name__ == "__main__":
    main()
