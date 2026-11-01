"""Composing methods from parts (docs/pipelines.md; phases 1 and 2).

A composition is a chain of steps over one state: the field (the candidates still in) and the readings
known so far. Sets narrow the field and pass the readings through; transforms replace a reading; a
reading or a final step takes the latest version of what it needs, restricted to the current field.
Readings that depend on the field (first preferences, rank points, scores stretched over the field) are
never taken from an earlier field: they are computed again on it, from totals over every subset of
candidates where the lab has them, else from the ballots restricted to the field (the exact fallback,
for the election count; polls then use the lab's default poll reading).

The same text works in Python (an add-on) and on the command line (read with Python's own parser,
only the parts below accepted):

    smith >> "Score"                                              Smith//Score
    buck(points) >> normalize >> "Score"                          ESR
    best(share(score) - worst(wv) / n)                            MMWV+Score
    ballots("rank") >> best(-worst(margins))                      Minimax
    top(2, score) >> runoff(score)                                Score runoff
    (top(1, score) | challenger(top(1, score), margins)) >> runoff(score)     SCR (add-on)

Parts:
    readings   score (totals; on two-round ballots the round-1 marks), margins, wv (winning votes), po
               (pairwise opposition: votes for the row over the column), soft(0.2 or 0.4) (soft margins),
               n (voters), levels (voters per score level, top first), points (the same times the
               level's score), approvals(k) (voters scoring at least k); share(values) = values /
               (n x top score). On two-round ballots the pairwise readings come from the round-1 marks
    pairs -> values   worst(pairs): each candidate's worst defeat (largest value a rival has against
               them, 0 if none)
    sets       smith, best(values) (all tied for highest), top(k, values) (ties at the k-th place
               branch), buck(levels or points), challenger(set, pairs) (the strongest beaters of the
               set; ties branch; none: an empty set); a | b, a & b. The same part twice in one set
               expression takes the same branch
    transforms normalize (score ballots stretched over the current field, not rounded)
    final      "Name": an existing method on the current field and readings; runoff, runoff(values):
               a separate runoff between the finalists by the runoff rankings of two-round ballots
               (a tie split, or to the higher value); a set at the end splits the win equally
    first      ballots("type"): the ballot type (default: that of the quoted method; score runoff
               ballots with runoff; else score)
    chaining   >> ("then"); arithmetic + - * / and unary minus on values and pairs

composed(expr) turns a composition into an ordinary Method (methods.py): ballot type, features,
decision, its poll view (model.md, What a poll shows: normalize is read as nothing in a poll, a quoted
method by its own poll view) and the settings of its final step.
"""
import ast
import hashlib
import itertools

import numpy as np

import methods as M
from ballots import BALLOTS

# ---------- how each feature narrows to a smaller field (docs/pipelines.md, the feature table) ----------
CAND0 = {"sum", "app", "run_sum", "maxs", "pos", "upper", "ge1", "ge2", "eq0", "grades"}   # axis 0 is the candidate
CAND1 = {"levels"}                                                                     # axis 1 is the candidate
PAIR = {"gt", "runoff_gt", "round1_gt", "excess", "soft20", "soft40", "levelled", "diffhist"}   # axes 0, 1
AMONG = {"firsts", "firsts_eq", "firsts_ex", "lasts", "keepnorm", "nscore_sub"}       # per subset, among it
SCALAR = {"n"}
# after normalize: features that keep their meaning (order among candidates, or already stretched per
# subset); "sum" becomes the normalized scores
AFTER_NORMALIZE = {"n", "sum", "gt", "firsts", "firsts_eq", "firsts_ex", "lasts", "nscore_sub"}
TWO_ROUND = {"fav_run", "app_run", "score_run"}
UNRANKED = {"rank", "weak", "trunc", "trunc3"}       # rankings where 0 can mean "not ranked"
ORDINAL = UNRANKED | {"strict"}
AS_CAST = {"fav", "veto"}                           # restricted to a field: as cast (a vote for a candidate
                                                    # who is out counts for nobody)


def rated(ballot):
    """Ballots whose values are per-candidate ratings (score, grades, approval; add-on types with a
    score range and no marks): their totals do not depend on the field."""
    b = BALLOTS[ballot]
    return ballot in ("score", "grade3", "approval") or (b.score_range and b.marks is None)


def restrictable(f, ballot):
    """Can feature f be read on a narrower field from its totals?"""
    if f == "sum":
        return rated(ballot)
    return f in CAND0 | CAND1 | PAIR | AMONG | SCALAR


def recastable(ballot):
    """Can ballots of this type be restricted to a narrower field (the exact fallback)?"""
    return rated(ballot) or ballot in ORDINAL | AS_CAST | TWO_ROUND


def _dense(X, keep_zero):
    """Per row: dense levels of the values (ties share a level), lowest 0; keep_zero: a 0 stays 0
    (not ranked) and the others are numbered from 1."""
    out = np.zeros_like(X, dtype=float)
    for i, row in enumerate(X):
        vals = np.unique(row[row > 0] if keep_zero else row)
        lev = np.searchsorted(vals, row) + (1 if keep_zero else 0)
        out[i] = np.where((row > 0) | (not keep_zero), lev, 0)
    return out


def restrict_ballots(B, field, ballot):
    """The ballots on a narrower field: scores and marks as cast; rankings as the induced ranking among
    the field (renumbered, ties kept, an unranked candidate stays unranked); two-round ballots both
    parts so (the runoff ranking renumbered)."""
    F = list(field)
    X = B[:, F]
    if rated(ballot) or ballot in AS_CAST:
        return X
    if ballot in ORDINAL:
        return _dense(X, ballot in UNRANKED)
    if ballot in TWO_ROUND:
        C, k = B.shape[1], len(F)
        marks, run = np.floor_divide(B, C)[:, F], np.mod(B, C)[:, F]
        return marks * k + _dense(run, False)
    raise CompositionError(f"ballots of type {ballot} cannot be restricted to a narrower field")


def _sub_index(field, C):
    """For the subsets of `field` (bits over its positions), the index of the same subset of all C."""
    k = len(field)
    idx = np.zeros(2 ** k, dtype=int)
    for i, c in enumerate(field):
        idx[(np.arange(2 ** k) >> i) & 1 == 1] += 1 << c
    return idx


def restrict(s, f, field, C, norm=None):
    """Feature f's sum on the field (a list of candidates). norm: the field normalize stretched over
    (then "sum" is the normalized scores)."""
    F = list(field)
    if f == "sum" and norm is not None:
        return s["nscore_sub"][int(sum(1 << c for c in norm))][F]
    x = s[f]
    if len(F) == C:
        return x
    if f in SCALAR:
        return x
    if f in CAND0:
        return x[F]
    if f in CAND1:
        return x[:, F]
    if f in PAIR:
        return x[np.ix_(F, F)]
    if f in AMONG:
        sub = x[_sub_index(F, C)]
        return sub[:, F] if sub.ndim == 2 else sub
    raise KeyError(f)


# ---------- the parts ----------
class CompositionError(ValueError):
    pass


def _part(x):
    """A Python value as a part: a string is a quoted method, a number a constant."""
    if isinstance(x, Part):
        return x
    if isinstance(x, str):
        return Quoted(x)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return Const(x)
    raise CompositionError(f"not a part of a composition: {x!r}")


class Part:
    kind = None              # values, pairs, scalar, levels, set, transform, final, ballots, chain
    prec = 9                 # for the canonical text, as Python binds: 9 atom, 6 unary, 5 * /, 4 + -, 3 >>, 2 &, 1 |

    # chaining and set operations
    def __rshift__(self, other):
        return Chain([self, _part(other)])

    def __rrshift__(self, other):
        return Chain([_part(other), self])

    def __or__(self, other):
        return SetOp("|", self, _part(other))

    def __and__(self, other):
        return SetOp("&", self, _part(other))

    # arithmetic on values and pairs
    def __add__(self, o): return BinOp("+", self, _part(o))
    def __radd__(self, o): return BinOp("+", _part(o), self)
    def __sub__(self, o): return BinOp("-", self, _part(o))
    def __rsub__(self, o): return BinOp("-", _part(o), self)
    def __mul__(self, o): return BinOp("*", self, _part(o))
    def __rmul__(self, o): return BinOp("*", _part(o), self)
    def __truediv__(self, o): return BinOp("/", self, _part(o))
    def __rtruediv__(self, o): return BinOp("/", _part(o), self)
    def __neg__(self): return Neg(self)

    def __repr__(self):
        return self.text()

    def text(self):
        raise NotImplementedError

    def check(self, ctx):
        """Static check on the context (ballot, narrowed, normalized, poll); returns the features
        needed from totals. A feature that depends on the field after a set is added to ctx["fallback"]."""
        return set()


def _wrap(p, prec):
    t = p.text()
    return f"({t})" if p.prec < prec else t


def _need(f, ctx):
    """Feature f in this context: from totals if it can be read on the current field, else marked for
    the exact fallback (ballots restricted to the field)."""
    if ctx["narrowed"] and not restrictable(f, ctx["ballot"]):
        if not recastable(ctx["ballot"]):
            raise CompositionError(f"{f} depends on the field, and {ctx['ballot']} ballots cannot be "
                                   "restricted to a narrower field")
        ctx["fallback"].add(f)
        return set()
    return {f}


class Reading(Part):
    """A reading of the ballots: feat(ballot) names the feature, fn(x, st) turns its sum on the field
    into the reading."""
    def __init__(self, name, kind, feat, fn, scored=False):
        self.name, self.kind, self.feat, self.fn, self.scored = name, kind, feat, fn, scored

    def text(self):
        return self.name

    def check(self, ctx):
        b = ctx["ballot"]
        if self.scored and not (rated(b) and BALLOTS[b].score_range):
            raise CompositionError(f"{self.name} needs score ballots, not {b}")
        if self.scored and ctx["normalized"] and not ctx["poll"]:
            raise CompositionError(f"{self.name} is not defined after normalize (stretched scores are not rounded)")
        f = self.feat(b)
        if f == "sum" and ctx["normalized"] and not ctx["poll"]:
            return {"nscore_sub"}
        return _need(f, ctx)

    def value(self, st):
        return self.fn(st.get(self.feat(st.ballot)), st)


def _score_feat(b):
    if b == "score_run":
        return "run_sum"
    if b in ("fav_run", "app_run"):
        return "app"
    return "sum"


def _pair_feat(b):
    return "round1_gt" if b in TWO_ROUND else "gt"


def _f(x):
    return np.asarray(x, float)


score = Reading("score", "values", _score_feat, lambda x, st: _f(x))
margins = Reading("margins", "pairs", _pair_feat, lambda x, st: _f(x) - _f(x).T)
wv = Reading("wv", "pairs", _pair_feat, lambda x, st: np.where(_f(x) > _f(x).T, _f(x), 0.0))
po = Reading("po", "pairs", _pair_feat, lambda x, st: _f(x) * (1 - np.eye(len(x))))
n = Reading("n", "scalar", lambda b: "n", lambda x, st: float(x))
levels = Reading("levels", "levels", lambda b: "levels", lambda x, st: _f(x), scored=True)
points = Reading("points", "levels", lambda b: "levels",
                 lambda x, st: (len(x) - np.arange(len(x)))[:, None] * _f(x), scored=True)


def soft(frac):
    """Soft margins (as Soft Minimax): per voter the score difference clipped to +-frac of the range."""
    if frac not in (0.2, 0.4):
        raise CompositionError(f"soft margins exist for 0.2 and 0.4, not {frac!r}")
    k = f"soft{int(round(frac * 100))}"
    r = Reading(f"soft({frac})", "pairs", lambda b: k, lambda x, st: _f(x), scored=True)
    return r


def approvals(level):
    """Voters scoring each candidate at least `level` (1 .. top score)."""
    if not isinstance(level, int) or level < 1:
        raise CompositionError(f"approvals needs a score level of at least 1, not {level!r}")

    def fn(x, st):
        if level > len(x):
            raise CompositionError(f"approvals({level}) is above the top score {len(x)}")
        return _f(x)[:len(x) - level + 1].sum(axis=0)       # levels: top score first
    return Reading(f"approvals({level})", "values", lambda b: "levels", fn, scored=True)


class Const(Part):
    kind = "scalar"

    def __init__(self, x):
        self.x = float(x)

    def text(self):
        return str(int(self.x)) if self.x == int(self.x) else repr(self.x)

    def value(self, st):
        return self.x


class Func(Part):
    """A part with arguments: share, worst."""
    def __init__(self, name, arg, kind, need, fn, extra=()):
        self.name, self.arg, self.kind, self.need, self.fn, self.extra = name, arg, kind, need, fn, extra
        if arg.kind not in need:
            raise CompositionError(f"{name} needs {' or '.join(need)}, {arg.text()} is {_kind_name(arg)}")

    def text(self):
        return f"{self.name}({self.arg.text()})"

    def check(self, ctx):
        return self.arg.check(ctx) | set().union(*[_need(f, ctx) for f in self.extra])

    def value(self, st):
        return self.fn(self.arg.value(st), st)


def _kind_name(p):
    return {"values": "values", "pairs": "pairs", "scalar": "a number", "levels": "a reading per level",
            "set": "a set", "transform": "a transform", "final": "a method", "ballots": "a ballot type",
            "chain": "a chain of steps"}.get(p.kind, p.kind)


def share(x):
    """values / (n x top score): a share of the maximum."""
    return Func("share", _part(x), "values", ("values",), lambda v, st: v / (st.get("n") * M.MAXS), extra=("n",))


def worst(x):
    """Each candidate's worst defeat: the largest value a rival has against them (0 if none)."""
    return Func("worst", _part(x), "values", ("pairs",), lambda p, st: np.clip(p, 0, None).max(axis=0))


class BinOp(Part):
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b
        self.prec = 4 if op in "+-" else 5
        kinds = {a.kind, b.kind} - {"scalar"}
        if not kinds <= {"values"} and not kinds <= {"pairs"}:
            raise CompositionError(f"{op} combines values with values or pairs with pairs (and numbers): "
                                   f"{a.text()} is {_kind_name(a)}, {b.text()} is {_kind_name(b)}")
        self.kind = kinds.pop() if kinds else "scalar"

    def text(self):
        right = self.prec + (1 if self.op in "-/" else 0)          # a - (b - c), a / (b * c)
        return f"{_wrap(self.a, self.prec)} {self.op} {_wrap(self.b, right)}"

    def check(self, ctx):
        return self.a.check(ctx) | self.b.check(ctx)

    def value(self, st):
        x, y = self.a.value(st), self.b.value(st)
        return {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}[self.op](x, y)


class Neg(Part):
    prec = 6

    def __init__(self, a):
        if a.kind not in ("values", "pairs", "scalar"):
            raise CompositionError(f"- needs values, pairs or a number, {a.text()} is {_kind_name(a)}")
        self.a, self.kind = a, a.kind

    def text(self):
        return f"-{_wrap(self.a, self.prec)}"

    def check(self, ctx):
        return self.a.check(ctx)

    def value(self, st):
        return -self.a.value(st)


# sets: branches(st, memo) -> list of (weight, field, memo); the field a list of candidates, memo the
# branch taken by each branching part so far (the same part twice in one set expression takes the same
# branch)
class Set(Part):
    kind = "set"

    def branches(self, st, memo):
        return [(1.0, self.field(st), memo)]


class Smith(Set):
    def text(self):
        return "smith"

    def check(self, ctx):
        return margins.check(ctx)

    def field(self, st):
        M_ = margins.value(st)
        return [st.field[i] for i in M.smith_set(M_)]


smith = Smith()


class Best(Set):
    def __init__(self, arg):
        if arg.kind != "values":
            raise CompositionError(f"best needs values, {arg.text()} is {_kind_name(arg)}")
        self.arg = arg

    def text(self):
        return f"best({self.arg.text()})"

    def check(self, ctx):
        return self.arg.check(ctx)

    def field(self, st):
        v = np.asarray(self.arg.value(st), float)
        return [st.field[i] for i in np.flatnonzero(v >= v.max() - M.EPS)]


def best(x):
    """All candidates tied for the highest value."""
    return Best(_part(x))


class Branching(Set):
    """A set whose ties branch: the memo keeps the branch for the same part elsewhere in the expression."""
    def branches(self, st, memo):
        key = self.text()
        if key in memo:
            return [(1.0, memo[key], memo)]
        return [(w, f, {**memo, key: f}) for w, f in self.options(st, memo)]


class Top(Branching):
    def __init__(self, k, arg):
        if not isinstance(k, int) or k < 1:
            raise CompositionError(f"top needs a whole number of candidates first, not {k!r}")
        if arg.kind != "values":
            raise CompositionError(f"top needs values, {arg.text()} is {_kind_name(arg)}")
        self.k, self.arg = k, arg

    def text(self):
        return f"top({self.k}, {self.arg.text()})"

    def check(self, ctx):
        return self.arg.check(ctx)

    def options(self, st, memo):
        F = st.field
        if len(F) <= self.k:
            return [(1.0, list(F))]
        v = np.asarray(self.arg.value(st), float)
        cut = v.max() if self.k == 1 else np.sort(v)[::-1][self.k - 1]
        above = np.flatnonzero(v > cut + M.EPS).tolist()
        tied = np.flatnonzero(np.abs(v - cut) <= M.EPS).tolist()
        if len(tied) == self.k - len(above):
            return [(1.0, [F[i] for i in sorted(above + tied)])]
        ways = list(itertools.combinations(tied, self.k - len(above)))
        return [(1.0 / len(ways), [F[i] for i in sorted(above + list(w))]) for w in ways]


def top(k, x):
    """The k highest by a measure; a tie at the k-th place branches with equal weights."""
    return Top(k, _part(x))


class Challenger(Set):
    def __init__(self, of, arg):
        if of.kind != "set":
            raise CompositionError(f"challenger needs a set first, {of.text()} is {_kind_name(of)}")
        if arg.kind != "pairs":
            raise CompositionError(f"challenger needs pairs second, {arg.text()} is {_kind_name(arg)}")
        self.of, self.arg = of, arg

    def text(self):
        return f"challenger({self.of.text()}, {self.arg.text()})"

    def check(self, ctx):
        return self.of.check(ctx) | self.arg.check(ctx)

    def branches(self, st, memo):
        key = self.text()
        if key in memo:
            return [(1.0, memo[key], memo)]
        out = []
        p = np.asarray(self.arg.value(st), float)
        pos = {c: i for i, c in enumerate(st.field)}
        for w, S, m in self.of.branches(st, memo):
            inside = [pos[c] for c in S]
            outside = [i for i in range(len(st.field)) if i not in inside]
            beat = p[outside][:, inside].max(axis=1) if outside and inside else np.zeros(0)
            if not len(beat) or beat.max() <= M.EPS:
                out.append((w, [], {**m, key: []}))            # nobody beats the set: no challenger
                continue
            ch = [st.field[outside[i]] for i in np.flatnonzero(beat >= beat.max() - M.EPS)]
            out += [(w / len(ch), [c], {**m, key: [c]}) for c in ch]
        return out


def challenger(of, x):
    """The strongest challengers of a set: the candidates outside it with the largest value (of the
    pairs) against one of its members; ties branch; nobody with a positive value: an empty set."""
    return Challenger(_part(of), _part(x))


class Buck(Set):
    def __init__(self, arg):
        if arg.kind != "levels":
            raise CompositionError(f"buck needs levels or points, {arg.text()} is {_kind_name(arg)}")
        self.arg = arg

    def text(self):
        return f"buck({self.arg.text()})"

    def check(self, ctx):
        return self.arg.check(ctx)

    def field(self, st):
        A = np.cumsum(self.arg.value(st), axis=0)           # per depth: totals from the top levels
        sup = sorted({int(c) for row in A for c in np.flatnonzero(row >= row.max() - M.EPS)})
        return [st.field[i] for i in sup]


def buck(x):
    """For every depth (the top k score levels), the candidates ahead by the reading from those levels
    (all tied for most), joined over the depths; at the full depth by points that is the score total."""
    return Buck(_part(x))


class SetOp(Set):
    def __init__(self, op, a, b):
        self.prec = 1 if op == "|" else 2
        for p in (a, b):
            if p.kind != "set":
                raise CompositionError(f"{op} joins sets, {p.text()} is {_kind_name(p)}")
        self.op, self.a, self.b = op, a, b

    def text(self):
        return f"{_wrap(self.a, self.prec)} {self.op} {_wrap(self.b, self.prec + 1)}"

    def check(self, ctx):
        return self.a.check(ctx) | self.b.check(ctx)

    def branches(self, st, memo):
        out = []
        for wa, fa, ma in self.a.branches(st, memo):
            for wb, fb, mb in self.b.branches(st, ma):
                f = sorted(set(fa) | set(fb)) if self.op == "|" else sorted(set(fa) & set(fb))
                out.append((wa * wb, f, mb))
        return out


class Normalize(Part):
    kind = "transform"

    def text(self):
        return "normalize"

    def check(self, ctx):
        if not rated(ctx["ballot"]) or not BALLOTS[ctx["ballot"]].score_range:
            raise CompositionError(f"normalize needs score ballots, not {ctx['ballot']}")
        return set()


normalize = Normalize()


class Ballots(Part):
    kind = "ballots"

    def __init__(self, name):
        if name not in BALLOTS:
            raise CompositionError(f"unknown ballot type {name!r}; the types: {', '.join(BALLOTS)}")
        self.name = name

    def text(self):
        return f'ballots("{self.name}")'


def ballots(name):
    """The ballot type of the composition (first step)."""
    return Ballots(name)


class Quoted(Part):
    kind = "final"

    def __init__(self, name):
        hit = name if name in M.METHODS else next((k for k in M.METHODS if k.lower() == name.lower()), None)
        if hit is None:
            hit = M.current_name(name) if name in M.RENAMED else None
        if hit is None or hit not in M.METHODS:
            raise CompositionError(f"unknown method {name!r}")
        self.name = hit
        self.method = M.METHODS[hit]
        if self.method.full is not None:
            raise CompositionError(f"{hit} has no feature sums (counted from the ballots): not usable as a step yet")

    def text(self):
        return '"' + self.name.replace('"', '\\"') + '"'

    def view(self, poll):
        return self.method.poll_view() if poll else self.method

    def check(self, ctx):
        m = self.view(ctx["poll"])
        if self.method.ballot != ctx["ballot"]:
            raise CompositionError(f"{self.name} uses {self.method.ballot} ballots, the composition {ctx['ballot']}")
        if ctx["normalized"] and not ctx["poll"]:
            bad = [f for f in m.feats if f not in AFTER_NORMALIZE]
            if bad:
                raise CompositionError(f"{self.name} reads {', '.join(bad)}, which normalize does not provide")
            return set().union(*[_need("nscore_sub" if f == "sum" else f, ctx) for f in m.feats])
        return set().union(*[_need(f, ctx) for f in m.feats])

    def lottery(self, st):
        m = self.view(st.poll)
        return np.asarray(m.decide({f: st.get(f) for f in m.feats}), float)


class Runoff(Part):
    kind = "final"

    def __init__(self, tiebreak=None):
        if tiebreak is not None and tiebreak.kind != "values":
            raise CompositionError(f"runoff breaks a tie by values, {tiebreak.text()} is {_kind_name(tiebreak)}")
        self.tiebreak = tiebreak

    def text(self):
        return "runoff" if self.tiebreak is None else f"runoff({self.tiebreak.text()})"

    def check(self, ctx):
        if ctx["ballot"] not in TWO_ROUND:
            raise CompositionError(f"runoff needs two-round ballots (score_run, app_run, fav_run), not {ctx['ballot']}")
        return _need("runoff_gt", ctx) | (self.tiebreak.check(ctx) if self.tiebreak is not None else set())

    def lottery(self, st):
        k = len(st.field)
        P = np.zeros(k)
        if k == 1:
            P[0] = 1.0
            return P
        g = _f(st.get("runoff_gt"))
        Mr = g - g.T
        tb = None if self.tiebreak is None else np.asarray(self.tiebreak.value(st), float)
        if k == 2:
            M.runoff(P, Mr, 0, 1, 1.0, tiebreak=tb)
            return P
        wins = (Mr > 0).sum(axis=1)                         # more than two finalists (a tie before):
        return M.uniform_max(wins.astype(float))            # most runoff wins, ties split


class _RunoffName(Runoff):
    """`runoff` as a plain name, and runoff(values) as a call."""
    def __call__(self, tiebreak):
        return Runoff(_part(tiebreak))


runoff = _RunoffName()


class Chain(Part):
    prec = 3

    def __init__(self, steps):
        flat = []
        for p in steps:
            flat += p.steps if isinstance(p, Chain) else [p]
        for i, p in enumerate(flat[:-1]):
            if p.kind not in ("set", "transform", "ballots"):
                raise CompositionError(f"{p.text()} is {_kind_name(p)}: only sets and transforms can come "
                                       "before >>")
            if p.kind == "ballots" and i > 0:
                raise CompositionError("ballots(...) comes first")
        self.steps = flat
        self.kind = "chain"

    def text(self):
        return " >> ".join(_wrap(p, 4) for p in self.steps)


# ---------- the state while counting ----------
class State:
    __slots__ = ("s", "field", "norm", "C", "poll", "ballot", "B", "cache", "exact")

    def __init__(self, s, field, norm, C, poll, ballot, B=None, exact=False):
        self.s, self.field, self.norm, self.C, self.poll = s, field, norm, C, poll
        self.ballot, self.B, self.cache, self.exact = ballot, B, {}, exact

    def get(self, f):
        """Feature f's sum on the field: from the totals where it can be read there, else from the
        ballots restricted to the field (the exact fallback)."""
        if f in self.cache:
            return self.cache[f]
        norm = None if self.poll else self.norm
        have = f in self.s or (f == "sum" and norm is not None and "nscore_sub" in self.s)
        full = len(self.field) == self.C and (f != "sum" or norm is None)
        if self.B is None or (have and (full or (not self.exact and
                                                  (restrictable(f, self.ballot) or (f == "sum" and norm is not None))))):
            x = restrict(self.s, f, self.field, self.C, norm)
        else:
            if f == "sum" and norm is not None:          # the ballots stretched over the field normalize saw
                Bn = normalize_ballots(self.B[:, list(norm)])
                pos = {c: i for i, c in enumerate(norm)}
                x = Bn[:, [pos[c] for c in self.field]].sum(axis=0)
            else:
                Bf = self.B if len(self.field) == self.C else restrict_ballots(self.B, self.field, self.ballot)
                k = len(self.field)
                x = M.FEATURES[f][0](Bf).reshape(len(Bf), -1).astype(float).sum(axis=0).reshape(M.FEATURES[f][1](k))
                x = x if x.shape else float(x)
        self.cache[f] = x
        return x


def normalize_ballots(X):
    """Score ballots stretched over their columns: lowest 0, highest the top score, not rounded; a
    ballot rating them all alike counts 0 for all (as the feature nscore_sub)."""
    lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    return np.where(hi > lo, (X - lo) * M.MAXS / np.where(hi > lo, hi - lo, 1), 0.0)


def _steps(expr):
    return expr.steps if isinstance(expr, Chain) else [expr]


def _final(expr):
    p = _steps(expr)[-1]
    if p.kind not in ("final", "set"):
        raise CompositionError(f"a composition ends in a method, a set or best(...); {p.text()} is {_kind_name(p)}")
    return p


def check(expr, ballot, poll=False):
    """Static check of the whole composition: (features needed from totals, features that need the
    exact fallback)."""
    ctx = {"ballot": ballot, "narrowed": False, "normalized": False, "poll": poll, "fallback": set()}
    feats = set()
    _final(expr)
    for p in _steps(expr):
        if p.kind == "ballots":
            continue
        feats |= p.check(ctx)
        if p.kind == "set":
            ctx["narrowed"] = True
        elif p.kind == "transform":
            ctx["normalized"] = True
    return feats, ctx["fallback"]


def evaluate(expr, s, C, ballot, poll=False, B=None, exact=False):
    """Winner probabilities (C,) of the composition from the summed features s (and the ballots B for
    the exact fallback; exact: every reading on a narrower field from the ballots, for the tests)."""
    P = np.zeros(C)
    steps = [p for p in _steps(expr) if p.kind != "ballots"]

    def go(i, st, w):
        p = steps[i]
        if p.kind == "transform":
            go(i + 1, State(st.s, st.field, tuple(st.field), C, poll, ballot, B, exact), w)
        elif p.kind == "set":
            for wb, F, _ in p.branches(st, {}):
                if not F:
                    F = list(st.field)                       # an empty set narrows nothing
                if i == len(steps) - 1:                  # a set at the end: the win split equally
                    P[F] += w * wb / len(F)
                else:
                    go(i + 1, State(st.s, F, st.norm, C, poll, ballot, B, exact), w * wb)
        else:                                           # a quoted method or a runoff
            P[st.field] += w * p.lottery(st)

    go(0, State(s, list(range(C)), None, C, poll, ballot, B, exact), 1.0)
    return P


def _walk(expr):
    out = []

    def walk(p):
        out.append(p)
        for a in ("a", "b", "arg", "of", "tiebreak"):
            if isinstance(getattr(p, a, None), Part):
                walk(getattr(p, a))
        for q in getattr(p, "steps", ()):
            walk(q)
    walk(expr)
    return out


def ballot_of(expr):
    """The composition's ballot type: ballots(...) if given, else that of its quoted methods, else score
    runoff ballots if it has a runoff, else score ballots."""
    steps = _steps(expr)
    if steps[0].kind == "ballots":
        return steps[0].name
    parts = _walk(expr)
    quoted = [p for p in parts if isinstance(p, Quoted)]
    if quoted:
        return quoted[0].method.ballot
    return "score_run" if any(isinstance(p, Runoff) for p in parts) else "score"


def _C_of(s):
    for f, x in s.items():
        if f in CAND0 or f in PAIR:
            return x.shape[0]
        if f in CAND1 or f in AMONG:
            return x.shape[-1]
    raise CompositionError("no feature with candidates")


def composed(expr, ballot=None):
    """The composition as a Method (methods.py)."""
    expr = _part(expr)
    ballot = ballot or ballot_of(expr)
    feats, fallback = check(expr, ballot)
    feats = tuple(sorted(feats))
    pfeats, pfallback = check(expr, ballot, poll=True)
    parts = _walk(expr)
    finals = [p for p in _steps(expr) if isinstance(p, Quoted)]
    settings = finals[-1].method.settings if finals else None
    poll = None
    if pfallback:
        poll = None                     # the lab's default poll reading (score totals or first preferences)
    elif fallback or any(isinstance(p, Normalize) for p in parts) or \
            any(p.method.poll_view() is not p.method for p in parts if isinstance(p, Quoted)):
        poll = (tuple(sorted(pfeats)), lambda s: evaluate(expr, s, _C_of(s), ballot, poll=True))
    if fallback:                        # counted from the ballots (features of the field as needed)
        inner = M.Method(ballot, None, feats) if feats else None

        def full(B):
            C = B.shape[1]
            s = {}
            if inner is not None:
                inner._C = C
                s = inner.unpack(inner.features(B).sum(axis=0))
            return evaluate(expr, s, C, ballot, B=B)
        m = M.Method(ballot, full=full, settings=settings, poll=poll)
    else:
        m = M.Method(ballot, lambda s: evaluate(expr, s, _C_of(s), ballot), feats, settings=settings, poll=poll)
    m.__dict__["composition"] = expr
    return m


# ---------- reading the text ----------
NAMES = {"score": score, "margins": margins, "wv": wv, "po": po, "n": n, "levels": levels, "points": points,
         "smith": smith, "normalize": normalize, "runoff": runoff}
CALLS = {"share": share, "worst": worst, "best": best, "top": top, "buck": buck, "ballots": ballots,
         "challenger": challenger, "soft": soft, "approvals": approvals, "runoff": runoff}


def parse(text):
    """A composition from its text (only the known parts; no other code runs)."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise CompositionError(f"cannot read {text!r}: {e.msg}") from None

    def conv(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value                           # a method name, or a ballot type in ballots(...)
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
        elif isinstance(node, ast.Name):
            if node.id in NAMES:
                return NAMES[node.id]
            raise CompositionError(f"unknown part {node.id!r}; parts: {', '.join(sorted(set(NAMES) | set(CALLS)))}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id not in CALLS:
                raise CompositionError(f"unknown part {node.func.id!r}(...); parts with arguments: {', '.join(CALLS)}")
            return CALLS[node.func.id](*[conv(a) for a in node.args])
        elif isinstance(node, ast.BinOp):
            ops = {ast.RShift: "__rshift__", ast.BitOr: "__or__", ast.BitAnd: "__and__", ast.Add: "__add__",
                   ast.Sub: "__sub__", ast.Mult: "__mul__", ast.Div: "__truediv__"}
            if type(node.op) in ops:
                a, b = conv(node.left), conv(node.right)
                return getattr(_part(a), ops[type(node.op)])(b)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            x = conv(node.operand)
            return -x
        raise CompositionError(f"not part of the notation: {ast.unparse(node)!r}")

    return _part(conv(tree.body))


def looks_like(name):
    """Is this method name a composition (not a plain method name)?"""
    return any(t in name for t in (">>", "(", '"')) or name.strip() in NAMES


def canonical(text_or_expr):
    """The canonical spelling of a composition (the same composition always gives the same name)."""
    e = parse(text_or_expr) if isinstance(text_or_expr, str) else _part(text_or_expr)
    return e.text()


def file_name(name):
    """A safe file name for a composition: letters, digits and dashes, and a short hash of the name."""
    base = "".join(ch if ch.isalnum() else "-" for ch in name)
    while "--" in base:
        base = base.replace("--", "-")
    return base.strip("-")[:60] + "-" + hashlib.sha1(name.encode()).hexdigest()[:6]
