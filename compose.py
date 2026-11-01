"""Composing methods from parts (docs/pipelines.md; phase 1).

A composition is a chain of steps over one state: the field (the candidates still in) and the readings
known so far. Sets narrow the field and pass the readings through; transforms replace a reading; a
reading or a final step takes the latest version of what it needs, restricted to the current field.
Readings that depend on the field (first preferences, rank points, scores stretched over the field) are
computed again on it from totals over every subset of candidates, or refused with an error; they are
never taken from an earlier field.

The same text works in Python (an add-on) and on the command line (read with Python's own parser,
only the parts below accepted):

    smith >> "Score"                          Smith//Score
    buck(points) >> normalize >> "Score"      ESR
    best(share(score) - worst(wv) / n)        MMWV+Score
    ballots("rank") >> best(-worst(margins))  Minimax

Parts (phase 1):
    readings   score (totals), margins, wv (winning votes), n (voters), levels (voters per score level),
               points (score points per level); share(values) = values / (n x top score)
    pairs -> values   worst(pairs): each candidate's worst defeat (largest value a rival has against
               them, 0 if none)
    sets       smith, best(values) (all tied for highest), top(k, values) (ties at the k-th place
               branch), buck(levels or points) (for every depth, the candidates ahead in the top
               levels, joined); a | b, a & b
    transforms normalize (score ballots stretched over the current field, not rounded)
    final      "Name": an existing method on the current field and readings; a set at the end
               splits the win equally among its members
    first      ballots("type"): the ballot type (default: that of the quoted method, else score)
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
PAIR = {"gt", "runoff_gt", "excess", "soft20", "soft40", "levelled", "diffhist"}      # axes 0 and 1
AMONG = {"firsts", "firsts_eq", "firsts_ex", "lasts", "keepnorm", "nscore_sub"}       # per subset, among it
SCALAR = {"n"}
# after normalize: features that keep their meaning (order among candidates, or already stretched per
# subset); "sum" becomes the normalized scores
AFTER_NORMALIZE = {"n", "sum", "gt", "firsts", "firsts_eq", "firsts_ex", "lasts", "nscore_sub"}


def rated(ballot):
    """Ballots whose values are per-candidate ratings (score, grades, approval; add-on types with a
    score range and no marks): their totals do not depend on the field."""
    b = BALLOTS[ballot]
    return ballot in ("score", "grade3", "approval") or (b.score_range and b.marks is None)


def restrictable(f, ballot):
    """Can feature f be read on a narrower field (from totals)?"""
    if f == "sum":
        return rated(ballot)
    return f in CAND0 | CAND1 | PAIR | AMONG | SCALAR


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
    x = s[f]
    F = list(field)
    if len(F) == C and (norm is None or f != "sum"):
        return x
    if f == "sum" and norm is not None:
        return s["nscore_sub"][int(sum(1 << c for c in norm))][F]
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
    kind = None              # values, pairs, scalar, levels, set, transform, final, ballots
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
        needed."""
        return set()


def _wrap(p, prec):
    t = p.text()
    return f"({t})" if p.prec < prec else t


class Reading(Part):
    def __init__(self, name, kind, feats, fn, needs_rated=False, field_free=True):
        self.name, self.kind, self._feats, self.fn = name, kind, feats, fn
        self.needs_rated, self.field_free = needs_rated, field_free

    def text(self):
        return self.name

    def check(self, ctx):
        if self.needs_rated and not rated(ctx["ballot"]):
            raise CompositionError(f"{self.name} needs rated ballots (score, grades, approval), not {ctx['ballot']}")
        if self.name == "score" and ctx["narrowed"] and not ctx["normalized"] and not rated(ctx["ballot"]):
            raise CompositionError(f"score on {ctx['ballot']} ballots is rank points, which depend on the field: "
                                   "not available after a set")
        if self.name in ("levels", "points") and ctx["normalized"] and not ctx["poll"]:
            raise CompositionError(f"{self.name} is not defined after normalize (stretched scores are not rounded)")
        if self.name == "score" and ctx["normalized"] and not ctx["poll"]:
            return {"nscore_sub"}
        return set(self._feats)

    def value(self, st):
        return self.fn(st)


def _score(st):
    return st.get("sum")


def _gt(st):
    return st.get("gt").astype(float)


score = Reading("score", "values", ("sum",), _score)
margins = Reading("margins", "pairs", ("gt",), lambda st: (lambda g: g - g.T)(_gt(st)))
wv = Reading("wv", "pairs", ("gt",), lambda st: (lambda g: np.where(g > g.T, g, 0.0))(_gt(st)))
n = Reading("n", "scalar", ("n",), lambda st: st.get("n"))
levels = Reading("levels", "levels", ("levels",), lambda st: st.get("levels"), needs_rated=True)
points = Reading("points", "levels", ("levels",),
                 lambda st: (lambda c: (c.shape[0] - np.arange(c.shape[0]))[:, None] * c)(st.get("levels")),
                 needs_rated=True)


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
        return self.arg.check(ctx) | set(self.extra)

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


# sets: value(st) -> list of (weight, field), the field as a list of candidates
class Set(Part):
    kind = "set"


class Smith(Set):
    def text(self):
        return "smith"

    def check(self, ctx):
        return {"gt"}

    def value(self, st):
        g = _gt(st)
        return [(1.0, [st.field[i] for i in M.smith_set(g - g.T)])]


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

    def value(self, st):
        v = np.asarray(self.arg.value(st), float)
        return [(1.0, [st.field[i] for i in np.flatnonzero(v >= v.max() - M.EPS)])]


def best(x):
    """All candidates tied for the highest value."""
    return Best(_part(x))


class Top(Set):
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

    def value(self, st):
        F = st.field
        if len(F) <= self.k:
            return [(1.0, list(F))]
        v = np.asarray(self.arg.value(st), float)
        cut = np.sort(v)[::-1][self.k - 1]
        above = [i for i in range(len(F)) if v[i] > cut + M.EPS]
        tied = [i for i in range(len(F)) if abs(v[i] - cut) <= M.EPS]
        ways = list(itertools.combinations(tied, self.k - len(above)))
        return [(1.0 / len(ways), [F[i] for i in sorted(above + list(w))]) for w in ways]


def top(k, x):
    """The k highest by a measure; a tie at the k-th place branches with equal weights."""
    return Top(k, _part(x))


class Buck(Set):
    def __init__(self, arg):
        if arg.kind != "levels":
            raise CompositionError(f"buck needs levels or points, {arg.text()} is {_kind_name(arg)}")
        self.arg = arg

    def text(self):
        return f"buck({self.arg.text()})"

    def check(self, ctx):
        return self.arg.check(ctx)

    def value(self, st):
        A = np.cumsum(self.arg.value(st), axis=0)           # per depth: totals from the top levels
        sup = sorted({int(c) for row in A for c in np.flatnonzero(row >= row.max() - M.EPS)})
        return [(1.0, [st.field[i] for i in sup])]


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

    def value(self, st):
        out = []
        for (wa, fa), (wb, fb) in itertools.product(self.a.value(st), self.b.value(st)):
            f = sorted(set(fa) | set(fb)) if self.op == "|" else sorted(set(fa) & set(fb))
            out.append((wa * wb, f or list(st.field)))     # an empty intersection narrows nothing
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
        if ctx["narrowed"]:
            bad = [f for f in m.feats if not restrictable(f, ctx["ballot"])]
            if bad:
                raise CompositionError(f"{self.name} reads {', '.join(bad)}, which "
                                       f"{'depends' if len(bad) == 1 else 'depend'} on the field: not available after a set")
        if ctx["normalized"] and not ctx["poll"]:
            bad = [f for f in m.feats if f not in AFTER_NORMALIZE]
            if bad:
                raise CompositionError(f"{self.name} reads {', '.join(bad)}, which normalize does not provide")
        return set(m.feats) | ({"nscore_sub"} if ctx["normalized"] and not ctx["poll"] and "sum" in m.feats else set())

    def lottery(self, st):
        m = self.view(st.poll)
        d = {f: st.get(f) for f in m.feats}
        return np.asarray(m.decide(d), float)


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
    __slots__ = ("s", "field", "norm", "C", "poll")

    def __init__(self, s, field, norm, C, poll):
        self.s, self.field, self.norm, self.C, self.poll = s, field, norm, C, poll

    def get(self, f):
        return restrict(self.s, f, self.field, self.C, None if self.poll else self.norm)


def _steps(expr):
    return expr.steps if isinstance(expr, Chain) else [expr]


def _final(expr):
    p = _steps(expr)[-1]
    if p.kind not in ("final", "set"):
        raise CompositionError(f"a composition ends in a method, a set or best(...); {p.text()} is {_kind_name(p)}")
    return p


def check(expr, ballot, poll=False):
    """Static check of the whole composition; the features it needs."""
    ctx = {"ballot": ballot, "narrowed": False, "normalized": False, "poll": poll}
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
    return feats


def evaluate(expr, s, C, poll=False):
    """Winner probabilities (C,) of the composition from the summed features s."""
    P = np.zeros(C)
    steps = [p for p in _steps(expr) if p.kind != "ballots"]

    def go(i, st, w):
        p = steps[i]
        if p.kind == "transform":
            go(i + 1, State(st.s, st.field, tuple(st.field), C, poll), w)
        elif p.kind == "set":
            for wb, F in p.value(st):
                if i == len(steps) - 1:                  # a set at the end: the win split equally
                    P[F] += w * wb / len(F)
                else:
                    go(i + 1, State(st.s, F, st.norm, C, poll), w * wb)
        else:                                           # a quoted method
            P[st.field] += w * p.lottery(st)

    go(0, State(s, list(range(C)), None, C, poll), 1.0)
    return P


def ballot_of(expr):
    """The composition's ballot type: ballots(...) if given, else that of its quoted methods, else score."""
    steps = _steps(expr)
    if steps[0].kind == "ballots":
        return steps[0].name
    quoted = [p for p in _quoted(expr)]
    return quoted[0].method.ballot if quoted else "score"


def _quoted(expr):
    out = []

    def walk(p):
        if isinstance(p, Quoted):
            out.append(p)
        for a in ("a", "b", "arg"):
            if hasattr(p, a) and isinstance(getattr(p, a), Part):
                walk(getattr(p, a))
        for q in getattr(p, "steps", ()):
            walk(q)
    walk(expr)
    return out


def _has_normalize(expr):
    return any(p.kind == "transform" for p in _steps(expr))


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
    feats = tuple(sorted(check(expr, ballot)))
    pfeats = tuple(sorted(check(expr, ballot, poll=True)))
    q = _quoted(expr)
    finals = [p for p in _steps(expr) if isinstance(p, Quoted)]
    settings = finals[-1].method.settings if finals else None
    poll = None
    if _has_normalize(expr) or any(p.method.poll_view() is not p.method for p in q):
        poll = (pfeats, lambda s: evaluate(expr, s, _C_of(s), poll=True))
    m = M.Method(ballot, lambda s: evaluate(expr, s, _C_of(s)), feats, settings=settings, poll=poll)
    m.__dict__["composition"] = expr
    return m


# ---------- reading the text ----------
NAMES = {"score": score, "margins": margins, "wv": wv, "n": n, "levels": levels, "points": points,
         "smith": smith, "normalize": normalize}
CALLS = {"share": share, "worst": worst, "best": best, "top": top, "buck": buck, "ballots": ballots}


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
            raise CompositionError(f"unknown part {node.id!r}; parts: {', '.join(list(NAMES) + list(CALLS))}")
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
