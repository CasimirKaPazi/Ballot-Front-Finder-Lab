"""Composing methods from parts (docs/compositions.md).

A composition is a chain of steps over one state: the field (the candidates still in) and the readings
known so far. Sets narrow the field and pass the readings through; transforms replace a reading; a
reading or a final step takes the latest version of what it needs, restricted to the current field.
Readings that depend on the field (first preferences, rank points, scores stretched over the field) are
never taken from an earlier field: they are computed again on it, from totals over every subset of
candidates where the lab has them, else from the ballots restricted to the field (the exact fallback,
for the election count; polls then use the lab's default poll reading).

The same text works in Python (an add-on) and on the command line (read with Python's own parser,
only the parts below accepted). Many built-in methods carry their composition as a comment in
methods.py ("# composed: ..."), e.g.

    smith >> "Score"                                              Smith//Score
    ballots("rank") >> repeat(drop(firsts))                       IRV
    btr(score, margins, score)                                    BTR-score
    top(2, score) >> runoff(score)                                Score runoff

Parts:
    readings   score (totals; on two-round ballots the round-1 marks), firsts (first preferences among
               the field, rankings), held (after proxy: the votes each candidate holds), margins, wv (winning votes), po (pairwise opposition), soft(k)
               and soft(k, "margins"/"wv"/"po") (soft counts, k a share of the range or a clip per pair in
               points), n (voters), levels (voters per score level, top first),
               points (the same times the level's score), approvals(k) (voters scoring at least k),
               median (each candidate's lower median score, as MJ)
    arithmetic share(values) = values / (n x top score); worst(pairs) (each candidate's worst defeat);
               total(pairs) (row sums); diff(values) (pairs: row minus column); + - * / and unary minus;
               abs(x), minimum(a, b), maximum(a, b) (element by element)
    sets       smith, schwartz, cdtt (majority defeats by more than half of all voters), banks (tops of
               the maximal chains), uncovered (covered by nobody), best(values), top(k, values), drop(values) (all but the lowest), buck(levels or
               points), challenger(set, pairs), bottom_two(order, pairs[, tiebreak]) (one round of
               bottom-two runoff), btr(order, pairs[, tiebreak]) (the bottom-two climb: the winner),
               repeat(steps) (until one is left or nothing changes); a | b, a & b. Ties branch; the
               same part twice in one set expression takes the same branch
    transforms normalize (score ballots stretched over the current field, not rounded); max_top (each
               ballot's highest-rated in the field count the top score), zero_bottom (its lowest-rated
               count 0), IMDS's steps; proxy,
               proxy("type") (asset voting: every vote given to candidates as the voter's ballot
               divides it, and cast from then on by them with their own ballots of the type; inside a
               repeat, the votes of candidates who are out are passed on round by round)
    final      "Name": an existing method on the current field and readings; runoff, runoff(values):
               a separate runoff by the runoff rankings of two-round ballots; a set at the end splits
               the win equally
    first      ballots("type"): the ballot type (default: that of the quoted method; score runoff
               ballots with runoff; else score); delegate (what lazy voters leave blank is filled from
               their favourite's own ballot)
    chaining   >> ("then")

composed(expr) turns a composition into an ordinary Method (methods.py): ballot type, features,
decision, its poll view (model.md, What a poll shows: normalize is read as nothing in a poll, a quoted
method by its own poll view, firsts by the first preferences and transfers as the IRV family's
stand-in) and the settings of its final step.
"""
import ast
import hashlib
import itertools

import numpy as np

import methods as M
from ballots import BALLOTS

# ---------- how each feature narrows to a smaller field (docs/compositions.md, Design notes) ----------
CAND0 = {"sum", "app", "run_sum", "maxs", "pos", "upper", "ge1", "ge2", "eq0", "grades"}   # axis 0 is the candidate
CAND1 = {"levels"}                                                                     # axis 1 is the candidate
PAIR = {"gt", "runoff_gt", "round1_gt", "excess", "soft20", "soft40", "levelled", "diffhist"}   # axes 0, 1
AMONG = {"firsts", "firsts_eq", "firsts_ex", "lasts", "keepnorm", "nscore_sub", "dmscore_sub", "rb_sub"}   # per subset, among it
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


ROUND1 = {"score_run": "score", "app_run": "approval", "fav_run": "fav"}   # two-round: the round-1 ballot


def base(f):
    """The feature a derived one is made from (r1_...: the same on the round-1 marks)."""
    return f[3:] if f.startswith("r1_") else f


def r1(f):
    """Feature f read on the round-1 marks of two-round ballots (made on first use)."""
    name = "r1_" + f
    if name not in M.FEATURES:
        fn, shape = M.FEATURES[f]
        M.FEATURES[name] = ((lambda fn: lambda B: fn(np.floor_divide(B, B.shape[1])))(fn), shape)
    return name


def is_pair(f):
    f = base(f)
    return f in PAIR or f.startswith("soft")


def restrictable(f, ballot):
    """Can feature f be read on a narrower field from its totals?"""
    if f.startswith("r1_"):
        return restrictable(base(f), ROUND1.get(ballot, ballot))
    if f == "sum":
        return rated(ballot)
    return f in CAND0 | CAND1 | AMONG | SCALAR or is_pair(f)


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


def restrict(s, f, field, C):
    """Feature f's sum on the field (a list of candidates)."""
    F = list(field)
    x = s[f]
    if len(F) == C:
        return x
    b = base(f)
    if b in SCALAR:
        return x
    if b in CAND0:
        return x[F]
    if b in CAND1:
        return x[:, F]
    if is_pair(f):
        return x[np.ix_(F, F)]
    if b in AMONG:
        sub = x[_sub_index(F, C)]
        return sub[:, F] if sub.ndim == 2 else sub
    raise KeyError(f)


# ---------- score transforms (normalize, max_top, zero_bottom) ----------
# The state keeps the transforms applied so far as a tuple of (name, field) steps. A score total after
# them comes from a table over every subset where the lab keeps one (nscore_sub: normalize; dmscore_sub:
# max_top and zero_bottom on the same field, as IMDS), else from the ballots (the exact fallback).
TRANSFORMS = ("normalize", "max_top", "zero_bottom")


def _table(ops):
    """The subset table that gives the score totals after these transforms, or None. ops: the names in
    order, each with a mark of the field it acted on (the same mark: the same field)."""
    names = [o for o, _ in ops]
    if names == ["normalize"]:
        return "nscore_sub"
    if sorted(names) == ["max_top", "zero_bottom"] and ops[0][1] == ops[1][1]:
        return "dmscore_sub"
    return None


def _apply(op, X):
    """One score transform on ballots X (voters x the field it acts on)."""
    if op == "normalize":
        return normalize_ballots(X)
    lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    if op == "max_top":                # the highest-rated count the top score; a ballot rating all alike stays
        return np.where((X >= hi) & (hi > lo), float(M.MAXS), X)
    return np.where(X <= lo, 0.0, X)    # zero_bottom: the lowest-rated count 0 (all alike: 0 for all)


def transform_ballots(B, norm, field):
    """Score ballots after the transforms norm ((name, field) steps, fields narrowing), on the field."""
    Y = np.asarray(B, float).copy()
    for op, F in norm:
        F = list(F)
        Y[:, F] = _apply(op, Y[:, F])
    return Y[:, list(field)]


def _sum_need(ctx):
    """What a score total needs after the transforms so far: a subset table, else the exact fallback."""
    t = _table(ctx["tops"])
    if t is not None:
        return _need(t, ctx)
    ctx["fallback"].add("sum")
    return set()


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
    def __abs__(self): return Abs(self)

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
    the exact fallback (ballots restricted to the field). After proxy every reading comes from the
    candidates' ballots, so nothing is needed from the voters' totals."""
    if ctx.get("proxied"):
        return set()
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
            raise CompositionError(f"{self.name} is not available after {', '.join(o for o, _ in ctx['tops'])}")
        f = self.feat(b)
        if f == "sum" and ctx["normalized"] and not ctx["poll"]:
            return _sum_need(ctx)
        return _need(f, ctx)

    def value(self, st):
        f = self.feat(st.ballot)
        norm = None if st.poll else st.norm
        if not st.exact and f in st.s and restrictable(f, st.ballot) and not (f == "sum" and norm is not None):
            # independent of the field: computed once on the whole field, then indexed (passed through)
            key = (self.name, st.poll)
            full = st.shared.get(key)
            if full is None:
                full = st.shared[key] = self.fn(st.s[f], st)
            if len(st.field) == st.C or self.kind == "scalar":
                return full
            F = st.field
            return full[np.ix_(F, F)] if self.kind == "pairs" else full[:, F] if self.kind == "levels" else full[F]
        return self.fn(st.get(f), st)


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


SOFT_COUNTS = ("margins", "wv", "po")


def soft(clip, count="margins"):
    """Soft pairwise counts (as Soft Minimax): every voter's score difference d for a pair counts by
    g(d) = clip(d, -k, k) / k, a difference of k or more as a full vote, smaller ones in proportion.
    clip: a number, k as a share of the score range (0 < k <= 1; 0.2 and 0.4 are the lab's features
    soft20 and soft40, another share gets its feature on first use), or pairs, k per pair in score
    points (soft(1 + abs(diff(score / n))) is Soft Minimax-D). count: "margins" (the soft support for
    the row minus that for the column), "wv" (the support for the row where it wins the pair, else 0)
    or "po" (the support for the row whether it wins or not), as margins, wv and po are for majorities."""
    if count not in SOFT_COUNTS:
        raise CompositionError(f"soft counts {', '.join(SOFT_COUNTS)}, not {count!r}")
    if isinstance(clip, Part):
        if clip.kind not in ("pairs", "scalar"):
            raise CompositionError(f"soft needs a share of the score range or pairs (a clip per pair, in score "
                                   f"points), {clip.text()} is {_kind_name(clip)}")
        if clip.kind == "pairs":
            return SoftPairs(clip, count)
        if not isinstance(clip, Const):
            raise CompositionError(f"soft needs a number (a share of the score range) or pairs, not {clip.text()}")
        clip = clip.x
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not 0 < clip <= 1:
        raise CompositionError(f"soft needs a share of the score range above 0 and at most 1 (or pairs: a clip "
                               f"per pair, in score points), not {clip!r}")
    frac = float(clip)
    if count != "margins":
        return SoftPairs(Const(frac), count, share=True)
    key = f"soft{round(frac * 100, 6):g}"
    if key not in M.FEATURES:
        M.FEATURES[key] = ((lambda f: lambda B: M._soft_margins(B, f))(frac), lambda C: (C, C))
    return Reading(f"soft({frac:g})", "pairs", lambda b: key, lambda x, st: _f(x), scored=True)


_diffhist = Reading("diffhist", "pairs", lambda b: "diffhist", lambda x, st: _f(x), scored=True)


def soft_counts(H, k, count):
    """Soft counts from the difference counts H (C, C, 2 MAXS + 1: voters per integer score difference
    of a pair) and a clip k per pair (C, C, points); a pair with k <= 0 counts 0."""
    x = np.arange(H.shape[2]) - M.MAXS
    kk = np.maximum(k, 1e-9)[:, :, None]
    if count == "margins":
        out = (H * np.clip(x[None, None, :], -kk, kk)).sum(axis=2) / kk[:, :, 0]
    else:
        out = (H * np.clip(x[None, None, :], 0, kk)).sum(axis=2) / kk[:, :, 0]     # support for the row
    out = np.where(k > 0, out, 0.0)
    if count == "wv":
        out = np.where(out > out.T, out, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


class SoftPairs(Part):
    """soft(clip, count) from the counts of score differences per pair (feature diffhist): any clip,
    also one per pair, and the one-sided counts (wv, po). Score ballots are whole numbers, so the
    counts are exact."""
    kind = "pairs"

    def __init__(self, clip, count, share=False):
        self.clip, self.count, self.share = clip, count, share

    def text(self):
        c = f"{self.clip.x:g}" if self.share else self.clip.text()
        return f"soft({c})" if self.count == "margins" else f'soft({c}, "{self.count}")'

    def check(self, ctx):
        b = ctx["ballot"]
        if not (rated(b) and BALLOTS[b].score_range):
            raise CompositionError(f"{self.text()} needs score ballots, not {b}")
        if ctx["normalized"] and not ctx["poll"]:
            raise CompositionError(f"{self.text()} is not available after {', '.join(o for o, _ in ctx['tops'])} "
                                   "(it counts whole score differences)")
        return _need("diffhist", ctx) | self.clip.check(ctx)

    def value(self, st):
        H = _diffhist.value(st)
        k = self.clip.value(st)
        k = np.full(H.shape[:2], k * M.MAXS if self.share else k, float) if np.ndim(k) == 0 else np.asarray(k, float)
        return soft_counts(H, k, self.count)


def approvals(level):
    """Voters scoring each candidate at least `level` (1 .. top score)."""
    if not isinstance(level, int) or level < 1:
        raise CompositionError(f"approvals needs a score level of at least 1, not {level!r}")

    def fn(x, st):
        if level > len(x):
            raise CompositionError(f"approvals({level}) is above the top score {len(x)}")
        return _f(x)[:len(x) - level + 1].sum(axis=0)       # levels: top score first
    return Reading(f"approvals({level})", "values", lambda b: "levels", fn, scored=True)


def _median(x, st):
    """Each candidate's median score (the lower median, as majority judgment reads it), from the voters
    per score (grades: h[c, g] voters giving c the score g)."""
    h = M._common_total(np.asarray(x, float))
    pos = M.mj_order(int(round(h[0].sum())))[0]
    return (np.cumsum(h, axis=1) <= pos).sum(axis=1).astype(float)


median = Reading("median", "values", lambda b: "grades", _median, scored=True)


FIRSTS = {"rank": "firsts", "strict": "firsts", "trunc": "firsts_ex", "trunc3": "firsts_ex", "weak": "firsts_eq"}


class Firsts(Reading):
    """First preferences among the current field (rankings): from the totals over every subset; in a
    poll, estimated from the first preferences and the transfer table, as the IRV family's stand-in
    (methods.among_subsets; on rankings with ties whole votes, as ER-IRV)."""
    def __init__(self):
        super().__init__("firsts", "values", lambda b: FIRSTS.get(b, "sum" if b == "fav" else None), None)

    def poll_feats(self, b):
        return ("top", "transfer_whole") if b == "weak" else ("first_split", "transfer")

    def check(self, ctx):
        b = ctx["ballot"]
        if b not in FIRSTS and b != "fav":
            raise CompositionError(f"firsts needs rankings (or one-name ballots), not {b}")
        if ctx["poll"] and b in FIRSTS:
            return set(self.poll_feats(b))
        return _need(self.feat(b), ctx)

    def value(self, st):
        b = st.ballot
        if st.poll and b in FIRSTS:
            am = st.cache.get("_among")
            if am is None:
                am = (M._Among(_f(st.s["top"]), st.s["transfer_whole"], whole=True) if b == "weak"
                      else M._Among(st.s["first_split"], st.s["transfer"]))
                st.cache["_among"] = am
            return np.asarray(am[int(sum(1 << c for c in st.field))], float)[st.field]
        f = self.feat(b)
        if b != "fav" and not st.exact and f in st.s:          # the row of the current field
            return _f(st.s[f][int(sum(1 << c for c in st.field))][st.field])
        x = st.get(f)
        return _f(x if b == "fav" else x[-1])


firsts = Firsts()


class Held(Part):
    """After proxy: the votes each candidate of the field holds (what they will cast)."""
    kind = "values"

    def text(self):
        return "held"

    def check(self, ctx):
        if not ctx.get("proxied"):
            raise CompositionError("held reads the votes the candidates hold: it needs a proxy before it")
        return set()

    def value(self, st):
        return np.asarray(st.W, float)[st.field]


held = Held()


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


def total(x):
    """Row sums of pairs: per candidate, the sum of their values against every other (total(po) is the
    number of candidates each voter ranks them above, Borda points among the field on strict rankings)."""
    return Func("total", _part(x), "values", ("pairs",), lambda p, st: (p * (1 - np.eye(len(p)))).sum(axis=1))


def diff(x):
    """Pairs from values: value of the row minus value of the column."""
    return Func("diff", _part(x), "pairs", ("values",), lambda v, st: np.subtract.outer(v, v))


def share(x):
    """values / (n x top score): a share of the maximum."""
    return Func("share", _part(x), "values", ("values",), lambda v, st: v / (st.get("n") * M.MAXS), extra=("n",))


def worst(x):
    """Each candidate's worst defeat: the largest value a rival has against them (0 if none; a candidate's
    value against themselves, the diagonal, does not count)."""
    return Func("worst", _part(x), "values", ("pairs",),
                lambda p, st: np.clip(np.where(np.eye(len(p), dtype=bool), 0.0, p), 0, None).max(axis=0))


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


class Abs(Part):
    """abs(x): the size of values, pairs or a number."""
    def __init__(self, a):
        if a.kind not in ("values", "pairs", "scalar"):
            raise CompositionError(f"abs needs values, pairs or a number, {a.text()} is {_kind_name(a)}")
        self.a, self.kind = a, a.kind

    def text(self):
        return f"abs({self.a.text()})"

    def check(self, ctx):
        return self.a.check(ctx)

    def value(self, st):
        return np.abs(self.a.value(st))


class Extreme(BinOp):
    """minimum(a, b), maximum(a, b): the smaller or larger of two values, pairs or numbers, element by
    element (a number with values or pairs: with each of them)."""
    def __init__(self, op, a, b):
        super().__init__(op, a, b)
        self.prec = 9

    def text(self):
        return f"{self.op}({self.a.text()}, {self.b.text()})"

    def value(self, st):
        return (np.minimum if self.op == "minimum" else np.maximum)(self.a.value(st), self.b.value(st))


def minimum(a, b):
    return Extreme("minimum", _part(a), _part(b))


def maximum(a, b):
    return Extreme("maximum", _part(a), _part(b))


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


def _closure(R):
    """Transitive closure of a boolean relation (beatpaths)."""
    R = R.copy()
    for k in range(len(R)):
        R |= R[:, [k]] & R[[k], :]
    return R


def _schwartz_of(D):
    """The candidates i such that whoever has a path of defeats D to i, i has one back: the union of
    the minimal sets no outsider defeats."""
    R = _closure(D)
    return [i for i in range(len(D)) if all(R[i, j] or not R[j, i] for j in range(len(D)))]


class Schwartz(Set):
    def text(self):
        return "schwartz"

    def check(self, ctx):
        return margins.check(ctx)

    def field(self, st):
        D = margins.value(st) > M.EPS                    # strict pairwise defeats
        return [st.field[i] for i in _schwartz_of(D)]


schwartz = Schwartz()


class CDTT(Set):
    def text(self):
        return "cdtt"

    def check(self, ctx):
        return po.check(ctx) | n.check(ctx)

    def field(self, st):
        D = po.value(st) > n.value(st) / 2 + M.EPS      # majority defeats: more than half of all voters
        return [st.field[i] for i in _schwartz_of(D)]


cdtt = CDTT()


class Uncovered(Set):
    def text(self):
        return "uncovered"

    def check(self, ctx):
        return margins.check(ctx)

    def field(self, st):
        D = margins.value(st) > M.EPS                    # strict pairwise defeats
        # A covers B: A defeats B and everyone B defeats (Miller); the uncovered are covered by nobody
        cov = D & ~(D[None, :, :] & ~D[:, None, :]).any(axis=2)
        return [st.field[j] for j in range(len(D)) if not cov[:, j].any()]


uncovered = Uncovered()

_BANKS = {}


def _banks_of(D):
    """The tops of the maximal chains: subsets ordered by strict defeats D with every pair decided and no
    cycle, that no further candidate can join; ties keep a pair out of a chain."""
    key = D.tobytes()
    if key in _BANKS:
        return _BANKS[key]
    k = len(D)
    beats = [sum(1 << j for j in range(k) if D[i, j]) for i in range(k)]
    dec = [sum(1 << j for j in range(k) if D[i, j] or D[j, i]) for i in range(k)]
    chain = [False] * (1 << k)
    chain[0] = True
    for S in range(1, 1 << k):
        i = (S & -S).bit_length() - 1
        rest = S & ~(1 << i)
        # S is a chain if S minus i is one, i is decided against all of it, and i's wins within S
        # differ from everyone else's (a transitive tournament has wins k-1, ..., 0)
        if chain[rest] and (dec[i] & rest) == rest:
            wins = sorted(bin(beats[j] & S).count("1") for j in range(k) if S >> j & 1)
            chain[S] = wins == list(range(len(wins)))
    out = set()
    for S in range(1, 1 << k):
        if chain[S] and not any(chain[S | 1 << x] for x in range(k) if not S >> x & 1):
            members = [j for j in range(k) if S >> j & 1]
            out.add(max(members, key=lambda j: bin(beats[j] & S).count("1")))
    res = sorted(out)
    if len(_BANKS) < 100000:
        _BANKS[key] = res
    return res


class Banks(Set):
    def text(self):
        return "banks"

    def check(self, ctx):
        return margins.check(ctx)

    def field(self, st):
        D = margins.value(st) > M.EPS
        return [st.field[i] for i in _banks_of(D)]


banks = Banks()


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


class Drop(Branching):
    def __init__(self, arg):
        if arg.kind != "values":
            raise CompositionError(f"drop needs values, {arg.text()} is {_kind_name(arg)}")
        self.arg = arg

    def text(self):
        return f"drop({self.arg.text()})"

    def check(self, ctx):
        return self.arg.check(ctx)

    def options(self, st, memo):
        F = st.field
        if len(F) <= 1:
            return [(1.0, list(F))]
        v = np.asarray(self.arg.value(st), float)
        low = np.flatnonzero(v <= v.min() + M.EPS)
        return [(1.0 / len(low), [c for j, c in enumerate(F) if j != i]) for i in low]


def drop(x):
    """Everyone but the lowest; a tie for lowest branches (each of them dropped, equal weights)."""
    return Drop(_part(x))


class BottomTwo(Branching):
    name = "bottom_two"

    def __init__(self, order, arg, tiebreak=None):
        if order.kind != "values":
            raise CompositionError(f"{self.name} needs values first (the order), {order.text()} is {_kind_name(order)}")
        if arg.kind != "pairs":
            raise CompositionError(f"{self.name} needs pairs second (the match), {arg.text()} is {_kind_name(arg)}")
        if tiebreak is not None and tiebreak.kind != "values":
            raise CompositionError(f"{self.name} breaks a tie by values, {tiebreak.text()} is {_kind_name(tiebreak)}")
        self.order, self.arg, self.tiebreak = order, arg, tiebreak

    def text(self):
        tb = "" if self.tiebreak is None else f", {self.tiebreak.text()}"
        return f"{self.name}({self.order.text()}, {self.arg.text()}{tb})"

    def check(self, ctx):
        return self.order.check(ctx) | self.arg.check(ctx) | (self.tiebreak.check(ctx) if self.tiebreak else set())

    def options(self, st, memo):
        F = st.field
        if len(F) <= 1:
            return [(1.0, list(F))]
        o = np.asarray(self.order.value(st), float)
        p = np.asarray(self.arg.value(st), float)
        tb = None if self.tiebreak is None else np.asarray(self.tiebreak.value(st), float)
        out = []
        for a, b, w in M.top_two(-o):                          # the bottom two by the order (ties branch)
            x = p[a, b]
            if abs(x) <= M.EPS and tb is not None:
                x = tb[a] - tb[b]
            losers = [(b, 1.0)] if x > M.EPS else [(a, 1.0)] if x < -M.EPS else [(a, .5), (b, .5)]
            out += [(w * wl, [c for j, c in enumerate(F) if j != l]) for l, wl in losers]
        return out


def bottom_two(order, x, tiebreak=None):
    """One round: the bottom two by the order meet, and the loser of the match (pairs: positive, the
    row wins) is out; a tie goes to the higher tiebreak value, else is split; ties for the bottom two
    branch. With repeat, the order is read again on the field left each round (BTR-IRV)."""
    return BottomTwo(_part(order), _part(x), None if tiebreak is None else _part(tiebreak))


class BTR(BottomTwo):
    """The bottom-two climb on a fixed order: a set of the winner."""
    name = "btr"

    def branches(self, st, memo):
        F = st.field
        if len(F) <= 1:
            return [(1.0, list(F), memo)]
        o = np.asarray(self.order.value(st), float)
        p = np.asarray(self.arg.value(st), float)
        tb = None if self.tiebreak is None else np.asarray(self.tiebreak.value(st), float)

        def match(x, y):                                   # x comes from below, y is the next one up
            v = p[x, y]
            if abs(v) > M.EPS or tb is None:
                return v
            d = tb[x] - tb[y]
            return d if abs(d) > M.EPS else -1.0            # still tied: the one higher in the order
        P = M.bottom_two_runoff(o, match)
        return [(float(P[i]), [F[i]], memo) for i in np.flatnonzero(P > 0)]


def btr(order, x, tiebreak=None):
    """The bottom-two climb: the field ordered high to low by `order` (every order of tied candidates
    equally); the lowest two meet, the winner meets the next one up, and so on; the match (pairs:
    positive, the row wins) decides; a tie is split, or with a tiebreak goes to the higher tiebreak
    value and, when those tie too, to the candidate higher in the order (methods.bottom_two_runoff)."""
    return BTR(_part(order), _part(x), None if tiebreak is None else _part(tiebreak))


class Repeat(Set):
    def __init__(self, inner):
        steps = _steps(inner)
        for p in steps:
            if p.kind not in ("set", "transform"):
                raise CompositionError(f"repeat takes sets and transforms, {p.text()} is {_kind_name(p)}")
        self.inner = inner if isinstance(inner, Chain) else Chain([inner])

    def text(self):
        return f"repeat({self.inner.text()})"

    def check(self, ctx):
        # every round after the first is on a narrower field; with a proxy inside, the voters' ballots
        # are only read in the first round (later rounds count the candidates' ballots)
        has_proxy = any(isinstance(p, Proxy) for p in _walk(self.inner))
        c = {**ctx, "tops": list(ctx["tops"]), "nset": ctx["nset"] + 1}
        if not has_proxy:
            c["narrowed"] = True
        feats = set()
        for p in self.inner.steps:
            feats |= p.check(c)
            if p.kind == "transform":
                p.after(c)
            elif p.kind == "set":
                c["narrowed"] = True
                c["nset"] += 1
        ctx["fallback"] |= c["fallback"]
        if c.get("proxied"):                                     # proxy changes the ballots for good
            ctx.update(proxied=True, normalized=False, poll=False, ballot=c["ballot"], tops=[])
        return feats

    def branches(self, st, memo):
        out = []

        def once(state):
            res = [(1.0, state)]
            for p in self.inner.steps:
                nxt = []
                for w, s_ in res:
                    if p.kind == "transform":
                        nxt.append((w, p.transform(s_)))
                    else:
                        for wb, F, _ in p.branches(s_, {}):
                            F = F or list(s_.field)
                            nxt.append((w * wb, s_.but(field=F)))
                res = nxt
            return res

        def go(state, w):
            if len(state.field) <= 1:
                out.append((w, list(state.field), memo))
                return
            for wb, s_ in once(state):
                if s_.field == state.field:                      # nothing changed: done
                    out.append((w * wb, list(s_.field), memo))
                else:                       # the next round: normalize again, but proxy's ballots carry over
                    go(s_.but(norm=st.norm if s_.W is None else None), w * wb)

        go(st, 1.0)
        return out


def repeat(x):
    """The steps (sets and transforms) again and again, until one candidate is left or a round changes
    nothing; each round starts from the readings before the repeat (a normalize inside acts per round),
    except that the ballots proxy gives carry over (a proxy inside passes votes on round by round)."""
    return Repeat(_part(x))


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


class ScoreTransform(Part):
    """A transform of the score ballots over the current field (normalize, max_top, zero_bottom)."""
    kind = "transform"

    def __init__(self, name):
        self.name = name

    def text(self):
        return self.name

    def check(self, ctx):
        if not rated(ctx["ballot"]) or not BALLOTS[ctx["ballot"]].score_range:
            raise CompositionError(f"{self.name} needs score ballots, not {ctx['ballot']}")
        return set()

    def after(self, ctx):
        ctx["normalized"] = True
        step = (self.name, ctx["nset"])
        # normalize after anything gives what normalize alone gives (it stretches every ballot anew)
        ctx["tops"] = [step] if self.name == "normalize" else ctx["tops"] + [step]

    def transform(self, st):
        step = (self.name, tuple(st.field))
        return st.but(norm=(step,) if self.name == "normalize" or st.norm is None else st.norm + (step,))


normalize = ScoreTransform("normalize")
max_top = ScoreTransform("max_top")
zero_bottom = ScoreTransform("zero_bottom")
Normalize = ScoreTransform


VOTER_SHARES = ("fav", "rank", "strict", "weak", "trunc", "trunc3", "approval", "score", "grade3")
CAND_BALLOTS = ("rank", "strict", "weak", "trunc", "trunc3", "approval", "score", "grade3")


class Proxy(Part):
    """Asset voting: every vote is given to candidates, and from then on cast by them, each with
    their own sincere ballot (methods.candidate_ballots, from the candidates' utilities over the
    field). The voters' ballots say how their vote is divided among the current field
    (methods.vote_shares): one name or a ranking give it to the first choice; approval splits it
    equally among the approved (a cumulative vote); scores and grades split it by the Kotze-Pereira
    transformation. The candidates' ballots are of the type given, proxy("score"); a bare proxy
    takes the ballot type of the composition's last quoted method, else rankings. The electorate is
    then the candidates, each weighted by the votes they hold, and every later reading counts their
    ballots (exactly, also in a poll: only who holds how much comes from the poll). A candidate who is
    out still holds their votes until a later proxy: then everything a candidate out of the field
    holds is passed on as their own ballot divides a vote among the field, and candidates in the field
    keep what they hold. Inside a repeat this carries over from round to round."""
    kind = "transform"

    def __init__(self, ptype=None):
        if ptype is not None and ptype not in CAND_BALLOTS:
            raise CompositionError(f"proxy: the candidates' ballots must order more than one candidate "
                                   f"({', '.join(CAND_BALLOTS)}), not {ptype!r}")
        self.ptype = ptype

    def __call__(self, ptype):
        return Proxy(ptype)

    def text(self):
        return "proxy" if self.ptype is None else f'proxy("{self.ptype}")'

    def _type(self, ctx_or_st):
        pt = self.ptype or (ctx_or_st["proxy_type"] if isinstance(ctx_or_st, dict) else ctx_or_st.pt)
        return pt or "rank"

    def check(self, ctx):
        b, pt = ctx["ballot"], self._type(ctx)
        if pt not in CAND_BALLOTS:
            raise CompositionError(f"proxy: the candidates' ballots must order more than one candidate "
                                   f"({', '.join(CAND_BALLOTS)}), not {pt} (give the type: proxy(\"rank\"))")
        if ctx.get("proxied"):
            return set()
        if b not in VOTER_SHARES:
            raise CompositionError(f"proxy needs voters' ballots that give a vote to candidates ({', '.join(VOTER_SHARES)}), "
                                   f"not {b}")
        return firsts.check(ctx) if b in FIRSTS or b == "fav" else _need(M.SHARE_FEATURE[b], ctx)

    def after(self, ctx):
        ctx.update(proxied=True, normalized=False, poll=False, ballot=self._type(ctx), tops=[])

    def transform(self, st):
        pt = self._type(st)
        K = M.candidate_ballots(pt, st.C)
        F = list(st.field)
        W = np.zeros(st.C)
        if st.W is None:                                       # the voters' votes, divided among the field
            b = st.ballot
            W[F] = np.asarray(firsts.value(st) if b in FIRSTS or b == "fav" else st.get(M.SHARE_FEATURE[b]), float)
        else:                                                  # candidates out pass on what they hold
            inn = np.zeros(st.C, bool)
            inn[F] = True
            W[inn] = st.W[inn]
            out = np.flatnonzero(~inn & (st.W > M.EPS))
            if len(out):
                sh = M.vote_shares(restrict_ballots(K[out], F, pt), pt)
                W[F] += st.W[out] @ sh
        return State({}, F, None, st.C, False, pt, K, True, None, W, pt=st.pt)


proxy = Proxy()


class Delegate(Part):
    """Delegation: what a lazy voter leaves blank is filled from their favourite's own ballot, below
    everything the voter marked (methods.delegate_fill). A property of the ballots, so it comes first
    (after ballots(...), if given); it changes nothing without laziness (--lazy)."""
    kind = "ballots"

    def text(self):
        return "delegate"


delegate = Delegate()


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
    """An existing method: as the last step its winner lottery; anywhere a set can stand, the set of its
    winner (a tie branches with the lottery's weights). On two-round ballots, a method of the round-1
    ballot type (score ballots for score runoff ballots, approval, one name) counts the round-1 marks."""
    kind = "set"

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

    def _feat(self, f, ballot):
        return f if self.method.ballot == ballot else r1(f)

    def check(self, ctx):
        m = self.view(ctx["poll"])
        if self.method.ballot != ctx["ballot"]:
            if ROUND1.get(ctx["ballot"]) != self.method.ballot:
                raise CompositionError(f"{self.name} uses {self.method.ballot} ballots, the composition {ctx['ballot']}")
            return set().union(*[_need(r1(f), ctx) for f in m.feats])
        if ctx["normalized"] and not ctx["poll"]:
            bad = [f for f in m.feats if f not in AFTER_NORMALIZE]
            if bad:
                raise CompositionError(f"{self.name} reads {', '.join(bad)}, which is not available after "
                                       f"{', '.join(o for o, _ in ctx['tops'])}")
            return set().union(*[_sum_need(ctx) if f == "sum" else _need(f, ctx) for f in m.feats])
        return set().union(*[_need(f, ctx) for f in m.feats])

    def lottery(self, st):
        m = self.view(st.poll)
        return np.asarray(m.decide({f: st.get(self._feat(f, st.ballot)) for f in m.feats}), float)

    def branches(self, st, memo):
        key = self.text()
        if key in memo:
            return [(1.0, memo[key], memo)]
        P = self.lottery(st)
        return [(float(P[i]), [st.field[i]], {**memo, key: [st.field[i]]}) for i in np.flatnonzero(P > 1e-12)]


class RandomBallot(Set):
    """A ballot drawn at random, then a candidate of the current field in proportion to what it gives
    them: on score, grade and approval ballots and single marks the values themselves (on two-round
    ballots the round-1 marks); on rankings the first preference among the field. A ballot that gives
    the field nothing is drawn again. As the last step its lottery; as a set, a branch per candidate
    with the lottery's weights. In a poll: in proportion to the totals (score totals, first preferences
    and transfers on rankings)."""
    def text(self):
        return "random_ballot"

    def _feat(self, b):
        if b in FIRSTS:
            return FIRSTS[b]
        if b == "veto":
            raise CompositionError("random_ballot needs ballots that give candidates something, not veto ballots")
        return r1("rb_sub") if b in TWO_ROUND else "rb_sub"

    def check(self, ctx):
        b = ctx["ballot"]
        if b in FIRSTS:
            return firsts.check(ctx)
        f = self._feat(b)
        if ctx["poll"]:
            return _need(r1("sum") if b in TWO_ROUND else "sum", ctx)
        return _need(f, ctx)

    def lottery(self, st):
        b = st.ballot
        if b in FIRSTS:
            v = firsts.value(st)
        elif st.poll:
            v = st.get(r1("sum") if b in TWO_ROUND else "sum")
        else:
            f = self._feat(b)
            if not st.exact and f in st.s:
                v = st.s[f][int(sum(1 << c for c in st.field))][st.field]
            else:
                v = st.get(f)[-1]
        return M.lottery_of(v)

    def branches(self, st, memo):
        key = self.text()
        if key in memo:
            return [(1.0, memo[key], memo)]
        P = self.lottery(st)
        return [(float(P[i]), [st.field[i]], {**memo, key: [st.field[i]]}) for i in np.flatnonzero(P > 1e-12)]


random_ballot = RandomBallot()


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
            if p.kind == "ballots" and i > 0 and not (isinstance(p, Delegate) and i == 1 and isinstance(flat[0], Ballots)):
                raise CompositionError(f"{p.text()} comes first" + (" (after ballots(...))" if isinstance(p, Delegate) else ""))
        self.steps = flat
        self.kind = "chain"

    def text(self):
        return " >> ".join(_wrap(p, 4) for p in self.steps)


# ---------- the state while counting ----------
class State:
    __slots__ = ("s", "field", "norm", "C", "poll", "ballot", "B", "cache", "exact", "shared", "W", "pt")

    def __init__(self, s, field, norm, C, poll, ballot, B=None, exact=False, shared=None, W=None, pt=None):
        self.s, self.field, self.norm, self.C, self.poll = s, field, norm, C, poll
        self.ballot, self.B, self.cache, self.exact = ballot, B, {}, exact
        self.shared = {} if shared is None else shared       # readings on the whole field, for one count
        self.W = W                  # after proxy: B are the candidates' ballots, W the votes each holds
        self.pt = pt                # the candidates' ballot type of a bare proxy (proxy_type)

    def but(self, **kw):
        """The same state with some slots changed (and nothing cached)."""
        a = {k: getattr(self, k) for k in ("s", "field", "norm", "C", "poll", "ballot", "B", "exact", "shared", "W", "pt")}
        a.update(kw)
        return State(**a)

    def get(self, f):
        """Feature f's sum on the field: from the totals where it can be read there, else from the
        ballots restricted to the field (the exact fallback)."""
        if f in self.cache:
            return self.cache[f]
        norm = None if self.poll else self.norm
        W = None if self.W is None else self.W[:, None]        # after proxy: each ballot weighted
        if f == "sum" and norm is not None:                   # scores after normalize, max_top, zero_bottom
            t = _table([(o, F) for o, F in norm])
            if t is not None and t in self.s and (self.B is None or not self.exact):
                x = self.s[t][int(sum(1 << c for c in norm[0][1]))][list(self.field)]
            else:                                              # the transformed ballots, on the field
                Bn = transform_ballots(self.B, norm, self.field)
                x = (Bn if W is None else Bn * W).sum(axis=0)
            self.cache[f] = x
            return x
        have = f in self.s
        full = len(self.field) == self.C
        if self.B is None or (have and (full or (not self.exact and restrictable(f, self.ballot)))):
            x = restrict(self.s, f, self.field, self.C)
        else:
            Bf = self.B if len(self.field) == self.C else restrict_ballots(self.B, self.field, self.ballot)
            k = len(self.field)
            X = M.FEATURES[f][0](Bf).reshape(len(Bf), -1).astype(float)
            x = (X if W is None else X * W).sum(axis=0).reshape(M.FEATURES[f][1](k))
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
    ctx = {"ballot": ballot, "narrowed": False, "normalized": False, "poll": poll, "fallback": set(),
           "proxy_type": proxy_type(expr), "tops": [], "nset": 0}
    feats = set()
    _final(expr)
    for p in _steps(expr):
        if p.kind == "ballots":
            continue
        feats |= p.check(ctx)
        if p.kind == "set":
            ctx["narrowed"] = True
            ctx["nset"] += 1
        elif p.kind == "transform":
            p.after(ctx)
    return feats, ctx["fallback"]


def proxy_type(expr):
    """The candidates' ballot type of a bare proxy: that of the composition's last quoted method,
    else rankings."""
    quoted = [p for p in _walk(expr) if isinstance(p, Quoted)]
    return quoted[-1].method.ballot if quoted else "rank"


def evaluate(expr, s, C, ballot, poll=False, B=None, exact=False, pt=None):
    """Winner probabilities (C,) of the composition from the summed features s (and the ballots B for
    the exact fallback; exact: every reading on a narrower field from the ballots, for the tests)."""
    P = np.zeros(C)
    steps = [p for p in _steps(expr) if p.kind != "ballots"]

    def go(i, st, w):
        p = steps[i]
        if p.kind == "transform":
            go(i + 1, p.transform(st), w)
        elif p.kind == "set":
            for wb, F, _ in p.branches(st, {}):
                if not F:
                    F = list(st.field)                       # an empty set narrows nothing
                if i == len(steps) - 1:                  # a set at the end: the win split equally
                    P[F] += w * wb / len(F)
                else:
                    go(i + 1, st.but(field=F), w * wb)
        else:                                           # a quoted method or a runoff
            P[st.field] += w * p.lottery(st)

    go(0, State(s, list(range(C)), None, C, poll, ballot, B, exact, pt=pt or proxy_type(expr)), 1.0)
    return P


def _walk(expr):
    out = []

    def walk(p):
        out.append(p)
        for a in ("a", "b", "arg", "of", "tiebreak", "order", "inner"):
            if isinstance(getattr(p, a, None), Part):
                walk(getattr(p, a))
        for q in getattr(p, "steps", ()):
            walk(q)
    walk(expr)
    return out


def ballot_of(expr):
    """The composition's ballot type: ballots(...) if given; with a runoff the two-round ballot whose
    first round is that of the first quoted method (score runoff ballots without one); else the ballot
    of the first quoted method, else score ballots."""
    steps = _steps(expr)
    if isinstance(steps[0], Ballots):
        return steps[0].name
    parts = _walk(expr)
    quoted = [p for p in parts if isinstance(p, Quoted)]
    if any(isinstance(p, Runoff) for p in parts):              # two-round: the one whose first round fits
        first = quoted[0].method.ballot if quoted else "score"
        return first if first in ROUND1 else next((k for k, v in ROUND1.items() if v == first), "score_run")
    return quoted[0].method.ballot if quoted else "score"


def _C_of(s):
    """The number of candidates, from the shape of a summed feature."""
    for f, x in s.items():
        shape = np.shape(x)
        if not shape:
            continue
        for C in range(1, 17):
            if tuple(M.FEATURES[f][1](C)) == shape:
                return C
    raise CompositionError("no feature with candidates")


def composed(expr, ballot=None):
    """The composition as a Method (methods.py)."""
    expr = _part(expr)
    ballot = ballot or ballot_of(expr)
    feats, fallback = check(expr, ballot)
    feats = tuple(sorted(feats))
    pt = proxy_type(expr)
    pfeats, pfallback = check(expr, ballot, poll=True)
    parts = _walk(expr)
    finals = [p for p in _steps(expr) if isinstance(p, Quoted)]
    settings = finals[-1].method.settings if finals else None
    poll = None
    if pfallback or not all(M.poll_like(f) for f in pfeats):
        poll = None                     # the lab's default poll reading (score totals or first preferences)
    elif fallback or set(pfeats) != set(feats) or any(isinstance(p, ScoreTransform) for p in parts) or \
            any(p.method.poll_view() is not p.method for p in parts if isinstance(p, Quoted)):
        poll = (tuple(sorted(pfeats)), lambda s: evaluate(expr, s, _C_of(s), ballot, poll=True, pt=pt))
    if fallback:                        # counted from the ballots (features of the field as needed)
        inner = M.Method(ballot, None, feats) if feats else None

        def full(B):
            C = B.shape[1]
            s = {}
            if inner is not None:
                inner._C = C
                s = inner.unpack(inner.totals(B))
            return evaluate(expr, s, C, ballot, B=B, pt=pt)
        m = M.Method(ballot, full=full, settings=settings, poll=poll)
    else:
        m = M.Method(ballot, lambda s: evaluate(expr, s, _C_of(s), ballot, pt=pt), feats, settings=settings, poll=poll)
    if any(isinstance(p, Delegate) for p in _steps(expr)):
        if ballot not in M.DELEGABLE:
            raise CompositionError(f"delegate needs ballots where a voter leaves candidates blank and a blank can "
                                   f"be filled below what they marked ({', '.join(M.DELEGABLE)}), not {ballot}")
        m.delegate = True
    m.__dict__["composition"] = expr
    return m


# ---------- reading the text ----------
NAMES = {"score": score, "margins": margins, "wv": wv, "po": po, "n": n, "levels": levels, "points": points,
         "firsts": firsts, "median": median, "smith": smith, "schwartz": schwartz, "cdtt": cdtt, "banks": banks, "uncovered": uncovered,
         "normalize": normalize, "max_top": max_top, "zero_bottom": zero_bottom, "runoff": runoff, "random_ballot": random_ballot,
         "proxy": proxy, "delegate": delegate, "held": held}
CALLS = {"share": share, "worst": worst, "total": total, "diff": diff, "best": best, "top": top, "buck": buck,
         "drop": drop, "btr": btr, "bottom_two": bottom_two, "repeat": repeat, "ballots": ballots, "challenger": challenger, "soft": soft,
         "approvals": approvals, "runoff": runoff, "proxy": proxy, "abs": lambda x: Abs(_part(x)), "minimum": minimum,
         "maximum": maximum}


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
