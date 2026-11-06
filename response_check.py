"""How wrong is the individual engine's response model? A diagnostic; it changes nothing in the engine.

The individual engine (individual.py) values a move by arithmetic: the change of the poll totals it makes
(with the voter's bloc) is worth so many voters switching from the poll winner A to the runner-up B, or
to the next place T (the projection on one switching voter's change), plus the cross-effects of the
scenario table. This script runs the engine as usual and, in every poll round, also works out the same
quantities exactly: it counts the averaged poll again with each move's change and finds the closeness
of B and T by the same bisection the engine uses for the poll itself (from 0 to n switching voters; for a candidate leading
after the change, from 3 sigma below 0, where the counts are still meaningful). Belief (the averaged poll, the
spread sigma, the chances from closeness) and the moves are the engine's; only the response to a
move differs. So the differences are what the arithmetic gets wrong, method by method.

Per method, over every (voter, move) with a change, in every round:
- err B, err T: mean |closeness by arithmetic - exact closeness|, in voters (sigma for scale);
- missed: moves the exact response values above 0.1% of the voter's utility range and the arithmetic
  at or below 0; phantom: the reverse (shares of all moves);
- best differs: share of voters whose decision differs (with no honesty cost: their most valuable move if
  it gains above 0.1% of their range, else sincere); moving: the share of voters who would move, by
  the arithmetic / exactly;
- new race: share of moves after which a candidate outside the races (B, T) needs fewer switching
  voters than T (exactly): a race the engine cannot see opens;
- flips: share of moves that change the winner of the averaged poll exactly / by the arithmetic.

Usage: python response_check.py [run.py options]   (e.g. --elections 200 --methods standard)
The run's own output goes to a temporary folder; the table is printed and written to
response_check.md in the current folder.
"""
import sys
import tempfile
from collections import defaultdict

import numpy as np

import individual as ind
import methods
import moves as mv

GAIN_TOL = 1e-3        # a gain counts above this share of the voter's utility range
NAMES = {id(M): name for name, M in methods.METHODS.items()}
STATS = defaultdict(lambda: defaultdict(float))


def predicted_k(r, S):
    """The closeness of each race after the change S (rows), by the engine's arithmetic (as in
    individual._value)."""
    races = r.races
    s = {c: (S @ r.delta[c]) / max(float(r.delta[c] @ r.delta[c]), 1e-12) for c in races}
    k = {}
    for c in races:
        if c in (r.B, r.T):
            k0 = min(r.k[c], float(r.n))
            base = np.full(len(S), k0)
            for d in races:
                if d != c and (c, d) in r.curve:
                    base = base + (r.cross(c, d, s[d]) - k0)
            k[c] = np.where(np.isfinite(r.k[c]) | (base < r.n), base, np.inf) - s[c]
        else:
            k[c] = r.k[c] - s[c]
    return k


def exact_k(r, S, cs):
    """The closeness of candidates cs after the change S, exactly: the poll counted again, the same
    bisection as for the poll (voters switching from A to c, from -n to n: below 0 c already leads)."""
    rows = r.T0[None] + S
    z, hi = np.zeros(len(S)), np.full(len(S), float(r.n))
    neg = np.full(len(S), -3 * r.sigma)       # below 0 only within 3 sigma: far beyond, the counts are meaningless
    out = {}
    for c in cs:
        k = ind._closeness(r.count, rows, c, r.C, r.delta[c], z, hi, ind.STEPS_FULL)    # as the engine does
        lead = k <= 0
        if lead.any():                         # c leads after the change: how far, voters switching back to A
            k[lead] = ind._closeness(r.count, rows[lead], c, r.C, r.delta[c], neg[lead], z[lead], ind.STEPS_FULL)
        out[c] = k
    return out


def chances(r, k, races):
    pA, ps = ind._chances_many([np.minimum(k[c], np.inf) for c in races], r.sigma)
    P = np.zeros((len(pA), r.C))
    P[:, r.A] = pA
    for c, p in zip(races, ps):
        P[:, c] += p
    return P


def options(method, u, sincere, r, V):
    """The engine's named moves for voters V (compromise, burial and both, for A, B, T), rendered."""
    places = [c for c in (r.A, r.B, r.T) if c is not None]
    strategies = []
    for c in places:
        strategies.append(ind.obvious(method.ballot, c, next(x for x in places if x != c)))
    for c in places:
        b = ind.burial(method.ballot, None, c)
        if b is not None:
            strategies.append(b)
            for d in places:
                if d != c:
                    strategies.append(ind.obvious(method.ballot, d, c) + b)
    strategies = list(dict.fromkeys(strategies))
    lazy, appr = mv.own_params(V)
    return [mv.render(method, u[V], m, sincere[V], lazy, appr) for m in strategies]


def check(method, count, u, B, sincere, r, threshold, followers):
    name = NAMES.get(id(method), "?")
    st = STATS[name]
    if r.B is None:
        st["no race"] += 1
        return
    V = np.flatnonzero(np.isfinite(threshold))
    if not len(V):
        return
    lo_u, hi_u = u[V].min(axis=1), u[V].max(axis=1)
    span = np.where(hi_u > lo_u, hi_u - lo_u, 1.0)
    f = (ind.bloc(u, float(followers)) if np.ndim(followers) == 0 else np.asarray(followers, float))[V]
    F_cur = count.features(B)[V]
    races = r.races
    others = [c for c in range(r.C) if c != r.A and c not in races]
    X0 = sincere[V]
    eu_p, eu_x = [], []
    for X in [X0] + options(method, u, sincere, r, V):
        S = f[:, None] * (count.features(X) - F_cur)
        Su, inv = np.unique(np.round(S, 9), axis=0, return_inverse=True)
        inv = inv.ravel()
        kp = {c: v[inv] for c, v in predicted_k(r, Su).items()}
        kx_all = exact_k(r, Su, races + others)
        kx = {c: v[inv] for c, v in kx_all.items()}
        Pp, Px = chances(r, kp, races), chances(r, kx, races)
        eu_p.append((Pp * u[V]).sum(axis=1))
        eu_x.append((Px * u[V]).sum(axis=1))
        if X is X0:                          # sanity: with no change the exact reading is the engine's
            z = np.abs(S).sum(axis=1) == 0
            if z.any():
                st["check"] = max(st["check"], max(float(np.abs(np.minimum(kx[c][z], r.n) - min(r.k[c], r.n)).max())
                                                   for c in races))
            continue
        moved = np.abs(S).sum(axis=1) > 0
        m = int(moved.sum())
        if not m:
            continue
        st["moves"] += m
        st["sigma"] += r.sigma * m
        for c, key in ((r.B, "err B"), (r.T, "err T")):
            if c is not None:
                a, b = np.minimum(kp[c], r.n)[moved], np.minimum(kx[c], r.n)[moved]
                st[key] += float(np.abs(a - b).sum())
                st[key + " n"] += m
        if r.T is not None and others:
            kT = kx[r.T][moved]
            st["new race"] += float((np.min([kx[c][moved] for c in others], axis=0) < kT).sum())
        st["flip exact"] += float((np.min([kx[c][moved] for c in races + others], axis=0) < 0).sum())
        st["flip arith"] += float((np.min([kp[c][moved] for c in races], axis=0) < 0).sum())
        before_p, before_x = eu_p[0][moved], eu_x[0][moved]
        gp = (eu_p[-1][moved] - before_p) / span[moved]
        gx = (eu_x[-1][moved] - before_x) / span[moved]
        st["missed"] += float(((gx > GAIN_TOL) & (gp <= 0)).sum())
        st["phantom"] += float(((gp > GAIN_TOL) & (gx <= 0)).sum())
        st["gain exact"] += float((gx > GAIN_TOL).sum())
        st["gain arith"] += float((gp > GAIN_TOL).sum())
    def decision(eu):                        # a voter without honesty cost: the best move if it gains
        E = np.stack(eu)
        best = np.argmax(E, axis=0)
        return np.where((E[best, np.arange(len(V))] - E[0]) / span > GAIN_TOL, best, 0)
    bp, bx = decision(eu_p), decision(eu_x)
    st["voters"] += len(V)
    st["best differs"] += float((bp != bx).sum())
    st["moving arith"] += float((bp > 0).sum())
    st["moving exact"] += float((bx > 0).sum())
    st["rounds"] += 1


_answer = ind.answer


def answer(method, count, u, B, sincere, reading, **kw):
    check(method, count, u, B, sincere, reading, kw["threshold"], kw["followers"])
    return _answer(method, count, u, B, sincere, reading, **kw)


def table():
    head = ("| Method | moves | sigma | err B | err T | gains (exact / arith.) | missed | phantom | moving (exact / arith.) "
            "| best differs | new race | flips (exact / arith.) | check |")
    out = [head, "|" + "---|" * 13]
    for name, st in STATS.items():
        m = st["moves"]
        if not m:
            out.append(f"| {name} | 0 | | | | | | | | | | | |")
            continue
        e = lambda k: st[k] / st[k + " n"] if st[k + " n"] else float("nan")
        out.append(f"| {name} | {int(m)} | {st['sigma'] / m:.1f} | {e('err B'):.2f} | {e('err T'):.2f} "
                   f"| {st['gain exact'] / m:.1%} / {st['gain arith'] / m:.1%} | {st['missed'] / m:.1%} "
                   f"| {st['phantom'] / m:.1%} | {st['moving exact'] / max(st['voters'], 1):.1%} / "
                   f"{st['moving arith'] / max(st['voters'], 1):.1%} | {st['best differs'] / max(st['voters'], 1):.1%} "
                   f"| {st['new race'] / m:.1%} | {st['flip exact'] / m:.1%} / {st['flip arith'] / m:.1%} "
                   f"| {st['check']:.2g} |")
    return "\n".join(out)


def main():
    ind.answer = answer                      # strategy.py looks it up at call time
    import run
    out = tempfile.mkdtemp(prefix="response_check_")
    sys.argv = ["run.py"] + sys.argv[1:] + ["--workers", "0", "--out", out, "--overwrite"]
    run.main()
    t = table()
    print(t)
    import os
    with open(os.environ.get("RESPONSE_CHECK_OUT", "response_check.md"), "w") as fh:
        fh.write("# Response check\n\n" + " ".join(sys.argv[1:-5]) + "\n\n" + t + "\n")


if __name__ == "__main__":
    main()
