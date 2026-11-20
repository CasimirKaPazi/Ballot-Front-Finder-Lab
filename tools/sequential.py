"""Sequential voting, a theoretical reference for strategy (not part of the model).

The voters of an electorate vote one at a time, in a random order. Each sees every ballot cast before
them and casts the ballot that gives them the best result if the count ended with their ballot: the
expected utility of the method's result on the ballots cast so far plus their own. They model nobody,
and nothing is free to set: the first voter sees nothing and votes honestly, the last one knows
everything there is to know. A voter casts their honest ballot unless another one is strictly better.

The ballots a voter considers are their honest ballot and the moves the strategy engines know
(strategy/moves.py): the naive exaggerations for every pair of candidates and push-over for every
triple, rendered in the method's format from the voter's own utilities.

What it shows, per method, over the elections (one random order each; --orders 2 or more draws more
orders of the same electorate):
    VSE, WME     honest ballots, as run.py
    VSE~, WME~   the sequential ballots
    CE, CE~      Condorcet efficiency, honest and sequential
    strategic    the share of voters casting another ballot than their honest one, overall and by
                 their place in the order (fifths: first fifth of the voters to the last)
    orders       with --orders 2+: how far the results of two orders of the same electorate lie apart
                 (half the summed absolute difference of the winning chances, averaged over pairs of
                 orders; 0 = the order never matters, 1 = different winners every time)

Usage (from the lab's folder):
    python tools/sequential.py --elections 200 --methods Plurality Approval Score IRV Schulze
    python tools/sequential.py --elections 100 --orders 3 --methods Plurality IRV
Writes nothing; prints the table. Electorates as run.py's defaults (--voters, --cands, --seed).
"""
import argparse
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the lab's modules
import run  # noqa: E402
import methods  # noqa: E402
from methods import METHODS, set_max_score, set_lazy, set_voter_lazy, set_voter_approval  # noqa: E402
from methods import set_candidates, candidate_utilities  # noqa: E402
from metrics import election_row, worst_margins, summarize  # noqa: E402
from models import draw_electorate  # noqa: E402
from nomination import cand_prefs  # noqa: E402
from run import lazy_cuts  # noqa: E402
from scenario import Scenario  # noqa: E402
from strategy import moves as mv  # noqa: E402


def move_list(ballot, C):
    """Every move the engines know for this format: the naive exaggerations for each ordered pair of
    candidates, push-over for each triple (duplicates removed)."""
    out = []
    for A in range(C):
        for B in range(C):
            if A == B:
                continue
            out += [tuple(m) for m in mv._naive(ballot, A, B)]
            for X in range(C):
                if X not in (A, B):
                    p = mv._push(ballot, A, B, X)
                    if p is not None:
                        out.append(tuple(p))
    return list(dict.fromkeys(out))


def sequential(m, u, S, order, moves):
    """The ballots cast one at a time in the order; returns the ballots and who cast another than their
    honest one."""
    n, C = u.shape
    cast = np.empty((0, C))
    B = S.copy()
    strategic = np.zeros(n, bool)
    for v in order:
        lazy, appr = mv.own_params(np.array([v]))
        rows = [S[v]] + [mv.render(m, u[v:v + 1], mo, S[v:v + 1], lazy, appr)[0] for mo in moves]
        rows = np.unique(np.array(rows, float), axis=0)
        best, best_eu = S[v], float(m.tally(np.vstack([cast, S[v:v + 1]])) @ u[v])
        tol = 1e-12 * (np.ptp(u[v]) + 1e-300)
        for r in rows:
            if np.array_equal(r, S[v]):
                continue
            eu = float(m.tally(np.vstack([cast, r[None]])) @ u[v])
            if eu > best_eu + tol:
                best, best_eu = r, eu
        B[v] = best
        strategic[v] = not np.array_equal(best, S[v])
        cast = np.vstack([cast, best[None]])
    return B, strategic


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--elections", type=int, default=100)
    ap.add_argument("--methods", nargs="+", default=["Plurality", "Approval", "Score", "IRV", "Schulze"])
    ap.add_argument("--orders", type=int, default=1, help="random orders per electorate (2+: how much the order matters)")
    ap.add_argument("--voters", type=int, default=99)
    ap.add_argument("--cands", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = run.default_params()
    p.update(voters=a.voters, cands=a.cands, seed=a.seed)
    sc = Scenario(p["voters"], p["cands"], p["dims"], p["clusters"], p["spread"], p["width"], p["zipf"],
                  p["cand_dispersion"], p["cand_shift"], p.get("model", "clusters"), p.get("ks_alpha", 1.5),
                  p.get("ks_cut", 0.2), p.get("ks_shared", 0), salience=p.get("salience", "none"),
                  salience_exponent=p.get("salience_exponent", 1.0), fixed_variance=p.get("fixed_variance", 0),
                  field_kappa=p.get("field_kappa", 1.5), field_salience=p.get("field_salience", 0.0),
                  noise_strength=p.get("noise_strength", 3.0), utility=p.get("utility", "exp"),
                  cands_vote=int(p.get("cands_vote", 0)))      # as run.simulate
    names = [n for n in a.methods if n in METHODS]
    for n in a.methods:
        if n not in METHODS:
            print(f"unknown method {n!r}; left out")
    set_max_score(p["max_score"])
    set_lazy(p.get("lazy", 0.0))
    rows = {n: {"h": [], "s": []} for n in names}
    strat = {n: [] for n in names}
    apart = {n: [] for n in names}
    moves = {}
    t0 = time.time()
    for e in range(a.elections):
        rng = np.random.default_rng([p["seed"], e])
        ele = draw_electorate(sc, rng)
        u = ele.u
        n, C = u.shape
        set_voter_lazy(lazy_cuts(np.random.default_rng([p["seed"], e, zlib.crc32(b"lazy")]), n, p.get("lazy", 0.0)))
        set_voter_approval(np.random.default_rng([p["seed"], e, zlib.crc32(b"approval")]).random(n))
        cu = cand_prefs(ele.cands, sc.width, ele.cand_weights, getattr(sc, "utility", "exp"))
        set_candidates(candidate_utilities(cu, [(d, 0) for d in range(C)]))
        m_cand = worst_margins(u)
        orders = [np.random.default_rng([p["seed"], e, zlib.crc32(b"sequential"), k]).permutation(n)
                  for k in range(a.orders)]
        for name in names:
            m = METHODS[name]
            if (m.ballot, C) not in moves:
                moves[(m.ballot, C)] = move_list(m.ballot, C)
            S = m.sincere(u)
            rows[name]["h"].append(election_row(u, m.honest(u), m_cand))
            Ps = []
            for k, order in enumerate(orders):
                B, st = sequential(m, u, S, order, moves[(m.ballot, C)])
                P = m.tally(B)
                Ps.append(P)
                rows[name]["s"].append(election_row(u, P, m_cand))
                strat[name].append(st[order])          # by place in the order
            for i in range(len(Ps)):
                for j in range(i + 1, len(Ps)):
                    apart[name].append(0.5 * float(np.abs(Ps[i] - Ps[j]).sum()))
        set_candidates(None)
        print(f"\r{e + 1}/{a.elections} elections, {time.time() - t0:.0f} s", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    print(f"Sequential voting: {a.elections} elections, {a.orders} order(s) each, {a.voters} voters, "
          f"{a.cands} candidates, seed {a.seed}. ~: sequential ballots.")
    head = f"{'method':<16}{'VSE':>7}{'WME':>7}{'VSE~':>7}{'WME~':>7}{'CE':>7}{'CE~':>7}{'strat':>7}  by fifth of the order"
    if a.orders > 1:
        head += "            orders"
    print(head)
    for name in names:
        h, s = summarize(rows[name]["h"]), summarize(rows[name]["s"])
        st = np.array(strat[name])
        fifths = [st[:, k * st.shape[1] // 5:(k + 1) * st.shape[1] // 5].mean() for k in range(5)]
        line = (f"{name:<16}{h['VSE'] * 100:7.1f}{h['WME'] * 100:7.1f}{s['VSE'] * 100:7.1f}{s['WME'] * 100:7.1f}"
                f"{h['CE'] * 100:7.1f}{s['CE'] * 100:7.1f}{st.mean() * 100:6.1f}%  "
                + " ".join(f"{f * 100:4.1f}" for f in fifths))
        if a.orders > 1:
            line += f"   {np.mean(apart[name]):6.3f}"
        print(line)


if __name__ == "__main__":
    main()
