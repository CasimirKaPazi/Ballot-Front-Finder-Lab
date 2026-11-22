"""Dark horses: how often strategy elects a weak candidate (a diagnostic, not part of the model).

The pooled engine builds its moves around the poll's top three (two leaders and the next place), so
the DH3 pathology may be out of its reach: three strong candidates whose supporters each bury both
rivals, so that a weak fourth candidate nobody buries wins. This script measures how often it happens
anyway, per method, with honest and with learned ballots (run.py's defaults otherwise).

A candidate's strength is method-free: their rank by the Borda count of the sincere preferences (each
voter's ranking by utility), 1 = strongest. A dark horse is a candidate ranked 4th or lower. Reported:
    dark       the winner's chance of being a dark horse, averaged over the elections
    >= half    the share of elections in which a dark horse wins with a chance of at least one half
    by total   the same chance with the rank by total utility instead (4th or lower)
with honest ballots and learned ones (*); the difference is what strategy adds.

Usage (from the lab's folder):
    python tools/darkhorse.py --elections 1000
    python tools/darkhorse.py --elections 500 --first 500      # elections 500..999 (to split a run)
    python tools/darkhorse.py --elections 400 --strategic 1    # every voter strategic
Writes nothing; prints the table.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the lab's modules
import run  # noqa: E402
from scenario import Scenario  # noqa: E402

DEFAULT = ["Borda", "Score", "STAR", "Approval", "Score runoff", "321", "MJ", "IRV", "Plurality", "Schulze"]


def ranks(x):
    """1 for the largest value, 2 for the next, ... (ties: the order of the candidates)."""
    r = np.empty(len(x), int)
    r[np.argsort(-np.asarray(x), kind="stable")] = np.arange(1, len(x) + 1)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--elections", type=int, default=500)
    ap.add_argument("--first", type=int, default=0, help="the number of the first election")
    ap.add_argument("--methods", nargs="+", default=DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategic", type=float, default=None, help="the strategic share (run.py --strategic)")
    a = ap.parse_args()
    p = run.default_params()
    p["seed"] = a.seed
    if a.strategic is not None:
        p["strategic"] = a.strategic
    sc = Scenario(p["voters"], p["cands"], p["dims"], p["clusters"], p["spread"], p["width"], p["zipf"],
                  p["cand_dispersion"], p["cand_shift"], p.get("model", "clusters"), p.get("ks_alpha", 1.5),
                  p.get("ks_cut", 0.2), p.get("ks_shared", 0), salience=p.get("salience", "none"),
                  salience_exponent=p.get("salience_exponent", 1.0), fixed_variance=p.get("fixed_variance", 0),
                  field_kappa=p.get("field_kappa", 1.5), field_salience=p.get("field_salience", 0.0),
                  noise_strength=p.get("noise_strength", 3.0), utility=p.get("utility", "exp"),
                  cands_vote=int(p.get("cands_vote", 0)))      # as run.simulate
    orig = run.election_row

    def row(u, P, m_cand):                     # run's election_row, plus the dark-horse measures
        r = orig(u, P, m_cand)
        C = u.shape[1]
        pos = np.argsort(np.argsort(u, axis=1), axis=1)             # each voter's ranking: 0 = least liked
        dark_b = ranks(pos.sum(axis=0)) >= 4
        dark_u = ranks(u.sum(axis=0)) >= 4
        P = np.asarray(P, float)[:C]
        r.update(_dark=float(P[dark_b].sum()), _dark_u=float(P[dark_u].sum()))
        return r
    run.election_row = row
    rounds = int(p.get("rounds", 3))
    acc = {}
    t0 = time.time()
    for k, e in enumerate(range(a.first, a.first + a.elections)):
        for r in run.one_election({"sc": sc, "p": p, "e": e, "rounds": [rounds], "names": a.methods,
                                   "participation": False, "manipulation": False, "dummy": None}):
            acc.setdefault((r["method"], r["ballots"]), []).append((r["_dark"], r["_dark_u"]))
        print(f"\r{k + 1}/{a.elections} elections, {time.time() - t0:.0f} s", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    print(f"Dark horses (ranked 4th or lower by the sincere Borda count; by total utility in the last two "
          f"columns): elections {a.first}..{a.first + a.elections - 1}, seed {a.seed}"
          + (f", strategic share {a.strategic}" if a.strategic is not None else "") + ", defaults otherwise. "
          f"*: learned ballots.")
    print(f"{'method':<16}{'dark':>8}{'dark*':>8}{'>=half':>8}{'>=half*':>9}{'by total':>10}{'by total*':>11}")
    for m in a.methods:
        h = np.array(acc[(m, "sincere")])
        s = np.array(acc[(m, "learned")])
        print(f"{m:<16}{h[:, 0].mean() * 100:7.2f}%{s[:, 0].mean() * 100:7.2f}%"
              f"{(h[:, 0] >= 0.5).mean() * 100:7.2f}%{(s[:, 0] >= 0.5).mean() * 100:8.2f}%"
              f"{h[:, 1].mean() * 100:9.2f}%{s[:, 1].mean() * 100:10.2f}%")


if __name__ == "__main__":
    main()
