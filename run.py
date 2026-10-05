"""Run honest and strategic elections for all methods on one scenario and write CSV.

Usage:
    python run.py --elections 200 --voters 99 --cands 4 --clusters 3 --spread 0.5 --width 1.0 --seed 1

Each electorate is tallied twice per method: with sincere ballots, and after the groups
(voters with the same sincere ranking) have learned shared ballots (see strategy.py).

Axes of the Pareto front:
    PE, WM     proportional efficiency and winner's worst margin, sincere ballots (see metrics.py)
    PE*, WM*   the same with learned ballots
CE, CE* (Condorcet efficiency) are reported alongside.
Also reported: VSE and RBE (sincere), cognitive dissonance of the learned ballots
(see strategy.dissonance), share of elections where learning settled (cycles never do).

Outputs (prefix set by --out):
    <out>-elections.csv   one row per (election, method, ballots), with the run's parameters
                          as columns, so several runs can be combined (see analyze.py)
    <out>-summary.csv     one row per method, with on_front for the PE/WM/PE*/WM* front
"""
import argparse
import csv
import zlib
from multiprocessing import Pool
import numpy as np

from scenario import Scenario, electorate
from methods import METHODS, select
from strategy import learn, strategy_types
from metrics import election_row, summarize, pareto, worst_margins


def one_election(args):
    sc, seed, e, rounds, samples, group_size, names = args
    rng = np.random.default_rng([seed, e])
    u = electorate(sc, rng)
    m_cand = worst_margins(u)                       # each candidate's worst head-to-head margin
    out = []
    for name in names:
        m = METHODS[name]
        out.append({"election": e, "method": name, "ballots": "sincere", **election_row(u, m.honest(u), m_cand),
                    "dissonance": 0.0, "settled": 1, "compromise": 0.0, "pushover": 0.0, "burial": 0.0})
        # Each method learns with its own random stream, keyed by name, so results stay
        # the same when methods are added or reordered.
        mrng = np.random.default_rng([seed, e, zlib.crc32(name.encode())])
        B, settled, dis = learn(m, u, mrng, rounds, samples, group_size)
        out.append({"election": e, "method": name, "ballots": "learned", **election_row(u, m.tally(B), m_cand),
                    "dissonance": dis, "settled": int(settled), **strategy_types(B, u)})
    return out


# Run parameters written into every row of the elections CSV
PARAMS = ["voters", "cands", "dims", "clusters", "spread", "width", "rounds", "samples", "group_size", "seed"]


def simulate(p, elections, workers=None, names=None):
    """Run one condition. p: dict with the keys in PARAMS; names: methods (default all).
    Returns rows with p as columns."""
    names = names or list(METHODS)
    sc = Scenario(p["voters"], p["cands"], p["dims"], p["clusters"], p["spread"], p["width"])
    jobs = [(sc, p["seed"], e, p["rounds"], p["samples"], p["group_size"], names) for e in range(elections)]
    with Pool(workers) as pool:
        rows = [r for batch in pool.imap(one_election, jobs, chunksize=4) for r in batch]
    meta = {k: p[k] for k in PARAMS}
    return [{**meta, **r} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elections", type=int, default=200)
    ap.add_argument("--voters", type=int, default=99)
    ap.add_argument("--cands", type=int, default=4)
    ap.add_argument("--dims", type=int, default=2)
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=20, help="learning rounds")
    ap.add_argument("--samples", type=int, default=10, help="poll samples per round")
    ap.add_argument("--group-size", type=int, default=0,
                    help="max voters coordinating as one group (0 = whole ranking group, 1 = individual)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None, help="run only these methods (start from none)")
    ap.add_argument("--exclude", nargs="+", default=None, help="run all methods except these")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    p = {k: getattr(a, k) for k in PARAMS}
    names = select(a.methods, a.exclude)
    results = simulate(p, a.elections, a.workers, names)

    with open(f"{a.out}-elections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    summ = {}
    for m in names:
        sinc = [r for r in results if r["method"] == m and r["ballots"] == "sincere"]
        lrn = [r for r in results if r["method"] == m and r["ballots"] == "learned"]
        s, l = summarize(sinc), summarize(lrn)
        summ[m] = {"PE": s["PE"], "WM": s["WM"], "PE_learned": l["PE"], "WM_learned": l["WM"],
                   "CE": s["CE"], "CE_learned": l["CE"],
                   "VSE": s["VSE"], "RBE": s["RBE"],
                   "dissonance": float(np.mean([r["dissonance"] for r in lrn])),
                   "settled": float(np.mean([r["settled"] for r in lrn]))}
    axes = ["PE", "WM", "PE_learned", "WM_learned"]
    front = set(pareto([[summ[m][x] for x in axes] for m in names]))

    cols = ["PE", "WM", "PE_learned", "WM_learned", "CE", "CE_learned", "VSE", "RBE", "dissonance", "settled"]
    with open(f"{a.out}-summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols + ["on_front"])
        for i, m in enumerate(names):
            w.writerow([m] + [f"{summ[m][c]:.4f}" for c in cols] + [int(i in front)])

    print(f"{a.elections} elections, {a.voters} voters, {a.cands} candidates, {a.clusters} clusters, "
          f"spread {a.spread}, width {a.width}; learning {a.rounds} rounds x {a.samples} poll samples, "
          f"group size {a.group_size or 'unlimited'}; seed {a.seed}")
    print(f"{'method':<26}{'PE':>7}{'WM':>7}{'PE*':>7}{'WM*':>7}{'CE':>7}{'CE*':>7}{'dissonance':>12}{'settled':>9}  front")
    for m in sorted(names, key=lambda m: -summ[m]["PE"]):
        s = summ[m]
        print(f"{m:<26}{s['PE']*100:7.1f}{s['WM']*100:7.1f}{s['PE_learned']*100:7.1f}{s['WM_learned']*100:7.1f}"
              f"{s['CE']*100:7.1f}{s['CE_learned']*100:7.1f}"
              f"{s['dissonance']*100:11.1f}%{s['settled']*100:8.0f}%"
              f"  {'*' if names.index(m) in front else ''}")
    print("PE*, WM*, CE*: with learned ballots; WM in % of voters; front on PE, WM, PE*, WM*")


if __name__ == "__main__":
    main()
