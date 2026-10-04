"""Run honest and strategic elections for all methods on one scenario and write CSV.

Usage:
    python run.py --elections 200 --voters 99 --cands 4 --clusters 3 --spread 0.5 --width 1.0 --seed 1

Each electorate is tallied twice per method: with sincere ballots, and after the groups
(voters with the same sincere ranking) have learned shared ballots (see strategy.py).

Axes of the Pareto front:
    PE, CE     proportional and Condorcet efficiency, sincere ballots
    PE*, CE*   the same with learned ballots
Also reported: VSE and RBE (sincere), cognitive dissonance of the learned ballots
(see strategy.dissonance), share of elections where learning settled (cycles never do).

Outputs (prefix set by --out):
    <out>-elections.csv   one row per (election, method, ballots)
    <out>-summary.csv     one row per method, with on_front for the PE/CE/SR front
"""
import argparse
import csv
import zlib
from multiprocessing import Pool
import numpy as np

from scenario import Scenario, electorate
from methods import METHODS
from strategy import learn
from metrics import election_row, summarize, pareto


def one_election(args):
    sc, seed, e, rounds, samples = args
    rng = np.random.default_rng([seed, e])
    u = electorate(sc, rng)
    out = []
    for name, m in METHODS.items():
        out.append({"election": e, "method": name, "ballots": "sincere", **election_row(u, m.honest(u)),
                    "dissonance": 0.0, "settled": 1})
        # Each method learns with its own random stream, keyed by name, so results stay
        # the same when methods are added or reordered.
        mrng = np.random.default_rng([seed, e, zlib.crc32(name.encode())])
        B, settled, dis = learn(m, u, mrng, rounds, samples)
        out.append({"election": e, "method": name, "ballots": "learned", **election_row(u, m.tally(B)),
                    "dissonance": dis, "settled": int(settled)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elections", type=int, default=200)
    ap.add_argument("--voters", type=int, default=99)
    ap.add_argument("--cands", type=int, default=4)
    ap.add_argument("--dims", type=int, default=2)
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=15, help="learning rounds")
    ap.add_argument("--samples", type=int, default=10, help="poll samples per round")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    sc = Scenario(a.voters, a.cands, a.dims, a.clusters, a.spread, a.width)
    jobs = [(sc, a.seed, e, a.rounds, a.samples) for e in range(a.elections)]
    with Pool(a.workers) as pool:
        results = [r for batch in pool.imap(one_election, jobs, chunksize=4) for r in batch]

    with open(f"{a.out}-elections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    names = list(METHODS)
    summ = {}
    for m in names:
        sinc = [r for r in results if r["method"] == m and r["ballots"] == "sincere"]
        lrn = [r for r in results if r["method"] == m and r["ballots"] == "learned"]
        s, l = summarize(sinc), summarize(lrn)
        summ[m] = {"PE": s["PE"], "CE": s["CE"], "PE_learned": l["PE"], "CE_learned": l["CE"],
                   "VSE": s["VSE"], "RBE": s["RBE"],
                   "dissonance": float(np.mean([r["dissonance"] for r in lrn])),
                   "settled": float(np.mean([r["settled"] for r in lrn]))}
    axes = ["PE", "CE", "PE_learned", "CE_learned"]
    front = set(pareto([[summ[m][x] for x in axes] for m in names]))

    cols = ["PE", "CE", "PE_learned", "CE_learned", "VSE", "RBE", "dissonance", "settled"]
    with open(f"{a.out}-summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols + ["on_front"])
        for i, m in enumerate(names):
            w.writerow([m] + [f"{summ[m][c]:.4f}" for c in cols] + [int(i in front)])

    print(f"{a.elections} elections, {a.voters} voters, {a.cands} candidates, {a.clusters} clusters, "
          f"spread {a.spread}, width {a.width}; learning {a.rounds} rounds x {a.samples} poll samples; seed {a.seed}")
    print(f"{'method':<20}{'PE':>7}{'CE':>7}{'PE*':>7}{'CE*':>7}{'dissonance':>12}{'settled':>9}  front")
    for m in sorted(names, key=lambda m: -summ[m]["PE"]):
        s = summ[m]
        print(f"{m:<20}{s['PE']*100:7.1f}{s['CE']*100:7.1f}{s['PE_learned']*100:7.1f}{s['CE_learned']*100:7.1f}"
              f"{s['dissonance']*100:11.1f}%{s['settled']*100:8.0f}%"
              f"  {'*' if names.index(m) in front else ''}")
    print("PE*, CE*: with learned ballots")


if __name__ == "__main__":
    main()
