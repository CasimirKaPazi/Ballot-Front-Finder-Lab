"""Run honest and strategic elections for all methods on one scenario and write CSV.

Usage:
    python run.py --elections 200                      # defaults: see README, Parameters

Each electorate is tallied twice per method: with sincere ballots, and after the groups
(voters with the same sincere ranking) have learned shared ballots (see strategy.py).

Axes of the Pareto front (0 = random ballot, 100 = best candidate; see metrics.py):
    VSE, WME     voter satisfaction and worst-margin efficiency, sincere ballots
    VSE*, WME*   the same with learned ballots
CE, CE* (Condorcet efficiency) are reported alongside, as are the sincerity of the learned
ballots (see strategy.sincerity) and the share of elections where learning settled.

Partial strategy: --strategic p makes a random share p of the voters strategic (the same voters
for every method); the rest always vote sincerely.

Outputs (prefix set by --out; a name without a directory goes into output/):
    <out>-elections.csv   one row per (election, method, ballots), with the run's parameters
                          as columns, so several runs can be combined (see analyze.py)
    <out>-summary.csv     one row per method, with on_front for the VSE/WME/VSE*/WME* front
"""
import argparse
import csv
import os
import zlib
from multiprocessing import Pool
import numpy as np

from scenario import Scenario, electorate
from methods import METHODS, select
from strategy import learn, strategy_types
from metrics import election_row, summarize, pareto, worst_margins, electorate_stats


def one_election(args):
    sc, seed, e, rounds, samples, group_size, share, names = args
    rng = np.random.default_rng([seed, e])
    u = electorate(sc, rng)
    m_cand = worst_margins(u)                       # each candidate's worst head-to-head margin
    estats = electorate_stats(u)                    # properties of the electorate, for the report
    # every candidate's total utility and worst margin, for the frontier of what any rule could reach
    estats["U_cands"] = ";".join(f"{x:.6g}" for x in u.sum(axis=0))
    estats["WM_cands"] = ";".join(f"{x:.6g}" for x in m_cand)
    # strategic voters: own random stream, the same voters for every method
    strategic = np.random.default_rng([seed, e, zlib.crc32(b"strategic")]).random(len(u)) < share
    out = []
    for name in names:
        m = METHODS[name]
        out.append({"election": e, "method": name, "ballots": "sincere", **election_row(u, m.honest(u), m_cand), **estats,
                    "sincerity": 1.0, "settled": 1, "compromise": 0.0, "pushover": 0.0, "burial": 0.0})
        # Each method learns with its own random stream, keyed by name, so results stay
        # the same when methods are added or reordered.
        mrng = np.random.default_rng([seed, e, zlib.crc32(name.encode())])
        B, settled, sin = learn(m, u, mrng, rounds, samples, group_cap(group_size, len(u)), strategic)
        out.append({"election": e, "method": name, "ballots": "learned", **election_row(u, m.tally(B), m_cand), **estats,
                    "sincerity": sin, "settled": int(settled), **strategy_types(B, u)})
    return out


OUTPUT_DIR = "output"


def out_path(name):
    """Output prefix or directory: a bare name goes into OUTPUT_DIR (created if needed)."""
    path = name if os.path.dirname(name) else os.path.join(OUTPUT_DIR, name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def group_size_arg(x):
    """--group-size: 0 (whole ranking groups), a whole number of voters, or a share of the electorate
    between 0 and 1 (e.g. 0.05 = 5% of the voters)."""
    v = float(x)
    if v < 0 or (v > 1 and v != int(v)):
        raise argparse.ArgumentTypeError("group size: 0, a whole number of voters, or a share between 0 and 1")
    return int(v) if v == int(v) else v


def group_cap(g, n):
    """Group size cap in voters: a share 0 < g < 1 of n voters (at least 1), else g itself."""
    return max(1, round(g * n)) if 0 < g < 1 else int(g)


def group_text(g):
    g = float(g)
    return "unlimited" if g == 0 else f"{g:.0%} of voters" if g < 1 else f"{int(g)}"


# Run parameters written into every row of the elections CSV
PARAMS = ["voters", "cands", "dims", "zipf", "clusters", "spread", "width", "rounds", "samples", "group_size",
          "strategic", "seed"]


def simulate(p, elections, workers=None, names=None):
    """Run one condition. p: dict with the keys in PARAMS; names: methods (default all).
    Returns rows with p as columns."""
    names = names or list(METHODS)
    sc = Scenario(p["voters"], p["cands"], p["dims"], p["clusters"], p["spread"], p["width"], p["zipf"])
    jobs = [(sc, p["seed"], e, p["rounds"], p["samples"], p["group_size"], p["strategic"], names) for e in range(elections)]
    with Pool(workers) as pool:
        rows = [r for batch in pool.imap(one_election, jobs, chunksize=4) for r in batch]
    meta = {k: p[k] for k in PARAMS}
    return [{**meta, **r} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elections", type=int, default=200)
    ap.add_argument("--voters", type=int, default=99)
    ap.add_argument("--cands", type=int, default=6)
    ap.add_argument("--dims", type=int, default=10)
    ap.add_argument("--zipf", type=int, default=1, choices=[0, 1],
                    help="1: dimension i has scale 1/i (Zipf dimensions, see scenario.py)")
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--width", type=float, default=1.5)
    ap.add_argument("--rounds", type=int, default=20, help="learning rounds")
    ap.add_argument("--samples", type=int, default=10, help="poll samples per round")
    ap.add_argument("--group-size", type=group_size_arg, default=10,
                    help="max voters coordinating as one group: a number of voters (1 = individual), a share of the electorate between 0 and 1 (0.05 = 5%%), or 0 = whole ranking group")
    ap.add_argument("--strategic", type=float, default=0.5,
                    help="share of voters who vote strategically (random, the same for every method)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None, help="run only these methods (start from none)")
    ap.add_argument("--exclude", nargs="+", default=None, help="run all methods except these")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    p = {k: getattr(a, k) for k in PARAMS}
    names = select(a.methods, a.exclude)
    results = simulate(p, a.elections, a.workers, names)

    out = out_path(a.out)
    with open(f"{out}-elections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    summ = {}
    for m in names:
        sinc = [r for r in results if r["method"] == m and r["ballots"] == "sincere"]
        lrn = [r for r in results if r["method"] == m and r["ballots"] == "learned"]
        s, l = summarize(sinc), summarize(lrn)
        summ[m] = {"VSE": s["VSE"], "WME": s["WME"], "VSE_learned": l["VSE"], "WME_learned": l["WME"],
                   "CE": s["CE"], "CE_learned": l["CE"],
                   "sincerity": float(np.mean([r["sincerity"] for r in lrn])),
                   "settled": float(np.mean([r["settled"] for r in lrn]))}
    axes = ["VSE", "WME", "VSE_learned", "WME_learned"]
    front = set(pareto([[summ[m][x] for x in axes] for m in names]))

    cols = ["VSE", "WME", "VSE_learned", "WME_learned", "CE", "CE_learned", "sincerity", "settled"]
    with open(f"{out}-summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols + ["on_front"])
        for i, m in enumerate(names):
            w.writerow([m] + [f"{summ[m][c]:.4f}" for c in cols] + [int(i in front)])

    print(f"{a.elections} elections, {a.voters} voters, {a.cands} candidates, {a.dims} dimensions"
          f"{' (Zipf)' if a.zipf else ''}, {a.clusters} clusters, "
          f"spread {a.spread}, width {a.width}; learning {a.rounds} rounds x {a.samples} poll samples, "
          f"group size {group_text(a.group_size)}, strategic share {a.strategic}; seed {a.seed}")
    print(f"{'method':<26}{'VSE':>7}{'WME':>7}{'VSE*':>7}{'WME*':>7}{'CE':>7}{'CE*':>7}{'sincerity':>11}{'settled':>9}  front")
    for m in sorted(names, key=lambda m: -summ[m]["VSE"]):
        s = summ[m]
        print(f"{m:<26}{s['VSE']*100:7.1f}{s['WME']*100:7.1f}{s['VSE_learned']*100:7.1f}{s['WME_learned']*100:7.1f}"
              f"{s['CE']*100:7.1f}{s['CE_learned']*100:7.1f}"
              f"{s['sincerity']*100:10.1f}%{s['settled']*100:8.0f}%"
              f"  {'*' if names.index(m) in front else ''}")
    print("0 = random ballot, 100 = best candidate; *: with learned ballots; front on VSE, WME, VSE*, WME*")


if __name__ == "__main__":
    main()
