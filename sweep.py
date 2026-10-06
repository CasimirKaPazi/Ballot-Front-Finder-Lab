"""Run a grid of conditions and seeds into one elections CSV, for analyze.py.

Every flag that takes several values spans one dimension of the grid; all combinations run.
With --one-at-a-time, the first value of every flag is the base condition, and each other value
is run with all remaining flags at their base (one parameter varied at a time), per seed.

Usage:
    python sweep.py --one-at-a-time --dims 2 1 3 4 --clusters 3 1 2 5 --spread 0.5 0.25 1.0 \
        --width 1.0 0.5 2.0 --group-size 10 --out model
    python sweep.py --group-size 1 10 0 --cands 4 5 --seeds 1 2 --elections 200 --out sweep
    python sweep.py --methods STAR "BTR-score" SICT --group-size 1 0
    python sweep.py --exclude "Random ballot" "Random pair runoff"

Output:
    <out>-elections.csv   rows from all runs, each carrying its parameters as columns
                          (written run by run, so a long sweep can be inspected while it runs)
"""
import argparse
import csv
import itertools
import time

from run import simulate, PARAMS
from methods import select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elections", type=int, default=200, help="elections per run")
    ap.add_argument("--voters", type=int, nargs="+", default=[99])
    ap.add_argument("--cands", type=int, nargs="+", default=[4])
    ap.add_argument("--dims", type=int, nargs="+", default=[2])
    ap.add_argument("--zipf", type=int, nargs="+", default=[0], choices=[0, 1])
    ap.add_argument("--clusters", type=int, nargs="+", default=[3])
    ap.add_argument("--spread", type=float, nargs="+", default=[0.5])
    ap.add_argument("--width", type=float, nargs="+", default=[1.0])
    ap.add_argument("--rounds", type=int, nargs="+", default=[20])
    ap.add_argument("--samples", type=int, nargs="+", default=[10])
    ap.add_argument("--group-size", type=int, nargs="+", default=[0])
    ap.add_argument("--strategic", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None, help="run only these methods (start from none)")
    ap.add_argument("--exclude", nargs="+", default=None, help="run all methods except these")
    ap.add_argument("--one-at-a-time", action="store_true",
                    help="vary one parameter at a time around the base (the first value of every flag)")
    ap.add_argument("--out", default="sweep")
    a = ap.parse_args()
    names = select(a.methods, a.exclude)

    values = {k: getattr(a, k if k != "seed" else "seeds") for k in PARAMS}
    if a.one_at_a_time:
        base = {k: v[0] for k, v in values.items() if k != "seed"}
        conds = [base] + [{**base, k: x} for k, v in values.items() if k != "seed" for x in v[1:]]
        grid = [{**c, "seed": sd} for c in conds for sd in values["seed"]]
    else:
        grid = [dict(zip(PARAMS, combo)) for combo in itertools.product(*values.values())]
    print(f"{len(grid)} runs x {a.elections} elections, {len(names)} methods")
    with open(f"{a.out}-elections.csv", "w", newline="") as f:
        w = None
        for i, p in enumerate(grid, 1):
            t = time.time()
            rows = simulate(p, a.elections, a.workers, names)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
            w.writerows(rows)
            f.flush()
            varying = ", ".join(f"{k}={p[k]}" for k in PARAMS if len(values[k]) > 1)
            print(f"[{i}/{len(grid)}] {varying or 'single run'}: {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
