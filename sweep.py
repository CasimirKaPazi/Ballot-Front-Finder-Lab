"""Run a grid of conditions and seeds into one elections CSV, for analyze.py.

Every flag that takes several values spans one dimension of the grid; all combinations run.
With --one-at-a-time, the first value of every flag is the base condition, and each other value
is run with all remaining flags at their base (one parameter varied at a time), per seed.

Usage:
    python sweep.py --one-at-a-time --dims 10 2 5 --clusters 5 3 10 --spread 0.5 0.25 1.0 \
        --width 1.0 0.5 2.0 --out model
    python sweep.py --rounds 1 2 3 4 5 --strategic 0.5 1 --elections 200 --out rounds
    python sweep.py --aggressive 0 0.5 1 --methods STAR "BTR-score" Schulze
    python sweep.py --converge 20 --samples 10 --group-size 1 10 0 --out convergent
    python sweep.py --exclude "Random pair runoff" TRASH

Conditions that differ only in --rounds come from one run (see strategy.respond).

Output:
    <out>-elections.csv   rows from all runs, each carrying its parameters as columns
                          (written run by run, so a long sweep can be inspected while it runs)
"""
import argparse
import csv
import itertools
import time

from run import simulate, PARAMS, out_path, group_size_arg, check_params
from methods import select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elections", type=int, default=200, help="elections per run")
    ap.add_argument("--voters", type=int, nargs="+", default=[99])
    ap.add_argument("--cands", type=int, nargs="+", default=[6])
    ap.add_argument("--dims", type=int, nargs="+", default=[10])
    ap.add_argument("--zipf", type=int, nargs="+", default=[1], choices=[0, 1])
    ap.add_argument("--clusters", type=int, nargs="+", default=[5])
    ap.add_argument("--spread", type=float, nargs="+", default=[0.5])
    ap.add_argument("--width", type=float, nargs="+", default=[1.0])
    ap.add_argument("--rounds", type=int, nargs="+", default=[1])
    ap.add_argument("--samples", type=int, nargs="+", default=[30])
    ap.add_argument("--group-size", type=group_size_arg, nargs="+", default=[10])
    ap.add_argument("--strategic", type=float, nargs="+", default=[0.5])
    ap.add_argument("--max-score", type=int, nargs="+", default=[5])
    ap.add_argument("--exits", type=int, nargs="+", default=[0], choices=[0, 1])
    ap.add_argument("--clones", type=int, nargs="+", default=[0])
    ap.add_argument("--poll-size", type=int, nargs="+", default=[0])
    ap.add_argument("--search", type=int, nargs="+", default=[10])
    ap.add_argument("--poll-error", type=float, nargs="+", default=[2.0])
    ap.add_argument("--aggressive", type=float, nargs="+", default=[0.5])
    ap.add_argument("--converge", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None, help="run only these methods (start from none)")
    ap.add_argument("--exclude", nargs="+", default=None, help="run all methods except these")
    ap.add_argument("--one-at-a-time", action="store_true",
                    help="vary one parameter at a time around the base (the first value of every flag)")
    ap.add_argument("--participation", action="store_true",
                    help="also check participation (no-show paradox) on sincere and learned ballots; slower")
    ap.add_argument("--out", default="sweep")
    a = ap.parse_args()
    names = select(a.methods, a.exclude)
    check_params(a, ap.error)
    # parameters that have no effect in the chosen strategy model give identical conditions
    conv = set(a.converge) - {0}
    if not conv and len(a.group_size) > 1:
        print("note: --group-size only matters with --converge; these conditions will be identical")
    if conv and set(a.converge) == conv:
        idle = [f for f, k in (("--rounds", "rounds"), ("--aggressive", "aggressive"), ("--poll-size", "poll_size"),
                               ("--poll-error", "poll_error"), ("--search", "search")) if len(getattr(a, k)) > 1]
        if idle:
            print(f"note: {', '.join(idle)} only matter for poll rounds, not with --converge; "
                  "these conditions will be identical")

    values = {k: getattr(a, k if k != "seed" else "seeds") for k in PARAMS}
    if a.one_at_a_time:
        base = {k: v[0] for k, v in values.items() if k != "seed"}
        conds = [base] + [{**base, k: x} for k, v in values.items() if k != "seed" for x in v[1:]]
        grid = [{**c, "seed": sd} for c in conds for sd in values["seed"]]
    else:
        grid = [dict(zip(PARAMS, combo)) for combo in itertools.product(*values.values())]
    # Conditions that differ only in the number of poll rounds come from one run (strategy.respond):
    # merge them, keeping the order of first appearance.
    runs = {}
    for p in grid:
        key = tuple(p[k] for k in PARAMS if k != "rounds")
        if key in runs:
            if p["rounds"] not in runs[key]["rounds"]:
                runs[key]["rounds"].append(p["rounds"])
        else:
            runs[key] = {**p, "rounds": [p["rounds"]]}
    grid = list(runs.values())
    print(f"{len(grid)} runs x {a.elections} elections, {len(names)} methods"
          + (f" (poll rounds {', '.join(map(str, values['rounds']))} from one run each)" if len(values["rounds"]) > 1 else ""))
    out = out_path(a.out)
    with open(f"{out}-elections.csv", "w", newline="") as f:
        w = None
        for i, p in enumerate(grid, 1):
            t = time.time()
            rows = simulate(p, a.elections, a.workers, names, a.participation)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
            w.writerows(rows)
            f.flush()
            varying = ", ".join(f"{k}={p[k]}" for k in PARAMS if len(values[k]) > 1)
            print(f"[{i}/{len(grid)}] {varying or 'single run'}: {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
