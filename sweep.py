"""Run a grid of conditions and seeds, one elections CSV per run, for analyze.py.

Every flag that takes several values spans one dimension of the grid; all combinations run.
With --one-at-a-time, the first value of every flag is the base condition, and each other value
is run with all remaining flags at their base (one parameter varied at a time), per seed.

Usage:
    python sweep.py --one-at-a-time --noise-strength 5 3 7 --dims 6 3 10 --width 0.7 0.5 1.0 --out model
    python sweep.py --model clusters --cluster-count 20 5 --out clusters
    python sweep.py --rounds 1 2 3 4 5 --strategic 0.5 1 --elections 200 --out rounds
    python sweep.py --aggressive 0 0.5 1 --methods STAR "BTR-score" Schulze
    python sweep.py --converge 20 --samples 10 --group-size 1 10 0 --out convergent
    python sweep.py --exclude "Random pair runoff" TRASH

Conditions that differ only in --rounds come from one run (see strategy.respond).

Output, in one folder per sweep (--out; a name without a directory goes into output/; default
sweep-<date>-<time>). `python analyze.py <folder>` puts its analysis into <folder>/analysis/.
    runs/NN-<varied parameters>.csv   one elections CSV per run (a combination of the varied
                    parameters; poll rounds of one run share its file), each row carrying its
                    parameters as columns; files can be moved, mixed with other runs' files or
                    analysed alone
    command.txt     the command, simulator version and start time (a resumed run adds its own)
Every election is saved as soon as it is done; after an interruption the same command with
--resume continues where it stopped (see docs/running.md, Progress, saving and resuming).
"""
import argparse
import csv
import itertools
import os
import time

from models import MODELS
from run import (simulate, PARAMS, run_dir, write_command, group_size_arg, check_params, run_key, valid_combination,
                 resume_state, CsvSink, interrupted)
from methods import run_selection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elections", type=int, default=200, help="elections per run")
    ap.add_argument("--voters", type=int, nargs="+", default=[99])
    ap.add_argument("--cands", type=int, nargs="+", default=[6])
    ap.add_argument("--model", nargs="+", default=["noise"], choices=list(MODELS))
    ap.add_argument("--dims", type=int, nargs="+", default=[6])
    ap.add_argument("--cluster-zipf", dest="zipf", type=int, nargs="+", default=[0], choices=[0, 1])
    ap.add_argument("--cluster-count", dest="clusters", metavar="CLUSTER_COUNT", type=int, nargs="+", default=[20])
    ap.add_argument("--cluster-spread", dest="spread", metavar="CLUSTER_SPREAD", type=float, nargs="+", default=[0.5])
    ap.add_argument("--cluster-salience", dest="salience", nargs="+", default=["cluster"], choices=["none", "cluster"])
    ap.add_argument("--cluster-salience-exponent", dest="salience_exponent", metavar="CLUSTER_SALIENCE_EXPONENT", type=float, nargs="+", default=[0.5])
    ap.add_argument("--cluster-fixed-variance", dest="fixed_variance", type=int, nargs="+", default=[1], choices=[0, 1])
    ap.add_argument("--field-kappa", type=float, nargs="+", default=[1.5])
    ap.add_argument("--field-salience", type=float, nargs="+", default=[0.0])
    ap.add_argument("--noise-strength", type=float, nargs="+", default=[5.0])
    ap.add_argument("--ks-alpha", type=float, nargs="+", default=[1.5])
    ap.add_argument("--ks-cut", type=float, nargs="+", default=[0.2])
    ap.add_argument("--ks-shared", type=int, nargs="+", default=[0], choices=[0, 1])
    ap.add_argument("--width", type=float, nargs="+", default=[0.7])
    ap.add_argument("--cand-dispersion", type=float, nargs="+", default=[1.0])
    ap.add_argument("--cand-shift", type=float, nargs="+", default=[0.0])
    ap.add_argument("--rounds", type=int, nargs="+", default=[3])
    ap.add_argument("--samples", type=int, nargs="+", default=[10])
    ap.add_argument("--group-size", type=group_size_arg, nargs="+", default=[10])
    ap.add_argument("--strategic", type=float, nargs="+", default=[0.5])
    ap.add_argument("--adaptive", type=float, nargs="+", default=[0.0])
    ap.add_argument("--max-score", type=int, nargs="+", default=[5])
    ap.add_argument("--exits", type=int, nargs="+", default=[0], choices=[0, 1])
    ap.add_argument("--clones", type=int, nargs="+", default=[0])
    ap.add_argument("--poll-size", type=int, nargs="+", default=[0])
    ap.add_argument("--search", type=int, nargs="+", default=[10])
    ap.add_argument("--start", nargs="+", default=["honest"], choices=["honest", "previous"])
    ap.add_argument("--poll-error", type=float, nargs="+", default=[2.0])
    ap.add_argument("--aggressive", type=float, nargs="+", default=[0.5])
    ap.add_argument("--lazy", type=float, nargs="+", default=[0.0])
    ap.add_argument("--converge", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None, help="run only these methods or groups (start from none); \"all\" for every method")
    ap.add_argument("--exclude", nargs="+", default=None, help="leave out these methods or groups (from the standard set, or from --methods)")
    ap.add_argument("--one-at-a-time", action="store_true",
                    help="vary one parameter at a time around the base (the first value of every flag)")
    ap.add_argument("--participation", action="store_true",
                    help="also check participation (no-show paradox) on sincere and learned ballots; slower")
    ap.add_argument("--manipulation", action="store_true",
                    help="also test coalition manipulation from honest ballots, one- and two-sided (manipulation.py)")
    ap.add_argument("--out", default=None,
                    help="folder for this sweep (in output/ unless it has a directory); default sweep-<date>-<time>")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing sweep in --out")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted sweep in --out (same command): finished elections are kept")
    a = ap.parse_args()
    names = run_selection(a.methods, a.exclude)
    check_params(a, ap.error)
    out = run_dir(a.out, "sweep", a.overwrite, a.resume)
    write_command(out, a.resume)
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
    skipped = [p for p in grid if not valid_combination(p)]
    if skipped:
        grid = [p for p in grid if valid_combination(p)]
        print(f"skipped {len(skipped)} combination(s) that cannot run (--strategic + --adaptive above 1, "
              "or --adaptive with --converge)")
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
    runs_dir = os.path.join(out, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    keys = [run_key(p, PARAMS) for p in grid]
    files = [os.path.join(runs_dir, run_file(i, p, values)) for i, p in enumerate(grid, 1)]
    try:
        for i, (p, key, path) in enumerate(zip(grid, keys, files), 1):
            done, header = (resume_state(path, names, {key: p["rounds"]}) if a.resume and os.path.exists(path)
                            else ({}, None))
            skip = done.get(key, set())
            if len(skip) >= a.elections:
                print(f"[{i}/{len(grid)}] done before")
                continue
            t = time.time()
            varying = ", ".join(f"{k}={p[k]}" for k in PARAMS if len(values[k]) > 1)
            sink = CsvSink(path, header)
            try:
                simulate(p, a.elections, a.workers, names, a.participation, sink=sink, skip=skip,
                         label=f"[{i}/{len(grid)}] ", manipulation=a.manipulation)
            finally:
                sink.close()
            print(f"[{i}/{len(grid)}] {varying or 'single run'}: {time.time() - t:.0f}s")
    except KeyboardInterrupt:
        n = 0
        for path in files:
            if os.path.exists(path):
                with open(path, newline="") as f:
                    n += len({r["election"] for r in csv.DictReader(f)})
        interrupted(out, n)
    print(f"wrote {out}/runs/ ({len(grid)} file(s), one per run) and command.txt; "
          f"analyse with: python analyze.py {out}")


def run_file(i, p, values):
    """File name of one run of the sweep: its number and the parameters that vary, e.g.
    03-lazy=0.5-seed=2.csv (single run: 01.csv)."""
    parts = [f"{k}={'-'.join(map(str, p[k])) if isinstance(p[k], list) else p[k]}"
             for k in PARAMS if len(values[k]) > 1 and k != "rounds"]
    name = "-".join([f"{i:02d}"] + parts)
    return "".join(ch if ch.isalnum() or ch in "=-._" else "_" for ch in name) + ".csv"


if __name__ == "__main__":
    main()
