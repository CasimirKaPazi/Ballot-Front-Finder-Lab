"""Regression check: does the simulator still give exactly the same numbers?

    python regression.py            compare a few small fixed runs with regression.csv
    python regression.py --update   write regression.csv anew (after a change that is meant to change results)

A refactor (moving code, renaming internals, splitting files) should not change a single number.
This script runs a few tiny conditions (a few elections each, every method including the
references, the Inverse-* methods and the add-ons) with fixed seeds, and compares every column
of every row with the stored reference, regression.csv. It takes about a minute.

    all equal                  nothing changed: the refactor kept the results
    differences                which methods and columns changed, with the first few values;
                               either a bug, or an intended change: then run --update and commit
                               regression.csv together with the change
    new / missing methods      listed, but not counted as differences: a method added (e.g. a new
                               add-on) or removed. --update takes them into the reference.
    new columns                listed, not counted: a column added to the results files (the values
                               already there are still compared). --update takes them in.

Numbers count as equal within a relative tolerance of 1e-9 (rounding in the last digits can differ
between numpy versions or machines); text columns must match exactly. The "version" column is not
compared.
"""
import argparse
import csv
import os
import sys

import numpy as np

import run
from methods import METHODS

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "regression.csv")
TOL = 1e-9
IGNORE = {"version"}

# The run parameters every condition starts from (small and fast), then what each condition changes.
BASE = {"voters": 25, "cands": 4, "model": "clusters", "dims": 3, "zipf": 1, "clusters": 3, "spread": 0.5,
        "ks_alpha": 1.5, "ks_cut": 0.2, "width": 1.0,
        "cand_dispersion": 1.0, "cand_shift": 0.0, "rounds": 2, "samples": 3, "group_size": 10,
        "strategic": 0.5, "aggressive": 0.5, "lazy": 0.0, "max_score": 5, "exits": 0, "clones": 0,
        "poll_size": 0, "poll_error": 2.0, "search": 4, "converge": 0, "start": "honest", "seed": 7}
CONDITIONS = [
    ("default", {}),                                          # poll rounds, full ballots
    ("five-cands", {"cands": 5, "seed": 8}),                  # odd candidate count, other seed
    ("lazy-score10", {"lazy": 0.4, "max_score": 10}),         # truncation / laziness and the score range
    ("converge", {"converge": 3, "group_size": 0.2}),         # convergent learning
    ("nomination", {"exits": 1, "clones": 1, "rounds": 1}),   # strategic nomination
    ("previous", {"start": "previous"}),                      # first poll from a previous election
    ("ks", {"model": "ks", "exits": 1, "clones": 1}),         # Quinn's hierarchical model, with nomination (weights)
]
ELECTIONS = 2


def simulate_all(workers=None):
    """All conditions, every method: rows with 'condition' as the first column."""
    names = list(METHODS)
    rows = []
    for label, change in CONDITIONS:
        p = {**BASE, **change}
        rows += [{"condition": label, **r} for r in run.simulate(p, ELECTIONS, workers, names, label=f"{label}: ")]
    return rows


def key(r):
    return (r["condition"], str(r["rounds"]), str(r["election"]), r["method"], r["ballots"])


def same(a, b):
    """Two values from the CSV (as text) count as equal: numbers within TOL, else the same text."""
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    if np.isnan(x) and np.isnan(y):
        return True
    return abs(x - y) <= TOL * max(1.0, abs(x), abs(y))


def as_text(rows):
    """Rows as the CSV writer would store them, so new and stored values compare alike."""
    return [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows]


def write(rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(REFERENCE, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def compare(new, old):
    """Print the differences; returns the number of differing values."""
    old_by = {key(r): r for r in old}
    new_by = {key(r): r for r in new}
    old_methods = {r["method"] for r in old}
    new_methods = {r["method"] for r in new}
    added, removed = sorted(new_methods - old_methods), sorted(old_methods - new_methods)
    if added:
        print(f"new methods (not in the reference, not compared): {', '.join(added)}")
    if removed:
        print(f"methods in the reference but not run any more: {', '.join(removed)}")
    diffs = {}                                     # (method, column) -> list of (row key, old, new)
    new_cols = set()
    for k, r in new_by.items():
        if r["method"] in added:
            continue
        o = old_by.get(k)
        if o is None:
            diffs.setdefault((r["method"], "(row missing in the reference)"), []).append((k, "", ""))
            continue
        for c in set(r) | set(o):
            if c in IGNORE:
                continue
            a, b = o.get(c), r.get(c)
            if a is None:                          # a column added since the reference: not compared
                new_cols.add(c)
            elif b is None:
                diffs.setdefault((r["method"], c), []).append((k, a, b))
            elif not same(a, b):
                diffs.setdefault((r["method"], c), []).append((k, a, b))
    for k, o in old_by.items():
        if o["method"] not in removed and k not in new_by:
            diffs.setdefault((o["method"], "(row not produced any more)"), []).append((k, "", ""))
    if new_cols:
        print(f"new columns (not in the reference, not compared): {', '.join(sorted(new_cols))}")
    n = sum(len(v) for v in diffs.values())
    if not diffs:
        return 0
    nm = len({m for m, _ in diffs})
    print(f"{n} differing values in {nm} method{'' if nm == 1 else 's'}:")
    for (m, c), v in sorted(diffs.items()):
        cond, rounds, e, _, ballots = v[0][0]
        extra = f" (and {len(v) - 1} more)" if len(v) > 1 else ""
        if v[0][1] is None or v[0][2] is None:
            what = "column only in the " + ("new run" if v[0][1] is None else "reference")
        else:
            what = f"{v[0][1]} -> {v[0][2]}"
        print(f"  {m:<24} {c:<22} {cond}, election {e}, {ballots}: {what}{extra}")
    return n


def main():
    ap = argparse.ArgumentParser(description="Compare a few small fixed runs with the stored reference.")
    ap.add_argument("--update", action="store_true", help="write regression.csv from this run")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    rows = as_text(simulate_all(a.workers))
    if a.update:
        write(rows)
        print(f"wrote {os.path.basename(REFERENCE)}: {len(rows)} rows, {len({r['method'] for r in rows})} methods")
        return
    if not os.path.exists(REFERENCE):
        sys.exit("no regression.csv yet: run  python regression.py --update  once, on a version you trust")
    with open(REFERENCE, newline="") as f:
        old = list(csv.DictReader(f))
    n = compare(rows, old)
    if n:
        sys.exit("differences found (see above). If they are intended: python regression.py --update")
    print(f"all equal: {len(rows)} rows, {len({r['method'] for r in rows})} methods")


if __name__ == "__main__":
    main()
