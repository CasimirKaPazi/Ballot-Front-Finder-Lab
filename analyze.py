"""Analyse one or more elections CSVs (from run.py or sweep.py): uncertainty, fronts, plots, report.

Usage:
    python analyze.py sweep-elections.csv --out analysis
    python analyze.py a-elections.csv b-elections.csv --compare "BTR-score:STAR" "Minimax:Baldwin"

Conditions: rows with identical parameters except the seed form one condition; seeds are
pooled as replications.

Uncertainty: paired bootstrap over elections. All methods share the same electorates, so each
bootstrap replicate redraws whole elections (with replacement) and recomputes every method
on that same draw. Confidence intervals are the 2.5% and 97.5% percentiles.

Axes (higher is better):
    PE, CE       proportional and Condorcet efficiency with sincere ballots
    PE*, CE*     the same with learned ballots
Dissonance (lower is better) is reported alongside, not used for the front.

Dominated by: for each method, the methods that are at least as good on all four axes and better
on at least one, judged on the point estimates. Each is followed by the share of bootstrap
replicates in which that dominance holds. This is a pointer for interpretation, not a verdict:
the values themselves are the result.

Outputs in --out (a directory):
    summary.csv                one row per (condition, method): estimates, intervals, dominated_by
    report.md                  tables per condition, comparisons, plots
    arrows-<c>.png             per condition: honest -> strategic in the PE/CE plane, all methods together
    shift-<c>.png              per condition: the same shifts as one row per method (PE and CE panels)
    dissonance-<c>.png         per condition: PE* and CE* against dissonance
    axes-heatmap.png           all four axes, methods x conditions (if several conditions)
"""
import argparse
import csv
import os
from collections import OrderedDict
import numpy as np

import plots

AXES = ["PE", "CE", "PE*", "CE*"]
TYPES = ["compromise", "burial", "pushover"]
PARAMS = ["voters", "cands", "dims", "clusters", "spread", "width", "rounds", "samples", "group_size"]


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows or "seed" not in rows[0]:
        raise SystemExit("CSV has no run parameters; regenerate it with the current run.py or sweep.py")
    return rows


def conditions(rows):
    """Group rows by parameters (seed excluded). Returns {key: rows} in first-seen order."""
    out = OrderedDict()
    for r in rows:
        out.setdefault(tuple(r[k] for k in PARAMS), []).append(r)
    return out


def arrays(rows):
    """Per (election, method) arrays for one condition. Elections are keyed by (seed, election)."""
    methods = list(OrderedDict.fromkeys(r["method"] for r in rows))
    eids = list(OrderedDict.fromkeys((r["seed"], r["election"]) for r in rows))
    ei = {e: i for i, e in enumerate(eids)}
    mi = {m: j for j, m in enumerate(methods)}
    E, M = len(eids), len(methods)
    A = {k: np.zeros((E, M)) for k in ["num_s", "num_l", "cw_s", "cw_l", "dis"] + TYPES}
    den = np.zeros(E)
    has = np.zeros(E)
    for r in rows:
        i, j = ei[(r["seed"], r["election"])], mi[r["method"]]
        num = float(r["U_method"]) - float(r["U_tau1"])
        pcw = float(r["p_cw"]) if r["has_cw"] == "1" else 0.0
        if r["ballots"] == "sincere":
            A["num_s"][i, j], A["cw_s"][i, j] = num, pcw
            den[i] = float(r["U_best"]) - float(r["U_tau1"])
            has[i] = float(r["has_cw"])
        else:
            A["num_l"][i, j], A["cw_l"][i, j] = num, pcw
            A["dis"][i, j] = float(r["dissonance"])
            for t in TYPES:
                A[t][i, j] = float(r.get(t) or "nan")      # older CSVs have no strategy types
    return methods, A, den, has


def axis_values(W, A, den, has):
    """Axis values for weight rows W (B x E). Returns dict axis -> (B x M)."""
    d = W @ den
    h = W @ has
    return {
        "PE": (W @ A["num_s"]) / d[:, None],
        "CE": (W @ A["cw_s"]) / h[:, None],
        "PE*": (W @ A["num_l"]) / d[:, None],
        "CE*": (W @ A["cw_l"]) / h[:, None],
        "dissonance": (W @ A["dis"]) / W.sum(axis=1)[:, None],
        **{t: (W @ A[t]) / W.sum(axis=1)[:, None] for t in TYPES},
    }


def dominance(P):
    """P: (M x K) points, higher is better. dom[i, j] is True when j dominates i:
    at least as good on every axis and better on at least one."""
    ge = (P[None, :, :] >= P[:, None, :]).all(axis=2)
    gt = (P[None, :, :] > P[:, None, :]).any(axis=2)
    dom = ge & gt
    np.fill_diagonal(dom, False)
    return dom


def analyse_condition(rows, boot, rng):
    methods, A, den, has = arrays(rows)
    E = len(den)
    W = rng.multinomial(E, np.full(E, 1 / E), size=boot).astype(float)
    point = axis_values(np.ones((1, E)), A, den, has)
    bs = axis_values(W, A, den, has)
    P_b = np.stack([bs[a] for a in AXES], axis=2)            # B x M x 4
    share = np.mean([dominance(P_b[b]) for b in range(boot)], axis=0)
    dom = dominance(np.stack([point[a][0] for a in AXES], axis=1))
    dominated_by = [[(methods[j], share[i, j]) for j in np.flatnonzero(dom[i])] for i in range(len(methods))]
    res = {"methods": methods, "elections": E, "with_cw": int(has.sum()), "dominated_by": dominated_by, "boot": bs}
    for a in AXES + ["dissonance"] + TYPES:
        res[a] = point[a][0]
        res[a + "_lo"], res[a + "_hi"] = np.percentile(bs[a], [2.5, 97.5], axis=0)
    return res


def label(key, varying):
    p = dict(zip(PARAMS, key))
    names = {"group_size": "group size", "cands": "candidates"}
    return ", ".join(f"{names.get(k, k)} {p[k] if not (k == 'group_size' and p[k] == '0') else 'unlimited'}"
                     for k in varying) or "single condition"


def dom_text(items, sep=", ", limit=None):
    items = sorted(items, key=lambda t: -t[1])
    shown = items[:limit] if limit else items
    text = sep.join(f"{m} ({s:.2f})" for m, s in shown) or "—"
    return text + (f" +{len(items) - len(shown)} more" if len(items) > len(shown) else "")


def fmt(x, lo, hi, pct=True):
    s = 100 if pct else 1
    return f"{x*s:.1f} [{lo*s:.1f}–{hi*s:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="elections CSV files")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--compare", nargs="*", default=[], help='pairs "A:B" for paired differences')
    ap.add_argument("--hide", nargs="*", default=["Random ballot", "Random pair runoff", "Anti-plurality", "TRASH"],
                    help="methods left out of the scatter plots (they stay in tables and heatmaps); "
                         "give --hide with no names to show all")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    conds = conditions(load(a.csv))
    keys = list(conds)
    varying = [k for i, k in enumerate(PARAMS) if len({key[i] for key in keys}) > 1]
    results = OrderedDict()
    for c, key in enumerate(keys, 1):
        r = analyse_condition(conds[key], a.boot, rng)
        r["label"] = label(key, varying)
        r["seeds"] = sorted({row["seed"] for row in conds[key]}, key=int)
        r["tag"] = f"c{c}"
        results[key] = r

    # summary.csv
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = []
        for x in AXES + ["dissonance"] + TYPES:
            cols += [x, x + "_lo", x + "_hi"]
        w.writerow(PARAMS + ["condition", "elections", "method"] + cols + ["dominated_by"])
        for key, r in results.items():
            for j, m in enumerate(r["methods"]):
                w.writerow(list(key) + [r["tag"], r["elections"], m]
                           + [f"{r[x][j]:.5f}" for x in cols] + [dom_text(r["dominated_by"][j], "; ")])

    # plots
    for key, r in results.items():
        plots.arrows(r, os.path.join(a.out, f"arrows-{r['tag']}.png"), a.hide)
        plots.shift(r, os.path.join(a.out, f"shift-{r['tag']}.png"), a.hide)
        plots.dissonance(r, os.path.join(a.out, f"dissonance-{r['tag']}.png"), a.hide)
    if len(results) > 1:
        plots.axes_heatmap(list(results.values()), AXES, os.path.join(a.out, "axes-heatmap.png"))

    # report.md
    L = ["# Voting Pareto Lab — analysis", ""]
    L += [f"Input: {', '.join(os.path.basename(p) for p in a.csv)}. Bootstrap: {a.boot} paired replicates "
          "over elections. Brackets are 95% intervals.", ""]
    L += ["How to read this:", "",
          "- The values are the result. **Dominated by** lists the methods that are at least as good on all four "
          "axes (PE, CE, PE*, CE*) and better on one, with the share of bootstrap replicates in which that holds. "
          "It is a pointer for interpretation, relative to the methods included, not a verdict.",
          "- The intervals in the tables are for each value on its own. They are wide because elections differ "
          "a lot from each other. Comparisons between methods are much more precise, because all methods face the "
          "same elections: use the paired differences for those, not overlapping intervals.",
          "- Two methods whose paired difference interval includes 0 are tied at this sample size; their order "
          "in the table is not a finding.",
          "- The strategic axes (PE*, CE*) depend on the group size: see the README section on coordination.",
          "- **Compromise, burial, push-over** are shares of voters whose learned ballot reverses a real preference, "
          "by kind (see strategy.strategy_types): compromise raises a liked second choice above the favourite, "
          "push-over raises a disliked candidate to the top, burial keeps the favourite on top and reverses a "
          "preference below it.", ""]
    if len(results) > 1:
        L += ["## Across conditions", "", "![axes](axes-heatmap.png)", ""]
    for key, r in results.items():
        p = dict(zip(PARAMS, key))
        L += [f"## {r['tag']}: {r['label']}", "",
              f"{r['elections']} elections (seeds {', '.join(r['seeds'])}), {r['with_cw']} with a Condorcet winner. "
              f"{p['voters']} voters, {p['cands']} candidates, {p['clusters']} clusters, spread {p['spread']}, "
              f"width {p['width']}; learning {p['rounds']} rounds × {p['samples']} poll samples, "
              f"group size {p['group_size'] if p['group_size'] != '0' else 'unlimited'}.", "",
              "| Method | PE | CE | PE* | CE* | Dissonance | Compromise | Burial | Push-over | Dominated by |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for j in np.argsort(-r["PE"], kind="stable"):
            L.append(f"| {r['methods'][j]} | "
                     + " | ".join(fmt(r[x][j], r[x + '_lo'][j], r[x + '_hi'][j]) for x in AXES + ["dissonance"])
                     + " | " + " | ".join(f"{r[t][j] * 100:.1f}%" for t in TYPES)
                     + f" | {dom_text(r['dominated_by'][j], limit=3)} |")
        L += ["", f"![strategy arrows](arrows-{r['tag']}.png)", "", f"![strategy shifts](shift-{r['tag']}.png)", "",
              f"![dissonance](dissonance-{r['tag']}.png)", ""]
        if a.compare:
            L += ["Paired differences (first minus second), percentage points:", "",
                  "| Pair | " + " | ".join(AXES + ["dissonance"]) + " |", "|---|" + "---|" * 5]
            for pair in a.compare:
                m1, m2 = pair.split(":")
                if m1 not in r["methods"] or m2 not in r["methods"]:
                    continue
                i, j = r["methods"].index(m1), r["methods"].index(m2)
                cells = []
                for x in AXES + ["dissonance"]:
                    d = (r["boot"][x][:, i] - r["boot"][x][:, j]) * 100
                    pt = (r[x][i] - r[x][j]) * 100
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    tie = " (tie)" if lo <= 0 <= hi else ""
                    cells.append(f"{pt:+.1f} [{lo:+.1f}, {hi:+.1f}]{tie}")
                L.append(f"| {m1} − {m2} | " + " | ".join(cells) + " |")
            L.append("")
    if a.hide:
        L += [f"Scatter plots leave out {', '.join(a.hide)} (far from the rest, they compress the scale); "
              "they are included in all tables and heatmaps.", ""]
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(L))
    print(f"{len(results)} condition(s); wrote {a.out}/summary.csv, report.md and plots")


if __name__ == "__main__":
    main()
