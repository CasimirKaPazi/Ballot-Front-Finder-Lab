"""Analyse one or more elections CSVs (from run.py or sweep.py): uncertainty, fronts, plots, report.

Usage:
    python analyze.py sweep-elections.csv --out analysis      # files without a directory: looked up in output/
    python analyze.py a-elections.csv b-elections.csv --compare "BTR-score:STAR" "Minimax:Baldwin"

Conditions: rows with identical parameters except the seed form one condition; seeds are
pooled as replications.

Uncertainty: paired bootstrap over elections. All methods share the same electorates, so each
bootstrap replicate redraws whole elections (with replacement) and recomputes every method
on that same draw. Confidence intervals are the 2.5% and 97.5% percentiles.

Axes (higher is better):
    VSE, WME     voter satisfaction and worst-margin efficiency with sincere ballots (see metrics.py);
                 0 = random ballot, 100 = best candidate, as ratios of sums over elections
    VSE*, WME*   the same with learned ballots
    sincerity    of the learned ballots (higher is more honest)
The front is VSE*, WME* and sincerity: what happens with real (strategic) voters, and how honest
they can afford to be. Honest VSE and WME depend more on modelling assumptions; they are reported
alongside, as are CE, CE* (Condorcet efficiency), not used for dominance.

Dominated by: for each method, the methods that are clearly better on at least one front axis and
clearly worse on none. Clearly: the paired 95% bootstrap interval of the difference excludes 0;
smaller differences count as ties (otherwise methods equal on one axis, like all Condorcet methods
with honest ballots, would be separated by noise-sized differences). Each is followed by the
axes on which it is clearly better. This is a pointer for interpretation, not a verdict: the
values themselves are the result.

Outputs in --out (a directory; a bare name goes into output/):
    summary.csv                one row per (condition, method): estimates, intervals, dominated_by
    report.md                  tables per condition, comparisons, plots
    arrows-<c>.png             per condition: honest -> strategic in the VSE/WME plane, all methods together
    shift-<c>.png              per condition: the same shifts as one row per method (VSE and WME panels)
    sincerity-<c>.png          per condition: VSE* and WME* against sincerity
    axes-heatmap.png           VSE*, WME*, sincerity, VSE, WME: methods x conditions (if several conditions)
    chain-<param>.png          per parameter varied around the base: each method's chain of results
"""
import argparse
import csv
import os
from collections import OrderedDict
import numpy as np

import plots
from run import out_path, group_text, strategy_text, cand_text, OUTPUT_DIR, VERSION, PARAMS as RUN_PARAMS
from methods import GROUPS, METHODS, select, group_of

AXES = ["VSE", "WME", "VSE*", "WME*"]
FRONT = ["VSE*", "WME*", "sincerity"]   # the axes dominance is judged on
EXTRA = ["CE", "CE*"]           # reported, not used for dominance
TYPES = ["compromise", "burial", "pushover"]
PARAMS = [k for k in RUN_PARAMS if k != "seed"]      # condition parameters, as run.py writes them
NOM = ["nom_changed", "nom_exits", "nom_clones", "nom_settled"]     # per method and election (sincere rows)
PART = ["noshow", "noshow_voters", "noshow_gain"]   # participation check (run.py --participation), both ballot sets
def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            missing = [k for k in RUN_PARAMS + ["WM_best", "U_rb", "sincerity", "U_cands", "nom_changed", "noshow"]
                       if k not in (reader.fieldnames or [])]
            if missing:
                raise SystemExit(f"{p} lacks {', '.join(missing)}; regenerate it with the current run.py or sweep.py")
            rows.extend(reader)
    if not rows:
        raise SystemExit("no rows in the input CSVs")
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
    A = {k: np.zeros((E, M)) for k in ["num_s", "num_l", "cw_s", "cw_l", "wm_s", "wm_l", "sin"] + TYPES + NOM}
    A.update({k + sfx: np.full((E, M), np.nan) for k in PART for sfx in ("_s", "_l")})
    den = np.zeros(E)          # per election: U_best - U_rb (VSE denominator)
    dwm = np.zeros(E)          # per election: WM_best - WM_rb (WME denominator)
    has = np.zeros(E)
    ele = {k: np.full(E, np.nan) for k in ["cw_is_util", "U_cw", "U_best", "fav_ratio", "U_rb", "WM_rb"]}
    cands = [None] * E           # per election: (total utility, worst margin) of every candidate, for the frontier
    for r in rows:
        i, j = ei[(r["seed"], r["election"])], mi[r["method"]]
        num = float(r["U_method"]) - float(r["U_rb"])
        nwm = float(r["WM"]) - float(r["WM_rb"])
        pcw = float(r["p_cw"]) if r["has_cw"] == "1" else 0.0
        if r["ballots"] == "sincere":
            A["num_s"][i, j], A["cw_s"][i, j], A["wm_s"][i, j] = num, pcw, nwm
            for k in NOM:
                A[k][i, j] = float(r[k])
            for k in PART:
                A[k + "_s"][i, j] = float(r[k])               # nan: participation not checked
            den[i] = float(r["U_best"]) - float(r["U_rb"])
            dwm[i] = float(r["WM_best"]) - float(r["WM_rb"])
            has[i] = float(r["has_cw"])
            for k in ele:
                ele[k][i] = float(r[k] or "nan")              # empty without a Condorcet winner
            if r["U_cands"]:
                cands[i] = (np.array(r["U_cands"].split(";"), float), np.array(r["WM_cands"].split(";"), float))
        else:
            A["num_l"][i, j], A["cw_l"][i, j], A["wm_l"][i, j] = num, pcw, nwm
            A["sin"][i, j] = float(r["sincerity"])
            for t in TYPES:
                A[t][i, j] = float(r[t])
            for k in PART:
                A[k + "_l"][i, j] = float(r[k])
    return methods, A, (den, dwm), has, ele, cands


def axis_values(W, A, dens, has):
    """Axis values for weight rows W (B x E). Returns dict axis -> (B x M)."""
    d, dw = W @ dens[0], W @ dens[1]
    h = W @ has
    return {
        "VSE": (W @ A["num_s"]) / d[:, None],
        "WME": (W @ A["wm_s"]) / dw[:, None],
        "VSE*": (W @ A["num_l"]) / d[:, None],
        "WME*": (W @ A["wm_l"]) / dw[:, None],
        "CE": (W @ A["cw_s"]) / np.where(h > 0, h, np.nan)[:, None],     # no Condorcet winner drawn: undefined
        "CE*": (W @ A["cw_l"]) / np.where(h > 0, h, np.nan)[:, None],
        "sincerity": (W @ A["sin"]) / W.sum(axis=1)[:, None],
        **{t: (W @ A[t]) / W.sum(axis=1)[:, None] for t in TYPES},
    }


def dominance(P_b):
    """P_b: (B x M x K) bootstrap points, higher is better. Differences whose paired 95% interval
    includes 0 count as ties. dom[i, j] is True when j dominates i: clearly better on at least one
    axis and clearly worse on none. Also returns better[i, j, k]: j clearly better than i on axis k."""
    D = P_b[:, None, :, :] - P_b[:, :, None, :]             # D[b, i, j, k] = j minus i
    lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
    better, worse = lo > 0, hi < 0
    dom = better.any(axis=2) & ~worse.any(axis=2)
    np.fill_diagonal(dom, False)
    return dom, better


def frontier(cands, dens, ele, steps=101):
    """The best combinations of VSE and WME any rule could reach in these elections, knowing every
    voter's sincere utilities: for each weight lam, elect in every election the candidate with the
    highest lam * VSE contribution + (1 - lam) * WME contribution. Both axes are ratios of sums with
    fixed denominators, so this traces the exact upper-right boundary of what is reachable (lotteries
    fill in between the points). Returns points (VSE, WME), from best WME to best VSE, or None."""
    if any(c is None for c in cands):
        return None
    du, dw = dens[0].sum(), dens[1].sum()
    pts = []
    for lam in np.linspace(0, 1, steps):
        nu = nw = 0.0
        for (U, W), urb, wrb in zip(cands, ele["U_rb"], ele["WM_rb"]):
            gu, gw = (U - urb) / du, (W - wrb) / dw
            c = np.argmax(lam * gu + (1 - lam) * gw + 1e-12 * gu + 1e-12 * gw)   # break exact ties toward both
            nu, nw = nu + gu[c], nw + gw[c]
        pts.append((nu, nw))
    out = []
    for p in pts:                                       # drop repeats
        if not out or abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return np.array(out)


def electorate_summary(dens, has, ele):
    """Electorate properties of a condition (point estimates), or None for CSVs without them."""
    if np.isnan(ele["fav_ratio"]).all():
        return None
    h = has == 1
    return {"cw": float(h.mean()),
            "cw_util": float(np.nanmean(ele["cw_is_util"][h])) if h.any() else float("nan"),
            # VSE lost by electing the Condorcet winner, as a ratio of sums like VSE itself
            "cw_loss": float((ele["U_best"][h] - ele["U_cw"][h]).sum() / dens[0][h].sum()) if h.any() else float("nan"),
            "fav": float(np.median(ele["fav_ratio"]))}


def frontier_text(F):
    if F is None:
        return ""
    k = int(np.argmax(F[:, 0] + F[:, 1]))
    return (f" Reachable frontier (any rule, knowing the sincere utilities): with the best WME (100) at most VSE "
            f"{F[0, 0] * 100:.1f}; with the best VSE (100) at most WME {F[-1, 1] * 100:.1f}; the most balanced point "
            f"VSE {F[k, 0] * 100:.1f} / WME {F[k, 1] * 100:.1f}.")


def electorate_text(e):
    return (f"Electorates: Condorcet winner in {e['cw']:.1%}; it also has the highest total utility in "
            f"{e['cw_util']:.1%} of those; electing it loses {e['cw_loss']:.1%} of VSE; "
            f"favourite ÷ least liked {e['fav']:.1f} (median voter).")


def analyse_condition(rows, boot, rng):
    methods, A, dens, has, ele, cands = arrays(rows)
    E = len(dens[0])
    W = rng.multinomial(E, np.full(E, 1 / E), size=boot).astype(float)
    point = axis_values(np.ones((1, E)), A, dens, has)
    halves = axis_values(np.stack([np.arange(E) % 2 == 0, np.arange(E) % 2 == 1]).astype(float), A, dens, has)
    bs = axis_values(W, A, dens, has)
    P_b = np.stack([bs[a] for a in FRONT], axis=2)           # B x M x 3
    dom, better = dominance(P_b)
    # each dominating method with the axes on which it is clearly better, most axes first
    dominated_by = [[(methods[j], [FRONT[k] for k in np.flatnonzero(better[i, j])]) for j in np.flatnonzero(dom[i])]
                    for i in range(len(methods))]
    res = {"methods": methods, "elections": E, "with_cw": int(has.sum()), "dominated_by": dominated_by, "boot": bs, "halves": halves, "frontier": frontier(cands, dens, ele),
           "electorate": electorate_summary(dens, has, ele),
           "nomination": {k: A[k].mean(axis=0) for k in NOM},
           "participation": participation_summary(A, W)}
    for a in AXES + EXTRA + ["sincerity"] + TYPES:
        res[a] = point[a][0]
        res[a + "_lo"], res[a + "_hi"] = np.nanpercentile(bs[a], [2.5, 97.5], axis=0)
    return res


def participation_summary(A, W):
    """Per method, sincere (_s) and learned (_l) ballots: share of elections with a no-show failure
    (with a bootstrap 95% interval), share of voters in failing groups, mean largest gain where it
    fails. None when the condition was run without --participation."""
    if np.isnan(A["noshow_s"]).all():
        return None
    out = {}
    for sfx in ("_s", "_l"):
        x = A["noshow" + sfx]
        out["noshow" + sfx] = x.mean(axis=0)
        bs = (W @ x) / W.sum(axis=1)[:, None]
        out["noshow" + sfx + "_lo"], out["noshow" + sfx + "_hi"] = np.percentile(bs, [2.5, 97.5], axis=0)
        out["noshow_voters" + sfx] = A["noshow_voters" + sfx].mean(axis=0)
        g = np.where(x > 0, A["noshow_gain" + sfx], 0.0)
        k = (x > 0).sum(axis=0)
        out["noshow_gain" + sfx] = np.where(k > 0, g.sum(axis=0) / np.maximum(k, 1), np.nan)
    return out


def label(key, varying, first):
    """Name a condition by the parameters in which it differs from the first condition (the base)."""
    p, b = dict(zip(PARAMS, key)), dict(zip(PARAMS, first))
    names = PARAM_NAMES
    diff = [k for k in varying if p[k] != b[k]]
    if not varying:
        return "single condition"
    return ", ".join(f"{names.get(k, k)} {group_text(p[k]) if k == 'group_size' else p[k]}"
                     for k in diff) or "base"


PARAM_NAMES = {"group_size": "group size", "cands": "candidates", "strategic": "strategic share", "max_score": "max score",
               "rounds": "poll rounds", "poll_size": "poll size", "poll_error": "poll error", "converge": "convergent rounds",
               "aggressive": "aggressive share", "start": "first poll", "cand_dispersion": "candidate dispersion",
               "cand_shift": "candidate shift"}
LEARNING_ONLY = {"rounds", "samples", "group_size", "strategic", "poll_size", "poll_error", "search", "converge",
                 "aggressive", "start"}   # same electorates
SAME_ELECTORATES = LEARNING_ONLY | {"max_score", "exits", "clones"}  # same electorates (honest results may differ)


def param_order(k, v):
    """Sort key for a parameter value; group size 0 (whole ranking groups) counts as the largest."""
    if k == "start":
        return ["honest", "previous"].index(v)
    x = float(v)
    return float("inf") if k == "group_size" and x == 0 else x


def sweep_chains(keys, varying):
    """For each varying parameter: the base condition (the first) and every condition that differs
    from it only in that parameter, ordered by the parameter's value. Only parameters with at
    least two such conditions form a chain."""
    base = dict(zip(PARAMS, keys[0]))
    out = {}
    for k in varying:
        members = [key for key in keys
                   if all(v == base[q] for q, v in zip(PARAMS, key) if q != k)]
        if len(members) >= 2:
            out[k] = sorted(members, key=lambda key: param_order(k, dict(zip(PARAMS, key))[k]))
    return out


def ranks(x):
    """Ranks 1..n, ties get their average rank."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        tie = x == v
        r[tie] = r[tie].mean()
    return r


def spearman(x, y):
    """Rank correlation of two value vectors (1 = same order, 0 = unrelated, -1 = reversed)."""
    rx, ry = ranks(np.round(x, 9)), ranks(np.round(y, 9))
    return float(np.corrcoef(rx, ry)[0, 1])


def dom_text(items, sep=", ", limit=None):
    items = sorted(items, key=lambda t: -len(t[1]))
    shown = items[:limit] if limit else items
    text = sep.join(f"{m} ({' '.join(ax)})" for m, ax in shown) or "—"
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
    ap.add_argument("--hide", nargs="*", default=["references", "inverse"],
                    help="methods or groups left out of the scatter plots (they stay in tables and heatmaps); "
                         "default: the references and inverse groups; give --hide with no names to show all")
    ap.add_argument("--show", nargs="+", default=None,
                    help="only these methods or groups in the scatter and chain plots (overrides --hide)")
    ap.add_argument("--errorbars", action="store_true",
                    help="arrow plots: 95%% intervals relative to the average of the methods shown; track plots: "
                         "each point's own interval")
    ap.add_argument("--track", nargs="+", default=None,
                    help="methods (or groups) to follow across all conditions: track-<method>.png each")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out = out_path(a.out)
    os.makedirs(a.out, exist_ok=True)
    # CSVs given without a directory are looked up in the output folder if not found here
    a.csv = [p if os.path.exists(p) or os.path.dirname(p) else os.path.join(OUTPUT_DIR, p) for p in a.csv]
    rng = np.random.default_rng(a.seed)

    if a.show:
        keep = set(select(a.show))
        a.hide = [m for m in METHODS if m not in keep]
    elif a.hide:
        a.hide = select(a.hide)
    rows_all = load(a.csv)
    versions = sorted({r.get("version") or "unknown" for r in rows_all})
    if len(versions) > 1:
        print(f"note: the input mixes simulator versions {', '.join(versions)}")
    conds = conditions(rows_all)
    keys = list(conds)
    varying = [k for i, k in enumerate(PARAMS) if len({key[i] for key in keys}) > 1]
    results = OrderedDict()
    for c, key in enumerate(keys, 1):
        r = analyse_condition(conds[key], a.boot, rng)
        r["label"] = label(key, varying, keys[0])
        r["seeds"] = sorted({row["seed"] for row in conds[key]}, key=int)
        r["tag"] = f"c{c}"
        results[key] = r

    # summary.csv
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = []
        for x in AXES + EXTRA + ["sincerity"] + TYPES:
            cols += [x, x + "_lo", x + "_hi"]
        w.writerow(PARAMS + ["condition", "elections", "method", "group"] + cols + ["dominated_by"])
        for key, r in results.items():
            for j, m in enumerate(r["methods"]):
                w.writerow(list(key) + [r["tag"], r["elections"], m, group_of(m)]
                           + [f"{r[x][j]:.5f}" for x in cols] + [dom_text(r["dominated_by"][j], "; ")])

    # plots
    for key, r in results.items():
        plots.arrows(r, os.path.join(a.out, f"arrows-{r['tag']}.png"), a.hide, a.errorbars)
        plots.shift(r, os.path.join(a.out, f"shift-{r['tag']}.png"), a.hide)
        plots.sincerity(r, os.path.join(a.out, f"sincerity-{r['tag']}.png"), a.hide)
    if len(results) > 1:
        plots.axes_heatmap(list(results.values()), ["VSE*", "WME*", "sincerity", "VSE", "WME"],
                           os.path.join(a.out, "axes-heatmap.png"))
    chains = sweep_chains(keys, varying)
    for k, chain_keys in chains.items():
        vals = [dict(zip(PARAMS, key))[k] for key in chain_keys]
        shown_vals = [group_text(v) if k == "group_size" else v for v in vals]
        plots.chain([results[key] for key in chain_keys], PARAM_NAMES.get(k, k), shown_vals,
                    os.path.join(a.out, f"chain-{k}.png"), a.hide, k in LEARNING_ONLY, GROUPS,
                    k in SAME_ELECTORATES)
    tracked = []
    if a.track and len(results) > 1:
        conds_p = [(results[key], dict(zip(PARAMS, key))) for key in keys]
        same = all(k in SAME_ELECTORATES for k in varying)
        for m in select(a.track):
            fn = "track-" + "".join(ch if ch.isalnum() else "_" for ch in m) + ".png"
            if plots.track(conds_p, m, varying, PARAM_NAMES, os.path.join(a.out, fn), same, a.errorbars):
                tracked.append((m, fn))

    # report.md
    L = ["# Voting Pareto Lab — analysis", ""]
    L += [f"Input: {', '.join(os.path.basename(p) for p in a.csv)} (simulator version {', '.join(versions)}"
          + ("; mixed versions" if len(versions) > 1 else "") + f"; analysis version {VERSION}). "
          f"Bootstrap: {a.boot} paired replicates over elections. Brackets are 95% intervals.", ""]
    L += ["How to read this:", "",
          "- The values are the result. The **front** is VSE*, WME* and sincerity: the result with strategic voters, "
          "and how honest they can afford to be. Honest VSE and WME depend more on modelling assumptions and are "
          "shown for context. **Dominated by** lists the methods that are clearly better on at least one of the three "
          "front axes and clearly worse on none, with the axes on which they are better in brackets. Sincerity "
          "differences are measured very precisely, so at equal VSE* and WME* the more honest method dominates. Clearly means the paired 95% interval of the difference excludes 0; smaller differences count "
          "as ties. It is a pointer for interpretation, relative to the methods included, not a verdict.",
          "- The intervals in the tables are for each value on its own. They are wide because elections differ "
          "a lot from each other. Comparisons between methods are much more precise, because all methods face the "
          "same elections: use the paired differences for those, not overlapping intervals.",
          "- Two methods whose paired difference interval includes 0 are tied at this sample size; their order "
          "in the table is not a finding.",
          "- **VSE** and **WME** are efficiencies with 0 = random ballot and 100 = the best candidate of each "
          "election (values above 100 cannot be reached). VSE measures total utility; WME the winner's worst "
          "head-to-head margin by sincere preferences (a refined Condorcet measure; every election counts; Minimax "
          "reaches 100 with honest ballots by construction). CE (Condorcet efficiency, elections with a Condorcet "
          "winner only, elected or not) is shown for comparison and not used for dominance."]
    if a.errorbars:
        L += ["- **Error bars** in the arrow plots are 95% intervals relative to the average of the methods shown: "
              "the variation all methods share (some elections are harder for everyone) is removed, so clearly "
              "separated bars suggest a real difference (an approximation of the paired differences). In track "
              "plots they are each point's own interval."]
    L += ["- The strategic axes (VSE*, WME*) give the result after the poll rounds: voters see polls of the previous round "
          "(the method's counts only) and answer them all at once (README, Strategic voting). Under convergent learning "
          "(--converge) they describe a long-run state instead and depend on the group size.",
          "- **Compromise, burial, push-over** are shares of voters whose learned ballot reverses a real preference, "
          "by kind (see strategy.strategy_types): compromise raises a liked second choice above the favourite, "
          "push-over raises a disliked candidate to the top, burial keeps the favourite on top and reverses a "
          "preference below it.", ""]
    if len(results) > 1:
        L += ["## Across conditions", "", "![axes](axes-heatmap.png)", ""]
        if chains:
            L += ["**Sweeps:** for each parameter varied around the base, the methods' learned values as a chain "
                  "of arrows in the order of the parameter's values (light to dark). Parameters that only affect "
                  "learning start each chain at the honest point; others show the honest values as a dashed chain.",
                  ""] + [f"![sweep over {k}](chain-{k}.png)\n" for k in chains]
        if tracked:
            L += ["**Tracked methods:** each method across all conditions, in the VSE*/WME* plane (learned ballots; "
                  "honest values open). With two varied parameters, lines follow one and colour shows the other.", ""]
            L += [f"![{m}]({fn})\n" for m, fn in tracked]
        # rank stability: does each condition order the methods like the base (first) condition?
        base = list(results.values())[0]
        shown = [m for m in base["methods"] if m not in a.hide]
        cols = AXES + ["sincerity"]
        L += ["**Rank stability:** rank correlation (Spearman) of the methods' values with those in the base "
              f"condition ({base['tag']}), per axis; 1 = same order, 0 = unrelated. Methods hidden in the plots "
              "are left out, since their extreme values would inflate the correlation. The first row is a noise "
              "reference: the agreement expected from sampling noise alone between two runs of the base condition "
              "(its even and odd elections correlated, then scaled up to full size with the Spearman-Brown "
              "formula). A condition near or above that row orders the methods as the base does; only values "
              "clearly below it show a real change.", "",
              "| Condition | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        ib = [base["methods"].index(m) for m in shown]

        def full_size(rh):     # Spearman-Brown: agreement expected between two full-size samples
            return 2 * rh / (1 + rh) if rh > -1 else rh
        L.append(f"| noise reference: {base['tag']} vs itself | "
                 + " | ".join(f"{full_size(spearman(base['halves'][x][0][ib], base['halves'][x][1][ib])):.2f}"
                              for x in cols) + " |")
        for r in list(results.values())[1:]:
            common = [m for m in shown if m in r["methods"]]
            if len(common) < 3:
                continue
            ib = [base["methods"].index(m) for m in common]
            ic = [r["methods"].index(m) for m in common]
            L.append(f"| {r['tag']}: {r['label']} | "
                     + " | ".join(f"{spearman(base[x][ib], r[x][ic]):.2f}" for x in cols) + " |")
        L.append("")
        if all(r["electorate"] for r in results.values()):
            L += ["**Electorates** (properties of the generated electorates, independent of the methods): how often "
                  "a Condorcet winner exists, how often it also has the highest total utility, how much VSE electing "
                  "it loses (ratio of sums, as VSE), and how steep preferences are (favourite ÷ least liked, median "
                  "voter). Differences between methods on VSE mostly come from the third column.", "",
                  "| Condition | Condorcet winner | … with highest total utility | VSE lost electing it | "
                  "favourite ÷ least liked |", "|---|---|---|---|---|"]
            for r in results.values():
                e = r["electorate"]
                L.append(f"| {r['tag']}: {r['label']} | {e['cw']:.1%} | {e['cw_util']:.1%} | {e['cw_loss']:.1%} "
                         f"| {e['fav']:.1f} |")
            L.append("")
    for key, r in results.items():
        p = dict(zip(PARAMS, key))
        L += [f"## {r['tag']}: {r['label']}", "",
              f"{r['elections']} elections (seeds {', '.join(r['seeds'])}), {r['with_cw']} with a Condorcet winner. "
              f"{p['voters']} voters, {p['cands']} candidates, {p['dims']} dimensions"
              f"{' (Zipf)' if p['zipf'] == '1' else ''}, {p['clusters']} clusters, spread {p['spread']}, "
              f"width {p['width']}{cand_text(p['cand_dispersion'], p['cand_shift'])}; {strategy_text(p, p['voters'])}, "
              f"strategic share {p['strategic']}, scores 0..{p['max_score']}"
              + (f"; nomination: exits {p['exits']}, clones {p['clones']}" if p['exits'] != '0' or p['clones'] != '0' else "")
              + "." + (" " + electorate_text(r["electorate"]) if r["electorate"] else "")
              + frontier_text(r["frontier"]),
              "",
              "**Front** (dominated by none on VSE*, WME*, sincerity): "
              + ", ".join(r["methods"][j] for j in np.argsort(-r["VSE*"], kind="stable") if not r["dominated_by"][j])
              + ".", "",
              "| Method | Group | VSE* | WME* | Sincerity | VSE | WME | CE | CE* | Compromise | Burial | Push-over | Dominated by |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for j in np.argsort(-r["VSE*"], kind="stable"):
            L.append(f"| {r['methods'][j]} | {group_of(r['methods'][j])} | "
                     + " | ".join(fmt(r[x][j], r[x + '_lo'][j], r[x + '_hi'][j])
                                  for x in ["VSE*", "WME*", "sincerity", "VSE", "WME"] + EXTRA)
                     + " | " + " | ".join(f"{r[t][j] * 100:.1f}%" for t in TYPES)
                     + f" | {dom_text(r['dominated_by'][j], limit=3)} |")
        L += [""]
        if p["exits"] != "0" or p["clones"] != "0":
            nm = r["nomination"]
            L += ["**Nomination** (with sincere ballots; per method, averaged over elections): how often the "
                  "honest result changed, how many candidates were out of the race and how many clones in it "
                  "at the end, and how often the candidates reached a field where nobody wanted to move "
                  "(settled). VSE and WME above are judged on the original field.", "",
                  "| Method | Result changed | Exits | Clones | Settled |", "|---|---|---|---|---|"]
            for j in np.argsort(-nm["nom_changed"], kind="stable"):
                L.append(f"| {r['methods'][j]} | {nm['nom_changed'][j]:.1%} | {nm['nom_exits'][j]:.2f} "
                         f"| {nm['nom_clones'][j]:.2f} | {nm['nom_settled'][j]:.0%} |")
            L += [""]
        pt = r["participation"]
        if pt is not None:
            L += ["**Participation** (no-show paradox, see participation.py): share of elections in which some "
                  "voters (1, half or all of a group with the same ranking and the same ballot) get a better "
                  "result by staying home while everyone else votes as before; the share of voters in such groups; "
                  "and, where it happens, the largest gain per absent voter in % of that voter's utility range. "
                  "Honest: everyone votes sincerely. Strategic: with the learned ballots.", "",
                  "| Method | Elections (honest) | Voters | Gain | Elections (strategic) | Voters | Gain |",
                  "|---|---|---|---|---|---|---|"]
            g = lambda x: "—" if np.isnan(x) else f"{x:.1%}"
            for j in np.argsort(-pt["noshow_s"], kind="stable"):
                L.append(f"| {r['methods'][j]} | {fmt(pt['noshow_s'][j], pt['noshow_s_lo'][j], pt['noshow_s_hi'][j])}% "
                         f"| {pt['noshow_voters_s'][j]:.1%} | {g(pt['noshow_gain_s'][j])} "
                         f"| {fmt(pt['noshow_l'][j], pt['noshow_l_lo'][j], pt['noshow_l_hi'][j])}% "
                         f"| {pt['noshow_voters_l'][j]:.1%} | {g(pt['noshow_gain_l'][j])} |")
            L += [""]
        L += [f"![strategy arrows](arrows-{r['tag']}.png)", "", f"![strategy shifts](shift-{r['tag']}.png)", "",
              f"![sincerity](sincerity-{r['tag']}.png)", ""]
        if a.compare:
            L += ["Paired differences (first minus second), percentage points:", "",
                  "| Pair | " + " | ".join(AXES + EXTRA + ["sincerity"]) + " |", "|---|" + "---|" * 7]
            for pair in a.compare:
                m1, m2 = pair.split(":")
                if m1 not in r["methods"] or m2 not in r["methods"]:
                    continue
                i, j = r["methods"].index(m1), r["methods"].index(m2)
                cells = []
                for x in AXES + EXTRA + ["sincerity"]:
                    d = (r["boot"][x][:, i] - r["boot"][x][:, j]) * 100
                    pt = (r[x][i] - r[x][j]) * 100
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    tie = " (tie)" if lo <= 0 <= hi else ""
                    cells.append(f"{pt:+.1f} [{lo:+.1f}, {hi:+.1f}]{tie}")
                L.append(f"| {m1} − {m2} | " + " | ".join(cells) + " |")
            L.append("")
    if a.show:
        L += [f"Scatter and chain plots show only {', '.join(a.show)}; all methods are in the tables and heatmaps.", ""]
    elif a.hide:
        L += [f"Scatter and chain plots leave out {', '.join(a.hide)}; they are in all tables and heatmaps.", ""]
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(L))
    print(f"{len(results)} condition(s); wrote {a.out}/summary.csv, report.md and plots")


if __name__ == "__main__":
    main()
