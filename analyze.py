"""Analyse one or more elections CSVs (from run.py or sweep.py): uncertainty, fronts, plots, report.

Usage:
    python analyze.py rounds                    # a run or sweep folder (looked up in output/) -> rounds/analysis/
    python analyze.py rounds mine --out compare --compare "BTR-score:STAR"   # several inputs: --out needed

Inputs are run or sweep folders (their elections.csv) or elections CSV files. With one input the
analysis goes into an analysis/ folder next to it, so results and their analysis stay together;
with several, --out names the folder.

Conditions: rows with identical parameters except the seed form one condition; seeds are
pooled as replications. Runs of different methods on the same elections can be combined, e.g. a
new method with the example output: see combine() and docs/running.md, Combining runs.

Uncertainty: paired bootstrap over elections. All methods share the same electorates, so each
bootstrap replicate redraws whole elections (with replacement) and recomputes every method
on that same draw. Confidence intervals are the 2.5% and 97.5% percentiles.

Axes (higher is better):
    VSE, WME     voter satisfaction and worst-margin efficiency with sincere ballots (see metrics.py);
                 0 = random ballot, 100 = best candidate, as ratios of sums over elections
    VSE*, WME*   the same with learned ballots
    sincerity    of the learned ballots, within each method's ballot format (strategy.sincerity;
                 higher is more honest); fidelity (how well the ballots reproduce the utility
                 differences, learned and honest) is reported alongside
The front is VSE*, WME* and sincerity: what happens with real (strategic) voters, and how honest
they can afford to be. Honest VSE and WME depend more on modelling assumptions; they are reported
alongside, as are CE, CE* (Condorcet efficiency), not used for dominance.

Dominated by: for each method, the methods that are clearly better on at least one front axis and
clearly worse on none. Clearly: the paired 95% bootstrap interval of the difference excludes 0;
smaller differences count as ties (otherwise methods equal on one axis, like all Condorcet methods
with honest ballots, would be separated by noise-sized differences). Each is followed by the
axes on which it is clearly better. This is a pointer for interpretation, not a verdict: the
values themselves are the result.

Outputs in the analysis folder (--out: a bare name goes into output/):
    summary.csv                one row per (condition, method): estimates, intervals, dominated_by
    report.md                  tables per condition, comparisons, plots
    arrows-<c>.png             per condition: honest -> strategic in the VSE/WME plane, all methods together
    shift-<c>.png              per condition: the same shifts as one row per method (VSE and WME panels)
    sincerity-<c>.png          per condition: VSE* and WME* against sincerity
    fidelity-<c>.png           per condition: honest -> learned ballots in the fidelity / sincerity plane
    tradeoff-<c>.png           per condition: ternary map of the Pareto front (best method per weighting)
    axes-heatmap.png           VSE*, WME*, sincerity, VSE, WME: methods x conditions (if several conditions)
    chain-<param>.png          per parameter varied around the base: each method's chain of results
"""
import argparse
import csv
import os
import re
import warnings
from collections import OrderedDict
import numpy as np

import plots
from metrics import pareto
from run import out_path, write_command, group_text, strategy_text, cand_text, OUTPUT_DIR, VERSION, PARAMS as RUN_PARAMS
from models import model_text, MODELS
from methods import GROUPS, METHODS, select, group_of, sincerity_defined, current_name

AXES = ["VSE", "WME", "VSE*", "WME*"]
FRONT = ["VSE*", "WME*", "sincerity"]   # the axes dominance is judged on
EXTRA = ["CE", "CE*", "Low", "Low*"]   # reported, not used for dominance (Low: winner below the median candidate)
TYPES = ["compromise", "burial", "pushover"]
PARAMS = [k for k in RUN_PARAMS if k != "seed"]      # condition parameters, as run.py writes them
NOM = ["nom_changed", "nom_exits", "nom_clones", "nom_settled"]     # per method and election (sincere rows)
PART = ["noshow", "noshow_voters", "noshow_gain"]   # participation check (run.py --participation), both ballot sets
MANIP = ["manip1", "manip2", "um1", "um2"]          # manipulation test (run.py --manipulation), sincere rows; um = U_manip - U_rb
def input_paths(p):
    """The elections CSVs of one input: a run folder (its elections.csv), a sweep folder (every CSV in
    its runs/ folder, one per run; also an older sweep's elections.csv), or a CSV file. A bare name
    not found here is looked up in output/. Returns (files, folder for the default analysis)."""
    if not os.path.exists(p) and not os.path.dirname(p):
        p = os.path.join(OUTPUT_DIR, p)
    if os.path.isdir(p):
        files = []
        if os.path.exists(os.path.join(p, "elections.csv")):
            files.append(os.path.join(p, "elections.csv"))
        runs = os.path.join(p, "runs")
        if os.path.isdir(runs):
            files += sorted(os.path.join(runs, f) for f in os.listdir(runs) if f.endswith(".csv"))
        if not files:
            raise SystemExit(f"no elections CSV in {p} (elections.csv or runs/*.csv)")
        return files, p
    if not os.path.exists(p):
        raise SystemExit(f"no elections CSV at {p}")
    return [p], os.path.dirname(p) or "."


def source_name(p):
    """How an input is named in notes and the report: its folder for elections.csv, folder/file for a
    sweep's run file, else the file."""
    d, f = os.path.split(os.path.abspath(p))
    if f == "elections.csv":
        return os.path.basename(d)
    if os.path.basename(d) == "runs":
        return f"{os.path.basename(os.path.dirname(d))}/{f[:-4]}"
    return f


# Older results files. When run.py gets a new column (a parameter or a measure), it goes into
# ADDED_COLUMNS, so files written before it can still be read and combined with newer ones:
#   a text value   what older runs implied (a parameter at the value it had before it could be set):
#                  filled in, and older rows count as that condition
#   None           not measured in older runs: left blank (tables show it as missing), and the report
#                  says which inputs lack it
# The note says when and what. Columns in REQUIRED_COLUMNS (or a run parameter not listed here) must
# be present; a file without them is too old and has to be regenerated.
ADDED_COLUMNS = {
    "model": ("clusters", "2026-10-14, electorate models: older runs used the clusters model"),
    "ks_alpha": ("1.5", "2026-10-14, parameters of the ks model (not used by older runs)"),
    "ks_cut": ("0.2", "2026-10-14, parameters of the ks model (not used by older runs)"),
    "ks_shared": ("0", "2026-10-14, parameters of the ks model (not used by older runs)"),
    "salience": ("none", "2026-10-15, salience per cluster: older runs weighed every dimension alike"),
    "salience_exponent": ("1.0", "2026-10-15, salience per cluster (not used by older runs)"),
    "field_kappa": ("1.5", "2026-10-15, parameters of the field model (not used by older runs)"),
    "field_salience": ("0.0", "2026-10-15, parameters of the field model (not used by older runs)"),
    "adaptive": ("0.0", "2026-10-16, adaptive voters: older runs had none"),
    "adapted": (None, "2026-10-16, share of adaptive voters who turned strategic"),
    "noise_strength": ("3.0", "2026-10-15, parameter of the noise model (not used by older runs)"),
    "fixed_variance": ("0", "2026-10-15, fixed total variance: older runs did not scale positions"),
    "lazy": ("0.0", "2026-10-13, laziness: older runs had full ballots"),
    "fidelity": (None, "2026-10-12, fidelity, with the current sincerity measure"),
    "bcw": (None, "2026-10-12, Condorcet winner in the ballots"),
    "manip1": (None, "2026-10-17, manipulation test (run.py --manipulation)"),
    "manip2": (None, "2026-10-17, manipulation test (run.py --manipulation)"),
    "U_manip1": (None, "2026-10-17, manipulation test (run.py --manipulation)"),
    "U_manip2": (None, "2026-10-17, manipulation test (run.py --manipulation)"),
    "p_bcw": (None, "2026-10-12, Condorcet winner in the ballots"),
}
REQUIRED_COLUMNS = ["WM_best", "U_rb", "sincerity", "U_cands", "nom_changed", "noshow"]
LACKING = {}           # input -> added columns it lacks (filled as in ADDED_COLUMNS)
OLD_SINCERITY = []     # inputs written before the current sincerity measure (no fidelity column)
RENAMED_SEEN = {}      # old method name -> current name, for names renamed since the input was written


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            have = reader.fieldnames or []
            fill = {k: v if v is not None else "" for k, (v, _) in ADDED_COLUMNS.items() if k not in have}
            missing = [k for k in RUN_PARAMS + REQUIRED_COLUMNS if k not in have and k not in fill]
            if missing:
                raise SystemExit(f"{p} lacks {', '.join(missing)}; regenerate it with the current run.py or sweep.py")
            if fill:
                LACKING[source_name(p)] = list(fill)
            if "fidelity" not in have:          # the sincerity column then holds the earlier measure
                OLD_SINCERITY.append(source_name(p))
            for r in reader:
                r.update(fill)                   # columns added later (ADDED_COLUMNS)
                new = current_name(r["method"])  # methods renamed since (methods.RENAMED)
                if new != r["method"]:
                    RENAMED_SEEN[r["method"]] = new
                    r["method"] = new
                r["_file"] = source_name(p)
                rows.append(r)
    if not rows:
        raise SystemExit("no rows in the input CSVs")
    return rows


def lacking_lines():
    """One line per input that lacks added columns: which were filled with an implied value, which left blank."""
    out = []
    for name, cols in LACKING.items():
        groups = {}                                       # note -> its columns, in ADDED_COLUMNS order
        for k in cols:
            groups.setdefault(ADDED_COLUMNS[k][1], []).append(k)
        parts = []
        for note, ks in groups.items():
            v = ADDED_COLUMNS[ks[0]][0]
            parts.append(f"{', '.join(ks)} {'read as ' + v if v is not None else 'blank'} ({note})")
        out.append(f"{name} lacks columns added later: {'; '.join(parts)}")
    return out


ELECTORATE_COLS = ["U_best", "U_rb", "WM_best", "WM_rb"]   # the same in every row of an election
RESULT_COLS = ["U_method", "WM", "sincerity"]               # compared when a method's rows are replaced


def _same(r1, r2, cols):
    def f(x):
        return float(x or "nan")
    return all((np.isnan(f(r1[k])) and np.isnan(f(r2[k])))
               or abs(f(r1[k]) - f(r2[k])) <= 1e-9 * max(1.0, abs(f(r1[k]))) for k in cols)


def combine(rows):
    """Combine rows from several files into one consistent set (see docs/running.md, Combining runs).

    An election is (condition, seed, election number). Every method's result in an election depends
    only on these (run.py gives each method the same random streams), so runs of different methods
    with the same parameters and seed can be pooled, and a new method compared with an existing run
    without rerunning the others.
    - Elections: rows for the same election must describe the same electorate; otherwise the files
      come from incompatible simulator versions and analysis stops.
    - Duplicates: a method in a later file replaces the same method from earlier files in the
      elections both have (later files win: give the newest run last).
    - Coverage: within a condition, only the elections every method has are kept, so all
      comparisons stay paired.
    Returns (rows, notes, cover): notes on replaced rows; cover[condition key] says which elections
    were left out, for conditions where not all methods had the same ones."""
    seen, elect = OrderedDict(), {}
    replaced = OrderedDict()        # (method, old file, new file) -> [elections, elections with other results]
    for r in rows:
        cond = tuple(r[k] for k in PARAMS)
        ek = cond + (r["seed"], r["election"])
        first = elect.setdefault(ek, r)
        if first is not r and not _same(first, r, ELECTORATE_COLS):
            raise SystemExit(f"election {r['election']} (seed {r['seed']}) differs between {first['_file']} and "
                             f"{r['_file']} although the parameters are the same: the files come from incompatible "
                             "simulator versions and cannot be combined")
        key = ek + (r["method"], r["ballots"])
        old = seen.get(key)
        if old is not None:
            c = replaced.setdefault((r["method"], old["_file"], r["_file"]), [set(), set()])
            c[0].add(ek)
            if not _same(old, r, RESULT_COLS):
                c[1].add(ek)
        seen[key] = r                 # an OrderedDict keeps the method's first position
    notes = [f"{m}: {len(c[0])} election(s) of {f_old} replaced by {f_new}, "
             + ("same results" if not c[1] else f"different results in {len(c[1])} (the method or simulator "
                                               f"changed between the runs; {f_new} is used)")
             for (m, f_old, f_new), c in replaced.items()]
    rows = list(seen.values())
    # coverage: per condition, keep the elections that every method has
    by_cond = OrderedDict()
    for r in rows:
        by_cond.setdefault(tuple(r[k] for k in PARAMS), []).append(r)
    out, cover = [], {}
    for cond, rs in by_cond.items():
        have = OrderedDict()
        for r in rs:
            have.setdefault(r["method"], set()).add((r["seed"], r["election"]))
        common = set.intersection(*have.values())
        if not common:
            raise SystemExit("some methods have no elections in common with the others under the same parameters "
                             f"({', '.join(have)}): to compare with an existing run, use the same parameters and "
                             "seed (and at most as many elections)")
        alle = set.union(*have.values())
        if len(common) < len(alle):
            least = min(len(e) for e in have.values())
            short = [m for m, e in have.items() if len(e) == least]
            cover[cond] = (f"{len(common)} of {len(alle)} elections used, the ones all methods have; "
                           f"fewest: {', '.join(short[:5])}{' …' if len(short) > 5 else ''}")
        out += [r for r in rs if (r["seed"], r["election"]) in common]
    return out, notes, cover


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
    A.update({k: np.full((E, M), np.nan) for k in ["fid_s", "fid_l", "pbcw_s", "pbcw_l", "adapted", "us", "ul"]})   # absent in older CSVs
    A.update({k: np.full((E, M), "", dtype=object) for k in ["bcw_s", "bcw_l"]})    # ballot Condorcet winner
    A.update({k + sfx: np.full((E, M), np.nan) for k in PART for sfx in ("_s", "_l")})
    A.update({k: np.full((E, M), np.nan) for k in MANIP})
    den = np.zeros(E)          # per election: U_best - U_rb (VSE denominator)
    dwm = np.zeros(E)          # per election: WM_best - WM_rb (WME denominator)
    has = np.zeros(E)
    ele = {k: np.full(E, np.nan) for k in ["cw_is_util", "U_cw", "U_best", "fav_ratio", "U_rb", "WM_rb", "U_rand"]}
    cands = [None] * E           # per election: (total utility, worst margin) of every candidate, for the frontier
    for r in rows:
        i, j = ei[(r["seed"], r["election"])], mi[r["method"]]
        num = float(r["U_method"]) - float(r["U_rb"])
        nwm = float(r["WM"]) - float(r["WM_rb"])
        pcw = float(r["p_cw"]) if r["has_cw"] == "1" else 0.0
        if r["ballots"] == "sincere":
            A["num_s"][i, j], A["cw_s"][i, j], A["wm_s"][i, j] = num, pcw, nwm
            A["us"][i, j] = float(r["U_method"])
            for k in NOM:
                A[k][i, j] = float(r[k])
            A["bcw_s"][i, j], A["pbcw_s"][i, j] = r.get("bcw", ""), float(r.get("p_bcw") or "nan")
            if sincerity_defined(r["method"]):            # blank for TRASH and Inverse-* (also in older CSVs)
                A["fid_s"][i, j] = float(r.get("fidelity") or "nan")
            for k in PART:
                A[k + "_s"][i, j] = float(r[k])               # nan: participation not checked
            if r.get("manip1"):                               # blank: manipulation not tested
                A["manip1"][i, j], A["manip2"][i, j] = float(r["manip1"]), float(r["manip2"])
                A["um1"][i, j] = float(r["U_manip1"]) - float(r["U_rb"])
                A["um2"][i, j] = float(r["U_manip2"]) - float(r["U_rb"])
            den[i] = float(r["U_best"]) - float(r["U_rb"])
            dwm[i] = float(r["WM_best"]) - float(r["WM_rb"])
            has[i] = float(r["has_cw"])
            for k in ele:
                ele[k][i] = float(r[k] or "nan")              # empty without a Condorcet winner
            if r["U_cands"]:
                cands[i] = (np.array(r["U_cands"].split(";"), float), np.array(r["WM_cands"].split(";"), float))
        else:
            A["num_l"][i, j], A["cw_l"][i, j], A["wm_l"][i, j] = num, pcw, nwm
            A["ul"][i, j] = float(r["U_method"])
            A["bcw_l"][i, j], A["pbcw_l"][i, j] = r.get("bcw", ""), float(r.get("p_bcw") or "nan")
            if sincerity_defined(r["method"]):
                A["sin"][i, j] = float(r["sincerity"] or "nan")
                A["fid_l"][i, j] = float(r.get("fidelity") or "nan")
            else:
                A["sin"][i, j] = np.nan
            for t in TYPES:
                A[t][i, j] = float(r[t])
            A["adapted"][i, j] = float(r.get("adapted") or "nan")   # blank without adaptive voters
            for k in PART:
                A[k + "_l"][i, j] = float(r[k])
    # tails: the winner (expected utility of the winner lottery) below the median candidate's total utility;
    # nan where the CSV has no candidate utilities (runs from before the frontier columns)
    med = np.array([np.median(c[0]) if c is not None else np.nan for c in cands])
    for k, src in (("low_s", "us"), ("low_l", "ul")):
        A[k] = np.where(np.isnan(med)[:, None], np.nan, (A[src] < med[:, None] - 1e-9 * np.abs(med)[:, None]).astype(float))
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
        **{x: (W @ np.nan_to_num(A[k])) / np.where(W @ ~np.isnan(A[k]) > 0, W @ ~np.isnan(A[k]), np.nan)
           for x, k in (("Low", "low_s"), ("Low*", "low_l"))},
        "sincerity": (W @ A["sin"]) / W.sum(axis=1)[:, None],
        "fidelity": (W @ A["fid_l"]) / W.sum(axis=1)[:, None],
        "fidelity_honest": (W @ A["fid_s"]) / W.sum(axis=1)[:, None],
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


def region(cands, dens, ele, steps=720):
    """The whole region of (VSE, WME) any rule could reach in these elections, knowing every voter's
    sincere utilities: the convex polygon of all combinations of one candidate per election (lotteries
    fill it in). For each direction (cos t, sin t) the polygon's support point elects, in every election,
    the candidate furthest in that direction; going round the circle gives its vertices in order
    (counter-clockwise). The upper-right part is the frontier (see frontier); the rest shows that
    near one end of the trade-off the other axis is pinned too: always electing the candidate with the
    best worst margin gives exactly one VSE (up to ties), not anything below it. Returns an array of
    vertices (VSE, WME) as fractions, or None."""
    if any(c is None for c in cands):
        return None
    du, dw = dens[0].sum(), dens[1].sum()
    t = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    D = np.stack([np.cos(t), np.sin(t)], axis=1)                     # K x 2
    pts = np.zeros((steps, 2))
    for (U, W), urb, wrb in zip(cands, ele["U_rb"], ele["WM_rb"]):
        G = np.stack([(U - urb) / du, (W - wrb) / dw], axis=1)       # C x 2
        c = np.argmax(G @ D.T, axis=0)
        pts += G[c]
    out = []
    for p in pts:                                       # drop repeats
        if not out or np.abs(p - out[-1]).max() > 1e-9:
            out.append(p)
    if len(out) > 1 and np.abs(out[0] - out[-1]).max() <= 1e-9:
        out.pop()
    return np.array(out)


def references(cands, dens, ele):
    """Reference points in the VSE/WME plane, as fractions: the random ballot (0, 0 by definition)
    and a random candidate (the zero of classical VSE; its WME is the average worst margin of the
    candidates). None when the candidates' values are missing."""
    if any(c is None for c in cands):
        return None
    du, dw = dens[0].sum(), dens[1].sum()
    v = sum(U.mean() - urb for (U, W), urb in zip(cands, ele["U_rb"])) / du
    w = sum(W.mean() - wrb for (U, W), wrb in zip(cands, ele["WM_rb"])) / dw
    return {"random ballot": (0.0, 0.0), "random candidate": (float(v), float(w))}


def region_span(R):
    """Ranges the region allows at its ends: VSE (min, max) where WME is best, WME (min, max) where
    VSE is best."""
    top = R[R[:, 1] >= R[:, 1].max() - 1e-9]
    right = R[R[:, 0] >= R[:, 0].max() - 1e-9]
    return (top[:, 0].min(), top[:, 0].max()), (right[:, 1].min(), right[:, 1].max())


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


def frontier_text(F, R=None, refs=None):
    if F is None:
        return ""
    rc = refs and refs.get("random candidate")
    rc_text = f" A random candidate (the zero of classical VSE) gives VSE {rc[0] * 100:.1f} / WME {rc[1] * 100:.1f}." if rc else ""
    k = int(np.argmax(F[:, 0] + F[:, 1]))

    def rng_(a, b, name):         # the other axis at one end of the region: one value, or a range (ties)
        return f"{name} {a * 100:.1f}" if b - a < 0.0005 else f"{name} {a * 100:.1f} to {b * 100:.1f}"
    if R is not None:
        (v0, v1), (w0, w1) = region_span(R)
        ends = (f"with the best WME (100) {rng_(v0, v1, 'VSE')}; with the best VSE (100) {rng_(w0, w1, 'WME')} "
                "(nothing lower is possible there either)")
    else:
        ends = (f"with the best WME (100) at most VSE {F[0, 0] * 100:.1f}; with the best VSE (100) at most WME "
                f"{F[-1, 1] * 100:.1f}")
    return (f" Reachable region (any rule, knowing the sincere utilities): {ends}; the most balanced point "
            f"VSE {F[k, 0] * 100:.1f} / WME {F[k, 1] * 100:.1f}." + rc_text)


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
           "region": region(cands, dens, ele),
           "references": references(cands, dens, ele),
           "electorate": electorate_summary(dens, has, ele),
           "nomination": {k: A[k].mean(axis=0) for k in NOM},
           "adapted": nanmean0(A["adapted"]),
           "participation": participation_summary(A, W),
           "manipulation": manipulation_summary(A, W, dens[0]),
           "ballot_cw": ballot_cw_summary(A, has),
           # Pareto front on point estimates; undefined sincerity (TRASH, Inverse-*) counts as worst
           "pareto": set(pareto(np.nan_to_num(np.stack([point[a][0] for a in FRONT], axis=1), nan=-np.inf)))}
    for a in AXES + EXTRA + ["sincerity", "fidelity", "fidelity_honest"] + TYPES:
        res[a] = point[a][0]
        with np.errstate(all="ignore"), warnings.catch_warnings():     # all-nan columns (e.g. CE without a
            warnings.simplefilter("ignore", RuntimeWarning)               # Condorcet winner) give nan, no warning
            res[a + "_lo"], res[a + "_hi"] = np.nanpercentile(bs[a], [2.5, 97.5], axis=0)
    return res


def ballot_cw_summary(A, has):
    """Per method (ordered ballots only, else None): how the Condorcet winner in the cast ballots
    relates to the real one (metrics.ballot_condorcet), for honest (_s) and learned (_l) ballots.
    Shares of the elections with a real Condorcet winner: same, false, none; of those without one:
    spurious. ce_ballots: how often the method elects the ballots' Condorcet winner, over the
    elections whose ballots have one (what ballot data would report as Condorcet efficiency)."""
    out = {}
    real = has == 1
    for j in range(A["bcw_l"].shape[1]):
        if not any(A["bcw_l"][:, j]):
            out[j] = None
            continue
        d = {}
        for sfx in ("_s", "_l"):
            k = A["bcw" + sfx][:, j]
            for c in ("same", "false", "none"):
                d[c + sfx] = float(np.mean(k[real] == c)) if real.any() else float("nan")
            d["spurious" + sfx] = float(np.mean(k[~real] == "spurious")) if (~real).any() else float("nan")
            p = A["pbcw" + sfx][:, j]
            d["ce_ballots" + sfx] = float(np.nanmean(p)) if (~np.isnan(p)).any() else float("nan")
        out[j] = d
    return out


def manipulation_summary(A, W, den):
    """Per method: share of elections in which a coalition can manipulate one-sided (manip1) and still
    against the defence (manip2), with bootstrap 95% intervals, and the VSE of the recorded outcomes
    (ratio of sums, as VSE). None when the condition was run without --manipulation; methods without
    values (from a run without it) are nan."""
    if np.isnan(A["manip1"]).all():
        return None
    out = {}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in ("manip1", "manip2"):
            x = A[k]
            out[k] = np.nanmean(x, axis=0)
            bs = (W @ np.nan_to_num(x)) / W.sum(axis=1)[:, None]
            out[k + "_lo"], out[k + "_hi"] = np.percentile(bs, [2.5, 97.5], axis=0)
        for k, v in (("um1", "vse1"), ("um2", "vse2")):
            x = A[k]
            has = ~np.isnan(x)
            out[v] = np.where(has.any(axis=0), np.nansum(x, axis=0) / np.maximum((den[:, None] * has).sum(axis=0), 1e-12), np.nan)
            bs = (W @ np.nan_to_num(x)) / np.maximum(W @ (den[:, None] * has), 1e-12)
            out[v + "_lo"], out[v + "_hi"] = np.percentile(bs, [2.5, 97.5], axis=0)
    return out


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
               "cand_shift": "candidate shift", "lazy": "laziness", "model": "electorate model",
               "ks_alpha": "ks faction alpha", "ks_cut": "ks cut",
               "ks_shared": "ks shared factions", "salience": "salience", "salience_exponent": "salience exponent", "fixed_variance": "fixed total variance",
               "field_kappa": "field saturation", "field_salience": "salience field",
               "noise_strength": "noise strength", "adaptive": "adaptive share"}
LEARNING_ONLY = {"rounds", "samples", "group_size", "strategic", "adaptive", "poll_size", "poll_error", "search", "converge",
                 "aggressive", "start"}   # same electorates
SAME_ELECTORATES = LEARNING_ONLY | {"max_score", "exits", "clones", "lazy"}  # same electorates (honest results may differ)


def param_order(k, v):
    """Sort key for a parameter value: numbers by value (group size 0, whole ranking groups, counts as
    the largest); words in their listed order (start: honest, previous; model: as in models.MODELS),
    other words alphabetically, after the numbers."""
    words = {"start": ["honest", "previous"], "model": list(MODELS), "salience": ["none", "cluster"]}
    try:
        x = float(v)
    except ValueError:
        known = words.get(k, [])
        return (1, known.index(v) if v in known else len(known), str(v))
    return (0, float("inf") if k == "group_size" and x == 0 else x, "")


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
    ok = ~(np.isnan(x) | np.isnan(y))          # undefined values (e.g. sincerity of TRASH) are left out
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(np.round(x, 9)), ranks(np.round(y, 9))
    if rx.std() == 0 or ry.std() == 0:      # all methods equal on this axis: no order to compare
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cor(x):
    return "—" if np.isnan(x) else f"{x:.2f}"


def dom_text(items, sep=", ", limit=None):
    items = sorted(items, key=lambda t: -len(t[1]))
    shown = items[:limit] if limit else items
    text = sep.join(f"{m} ({' '.join(ax)})" for m, ax in shown) or "—"
    return text + (f" +{len(items) - len(shown)} more" if len(items) > len(shown) else "")


def fmt(x, lo, hi, pct=True):
    if np.isnan(x):
        return "—"          # undefined (sincerity and fidelity of TRASH and the Inverse-* methods)
    s = 100 if pct else 1
    return f"{x*s:.1f} [{lo*s:.1f}–{hi*s:.1f}]"




def nanmean0(X):
    """Column means ignoring nan; nan where a column has no values (no warning)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(X, axis=0)

MD_STARRED = re.compile(r"(?<![\\\w])(VSE|WME|CE|Low|Inverse-)\*(?!\*)")


def md_stars(text):
    """Escape the asterisk of VSE*, WME*, CE* and Inverse-* outside code spans, so Markdown does not read
    "VSE*, WME*" as emphasis."""
    parts = text.split("`")
    return "`".join(MD_STARRED.sub(r"\1\\*", p) if i % 2 == 0 else p for i, p in enumerate(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="run or sweep folders, or elections CSV files")
    ap.add_argument("--out", default=None,
                    help="analysis folder; default: analysis/ next to the input (needed with several inputs)")
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
    found = [input_paths(p) for p in a.csv]
    a.csv = [f for files, _ in found for f in files]
    if a.out is None:
        if len(found) > 1:
            ap.error("with several inputs, name the analysis folder with --out")
        a.out = os.path.join(found[0][1], "analysis")
    else:
        a.out = out_path(a.out)
    os.makedirs(a.out, exist_ok=True)
    write_command(a.out)
    rng = np.random.default_rng(a.seed)

    if a.show:
        keep = set(select(a.show))
        a.hide = [m for m in METHODS if m not in keep]
    elif a.hide:
        a.hide = select(a.hide)
    rows_all, combined, cover = combine(load(a.csv))
    for line in combined:
        print("note:", line)
    for line in lacking_lines():
        print("note:", line)
    if RENAMED_SEEN:
        print("note: old method names read as the current ones: "
              + ", ".join(f"{o} -> {n}" for o, n in sorted(RENAMED_SEEN.items())))
    if OLD_SINCERITY:
        print(f"note: {', '.join(OLD_SINCERITY)} predate(s) the current sincerity measure: their sincerity is an "
              "earlier one, and fidelity is missing; rerun for the current measures")
    present = {r["method"] for r in rows_all}
    all_hidden = not (present - set(a.hide))
    if all_hidden:          # e.g. a run of the inverse group only, with the default --hide
        print("note: --hide/--show would leave the plots empty; they show all methods instead")
        a.show, a.hide = None, []
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
        r["cover"] = cover.get(key)
        if r["cover"]:
            print(f"note: {r['tag']} ({r['label']}): {r['cover']}")
        results[key] = r

    # summary.csv
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = []
        for x in AXES + EXTRA + ["sincerity", "fidelity", "fidelity_honest"] + TYPES:
            cols += [x, x + "_lo", x + "_hi"]
        w.writerow(PARAMS + ["condition", "elections", "method", "group"] + cols + ["pareto_front", "dominated_by"])
        for key, r in results.items():
            for j, m in enumerate(r["methods"]):
                w.writerow(list(key) + [r["tag"], r["elections"], m, group_of(m)]
                           + [f"{r[x][j]:.5f}" for x in cols] + [int(j in r["pareto"]), dom_text(r["dominated_by"][j], "; ")])

    # plots
    for key, r in results.items():
        plots.arrows(r, os.path.join(a.out, f"arrows-{r['tag']}.png"), a.hide, a.errorbars)
        plots.shift(r, os.path.join(a.out, f"shift-{r['tag']}.png"), a.hide)
        plots.sincerity(r, os.path.join(a.out, f"sincerity-{r['tag']}.png"), a.hide)
        r["fid_plot"] = plots.fidelity(r, os.path.join(a.out, f"fidelity-{r['tag']}.png"), a.hide)
        r["tri_plot"] = plots.tradeoff(r, os.path.join(a.out, f"tradeoff-{r['tag']}.png"), FRONT, a.hide)
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
    L += [f"Input: {', '.join(source_name(p) for p in a.csv)} (simulator version {', '.join(versions)}"
          + ("; mixed versions" if len(versions) > 1 else "") + f"; analysis version {VERSION}). "
          f"Bootstrap: {a.boot} paired replicates over elections. Brackets are 95% intervals.", ""]
    if LACKING:
        L += ["Older inputs (columns added since are filled with the value those runs implied, or left blank where "
              "they were not measured):", ""] + [f"- {x}" for x in lacking_lines()] + [""]
    if RENAMED_SEEN:
        L += ["Methods renamed since the input was written, reported under their current names: "
              + ", ".join(f"{o} → {n}" for o, n in sorted(RENAMED_SEEN.items())) + ".", ""]
    if OLD_SINCERITY:
        L += [f"**Note:** {', '.join(OLD_SINCERITY)} predate(s) the current sincerity measure (2026-10-12): the "
              "Sincerity values from there are an earlier measure (not comparable), and so is the sincerity axis of "
              "the front; fidelity is missing. "
              + ("Combined with newer runs, methods from different files use different measures and should not be "
                 "compared on sincerity. " if len(OLD_SINCERITY) < len(a.csv) else "")
              + "Rerun for the current measure.", ""]
    if combined:
        L += ["Combined runs (a method in a later file replaces the same method from earlier files in the "
              "elections both have):", ""] + [f"- {x}" for x in combined] + [""]
    L += ["How to read this:", "",
          "- The values are the result. The **front** is judged on VSE*, WME* and sincerity: the result with strategic "
          "voters, and how honest they can afford to be. The **Pareto front** uses the point estimates as they are: "
          "a method is on it if no other is at least as good on all three and better on one, however small the "
          "difference. **Undominated** asks for clear differences instead (see Dominated by). The two need not agree: a "
          "noise-sized advantage on one axis keeps a method on the Pareto front, but does not protect it from a method "
          "clearly better on another axis; a method beaten by a hair on every axis leaves the Pareto front but can stay "
          "undominated. Honest VSE and WME depend more on modelling assumptions and are "
          "shown for context. **Dominated by** lists the methods that are clearly better on at least one of the three "
          "front axes and clearly worse on none, with the axes on which they are better in brackets. Sincerity "
          "differences are measured very precisely, so at equal VSE* and WME* the more honest method dominates. Clearly means the paired 95% interval of the difference excludes 0; smaller differences count "
          "as ties. It is a pointer for interpretation, relative to the methods included, not a verdict.",
          "- **Sincerity**: how far the learned ballots go against the voters' own preferences, within each "
          "method's ballot format: the utility-weighted pairs a ballot puts in the wrong order, relative to the most "
          "dishonest ballot the format allows (100 = no reversed preference, 0 = as far against one's preferences as "
          "the format permits). Ties (exaggeration, levelling) are not reversals. **Fidelity**: how well the ballots "
          "reproduce the voters' utility differences (100 = exactly, a tie loses half a pair, a mirrored pair all of "
          "it); it includes what the format cannot express. **(honest)** is the fidelity of the sincere ballots: the "
          "format's own loss; the drop from there to Fidelity is what strategy adds (docs/analysis.md, Sincerity and fidelity). "
          "Both are undefined (—) for TRASH and the Inverse-* methods, whose honest ballot works against the voter's "
          "own interest; on the front they count as ties on sincerity.",
          "- The intervals in the tables are for each value on its own. They are wide because elections differ "
          "a lot from each other. Comparisons between methods are much more precise, because all methods face the "
          "same elections: use the paired differences for those, not overlapping intervals.",
          "- Two methods whose paired difference interval includes 0 are tied at this sample size; their order "
          "in the table is not a finding.",
          "- **VSE** and **WME** are efficiencies with 0 = random ballot and 100 = the best candidate of each "
          "election (values above 100 cannot be reached). VSE measures total utility; WME the winner's worst "
          "head-to-head margin by sincere preferences (a refined Condorcet measure; every election counts; Minimax "
          "reaches 100 with honest ballots by construction). CE (Condorcet efficiency, elections with a Condorcet "
          "winner only, elected or not) is shown for comparison and not used for dominance. **Low** and **Low\\***: "
          "the share of elections whose winner (the expected total utility of the winner lottery) lies below the "
          "median candidate's total utility, with honest and with learned ballots: how often a method elects a "
          "poor candidate, which the averages of VSE hide. Not used for dominance."]
    if a.errorbars:
        L += ["- **Error bars** in the arrow plots are 95% intervals relative to the average of the methods shown: "
              "the variation all methods share (some elections are harder for everyone) is removed, so clearly "
              "separated bars suggest a real difference (an approximation of the paired differences). In track "
              "plots they are each point's own interval."]
    L += ["- The strategic axes (VSE*, WME*) give the result after the poll rounds: voters see polls of the previous round "
          "(the method's counts only) and answer them all at once (docs/model.md, Strategy engine). Under convergent learning "
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
                 + " | ".join(cor(full_size(spearman(base['halves'][x][0][ib], base['halves'][x][1][ib])))
                              for x in cols) + " |")
        for r in list(results.values())[1:]:
            common = [m for m in shown if m in r["methods"]]
            if len(common) < 3:
                continue
            ib = [base["methods"].index(m) for m in common]
            ic = [r["methods"].index(m) for m in common]
            L.append(f"| {r['tag']}: {r['label']} | "
                     + " | ".join(cor(spearman(base[x][ib], r[x][ic])) for x in cols) + " |")
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
              f"{r['elections']} elections (seeds {', '.join(r['seeds'])}"
              + (f"; {r['cover']}" if r["cover"] else "") + f"), {r['with_cw']} with a Condorcet winner. "
              f"{p['voters']} voters, {p['cands']} candidates, {model_text(p)}, "
              f"width {p['width']}{cand_text(p['cand_dispersion'], p['cand_shift'])}; {strategy_text(p, p['voters'])}, "
              f"strategic share {p['strategic']}"
              + (f", adaptive share {p['adaptive']}" if float(p.get('adaptive') or 0) > 0 else "")
              + f", scores 0..{p['max_score']}"
              + (f"; nomination: exits {p['exits']}, clones {p['clones']}" if p['exits'] != '0' or p['clones'] != '0' else "")
              + "." + (" " + electorate_text(r["electorate"]) if r["electorate"] else "")
              + frontier_text(r["frontier"], r["region"], r["references"]),
              "",
              "**Pareto front** (point estimates: no other method at least as good on all of VSE*, WME*, "
              "sincerity and better on one): "
              + ", ".join(r["methods"][j] for j in np.argsort(-r["VSE*"], kind="stable") if j in r["pareto"]) + ".", "",
              "**Undominated** (no other method clearly better, i.e. beyond the paired 95% interval, on one of "
              "these axes and clearly worse on none): "
              + ", ".join(r["methods"][j] for j in np.argsort(-r["VSE*"], kind="stable") if not r["dominated_by"][j])
              + ".", "",
              "| Method | Group | VSE* | WME* | Sincerity | Fidelity | (honest) | VSE | WME | CE | CE* | Low | Low* "
              "| Compromise | Burial | Push-over | Dominated by |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for j in np.argsort(-r["VSE*"], kind="stable"):
            L.append(f"| {r['methods'][j]} | {group_of(r['methods'][j])} | "
                     + " | ".join(fmt(r[x][j], r[x + '_lo'][j], r[x + '_hi'][j]) for x in ["VSE*", "WME*", "sincerity"])
                     + " | " + " | ".join("—" if np.isnan(r[x][j]) else f"{r[x][j] * 100:.1f}"
                                          for x in ["fidelity", "fidelity_honest"])
                     + " | " + " | ".join(fmt(r[x][j], r[x + '_lo'][j], r[x + '_hi'][j]) for x in ["VSE", "WME"] + EXTRA)
                     + " | " + " | ".join(f"{r[t][j] * 100:.1f}%" for t in TYPES)
                     + f" | {dom_text(r['dominated_by'][j], limit=3)} |")
        L += [""]
        if float(p.get("adaptive") or 0) > 0:
            ad = r["adapted"]
            L += [f"**Adaptive voters** (a share {p['adaptive']} of the electorate, strategic in a round when the race "
                  "between the two leaders matters enough to them): the share of them voting strategically in the "
                  "reported round, per method, averaged over elections. A method whose leading candidates are both "
                  "broadly acceptable gives fewer voters a reason to strategise.", "",
                  "| Method | Adaptive voters strategic |", "|---|---|"]
            for j in np.argsort(-np.nan_to_num(ad, nan=-1), kind="stable"):
                L.append(f"| {r['methods'][j]} | {'—' if np.isnan(ad[j]) else f'{ad[j]:.1%}'} |")
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
            # methods from a run without --participation, combined with one that had it: no values
            fp = lambda x, lo, hi: "—" if np.isnan(x) else fmt(x, lo, hi) + "%"
            for j in np.argsort(-pt["noshow_s"], kind="stable"):
                L.append(f"| {r['methods'][j]} | {fp(pt['noshow_s'][j], pt['noshow_s_lo'][j], pt['noshow_s_hi'][j])} "
                         f"| {g(pt['noshow_voters_s'][j])} | {g(pt['noshow_gain_s'][j])} "
                         f"| {fp(pt['noshow_l'][j], pt['noshow_l_lo'][j], pt['noshow_l_hi'][j])} "
                         f"| {g(pt['noshow_voters_l'][j])} | {g(pt['noshow_gain_l'][j])} |")
            L += [""]
        mp = r["manipulation"]
        if mp is not None:
            L += ["**Manipulation** (from honest ballots, see manipulation.py; after John Huang's votesim): for each "
                  "candidate X other than the honest winner T, the voters who prefer X to T try the naive "
                  "exaggerations of their ballot type for X against T, everyone else votes sincerely. One-sided: the "
                  "share of elections in which some such coalition gains, and the VSE when the worst successful "
                  "manipulation for the electorate is recorded (the honest result where none succeeds). Two-sided: "
                  "T's supporters answer with whichever exaggeration for T counters it best; the share in which a coalition still "
                  "gains, and the VSE recorded the same way. Compare VSE in the main table (honest).", "",
                  "| Method | Manipulable (one-sided) | VSE (one-sided) | Manipulable (two-sided) | VSE (two-sided) |",
                  "|---|---|---|---|---|"]
            fp = lambda k, j: "—" if np.isnan(mp[k][j]) else fmt(mp[k][j], mp[k + "_lo"][j], mp[k + "_hi"][j]) + "%"
            fv = lambda k, j: "—" if np.isnan(mp[k][j]) else fmt(mp[k][j], mp[k + "_lo"][j], mp[k + "_hi"][j])
            for j in np.argsort(-np.nan_to_num(mp["vse1"], nan=-9), kind="stable"):
                L.append(f"| {r['methods'][j]} | {fp('manip1', j)} | {fv('vse1', j)} | {fp('manip2', j)} | {fv('vse2', j)} |")
            L += [""]
        bc = r["ballot_cw"]
        if any(bc[j] for j in bc):
            n_real, n_none = r["with_cw"], r["elections"] - r["with_cw"]
            pc = lambda x: "—" if np.isnan(x) else f"{x:.1%}"
            L += ["**Condorcet winner in the cast ballots** (methods with ordered ballots: rankings, scores, grades): "
                  "what a study of the ballot data would see. Of the "
                  f"{n_real} elections with a real (sincere) Condorcet winner: the ballots show the same one, another "
                  f"one (false), or none (a cycle or ties). Of the {n_none} without one: the ballots show one anyway "
                  "(spurious). CE* (ballots): how often the method elects the ballots' Condorcet winner, where they "
                  "have one; compare CE* (real) in the main table. Honest ballots in brackets (rounding and ties of "
                  "the format alone).", "",
                  "| Method | Same | False | None | Spurious | CE* (ballots) | CE* (real) |", "|---|---|---|---|---|---|---|"]
            for j in np.argsort(-r["VSE*"], kind="stable"):
                d = bc[j]
                if not d:
                    continue
                L.append(f"| {r['methods'][j]} | " + " | ".join(f"{pc(d[c + '_l'])} ({pc(d[c + '_s'])})"
                         for c in ("same", "false", "none", "spurious", "ce_ballots"))
                         + f" | {pc(r['CE*'][j])} |")
            L += [""]
        L += [f"![strategy arrows](arrows-{r['tag']}.png)", "", f"![strategy shifts](shift-{r['tag']}.png)", "",
              f"![sincerity](sincerity-{r['tag']}.png)", ""]
        if r.get("fid_plot"):
            L += [f"![fidelity](fidelity-{r['tag']}.png)", ""]
        if r.get("tri_plot"):
            L += ["**Trade-off on the Pareto front:** for each weighting of VSE*, WME* and sincerity (a point of the "
                  "triangle; corners: all weight on one axis), the Pareto method with the highest weighted sum. Each "
                  "axis is stretched to the range the Pareto methods span on it, so the picture shows the trade-offs "
                  "within the front, not their absolute size. Point estimates; the methods hidden in the scatter plots are "
                  "left out, so it is the Pareto front of the methods shown.", "",
                  f"![trade-off](tradeoff-{r['tag']}.png)", ""]
        if a.compare:
            L += ["Paired differences (first minus second), percentage points:", "",
                  "| Pair | " + " | ".join(AXES + EXTRA + ["sincerity"]) + " |", "|---|" + "---|" * len(AXES + EXTRA + ["sincerity"])]
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
    if all_hidden:
        L += ["The plots show all methods: hiding as requested (by default the references and inverse groups) "
              "would have left them empty.", ""]
    elif a.show:
        L += [f"Scatter and chain plots show only {', '.join(a.show)}; all methods are in the tables and heatmaps.", ""]
    elif a.hide:
        L += [f"Scatter and chain plots leave out {', '.join(a.hide)}; they are in all tables and heatmaps.", ""]
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write(md_stars("\n".join(L)))
    print(f"{len(results)} condition(s); wrote {a.out}/summary.csv, report.md and plots")


if __name__ == "__main__":
    main()
