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
                 higher is more honest); fidelity (how much of what the honest ballot says the
                 learned ballots keep; strategy.fidelity) is reported alongside
The front is VSE*, WME* and sincerity: what happens with real (strategic) voters, and how honest
they can afford to be. Honest VSE and WME depend more on modelling assumptions; they are reported
alongside, as are CE, CE* (Condorcet efficiency), not used for dominance.

Dominated by: for each method, the methods that are clearly better on at least one front axis and
clearly worse on none. Clearly: the paired 95% bootstrap interval of the difference excludes 0, and
on sincerity the difference is at least 1 point (metrics.FRONT_TOLERANCE); smaller differences count as ties (otherwise methods equal on one axis, like all Condorcet methods
with honest ballots, would be separated by noise-sized differences). Each is followed by the
axes on which it is clearly better. This is a pointer for interpretation, not a verdict: the
values themselves are the result.

Outputs in the analysis folder (--out: a bare name goes into output/):
    summary.csv                one row per (condition, method): estimates, intervals, dominated_by
    report.md                  tables per condition, comparisons, plots
    (plots as .png by default; --format svg or pdf for vector graphics, --dpi for the resolution)
    arrows-<c>.png             per condition: honest -> strategic in the VSE/WME plane, all methods together
    shift-<c>.png              per condition: the same shifts as one row per method (VSE and WME panels)
    sincerity-<c>.png          per condition: VSE* and WME* against sincerity
    fidelity-<c>.png           per condition: the learned ballots in the fidelity / sincerity plane
    tradeoff-<c>.png           per condition: ternary map of the Pareto front (best method per weighting)
    axes-heatmap.png           VSE*, WME*, sincerity, VSE, WME: methods x conditions (if several conditions)
    chain-<param>.png          per parameter varied around the base: each method's chain of results
"""
import argparse
import csv
import hashlib
import os
import re
import warnings
from collections import OrderedDict
import numpy as np

import plots
from metrics import pareto, FRONT_TOLERANCE, REF_POINTS, parse_jury
from run import out_path, write_command, strategy_text, cand_text, width_text, check_folder, ANALYSIS_FILES, OUTPUT_DIR, VERSION, PARAMS as RUN_PARAMS
from models import model_text, MODELS
from methods import GROUPS, METHODS, select, group_of, sincerity_defined, current_name, file_name

AXES = ["VSE", "WME", "VSE*", "WME*"]
FRONT = ["VSE*", "WME*", "sincerity"]   # the axes dominance is judged on
EXTRA = ["CE", "CE*", "NUCW", "NUCW*", "Span", "Span*", "Low", "Low*"]   # reported, not used for dominance (NUCW:
# the Condorcet winner elected where it also has the highest normalized total; Span: elected from the trade-off
# span, metrics.span; Low: winner below the median candidate)
# reference points the analysis derives from the candidates' values (besides metrics.REF_POINTS)
DERIVED_POINTS = {"coin": "a coin between the best worst margin and the best normalized total",
                  "nuw": "normalized utility winner",
                  "mwm": "best worst margin"}
# the trivial rules of the baseline: random ballot, CW//random ballot, the mirror of CW//random ballot on the
# utility side (the normalized utility winner weakened as much as CW//random ballot is against the best
# worst margin, see trivial_points), random pair
TRIVIAL = ["rb", "cwrb", "mirror", "pair"]
MIRROR = "mirror of CW//random ballot"      # its name among the reference points
BASE = ["Lead", "Lead*", "Gap", "Gap*"]   # beyond the lotteries of trivial rules; short of the normalized frontier
JURY = ["Jury", "Jury*"]   # equivalent jury size (honest, learned ballots): the smallest random jury reaching the method
TYPES = ["compromise", "burial", "pushover", "exaggeration"]
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


def is_run_folder(p):
    """A run or sweep folder (its own elections.csv or runs/): analyses go inside, never over its files."""
    return os.path.isfile(os.path.join(p, "elections.csv")) or os.path.isdir(os.path.join(p, "runs"))


def analysis_folder(out, folders):
    """Where the analysis goes. Default: analysis/ in the input's folder; with several inputs, in the
    last input's folder as analysis-vs-<first input> (the first input is the base, so a test analysed
    against it keeps its comparison next to its data). An --out that is a run or sweep folder gets the
    same naming inside it (analysis-vs-<the other inputs>), so its command.txt and data stay untouched."""
    names = [os.path.basename(os.path.normpath(f)) for f in folders]
    if out is None:
        host = folders[-1]
        return os.path.join(host, "analysis" if len(folders) == 1 else f"analysis-vs-{names[0]}")
    out = out_path(out)
    if not is_run_folder(out):
        return out
    others = [n for f, n in zip(folders, names) if os.path.normpath(f) != os.path.normpath(out)]
    return os.path.join(out, "analysis" if not others or others == names else "analysis-vs-" + "-".join(others))


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
    "honesty": ("none", "2026-11-04, cost of voting insincerely (individual voters): older runs used a stake threshold and a worth test instead"),
    "cands_vote": ("0", "2026-11-03, candidates who vote: older runs had candidates apart from the voters"),
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
    "utility": ("exp", "2026-10-21, utility of distance: older runs used exp(-distance / width)"),
    "fingerprint": (None, "2026-10-23, fingerprint of each method's results (run.fingerprints)"),
    "fidelity": (None, "2026-10-12, fidelity, with the current sincerity measure"),
    "bcw": (None, "2026-10-12, Condorcet winner in the ballots"),
    "manip1": (None, "2026-10-17, manipulation test (run.py --manipulation)"),
    "manip2": (None, "2026-10-17, manipulation test (run.py --manipulation)"),
    "U_manip1": (None, "2026-10-17, manipulation test (run.py --manipulation)"),
    "U_manip2": (None, "2026-10-17, manipulation test (run.py --manipulation)"),
    "p_bcw": (None, "2026-10-12, Condorcet winner in the ballots"),
    "split": (None, "2026-10-26, ties"),
    "zero_margins": (None, "2026-10-26, ties"),
    "U_ml": (None, "2026-10-26, the maximal lottery as a reference point"),
    "Ur_cands": (None, "2026-10-27, the line of rounded ballots"),
    **{f"{a}_{k}": (None, "2026-10-27, further reference points") for k in REF_POINTS for a in ("U", "WM")},
    "p_span": (None, "2026-11-04, the chance of electing from the trade-off span"),
    "jury": (None, "2026-11-05, random juries (equivalent jury size)"),
    "response": ("arithmetic", "2026-11-06, how individual voters value a move (arithmetic or exact)"),
    "campaign": ("0.0", "2026-10-27, campaigns: older runs had none"),
    "individual": ("0", "2026-10-30, individual voters reading one poll: older runs used ranking groups"),
    "coordination": ("0.25", "2026-10-29, coordination share: older runs had campaigns planned with the "
                             "--campaign value"),
    "risk": ("0", "2026-10-27, risk by satisfaction: older runs used the fixed aggressive share"),
    "risk_tolerance": ("0.1", "2026-10-27, risk by satisfaction (not used by older runs)"),
    "camp_follow": (None, "2026-10-27, campaigns"),
    "camp_top": (None, "2026-10-27, campaigns"),
    "bold": (None, "2026-10-27, risk by satisfaction"),
    "exaggeration": (None, "2026-10-30, strategy type: exaggeration without reversal"),
    "WM_ml": (None, "2026-10-26, the maximal lottery as a reference point"),
}
REQUIRED_COLUMNS = ["WM_best", "U_rb", "sincerity", "U_cands", "nom_changed", "noshow"]
LACKING = {}           # input -> added columns it lacks (filled as in ADDED_COLUMNS)
OLD_SINCERITY = []     # inputs written before the current sincerity measure (no fidelity column)
OLD_FIDELITY = []      # inputs before version 2.2.0: their fidelity column holds the earlier measure (left out)
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
                if r.get("moves", "0") not in ("", "0"):
                    raise SystemExit(f"{p} holds results of the shared-moves experiment (--moves 1), which was "
                                     "removed on 2026-10-31")
                if r.get("converge", "0") not in ("", "0"):
                    raise SystemExit(f"{p} holds results of convergent learning (--converge), which was removed "
                                     "after version 1.0.0; analyse it with that version")
                r.update(fill)                   # columns added later (ADDED_COLUMNS)
                if r.get("fidelity") and _older(r.get("version"), (2, 2, 0)):
                    r["fidelity"] = ""           # the earlier fidelity (format included), not comparable
                    if source_name(p) not in OLD_FIDELITY:
                        OLD_FIDELITY.append(source_name(p))
                if "coordination" in fill and r["campaign"] not in ("", "0", "0.0"):
                    r["coordination"], r["campaign"] = r["campaign"], "1.0"   # until 2026-10-29 --campaign was the share
                new = current_name(r["method"])  # methods renamed since (methods.RENAMED)
                if new != r["method"]:
                    RENAMED_SEEN[r["method"]] = new
                    r["method"] = new
                r["_file"] = source_name(p)
                rows.append(r)
    if not rows:
        raise SystemExit("no rows in the input CSVs")
    return rows


def _older(version, than):
    """Whether a version string (major.minor.patch) is older than the tuple than; unknown: older."""
    try:
        return tuple(int(x) for x in str(version).split(".")[:3]) < than
    except ValueError:
        return True


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


def election_ids(rows, raw=False):
    """The elections of one condition as (seed, election), sorted by number. raw: as the strings in the
    rows (keys for arrays), otherwise as integers (for bootstrap_weights)."""
    ids = sorted({(r["seed"], r["election"]) for r in rows}, key=lambda e: (int(e[0]), int(e[1])))
    return ids if raw else [(int(s), int(e)) for s, e in ids]


def bootstrap_weights(ids, boot, seed=0):
    """Paired bootstrap weights (boot x E) over the elections ids, a sorted list of (seed, election)
    integers. The draws depend only on these elections and on --seed, not on what else is analysed:
    the same condition gets the same draws here and in compare.py, and adding runs to an analysis or a
    comparison leaves the other conditions' results as they were (until 2026-10-25 one random stream ran
    through all conditions, so a borderline dominance could change with the set of inputs)."""
    h = hashlib.sha256(",".join(f"{s}:{e}" for s, e in ids).encode()).digest()
    rng = np.random.default_rng([int(seed), int.from_bytes(h[:8], "little")])
    E = len(ids)
    return rng.multinomial(E, np.full(E, 1 / E), size=boot).astype(float)


def arrays(rows):
    """Per (election, method) arrays for one condition. Elections are keyed by (seed, election)."""
    methods = list(OrderedDict.fromkeys(r["method"] for r in rows))
    eids = election_ids(rows, raw=True)
    ei = {e: i for i, e in enumerate(eids)}
    mi = {m: j for j, m in enumerate(methods)}
    E, M = len(eids), len(methods)
    A = {k: np.zeros((E, M)) for k in ["num_s", "num_l", "cw_s", "cw_l", "wm_s", "wm_l", "sin"] + TYPES + NOM}
    A.update({k: np.full((E, M), np.nan) for k in ["fid_s", "fid_l", "pbcw_s", "pbcw_l", "adapted", "us", "ul"]})   # absent in older CSVs
    A.update({k: np.full((E, M), "", dtype=object) for k in ["bcw_s", "bcw_l"]})    # ballot Condorcet winner
    A.update({k: np.full((E, M), np.nan) for k in ["split_s", "split_l", "zero_s", "zero_l"]})   # ties (2026-10-26)
    A.update({k: np.full((E, M), np.nan) for k in ["span_s", "span_l"]})          # trade-off span (2026-11-04)
    A.update({k + sfx: np.full((E, M), np.nan) for k in PART for sfx in ("_s", "_l")})
    A.update({k: np.full((E, M), np.nan) for k in MANIP})
    den = np.zeros(E)          # per election: U_best - U_rb (VSE denominator)
    dwm = np.zeros(E)          # per election: WM_best - WM_rb (WME denominator)
    has = np.zeros(E)
    ele = {k: np.full(E, np.nan) for k in ["cw_is_util", "U_cw", "U_best", "fav_ratio", "U_rb", "WM_rb", "U_rand",
                                         "U_ml", "WM_ml"] + [f"{a}_{k}" for k in REF_POINTS for a in ("U", "WM")]}
    jury = [None] * E            # per election: metrics.juries, or None (runs before 2026-11-05)
    cands = [None] * E           # per election: (total utility, worst margin, normalized total or None) of every
                                 # candidate, for the frontier
    for r in rows:
        i, j = ei[(r["seed"], r["election"])], mi[r["method"]]
        num = float(r["U_method"]) - float(r["U_rb"])
        nwm = float(r["WM"]) - float(r["WM_rb"])
        pcw = float(r["p_cw"]) if r["has_cw"] == "1" else 0.0
        if r.get("jury"):                                     # random juries: once per election (since 2026-11-05)
            jury[i] = parse_jury(r["jury"])
        if r["ballots"] == "sincere":
            A["num_s"][i, j], A["cw_s"][i, j], A["wm_s"][i, j] = num, pcw, nwm
            A["us"][i, j] = float(r["U_method"])
            for k in NOM:
                A[k][i, j] = float(r[k])
            A["bcw_s"][i, j], A["pbcw_s"][i, j] = r.get("bcw", ""), float(r.get("p_bcw") or "nan")
            A["span_s"][i, j] = float(r.get("p_span") or "nan")
            A["split_s"][i, j], A["zero_s"][i, j] = float(r.get("split") or "nan"), float(r.get("zero_margins") or "nan")
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
                cands[i] = (np.array(r["U_cands"].split(";"), float), np.array(r["WM_cands"].split(";"), float),
                            np.array(r["Un_cands"].split(";"), float) if r.get("Un_cands") else None,
                            np.array(r["Ur_cands"].split(";"), float) if r.get("Ur_cands") else None)
        else:
            A["num_l"][i, j], A["cw_l"][i, j], A["wm_l"][i, j] = num, pcw, nwm
            A["ul"][i, j] = float(r["U_method"])
            A["bcw_l"][i, j], A["pbcw_l"][i, j] = r.get("bcw", ""), float(r.get("p_bcw") or "nan")
            A["span_l"][i, j] = float(r.get("p_span") or "nan")
            A["split_l"][i, j], A["zero_l"][i, j] = float(r.get("split") or "nan"), float(r.get("zero_margins") or "nan")
            if sincerity_defined(r["method"]):
                A["sin"][i, j] = float(r["sincerity"] or "nan")
                A["fid_l"][i, j] = float(r.get("fidelity") or "nan")
            else:
                A["sin"][i, j] = np.nan
            for t in TYPES:
                A[t][i, j] = float(r.get(t) or "nan")          # exaggeration: blank before 2026-10-30
            A["adapted"][i, j] = float(r.get("adapted") or "nan")   # blank without adaptive voters
            for k in PART:
                A[k + "_l"][i, j] = float(r[k])
    # tails: the winner (expected utility of the winner lottery) below the median candidate's total utility;
    # nan where the CSV has no candidate utilities (runs from before the frontier columns)
    # the normalized utility Condorcet winner (the Condorcet winner where it also has the highest total of
    # normalized utilities, the total ballots can see): where there is one, how often the method elects it;
    # and the reference point "a coin between the best worst margin and the best normalized total" (the one
    # candidate where they agree), and the normalized utility winner itself (ties split evenly). All from the
    # candidates' values (Un_cands: runs since 2026-10-20).
    hu = np.zeros(E)
    for a_ in ("U", "WM"):
        for k_ in DERIVED_POINTS:
            ele[f"{a_}_{k_}"] = np.full(E, np.nan)
    for i, c in enumerate(cands):
        if c is None or c[2] is None:
            continue
        U_, W_, Un_ = c[0], c[1], c[2]
        mw = np.flatnonzero(W_ >= W_.max() - 1e-12)
        nw = np.flatnonzero(Un_ >= Un_.max() - 1e-9 * max(1.0, abs(Un_.max())))
        hu[i] = float(has[i] == 1 and len(nw) == 1 and int(np.argmax(W_)) == nw[0])
        ele["U_coin"][i] = 0.5 * U_[mw].mean() + 0.5 * U_[nw].mean()
        ele["WM_coin"][i] = 0.5 * W_[mw].mean() + 0.5 * W_[nw].mean()
        ele["U_nuw"][i], ele["WM_nuw"][i] = U_[nw].mean(), W_[nw].mean()      # ties split evenly
        ele["U_mwm"][i], ele["WM_mwm"][i] = U_[mw].mean(), W_[mw].mean()
    for sfx in ("_s", "_l"):
        A["ucw" + sfx] = A["cw" + sfx] * hu[:, None]
    med = np.array([np.median(c[0]) if c is not None else np.nan for c in cands])
    for k, src in (("low_s", "us"), ("low_l", "ul")):
        A[k] = np.where(np.isnan(med)[:, None], np.nan, (A[src] < med[:, None] - 1e-9 * np.abs(med)[:, None]).astype(float))
    ele["_jury"] = jury
    return methods, A, (den, dwm, hu), has, ele, cands


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
        "NUCW": (W @ A["ucw_s"]) / np.where(W @ dens[2] > 0, W @ dens[2], np.nan)[:, None],
        "NUCW*": (W @ A["ucw_l"]) / np.where(W @ dens[2] > 0, W @ dens[2], np.nan)[:, None],
        **{x: (W @ np.nan_to_num(A[k])) / np.where(W @ ~np.isnan(A[k]) > 0, W @ ~np.isnan(A[k]), np.nan)
           for x, k in (("Span", "span_s"), ("Span*", "span_l"))},
        **{x: (W @ np.nan_to_num(A[k])) / np.where(W @ ~np.isnan(A[k]) > 0, W @ ~np.isnan(A[k]), np.nan)
           for x, k in (("Low", "low_s"), ("Low*", "low_l"))},
        "sincerity": (W @ A["sin"]) / W.sum(axis=1)[:, None],
        "fidelity": (W @ A["fid_l"]) / W.sum(axis=1)[:, None],
        "fidelity_honest": (W @ A["fid_s"]) / W.sum(axis=1)[:, None],
        **{t: (W @ A[t]) / W.sum(axis=1)[:, None] for t in TYPES},
    }


def dominance(P_b, tol=0.0):
    """P_b: (B x M x K) bootstrap points, higher is better. Differences whose paired 95% interval
    includes 0 count as ties, and so do differences smaller than tol on that axis (per axis; see
    metrics.FRONT_TOLERANCE: sincerity below 1 point). dom[i, j] is True when j dominates i: clearly
    better on at least one axis and clearly worse on none. Also returns better[i, j, k]: j clearly
    better than i on axis k."""
    D = P_b[:, None, :, :] - P_b[:, :, None, :]             # D[b, i, j, k] = j minus i
    lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
    mid = np.median(D, axis=0)
    tol = np.asarray(tol, float)
    better, worse = (lo > 0) & (mid >= tol), (hi < 0) & (mid <= -tol)
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
        for (U, W, *_), urb, wrb in zip(cands, ele["U_rb"], ele["WM_rb"]):
            gu, gw = (U - urb) / du, (W - wrb) / dw
            c = np.argmax(lam * gu + (1 - lam) * gw + 1e-12 * gu + 1e-12 * gw)   # break exact ties toward both
            nu, nw = nu + gu[c], nw + gw[c]
        pts.append((nu, nw))
    out = []
    for p in pts:                                       # drop repeats
        if not out or abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return np.array(out)


def normalized_frontier(cands, dens, ele, steps=101, which=2):
    """What a rule could reach that sees the voters' utilities only as each voter would normalize them
    on a ballot: stretched to 0..1 between their least and most liked candidate, so how much one voter
    cares compared with another is lost (pairwise preferences, and so the worst margins, stay as they
    are). For each weight lam it elects, in every election, the candidate with the highest
    lam * normalized-utility contribution + (1 - lam) * WME contribution, and scores the result on the
    real VSE and WME. Points not dominated by another point of the curve, from best WME to best VSE;
    None without the normalized totals (runs from before 2026-10-20). Not a strict bound (a rule could
    in principle infer more from the ballots), but the limit of taking normalized ballots at face value;
    rounding to the score range comes on top.
    which=3: the same with the normalized utilities rounded to the score range (the honest score
    ballots, rounded_frontier); there equal totals are frequent, so ties are split evenly, as Score
    splits them."""
    if any(c is None or len(c) <= which or c[which] is None for c in cands):
        return None
    du, dw = dens[0].sum(), dens[1].sum()
    dun = sum(c[which].max() - c[which].min() for c in cands) or 1.0   # one scale for all elections, as du
    pts = []
    for lam in np.linspace(0, 1, steps):
        nu = nw = 0.0
        for c_, urb, wrb in zip(cands, ele["U_rb"], ele["WM_rb"]):
            U, W, Un = c_[0], c_[1], c_[which]
            gu, gw = (U - urb) / du, (W - wrb) / dw
            gn = Un / dun
            if which == 2:
                c = np.argmax(lam * gn + (1 - lam) * gw + 1e-12 * gn + 1e-12 * gw)
                nu, nw = nu + gu[c], nw + gw[c]
            else:
                v = lam * gn + (1 - lam) * gw
                win = v >= v.max() - 1e-12
                nu, nw = nu + gu[win].mean(), nw + gw[win].mean()
        pts.append((nu, nw))
    pts = np.array(pts)
    keep = [k for k, p in enumerate(pts)
            if not any((q >= p - 1e-12).all() and (q > p + 1e-12).any() for q in pts)]
    out = []
    for p in pts[keep][np.argsort(pts[keep][:, 0], kind="stable")]:
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
    for (U, W, *_), urb, wrb in zip(cands, ele["U_rb"], ele["WM_rb"]):
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
    """Reference points in the VSE/WME plane, as fractions: the random ballot (0, 0 by definition),
    a random candidate (the zero of classical VSE; its WME is the average worst margin of the
    candidates) and the maximal lottery of the sincere preferences (metrics.maximal_lottery; runs
    since 2026-10-26). None when the candidates' values are missing."""
    if any(c is None for c in cands):
        return None
    du, dw = dens[0].sum(), dens[1].sum()
    v = sum(U.mean() - urb for (U, W, *_), urb in zip(cands, ele["U_rb"])) / du
    w = sum(W.mean() - wrb for (U, W, *_), wrb in zip(cands, ele["WM_rb"])) / dw
    out = {"random ballot": (0.0, 0.0), "random candidate": (float(v), float(w))}
    for k, name in [("ml", "maximal lottery")] + list(DERIVED_POINTS.items()) + list(REF_POINTS.items()):
        if not np.isnan(ele[f"U_{k}"]).any():
            out[name] = (float((ele[f"U_{k}"] - ele["U_rb"]).sum() / du),
                         float((ele[f"WM_{k}"] - ele["WM_rb"]).sum() / dw))
    T = trivial_points(np.ones((1, len(cands))), ele, dens)[0]
    if not np.isnan(T).any():
        out[MIRROR] = tuple(float(x) for x in T[TRIVIAL.index("mirror")])
    return out


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _up_normals(P):
    """Unit normals w >= 0 of the upper-right boundary of the down-closure of conv(P), P: K x 2, with
    the two axis directions: along any ray the support line that binds is one of these (ray_reach)."""
    P = np.unique(np.round(P[~np.isnan(P).any(axis=1)], 12), axis=0)        # sorted by x, then y
    keep = [p for p in P if not ((P[:, 0] >= p[0]) & (P[:, 1] >= p[1]) & ((P[:, 0] > p[0]) | (P[:, 1] > p[1]))).any()]
    chain = []                                            # undominated points by rising x: the concave
    for p in keep:                                        # hull of them (y falls along it)
        while len(chain) >= 2 and _cross(chain[-1] - chain[-2], p - chain[-2]) >= -1e-15:
            chain.pop()
        chain.append(p)
    W = [(1.0, 0.0), (0.0, 1.0)]
    for a, b in zip(chain, chain[1:]):
        n = np.array([a[1] - b[1], b[0] - a[0]])         # (dy, dx) of a falling edge: both >= 0
        W.append(tuple(n / np.linalg.norm(n)))
    return np.array(W)


def lead(P, X):
    """How far points X (M x 2, or B x M x 2) lie beyond the lotteries of points P (K x 2, or B x K x 2),
    in points on both axes at once: the least, over every lottery q of P, of the larger of X - q on the
    two axes. Positive: against every lottery the point is ahead by at least that much on one axis;
    negative: some lottery is better by that much on both (anything worse than a lottery on both axes
    counts as reachable too). Through the support lines of the down-closure of conv(P): the smallest
    delta with X - delta * (1, 1) in it is the largest (w . X - h(w)) / (w . (1, 1)) over its normals w."""
    P, X = np.asarray(P, float), np.asarray(X, float)
    one = P.ndim == 2
    if one:
        P, X = P[None], X[None]
    out = np.empty(X.shape[:2])
    for b in range(len(P)):
        Wn = _up_normals(P[b])                               # N x 2
        h = (P[b] @ Wn.T).max(axis=0)                        # N
        out[b] = ((X[b] @ Wn.T - h[None, :]) / Wn.sum(axis=1)[None, :]).max(axis=1)
    return out[0] if one else out


def trivial_points(W, ele, dens):
    """The trivial rules of TRIVIAL as (VSE, WME) fractions for weight rows W (B x E): B x K x 2.
    The mirror: CW//random ballot reaches a share f_w of the WME and f_v of the VSE of the best worst
    margin (the end of the trade-off on the WME side); the mirror takes the normalized utility winner (the
    end on the VSE side) to f_w of its VSE and f_v of its WME, so both ends are weakened alike.
    nan where the candidates' values are missing."""
    d, dw = W @ dens[0], W @ dens[1]
    P = {k: np.stack([(W @ (ele[f"U_{k}"] - ele["U_rb"])) / d, (W @ (ele[f"WM_{k}"] - ele["WM_rb"])) / dw], axis=1)
         for k in ("cwrb", "pair", "nuw", "mwm")}
    with np.errstate(divide="ignore", invalid="ignore"):
        fv, fw = P["cwrb"][:, 0] / P["mwm"][:, 0], P["cwrb"][:, 1] / P["mwm"][:, 1]
    mirror = np.stack([P["nuw"][:, 0] * fw, P["nuw"][:, 1] * fv], axis=1)
    pts = {"rb": np.zeros((len(W), 2)), "cwrb": P["cwrb"], "mirror": mirror, "pair": P["pair"]}
    return np.stack([pts[k] for k in TRIVIAL], axis=1)


def baseline(V, Wm, T, F):
    """The natural baseline of methods at (V, Wm) (B x M fractions): their lead over the lotteries of the
    trivial rules T (B x K x 2) and their gap to the frontier F of normalized ballots (K' x 2; with T
    added, all of them rules that read ballots), both in the sense of lead(): (lead, gap), gap = -lead
    over the frontier, positive below it. Gap nan without a frontier, both nan without the trivial rules."""
    X = np.stack([V, Wm], axis=2)                           # B x M x 2
    if np.isnan(T).any():
        return np.full(V.shape, np.nan), np.full(V.shape, np.nan)
    ld = lead(T, X)
    if F is None:
        return ld, np.full(V.shape, np.nan)
    FF = np.concatenate([np.broadcast_to(F, (len(T),) + F.shape), T], axis=1)
    return ld, -lead(FF, X)


def jury_points(W, ele, dens):
    """What random juries reach (metrics.juries) for weight rows W (B x E): {k: B x L x 2} as (VSE, WME)
    fractions, the size 1 (the random ballot) as the origin; None where an election lacks the juries or
    the sizes differ between elections (electorates smaller than a size)."""
    J = ele.get("_jury")
    if not J or any(j is None for j in J):
        return None
    sizes = sorted(set.intersection(*(set(j) for j in J)))
    if not sizes:
        return None
    d, dw = W @ dens[0], W @ dens[1]
    out = {1: np.zeros((len(W), 1, 2))}
    for k in sizes:
        GU = np.stack([J[i][k][0] - ele["U_rb"][i] for i in range(len(J))])      # E x L
        GW = np.stack([J[i][k][1] - ele["WM_rb"][i] for i in range(len(J))])
        out[k] = np.stack([(W @ GU) / d[:, None], (W @ GW) / dw[:, None]], axis=2)
    return out


def jury_size(V, Wm, JP):
    """Equivalent jury size of methods at (V, Wm) (B x M fractions): the smallest random jury whose
    lotteries (and anything worse on both axes) reach the method, i.e. lead() <= 0 against them; between the
    sizes run, interpolated on a log scale in the lead. inf: beyond the largest jury; nan without juries."""
    if JP is None:
        return np.full(V.shape, np.nan)
    X = np.stack([V, Wm], axis=2)
    ks = sorted(JP)
    L = np.stack([lead(np.concatenate([JP[k], np.zeros((len(X), 1, 2))], axis=1), X) for k in ks])   # K x B x M
    lk = np.log(np.array(ks, float))
    out = np.full(V.shape, np.inf)
    done = np.zeros(V.shape, bool)
    for a in range(len(ks)):
        hit = ~done & (L[a] <= 0)
        if a == 0:
            out[hit] = ks[0]
        else:
            la, lb = L[a - 1][hit], L[a][hit]
            out[hit] = np.exp(lk[a - 1] + (lk[a] - lk[a - 1]) * la / np.where(la - lb > 1e-15, la - lb, 1))
        done |= hit
    return out


def jury_curves(JP):
    """Per jury size, the undominated points of what it reaches (full sample), sorted by VSE: for the plots."""
    if JP is None:
        return None
    out = {}
    for k, P in JP.items():
        if k == 1:
            continue
        P = P[0]
        keep = [p for p in P if not ((P[:, 0] >= p[0]) & (P[:, 1] >= p[1]) & ((P[:, 0] > p[0]) | (P[:, 1] > p[1]))).any()]
        out[k] = np.array(sorted(keep, key=lambda p: p[0]))
    return out


def region_span(R):
    """Ranges the region allows at its ends: VSE (min, max) where WME is best, WME (min, max) where
    VSE is best."""
    top = R[R[:, 1] >= R[:, 1].max() - 1e-9]
    right = R[R[:, 0] >= R[:, 0].max() - 1e-9]
    return (top[:, 0].min(), top[:, 0].max()), (right[:, 1].min(), right[:, 1].max())


def electorate_summary(dens, has, ele):
    """Electorate properties of a condition (point estimates), or None for CSVs without them."""
    if np.isnan(ele["U_best"]).all():
        return None
    h = has == 1
    return {"cw": float(h.mean()),
            "cw_util": float(np.nanmean(ele["cw_is_util"][h])) if h.any() else float("nan"),
            # VSE lost by electing the Condorcet winner, as a ratio of sums like VSE itself
            "cw_loss": float((ele["U_best"][h] - ele["U_cw"][h]).sum() / dens[0][h].sum()) if h.any() else float("nan"),
            "fav": float(np.nanmedian(ele["fav_ratio"])) if not np.isnan(ele["fav_ratio"]).all() else float("nan")}


def frontier_text(F, R=None, refs=None, N=None, NR=None, max_score=None):
    if F is None:
        return ""
    n_text = ""
    if N is not None and len(N):
        n_text = (f" With normalized ballots (each voter's utilities stretched between their least and most "
                  f"liked candidate, taken at face value) at most VSE {N[-1, 0] * 100:.1f} (at WME "
                  f"{N[-1, 1] * 100:.1f}); the dotted line in the plots.")
    if NR is not None and len(NR):
        n_text += (f" With the normalized ballots rounded to the score range (the honest score ballots"
                   + (f", 0–{max_score}" if max_score else "") + f") at most VSE {NR[-1, 0] * 100:.1f} (at WME "
                   f"{NR[-1, 1] * 100:.1f}); the dash-dotted line.")
    rc = refs and refs.get("random candidate")
    rc_text = f" A random candidate (the zero of classical VSE) gives VSE {rc[0] * 100:.1f} / WME {rc[1] * 100:.1f}." if rc else ""
    ml = refs and refs.get("maximal lottery")
    if ml:
        rc_text += (f" The maximal lottery of the sincere preferences (the Condorcet winner if there is one, else a "
                    f"lottery no candidate beats in expectation) gives VSE {ml[0] * 100:.1f} / WME {ml[1] * 100:.1f}.")
    more = [(name, refs[name]) for name in list(DERIVED_POINTS.values()) + list(REF_POINTS.values()) if refs and name in refs]
    if more:
        rc_text += " Further reference points (VSE / WME): " + "; ".join(
            f"{name} {p[0] * 100:.1f} / {p[1] * 100:.1f}" for name, p in more) + "."

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
            f"VSE {F[k, 0] * 100:.1f} / WME {F[k, 1] * 100:.1f}." + n_text + rc_text)


def baseline_text(r):
    """The trivial rules of the condition and the methods no better than a lottery of them."""
    T = r.get("trivial")
    if T is None:
        return []
    names = {"rb": "random ballot", "cwrb": "CW//random ballot", "mirror": "its mirror on the utility side", "pair": "random pair"}
    order = np.argsort(-r["VSE*"], kind="stable")
    out = [f"with {how} ballots " + (", ".join(r["methods"][j] for j in order if r[x][j] <= 0) or "none")
           for x, how in (("Lead*", "learned"), ("Lead", "honest"))]
    return ["**Natural baseline:** trivial rules (VSE / WME) " + "; ".join(
        f"{names[k]} {p[0] * 100:.1f} / {p[1] * 100:.1f}" for k, p in zip(TRIVIAL, T))
        + ". No better than a lottery of them (Lead ≤ 0): " + "; ".join(out) + ".", ""]


def electorate_text(e):
    return (f"Electorates: Condorcet winner in {e['cw']:.1%}; it also has the highest total utility in "
            f"{e['cw_util']:.1%} of those; electing it loses {e['cw_loss']:.1%} of VSE; "
            + (f"favourite ÷ least liked {e['fav']:.1f} (median voter)." if not np.isnan(e['fav']) else
               "favourite ÷ least liked: — (utility not positive)."))


def analyse_condition(rows, boot, seed=0):
    methods, A, dens, has, ele, cands = arrays(rows)
    E = len(dens[0])
    W = bootstrap_weights(election_ids(rows), boot, seed)
    point = axis_values(np.ones((1, E)), A, dens, has)
    halves = axis_values(np.stack([np.arange(E) % 2 == 0, np.arange(E) % 2 == 1]).astype(float), A, dens, has)
    bs = axis_values(W, A, dens, has)
    NF = normalized_frontier(cands, dens, ele)
    for Wr, vals in ((np.ones((1, E)), point), (np.stack([np.arange(E) % 2 == 0, np.arange(E) % 2 == 1]).astype(float), halves), (W, bs)):
        T = trivial_points(Wr, ele, dens)
        vals["Lead"], vals["Gap"] = baseline(vals["VSE"], vals["WME"], T, NF)      # the frontier is the full
        vals["Lead*"], vals["Gap*"] = baseline(vals["VSE*"], vals["WME*"], T, NF)  # sample's in every resample
        JP = jury_points(Wr, ele, dens)
        vals["Jury"], vals["Jury*"] = jury_size(vals["VSE"], vals["WME"], JP), jury_size(vals["VSE*"], vals["WME*"], JP)
    P_b = np.stack([bs[a] for a in FRONT], axis=2)           # B x M x 3
    tol = [FRONT_TOLERANCE[a] for a in FRONT]
    dom, better = dominance(P_b, tol)
    # each dominating method with the axes on which it is clearly better, most axes first
    dominated_by = [[(methods[j], [FRONT[k] for k in np.flatnonzero(better[i, j])]) for j in np.flatnonzero(dom[i])]
                    for i in range(len(methods))]
    res = {"methods": methods, "elections": E, "with_cw": int(has.sum()), "dominated_by": dominated_by, "boot": bs, "halves": halves, "frontier": frontier(cands, dens, ele),
           "region": region(cands, dens, ele),
           "nfrontier": NF,
           "juries": jury_curves(jury_points(np.ones((1, E)), ele, dens)),
           "jury_max": (lambda J: max(J) if J else None)(jury_points(np.ones((1, E)), ele, dens)),
           "trivial": None if np.isnan(trivial_points(np.ones((1, E)), ele, dens)).any() else trivial_points(np.ones((1, E)), ele, dens)[0],
           "rfrontier": normalized_frontier(cands, dens, ele, which=3),
           "references": references(cands, dens, ele),
           "electorate": electorate_summary(dens, has, ele),
           "nomination": {k: A[k].mean(axis=0) for k in NOM},
           "adapted": nanmean0(A["adapted"]),
           "participation": participation_summary(A, W),
           "manipulation": manipulation_summary(A, W, dens[0]),
           "ballot_cw": ballot_cw_summary(A, has),
           "ties": ties_summary(A),
           # Pareto front on point estimates; undefined sincerity (TRASH, Inverse-*) counts as worst
           "pareto": set(pareto(np.nan_to_num(np.stack([point[a][0] for a in FRONT], axis=1), nan=-np.inf), tol))}
    for a in AXES + EXTRA + BASE + JURY + ["sincerity", "fidelity", "fidelity_honest"] + TYPES:
        res[a] = point[a][0]
        with np.errstate(all="ignore"), warnings.catch_warnings():     # all-nan columns (e.g. CE without a
            warnings.simplefilter("ignore", RuntimeWarning)               # Condorcet winner) give nan, no warning
            res[a + "_lo"], res[a + "_hi"] = np.nanpercentile(bs[a], [2.5, 97.5], axis=0,
                                                              method="nearest" if a in JURY else "linear")   # inf: beyond
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


def ties_summary(A):
    """Per method, for honest (_s) and learned (_l) ballots (metrics.ties): split, the share of
    elections whose winner is drawn by lot; zero_any, the share of elections whose cast ballots leave
    at least one pair of candidates tied; zero_pairs, the share of tied pairs, averaged over elections.
    The zero shares are nan for ballots that record no order. None for runs from before 2026-10-26."""
    if np.isnan(A["split_l"]).all():
        return None
    out = {}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for sfx in ("_s", "_l"):
            out["split" + sfx] = np.nanmean(A["split" + sfx], axis=0)
            z = A["zero" + sfx]
            out["zero_pairs" + sfx] = np.nanmean(z, axis=0)
            out["zero_any" + sfx] = np.where(np.isnan(z).all(axis=0), np.nan, np.nanmean(np.where(np.isnan(z), np.nan, z > 0), axis=0))
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
    return ", ".join(f"{names.get(k, k)} {p[k]}" for k in diff) or "base"


POINT_KEYS = {"rb": "random ballot", "rc": "random candidate", "ml": "maximal lottery", **DERIVED_POINTS, **REF_POINTS,
              "mirror": MIRROR}   # --points
PARAM_NAMES = {"response": "response to a move", "honesty": "honesty cost", "campaign": "campaigns", "coordination": "coordination share", "individual": "individual voters", "risk": "risk by satisfaction",
               "risk_tolerance": "risk tolerance", "cands": "candidates", "strategic": "strategic share", "max_score": "max score",
               "rounds": "poll rounds", "poll_size": "poll size", "poll_error": "poll error",
               "aggressive": "aggressive share", "start": "first poll", "cand_dispersion": "candidate dispersion", "cands_vote": "candidates vote",
               "cand_shift": "candidate shift", "lazy": "laziness", "model": "electorate model",
               "ks_alpha": "ks faction alpha", "ks_cut": "ks cut",
               "ks_shared": "ks shared factions", "clusters": "cluster count", "spread": "cluster spread", "zipf": "cluster Zipf",
               "salience": "cluster salience", "salience_exponent": "cluster salience exponent", "fixed_variance": "cluster fixed variance",
               "field_kappa": "field saturation", "field_salience": "salience field",
               "noise_strength": "noise strength", "adaptive": "adaptive share", "utility": "utility"}
LEARNING_ONLY = {"rounds", "samples", "strategic", "adaptive", "poll_size", "poll_error", "search",
                 "aggressive", "start", "campaign", "risk", "risk_tolerance", "coordination", "individual"}   # same electorates
SAME_ELECTORATES = LEARNING_ONLY | {"max_score", "exits", "clones", "lazy"}  # same electorates (honest results may differ)


def param_order(k, v):
    """Sort key for a parameter value: numbers by value; words in their listed order (start: honest, previous; model: as in models.MODELS),
    other words alphabetically, after the numbers."""
    words = {"start": ["honest", "previous"], "model": list(MODELS), "salience": ["none", "cluster"]}
    try:
        x = float(v)
    except ValueError:
        known = words.get(k, [])
        return (1, known.index(v) if v in known else len(known), str(v))
    return (0, x, "")


DERIVED = {"salience": "salience_exponent"}   # columns set from another parameter (run.resolve_salience)


def sweep_chains(keys, varying):
    """Chains around the base condition (the first): for each varying parameter, the base and every
    condition that differs from it only in that parameter, ordered by its value. Parameters that
    change together (e.g. --exits 1 --clones 2 against the base) form one chain, keyed by the tuple
    of their names, if every step changes all of them and none of them is also varied on its own
    (in a grid, the diagonal is not a chain). A derived column (DERIVED) does not count as
    a difference of its own. Returns {param or tuple of params: [keys in order]}; only chains with
    at least two conditions."""
    base = dict(zip(PARAMS, keys[0]))
    skip = {d for d, src in DERIVED.items() if src in varying}

    def diff(key):
        return tuple(q for q, v in zip(PARAMS, key) if q not in skip and v != base[q])

    groups = {}
    for key in keys[1:]:
        groups.setdefault(diff(key), []).append(key)
    out = {}
    for d, members in groups.items():
        if not d:
            continue
        chain = [keys[0]] + members
        if len(d) == 1:
            k = d[0]
            out[k] = sorted(chain, key=lambda key: param_order(k, dict(zip(PARAMS, key))[k]))
        elif (not any(g and set(g) < set(d) for g in groups)          # never varied separately (not a grid)
              and all(len({dict(zip(PARAMS, key))[q] for key in chain}) == len(chain) for q in d)):
            out[d] = sorted(chain, key=lambda key: tuple(param_order(q, dict(zip(PARAMS, key))[q]) for q in d))
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


def fmt_jury(r, x, j):
    """An equivalent jury size with its interval: one decimal below 10, whole numbers above; > the
    largest jury run where it is beyond it."""
    big = r.get("jury_max")

    def one(v):
        if np.isnan(v):
            return "—"
        if np.isinf(v):
            return f">{big}" if big else "—"
        return f"{v:.1f}" if v < 10 else f"{v:.0f}"
    if np.isnan(r[x][j]):
        return "—"
    return f"{one(r[x][j])} [{one(r[x + '_lo'][j])}–{one(r[x + '_hi'][j])}]"


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

MD_STARRED = re.compile(r"(?<![\\\w])(VSE|WME|Lead|Gap|Jury|CE|NUCW|Span|Low|Inverse-)\*(?!\*)")


def md_stars(text):
    """Escape the asterisk of VSE*, WME*, CE* and Inverse-* outside code spans, so Markdown does not read
    "VSE*, WME*" as emphasis; and | * _ in the names of compositions."""
    parts = text.split("`")
    text = "`".join(MD_STARRED.sub(r"\1\\*", p) if i % 2 == 0 else p for i, p in enumerate(parts))
    # compositions (compose.py) may contain | * _, which Markdown reads as table cells or emphasis
    for name in sorted((k for k, m in METHODS.items() if "composition" in m.__dict__), key=len, reverse=True):
        if any(ch in name for ch in "|*_"):
            text = text.replace(name, "".join("\\" + ch if ch in "|*_" else ch for ch in name))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="run or sweep folders, or elections CSV files")
    ap.add_argument("--out", default=None,
                    help="analysis folder; default: analysis/ in the input's folder, or with several inputs "
                         "analysis-vs-<first>/ in the last input's folder (the first input is the base). "
                         "A run or sweep folder given here gets the analysis inside it, never over its files")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--format", nargs="+", default=["png"], choices=["png", "svg", "pdf"],
                    help="file format(s) of the plots; the report links the first (svg and pdf: vector, sharp at any zoom)")
    ap.add_argument("--dpi", type=int, default=150, help="resolution of png plots (default 150)")
    ap.add_argument("--overlay", nargs="*", default=["frontier", "normalized", "rounded", "points", "trivial", "juries"],
                    choices=["frontier", "guides", "normalized", "rounded", "points", "trivial", "juries"],
                    help="what the VSE/WME plots draw besides the methods: the reachable frontier and region, the "
                         "guide lines (1, 2, 4, 8 points below the frontier; not drawn by default), the normalized-ballots line, the line of rounded ballots, reference points, "
                         "the edge of the trivial rules (natural baseline), the lines of random juries "
                         "(default: all; --overlay with "
                         "no names: none)")
    ap.add_argument("--points", nargs="+", default=["rb", "rc", "coin", "nuw", "mwm", "mirror", "cwrb", "pair", "median", "mean"], choices=list(POINT_KEYS),
                    help="which reference points to draw (default all but ml and cwuw, which lie next to cwrb; "
                         "all values are in the report): " + ", ".join(f"{k} = {v}" for k, v in POINT_KEYS.items()))
    ap.add_argument("--compare", nargs="*", default=[], help='pairs "A:B" for paired differences')
    ap.add_argument("--hide", nargs="*", default=["references", "inverse"],
                    help="methods or groups left out of the scatter plots (they stay in tables and heatmaps); "
                         "default: the references and inverse groups; give --hide with no names to show all")
    ap.add_argument("--show", nargs="+", default=None,
                    help="only these methods or groups in the scatter and chain plots; --hide still leaves out its "
                         "methods (by default the references and inverse groups) unless --show names them")
    ap.add_argument("--errorbars", action="store_true",
                    help="arrow plots: 95%% intervals relative to the average of the methods shown; track plots: "
                         "each point's own interval")
    ap.add_argument("--track", nargs="+", default=None,
                    help="methods (or groups) to follow across all conditions: track-<method>.png each")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    plots.FORMATS, plots.DPI = list(dict.fromkeys(a.format)), a.dpi
    plots.OVERLAY = set(a.overlay)
    plots.POINTS = {POINT_KEYS[k] for k in a.points}
    found = [input_paths(p) for p in a.csv]
    a.csv = [f for files, _ in found for f in files]
    a.out = analysis_folder(a.out, [folder for _, folder in found])
    check_folder(a.out, ANALYSIS_FILES, hint="give another --out (an analysis folder is new, empty or an earlier analysis)")
    os.makedirs(a.out, exist_ok=True)
    write_command(a.out)
    if a.show:              # --hide still applies within --show (by default the references and inverse
        # groups, which some sets contain), except for what --show names itself
        keep = set(select(a.show))
        drop = [h for h in a.hide if h not in a.show]
        if drop:
            keep -= set(select(drop))
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
    if OLD_FIDELITY:
        print(f"note: {', '.join(OLD_FIDELITY)} predate(s) version 2.2.0: their fidelity is the earlier measure "
              "and is left out; rerun for the current one")
    present = {r["method"] for r in rows_all}
    all_hidden = not (present - set(a.hide))
    if all_hidden:          # e.g. a run of the inverse group only, with the default --hide
        print("note: --hide/--show would leave the plots empty; they show all methods instead")
        a.show, a.hide = None, []
    fps = {}
    for r in rows_all:
        if r.get("fingerprint"):
            fps.setdefault(r["method"], {}).setdefault(r["fingerprint"], set()).add(r["_file"])
    for m, by in sorted(fps.items()):
        if len(by) > 1:
            print(f"note: {m} gives different results in different inputs (fingerprints "
                  + "; ".join(f"{f}: {', '.join(sorted(fs)[:3])}" for f, fs in by.items())
                  + "): the code changed between these runs, so they should not be pooled")
    versions = sorted({r.get("version") or "unknown" for r in rows_all})
    if len(versions) > 1:
        print(f"note: the input mixes simulator versions {', '.join(versions)}")
    conds = conditions(rows_all)
    keys = list(conds)
    varying = [k for i, k in enumerate(PARAMS) if len({key[i] for key in keys}) > 1]
    results = OrderedDict()
    for c, key in enumerate(keys, 1):
        r = analyse_condition(conds[key], a.boot, a.seed)
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
        for x in AXES + EXTRA + BASE + JURY + ["sincerity", "fidelity", "fidelity_honest"] + TYPES:
            cols += [x, x + "_lo", x + "_hi"]
        tie_cols = [c + s for c in ("split", "zero_any", "zero_pairs") for s in ("_honest", "")]   # blank before 2026-10-26
        w.writerow(PARAMS + ["condition", "elections", "method", "group"] + cols + ["pareto_front", "dominated_by"] + tie_cols)
        for key, r in results.items():
            ti = r["ties"]
            for j, m in enumerate(r["methods"]):
                tv = ["" if ti is None or np.isnan(ti[c][j]) else f"{ti[c][j]:.5f}"
                      for c in (k + s for k in ("split", "zero_any", "zero_pairs") for s in ("_s", "_l"))]
                w.writerow(list(key) + [r["tag"], r["elections"], m, group_of(m)]
                           + [f"{r[x][j]:.5f}" for x in cols] + [int(j in r["pareto"]), dom_text(r["dominated_by"][j], "; ")]
                           + tv)

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
    chains = {}
    for k, chain_keys in sweep_chains(keys, varying).items():
        ps = k if isinstance(k, tuple) else (k,)
        shown_vals = [" / ".join(str(dict(zip(PARAMS, key))[q]) for q in ps) for key in chain_keys]
        name = "-".join(ps)
        chains[name] = " + ".join(PARAM_NAMES.get(q, q) for q in ps)
        # the line of rounded ballots depends on the score range: not shared across a max-score sweep
        chain_rs = [dict(results[key], rfrontier=None) if "max_score" in ps else results[key] for key in chain_keys]
        plots.chain(chain_rs, chains[name], shown_vals,
                    os.path.join(a.out, f"chain-{name}.png"), a.hide, all(q in LEARNING_ONLY for q in ps), GROUPS,
                    all(q in SAME_ELECTORATES for q in ps))
    tracked = []
    if a.track and len(results) > 1:
        conds_p = [(dict(results[key], rfrontier=None) if "max_score" in varying else results[key], dict(zip(PARAMS, key)))
                   for key in keys]
        same = all(k in SAME_ELECTORATES for k in varying)
        for m in select(a.track):
            fn = "track-" + file_name(m) + "." + plots.ext()
            if plots.track(conds_p, m, varying, PARAM_NAMES, os.path.join(a.out, fn), same, a.errorbars):
                tracked.append((m, fn))

    # report.md
    L = ["# Ballot Front Finder Lab — analysis", ""]
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
    if OLD_FIDELITY:
        L += [f"**Note:** {', '.join(OLD_FIDELITY)} predate(s) version 2.2.0, whose fidelity measures what strategy "
              "takes from the honest ballot; their fidelity column holds the earlier measure (the format included) "
              "and is left out (—). Rerun for the current one.", ""]
    if combined:
        L += ["Combined runs (a method in a later file replaces the same method from earlier files in the "
              "elections both have):", ""] + [f"- {x}" for x in combined] + [""]
    L += ["How to read this:", "",
          "- The values are the result. The **front** is judged on VSE*, WME* and sincerity: the result with strategic "
          "voters, and how honest they can afford to be. The **Pareto front** uses the point estimates as they are: "
          "a method is on it if no other is at least as good on all three and better on one, however small the "
          "difference (sincerity: below 1 point counts as a tie). **Undominated** asks for clear differences instead (see Dominated by). The two need not agree: a "
          "noise-sized advantage on one axis keeps a method on the Pareto front, but does not protect it from a method "
          "clearly better on another axis; a method beaten by a hair on every axis leaves the Pareto front but can stay "
          "undominated. Honest VSE and WME depend more on modelling assumptions and are "
          "shown for context. **Dominated by** lists the methods that are clearly better on at least one of the three "
          "front axes and clearly worse on none, with the axes on which they are better in brackets. Clearly means "
          "the paired 95% interval of the difference excludes 0; smaller differences count as ties. Sincerity "
          "differences are measured very precisely, but below 1 point no voter is noticeably more honest, so they "
          "count as ties too, here and on the Pareto front. It is a pointer for interpretation, relative to the methods included, not a verdict.",
          "- **Sincerity**: how far the learned ballots go against the voters' own preferences, within each "
          "method's ballot format: the utility-weighted pairs a ballot puts in the wrong order, relative to the most "
          "dishonest ballot the format allows (100 = no reversed preference, 0 = as far against one's preferences as "
          "the format permits). Ties (exaggeration, levelling) are not reversals. **Fidelity**: how much of what the "
          "honest ballot says the learned ballots keep, over the utility-weighted pairs the honest ballot puts in order "
          "(100 = all of it; a pair tied loses half, a pair reversed all of it). It counts exaggeration and levelling, "
          "which sincerity leaves out, and not what the format itself cannot express, since the honest ballot is 100 "
          "(docs/analysis.md, Sincerity and fidelity; since version 2.2.0). "
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
          "winner only, elected or not) is shown for comparison and not used for dominance; **NUCW** likewise, for "
          "the elections where the Condorcet winner also has the highest total of normalized utilities (each voter's "
          "utilities stretched between their least and most liked candidate, as ballots see them): there the two "
          "measures agree as far as ballots can tell, and a good method should elect it nearly always. **Span**: the "
          "chance of electing from the trade-off span, the candidates no other beats on both the normalized total "
          "and the worst margin (where the two agree, only that candidate; runs since 2026-11-04). **Low** and **Low\\***: "
          "the share of elections whose winner (the expected total utility of the winner lottery) lies below the "
          "median candidate's total utility, with honest and with learned ballots: how often a method elects a "
          "poor candidate, which the averages of VSE hide. Not used for dominance.",
          "- **Lead** and **Gap** (with honest ballots; **Lead\\*** and **Gap\\*** with learned ones), a natural "
          "baseline: the trivial rules are the random ballot; the Condorcet winner, else random ballot; its mirror on the "
          "utility side (the normalized utility winner, weakened as much as CW//random ballot is against the best worst "
          "margin); a random pair decided by majority. **Lead**: "
          "against every lottery of them, the method is ahead by at least this many points on VSE or on WME; negative: "
          "some lottery of them is better by that much on both, so the method is no better than a coin among trivial "
          "rules. **Gap**: how far the method stays below the frontier of normalized ballots (the dotted line in the "
          "plots) in the same sense, by that many points on both axes; negative beyond it (the frontier is not a strict "
          "bound). No weights between VSE and WME are involved. The intervals leave out the noise "
          "of the frontier, which is the full sample's. Not used for dominance.",
          "- **Jury** and **Jury\\*** (equivalent jury size, honest and learned ballots; runs since 2026-11-05): a "
          "random jury of k voters decides with its honest normalized ballots, ideally for each trade-off between "
          "the normalized total and the worst margin among the jurors, and is judged on the whole electorate. A "
          "jury of one is the random ballot; the larger the jury, the closer it comes to the frontier of normalized "
          "ballots. The equivalent jury size is the smallest jury whose lotteries reach the method (Lead against "
          "them at most 0), interpolated on a log scale between the sizes run; > the largest size where no jury "
          "run reaches it. How many voters' honest judgment a method turns into its result. Not used for dominance."]
    if a.errorbars:
        L += ["- **Error bars** in the arrow plots are 95% intervals relative to the average of the methods shown: "
              "the variation all methods share (some elections are harder for everyone) is removed, so clearly "
              "separated bars suggest a real difference (an approximation of the paired differences). In track "
              "plots they are each point's own interval."]
    L += ["- The strategic axes (VSE*, WME*) give the result after the poll rounds: voters see polls of the previous round "
          "(the method's counts only) and answer them all at once (docs/model.md, Strategy engine).",
          "- **Compromise, burial, push-over** are shares of voters whose learned ballot reverses a real preference, "
          "by kind (see strategy.strategy_types): compromise raises a liked second choice, or the lesser evil of the "
          "two poll leaders, above the favourite; push-over raises a disliked candidate who is no poll leader to the "
          "top (to steer a runoff or an elimination); burial keeps the favourite on top and reverses a preference "
          "below it. **Exaggeration** counts the voters whose ballot differs from the sincere one without reversing "
          "any preference: candidates tied, left unranked or given the lowest or highest score together with others "
          "(a candidate buried or raised as far as the format allows without a reversal).", ""]
    if len(results) > 1:
        L += ["## Across conditions", "", f"![axes](axes-heatmap.{plots.ext()})", ""]
        if chains:
            L += ["**Sweeps:** for each parameter varied around the base, the methods' learned values as a chain "
                  "of arrows in the order of the parameter's values (light to dark). Parameters that only affect "
                  "learning start each chain at the honest point; others show the honest values as a dashed chain. "
                  "Parameters changed together (e.g. exits and clones) form one chain.",
                  ""] + [f"![sweep over {label}](chain-{k}.{plots.ext()})\n" for k, label in chains.items()]
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
              "formula; — where the halves do not agree at all, so the base itself gives no stable order on that "
              "axis); for conditions with fewer elections than the base, a further row gives the agreement "
              "expected between the base and a run of that size. A condition near or above its row orders the "
              "methods as the base does; only values clearly below it show a real change.", "",
              "| Condition | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        ib = [base["methods"].index(m) for m in shown]

        def at_size(rh, k):    # Spearman-Brown: the reliability of k elections from the agreement rh of two
            # halves of the base. It only holds for a positive half-sample agreement: at 0 or below (noise
            # alone, e.g. a few methods with nearly equal values) there is no usable reference, shown as —
            c = k / (base["elections"] / 2)
            return c * rh / (1 + (c - 1) * rh) if rh > 0 else float("nan")
        rh = {x: spearman(base['halves'][x][0][ib], base['halves'][x][1][ib]) for x in cols}
        L.append(f"| noise reference: {base['tag']} vs itself | "
                 + " | ".join(cor(at_size(rh[x], base["elections"])) for x in cols) + " |")
        # conditions with fewer elections than the base agree less with it by noise alone: the agreement
        # expected between the base and a run of m elections, sqrt(rel(n) rel(m))
        for m in sorted({r["elections"] for r in list(results.values())[1:] if r["elections"] < base["elections"]},
                        reverse=True):
            L.append(f"| noise reference: {base['tag']} vs a run of {m} elections | "
                     + " | ".join(cor(np.sqrt(at_size(rh[x], base["elections"]) * at_size(rh[x], m)))
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
                         f"| {'—' if np.isnan(e['fav']) else format(e['fav'], '.1f')} |")
            L.append("")
    for key, r in results.items():
        p = dict(zip(PARAMS, key))
        L += [f"## {r['tag']}: {r['label']}", "",
              f"{r['elections']} elections (seeds {', '.join(r['seeds'])}"
              + (f"; {r['cover']}" if r["cover"] else "") + f"), {r['with_cw']} with a Condorcet winner. "
              f"{p['voters']} voters, {p['cands']} candidates, {model_text(p)}, "
              f"{width_text(p)}{cand_text(p['cand_dispersion'], p['cand_shift'], p.get('cands_vote', 0))}; {strategy_text(p, p['voters'])}, "
              f"strategic share {p['strategic']}"
              + (f", adaptive share {p['adaptive']}" if float(p.get('adaptive') or 0) > 0 else "")
              + f", scores 0..{p['max_score']}"
              + (f"; nomination: exits {p['exits']}, clones {p['clones']}" if p['exits'] != '0' or p['clones'] != '0' else "")
              + "." + (" " + electorate_text(r["electorate"]) if r["electorate"] else "")
              + frontier_text(r["frontier"], r["region"], r["references"], r.get("nfrontier"), r.get("rfrontier"),
                              dict(zip(PARAMS, key)).get("max_score")),
              "",
              "**Pareto front** (point estimates: no other method at least as good on all of VSE*, WME*, "
              "sincerity and better on one): "
              + ", ".join(r["methods"][j] for j in np.argsort(-r["VSE*"], kind="stable") if j in r["pareto"]) + ".", "",
              "**Undominated** (no other method clearly better, i.e. beyond the paired 95% interval, on one of "
              "these axes and clearly worse on none): "
              + ", ".join(r["methods"][j] for j in np.argsort(-r["VSE*"], kind="stable") if not r["dominated_by"][j])
              + ".", "",
              ] + baseline_text(r) + [
              "| Method | Group | VSE* | WME* | Lead* | Gap* | Jury* | Sincerity | Fidelity | VSE | WME | Lead | Gap | Jury | CE | CE* | NUCW | NUCW* "
              "| Span | Span* | Low | Low* | Compromise | Burial | Push-over | Exaggeration | Dominated by |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for j in np.argsort(-r["VSE*"], kind="stable"):
            L.append(f"| {r['methods'][j]} | {group_of(r['methods'][j])} | "
                     + " | ".join(fmt(r[x][j], r[x + '_lo'][j], r[x + '_hi'][j]) for x in ["VSE*", "WME*", "Lead*", "Gap*"])
                     + " | " + fmt_jury(r, "Jury*", j) + " | " + fmt(r["sincerity"][j], r["sincerity_lo"][j], r["sincerity_hi"][j])
                     + " | " + " | ".join("—" if np.isnan(r[x][j]) else f"{r[x][j] * 100:.1f}"
                                          for x in ["fidelity"])
                     + " | " + " | ".join(fmt(r[x][j], r[x + '_lo'][j], r[x + '_hi'][j]) for x in ["VSE", "WME", "Lead", "Gap"])
                     + " | " + fmt_jury(r, "Jury", j)
                     + " | " + " | ".join(fmt(r[x][j], r[x + '_lo'][j], r[x + '_hi'][j]) for x in EXTRA)
                     + " | " + " | ".join("—" if np.isnan(r[t][j]) else f"{r[t][j] * 100:.1f}%" for t in TYPES)
                     + f" | {dom_text(r['dominated_by'][j], limit=3)} |")
        L += [""]
        if float(p.get("adaptive") or 0) > 0:
            ad = r["adapted"]
            why = ("strategic in a round when their best move gains more than their cost of voting insincerely"
                   if int(p.get("individual") or 0) else
                   "strategic in a round when the race between the two leaders matters enough to them")
            L += [f"**Adaptive voters** (a share {p['adaptive']} of the electorate, {why}): the share of them voting "
                  "strategically in the reported round, per method, averaged over elections. A method whose leading "
                  "candidates are both broadly acceptable gives fewer voters a reason to strategise.", "",
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
        ti = r["ties"]
        if ti is not None:
            pt = lambda x: "—" if np.isnan(x) else f"{x:.1%}"
            L += ["**Ties.** Split: the share of elections whose winner is drawn by lot among several candidates (a tie "
                  "the method breaks at random, or a method that draws). Tied pairs, for ordered ballots (rankings, "
                  "scores, grades): the share of elections in which the cast ballots leave at least one pair of "
                  "candidates tied (as many voters put one above the other as the other way round), and the share of "
                  "all pairs that are tied. Learned ballots, honest in brackets.", "",
                  "| Method | Split | Elections with a tied pair | Tied pairs |", "|---|---|---|---|"]
            for j in np.argsort(-r["VSE*"], kind="stable"):
                L.append(f"| {r['methods'][j]} | " + " | ".join(f"{pt(ti[c + '_l'][j])} ({pt(ti[c + '_s'][j])})"
                                                                 for c in ("split", "zero_any", "zero_pairs")) + " |")
            L += [""]
        L += [f"![strategy arrows](arrows-{r['tag']}.{plots.ext()})", "", f"![strategy shifts](shift-{r['tag']}.{plots.ext()})", "",
              f"![sincerity](sincerity-{r['tag']}.{plots.ext()})", ""]
        if r.get("fid_plot"):
            L += [f"![fidelity](fidelity-{r['tag']}.{plots.ext()})", ""]
        if r.get("tri_plot"):
            L += ["**Trade-off on the Pareto front:** for each weighting of VSE*, WME* and sincerity (a point of the "
                  "triangle; corners: all weight on one axis), the Pareto method with the highest weighted sum. Each "
                  "axis is stretched to the range the Pareto methods span on it, so the picture shows the trade-offs "
                  "within the front, not their absolute size. Point estimates; the methods hidden in the scatter plots are "
                  "left out, so it is the Pareto front of the methods shown.", "",
                  f"![trade-off](tradeoff-{r['tag']}.{plots.ext()})", ""]
        if a.compare:
            L += ["Paired differences (first minus second), percentage points:", "",
                  "| Pair | " + " | ".join(AXES + BASE + EXTRA + ["sincerity"]) + " |", "|---|" + "---|" * len(AXES + BASE + EXTRA + ["sincerity"])]
            for pair in a.compare:
                m1, m2 = pair.split(":")
                if m1 not in r["methods"] or m2 not in r["methods"]:
                    continue
                i, j = r["methods"].index(m1), r["methods"].index(m2)
                cells = []
                for x in AXES + BASE + EXTRA + ["sincerity"]:
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
