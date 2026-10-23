"""Compare methods across many runs: how each does over all the conditions in a folder, and how
often one method beats another, with the exceptions.

    python compare.py RUNS_FOLDER [--out FOLDER] [--methods ...] [--exclude ...] [--pairs "A:B" ...]

RUNS_FOLDER is searched for run and sweep folders (with elections.csv or runs/), at most --depth
levels down (default 2); everything found is read, as analyze.py reads its inputs. Nothing outside
the folder is used, so collect the runs that belong together in one folder (a repository of runs, or
a subfolder of output/). As a precaution against a wrong folder (a home or system folder), any other
subfolder without run data stops the search with a list of them (hidden folders, symbolic links and
the lab's report folders are skipped), and the output folder must be new, empty or an earlier
comparison: nothing else is overwritten.

Conditions. A condition is one set of run parameters (the seed aside). Runs with the same parameters
are pooled, as in analyze.py. Within a condition, runs whose method sets differ and that share no
elections (other seeds) are analysed separately, as replicates of the condition; each condition then
counts once in total, its replicates sharing that weight. So a parameter swept in many runs does not
outweigh one tested once, and "in 7 of 9 conditions" means 9 distinct settings. The report lists
every condition, named by what differs from the defaults of run.py.

Per condition, the comparisons are those of analyze.py: paired bootstrap intervals over the elections
of the condition; "clearly better" when the interval excludes 0 (and, for sincerity, the difference
is at least 1 point, metrics.FRONT_TOLERANCE); dominance on VSE*, WME* and sincerity.

Memory. Each file is read once, and only the columns the comparison needs are kept (as compact
numbers, see VALUES), so several gigabytes of results fit in memory. Progress goes to the terminal.

Versions. All inputs must come from the same simulator version, and every method must have the same
fingerprint in all of them (run.fingerprints: its results in a small fixed test, recorded by run.py
and sweep.py since 2026-10-23, so a change of the code between runs is seen even when the version
number stayed). Otherwise the comparison stops, unless --force-versions is given, which the report
then states at the top. Files from before fingerprints cannot be checked; the report lists them.

Outputs (default: RUNS_FOLDER/comparison/):
    report.md        the conditions; per method: averages, worst case, how often undominated, on the
                     Pareto front, near the front; pairs of the leading methods: who dominates whom in
                     how many conditions, and the exceptions to the usual outcome
    methods.csv      one row per method
    pairs.csv        one row per ordered pair of methods: conditions shared, clearly better per axis,
                     dominates, and the conditions where it does
    conditions.csv   one row per analysed condition (or replicate): its parameters and label
    command.txt      the command
"""
import argparse
import csv
import os
import sys
import time
from array import array
from collections import OrderedDict

import numpy as np

import analyze as an
from analyze import PARAMS, PARAM_NAMES, FRONT, ADDED_COLUMNS, dominance, input_paths
from metrics import pareto, FRONT_TOLERANCE
from methods import select, GROUPS, group_of, current_name, sincerity_defined
from run import write_command, default_params, group_text, VERSION

AXES_SHOWN = ["VSE*", "WME*", "sincerity"]
NEAR = (1.0, 2.0)          # points: "near the front" if no method is better by this much on both VSE* and WME*


OUTPUT_FILES = {"report.md", "methods.csv", "pairs.csv", "conditions.csv", "command.txt"}   # what compare.py writes


def listdir(d):
    """Folder contents, or none if it cannot be read."""
    try:
        return os.listdir(d)
    except OSError:
        return []


def is_output_folder(d):
    """A folder written by analyze.py or compare.py (a report with its tables), not run data."""
    names = set(listdir(d))
    return "report.md" in names and ("summary.csv" in names or "methods.csv" in names)


def find_runs(folder, depth):
    """Run and sweep folders in folder, at most `depth` levels down, not descending into them.

    Precautions against pointing at the wrong folder (a home or system folder): the search goes no
    deeper than `depth`, follows no symbolic links, skips hidden folders and the lab's own report
    folders, and stops with a list if any other folder on the way holds no run data at all, instead of
    reading whatever it finds. Returns the run and sweep folders."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"{folder} is not a folder")
    if an.is_run_folder(folder):
        return [folder]
    runs, empty = [], []

    def visit(d, level):
        """Run folders under d; True if any."""
        found = False
        for name in sorted(listdir(d)):
            if len(empty) > 20:                  # clearly the wrong folder: stop looking
                break
            sub = os.path.join(d, name)
            if name.startswith(".") or os.path.islink(sub) or not os.path.isdir(sub):
                continue
            if an.is_run_folder(sub):
                runs.append(sub)
                found = True
            elif is_output_folder(sub):
                continue
            elif level < depth and visit(sub, level + 1):
                found = True
            else:
                empty.append(sub)
        return found

    visit(folder, 1)
    if empty:
        shown = "\n".join(f"  {e}" for e in empty[:20]) + ("\n  …" if len(empty) > 20 else "")
        raise SystemExit(f"{folder} contains folders without run data (no elections.csv or runs/ within "
                         f"{depth} level(s)):\n{shown}\nGive a folder that holds only runs (and reports), "
                         "move the others out, or raise --depth if the runs lie deeper")
    if not runs:
        raise SystemExit(f"no run or sweep folders in {folder}")
    return runs


def check_out(out):
    """The output folder must be new, empty or an earlier comparison: nothing else is overwritten."""
    if not os.path.exists(out):
        return
    if not os.path.isdir(out):
        raise SystemExit(f"--out {out} exists and is not a folder")
    other = set(os.listdir(out)) - OUTPUT_FILES
    if other:
        raise SystemExit(f"{out} exists and holds other files ({', '.join(sorted(other)[:5])}"
                         f"{' …' if len(other) > 5 else ''}): give another --out")


def same_value(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def describe(params, defaults):
    """The parameters in which a condition differs from the defaults: {name: value}."""
    return OrderedDict((k, params[k]) for k in PARAMS
                       if k in defaults and not same_value(params[k], defaults[k]) and k != "salience")


def label_of(diff):
    if not diff:
        return "defaults"
    return ", ".join(f"{PARAM_NAMES.get(k, k)} {group_text(v) if k == 'group_size' else v}" for k, v in diff.items())


def say(*parts):
    """Progress line (stderr, flushed), so long comparisons show where they are."""
    print(*parts, file=sys.stderr, flush=True)


# Only these columns are kept from the results files (as compact numbers), so that runs of several
# gigabytes fit in memory: per row about 70 bytes instead of a dictionary of every column.
VALUES = ["U_method", "U_rb", "U_best", "WM", "WM_rb", "WM_best", "sincerity"]


class Part:
    """The rows of one condition in one results file, as compact columns."""
    def __init__(self, name):
        self.file = name
        self.seed, self.election, self.method, self.learned = array("l"), array("l"), array("h"), array("b")
        self.vals = array("d")
        self.names = []                # method names; self.method holds indices into it
        self._idx = {}

    def add(self, seed, election, method, learned, vals):
        k = self._idx.get(method)
        if k is None:
            k = self._idx[method] = len(self.names)
            self.names.append(method)
        self.seed.append(seed)
        self.election.append(election)
        self.method.append(k)
        self.learned.append(learned)
        self.vals.extend(vals)

    def finish(self):
        self.seed = np.array(self.seed, dtype=np.int64)
        self.election = np.array(self.election, dtype=np.int64)
        self.method = np.array(self.method, dtype=np.int64)
        self.learned = np.array(self.learned, dtype=bool)
        self.vals = np.array(self.vals).reshape(-1, len(VALUES))
        self.code = self.seed * 10_000_000 + self.election          # one number per election
        self.elections = frozenset(np.unique(self.code).tolist())


def num(x):
    return float(x) if x not in ("", None) else np.nan


def scan(path, keep, versions, notes, prints):
    """Read one results file: {condition key: Part}, methods in keep only (None: all). Fills the
    run parameters an older file lacks (ADDED_COLUMNS) and reads renamed methods by their current name."""
    name = an.source_name(path)
    parts = OrderedDict()
    with open(path, newline="") as f:
        rd = csv.reader(f)
        head = next(rd, None) or []
        col = {c: i for i, c in enumerate(head)}
        missing = [k for k in PARAMS + ["seed", "election", "method", "ballots"] + VALUES
                   if k not in col and k not in ADDED_COLUMNS]
        if missing:
            raise SystemExit(f"{path} lacks {', '.join(missing)}; regenerate it with the current run.py or sweep.py")
        fill = {k: ADDED_COLUMNS[k][0] for k in PARAMS if k not in col}
        if fill:
            notes.setdefault("lack columns added later: " + ", ".join(f"{k} read as {v}" for k, v in fill.items()),
                             []).append(name)
        pidx = [(k, col.get(k)) for k in PARAMS]
        vidx = [col[k] for k in VALUES]
        iv, im, ib, isd, iel = col.get("version"), col["method"], col["ballots"], col["seed"], col["election"]
        ifp = col.get("fingerprint")
        if ifp is None:
            notes.setdefault("have no fingerprints (made before 2026-10-23): changes of the code since cannot be "
                             "detected for them", []).append(name)
        renamed = {}
        n = 0
        for row in rd:
            n += 1
            if n % 1_000_000 == 0:
                say(f"    {name}: {n:,} rows")
            m = row[im]
            if m not in renamed:
                renamed[m] = current_name(m)
            m = renamed[m]
            if keep is not None and m not in keep:
                continue
            versions.add(row[iv] if iv is not None and row[iv] else "unknown")
            if ifp is not None and row[ifp]:
                prints.setdefault(m, {}).setdefault(row[ifp], set()).add(name)
            key = tuple(row[i] if i is not None else fill[k] for k, i in pidx)
            part = parts.get(key)
            if part is None:
                part = parts[key] = Part(name)
            part.add(int(row[isd]), int(row[iel]), m, row[ib] != "sincere", [num(row[i]) for i in vidx])
    for part in parts.values():
        part.finish()
    old = {o: c for o, c in renamed.items() if o != c}
    if old:
        notes.setdefault("old method names read as the current ones: "
                         + ", ".join(f"{o} -> {c}" for o, c in sorted(old.items())), []).append(name)
    return parts, n


def units(parts):
    """Analysable units: per condition, the parts (results files) covering exactly the same elections
    are pooled, so runs of different methods on the same elections are compared directly; parts with
    other elections (another seed, or more or fewer elections) become replicates of the condition.
    parts: [(key, Part)] in file order. Returns [(key, [Part])]."""
    by_cond = OrderedDict()
    for key, part in parts:
        by_cond.setdefault(key, OrderedDict()).setdefault(part.elections, []).append(part)
    return [(key, ps) for key, groups in by_cond.items() for ps in groups.values()]


def analyse_unit(parts, boot, rng):
    """Point values, dominance and clearly-better per axis for one unit. A method in several parts
    keeps the values of the last (later files win, as in analyze.py); only the elections every method
    has are used."""
    methods = list(OrderedDict.fromkeys(m for p in parts for m in p.names))
    mi = {m: j for j, m in enumerate(methods)}
    codes = np.array(sorted(parts[0].elections), dtype=np.int64)
    E, M = len(codes), len(methods)
    A = {k: np.full((E, M), np.nan) for k in ("num_s", "wm_s", "num_l", "wm_l", "sin")}
    den, dwm = np.full(E, np.nan), np.full(E, np.nan)
    for p in parts:
        e = np.searchsorted(codes, p.code)
        m = np.array([mi[x] for x in p.names])[p.method]
        V = p.vals
        d_u, d_w = V[:, 2] - V[:, 1], V[:, 5] - V[:, 4]
        known = ~np.isnan(den[e])
        if np.any(np.abs(den[e][known] - d_u[known]) > 1e-6 * np.maximum(1, np.abs(d_u[known]))):
            raise SystemExit(f"{p.file}: elections differ from another file with the same parameters; the files "
                             "come from incompatible simulator versions and cannot be combined")
        den[e], dwm[e] = d_u, d_w
        s, l = ~p.learned, p.learned
        A["num_s"][e[s], m[s]] = V[s, 0] - V[s, 1]
        A["wm_s"][e[s], m[s]] = V[s, 3] - V[s, 4]
        A["num_l"][e[l], m[l]] = V[l, 0] - V[l, 1]
        A["wm_l"][e[l], m[l]] = V[l, 3] - V[l, 4]
        A["sin"][e[l], m[l]] = V[l, 6]
    for j, x in enumerate(methods):
        if not sincerity_defined(x):
            A["sin"][:, j] = np.nan
    ok = ~np.isnan(A["num_s"]).any(axis=1) & ~np.isnan(A["num_l"]).any(axis=1)
    cover = None if ok.all() else f"{int(ok.sum())} of {E} elections used, the ones all methods have"
    A = {k: v[ok] for k, v in A.items()}
    den, dwm, E = den[ok], dwm[ok], int(ok.sum())
    if E == 0:
        return None

    def axes(W):
        d, dw = W @ den, W @ dwm
        sw = W.sum(axis=1)[:, None]
        return {"VSE": (W @ A["num_s"]) / d[:, None], "WME": (W @ A["wm_s"]) / dw[:, None],
                "VSE*": (W @ A["num_l"]) / d[:, None], "WME*": (W @ A["wm_l"]) / dw[:, None],
                "sincerity": (W @ np.nan_to_num(A["sin"])) / sw + np.where(np.isnan(A["sin"]).all(axis=0), np.nan, 0)}
    point = axes(np.ones((1, E)))
    W = rng.multinomial(E, np.full(E, 1 / E), size=boot).astype(float)
    bs = axes(W)
    P_b = np.stack([bs[a] for a in FRONT], axis=2)
    tol = [FRONT_TOLERANCE[a] for a in FRONT]
    dom, better = dominance(P_b, tol)
    pts = np.nan_to_num(np.stack([point[a][0] for a in FRONT], axis=1), nan=-np.inf)
    par = set(pareto(pts, tol))
    vals = {a: point[a][0] for a in AXES_SHOWN + ["VSE", "WME"]}
    # near the front: no method better by d points on both VSE* and WME* at once (point estimates)
    v, w = vals["VSE*"] * 100, vals["WME*"] * 100
    near = {d: np.array([not np.any((v >= v[i] + d) & (w >= w[i] + d)) for i in range(M)]) for d in NEAR}
    seeds = sorted({int(c // 10_000_000) for c in codes})
    return {"methods": methods, "elections": E, "vals": vals, "dom": dom, "better": better,
            "pareto": par, "near": near, "cover": cover, "seeds": [str(x) for x in seeds],
            "files": sorted({p.file.split("/")[0] for p in parts})}


def parse_pairs(pairs):
    """--pairs "A:B": method names (as in the lab, case ignored) checked before any data is read; a name
    not found gets the closest names as suggestions."""
    import difflib
    from methods import METHODS, RENAMED
    known = {m.lower(): m for m in METHODS} | {o.lower(): current_name(o) for o in RENAMED}
    out, bad = [], []
    for pr in pairs:
        if pr.count(":") != 1:
            bad.append(f'"{pr}": write a pair as "A:B"')
            continue
        names = []
        for x in (y.strip() for y in pr.split(":")):
            m = known.get(x.lower())
            if m is None:
                close = difflib.get_close_matches(x.lower(), list(known), n=3, cutoff=0.6)
                bad.append(f'"{x}" is not a method' + (f" (did you mean {', '.join(known[c] for c in close)}?)" if close else ""))
            names.append(m)
        if None not in names:
            out.append(tuple(names))
    if bad:
        raise SystemExit("--pairs: " + "; ".join(bad))
    return out


def frac(x):
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def common_setting(diffs):
    """Settings shared by all the given conditions (differences from the defaults), as text."""
    if not diffs:
        return ""
    shared = [k for k in diffs[0] if all(k in d and same_value(d[k], diffs[0][k]) for d in diffs)]
    return label_of(OrderedDict((k, diffs[0][k]) for k in shared)) if shared else ""


def main():
    ap = argparse.ArgumentParser(description="Compare methods across all runs in a folder.")
    ap.add_argument("folder", help="folder with run and sweep folders (searched recursively)")
    ap.add_argument("--out", default=None, help="output folder (default: FOLDER/comparison)")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="only these methods, groups or sets (default: all found, except references and inverse)")
    ap.add_argument("--exclude", nargs="*", default=None, help="methods or groups left out")
    ap.add_argument("--pairs", nargs="*", default=[], help='pairs "A:B" to report in detail, besides the leading methods')
    ap.add_argument("--top", type=int, default=12,
                    help="how many leading methods (most often undominated) get the pair matrix and exceptions")
    ap.add_argument("--min-conditions", type=int, default=None,
                    help="methods run in fewer conditions are listed separately and left out of the pair "
                         "matrix (default: half of the conditions)")
    ap.add_argument("--min-elections", type=int, default=100,
                    help="conditions (or replicates) with fewer elections are left out and listed (default 100)")
    ap.add_argument("--boot", type=int, default=400, help="bootstrap replicates per condition")
    ap.add_argument("--force-versions", action="store_true",
                    help="compare even if the inputs come from different simulator versions or code (fingerprints) "
                         "(stated in the report)")
    ap.add_argument("--depth", type=int, default=2,
                    help="how many folder levels to search for runs (default 2: FOLDER/test/ and FOLDER/group/test/)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    requested = parse_pairs(a.pairs)          # checked before the (possibly long) reading
    folders = find_runs(a.folder, a.depth)
    if not folders:
        raise SystemExit(f"no run or sweep folders under {a.folder}")
    paths = [f for d in folders for f in input_paths(d)[0]]
    out = a.out or os.path.join(a.folder, "comparison")
    check_out(out)
    t0 = time.time()
    keep_sel = None
    if a.methods:
        keep_sel = set(select(a.methods))
    drop = set(select(a.exclude)) if a.exclude else set()

    # 1. read every file once, keeping only what the comparison needs
    size = sum(os.path.getsize(p) for p in paths)
    say(f"[1/3] reading {len(paths)} results file(s), {size / 1e9:.2f} GB")
    vers, notes, all_parts, prints = OrderedDict(), OrderedDict(), [], {}
    for k, p in enumerate(paths, 1):
        t = time.time()
        vs = set()
        parts, n = scan(p, keep_sel, vs, notes, prints)
        name = an.source_name(p)
        vers[name] = sorted(vs)
        all_parts += [(key, part) for key, part in parts.items()]
        say(f"  {k}/{len(paths)} {name}: {n:,} rows, {len(parts)} condition(s), {time.time() - t:.0f} s")
    all_v = sorted({v for vs in vers.values() for v in vs})
    forced = None
    if len(all_v) > 1:
        lines = [f"  {name}: {', '.join(vs)}" for name, vs in vers.items()]
        if not a.force_versions:
            raise SystemExit("the runs come from different simulator versions (" + ", ".join(all_v) + "):\n"
                             + "\n".join(lines) + "\ncompare runs of one version, or give --force-versions "
                             "(the report will say so)")
        forced = all_v
    mixed = {m: by for m, by in prints.items() if len(by) > 1}
    if mixed:
        lines = [f"  {m}: " + "; ".join(f"{f} in {', '.join(sorted(fs)[:3])}{' …' if len(fs) > 3 else ''}"
                                       for f, fs in by.items()) for m, by in sorted(mixed.items())]
        if not a.force_versions:
            raise SystemExit(f"{len(mixed)} method(s) give different results in different runs (fingerprints differ: "
                             "the code changed between the runs):\n" + "\n".join(lines[:20])
                             + ("\n  …" if len(lines) > 20 else "") + "\ncompare runs of the same code, or give "
                             "--force-versions (the report will say so)")
        forced_fp = sorted(mixed)
    else:
        forced_fp = None
    found = sorted({m for _, part in all_parts for m in part.names})
    keep = set(found) if keep_sel is not None else set(found) - set(GROUPS.get("references", [])) - set(GROUPS.get("inverse", []))
    keep -= drop
    if not keep:
        raise SystemExit("no methods left to compare")
    if keep != set(found):                      # leave the others out of every part
        for _, part in all_parts:
            sel = np.array([part.names[i] in keep for i in range(len(part.names))])[part.method] if part.names else []
            if len(sel) and not sel.all():
                old_names = part.names
                used = [i for i, x in enumerate(old_names) if x in keep]
                remap = {i: k for k, i in enumerate(used)}
                part.names = [old_names[i] for i in used]
                part.method = np.array([remap[i] for i in part.method[sel]], dtype=np.int64)
                part.seed, part.election, part.learned = part.seed[sel], part.election[sel], part.learned[sel]
                part.vals, part.code = part.vals[sel], part.code[sel]
        all_parts = [(key, part) for key, part in all_parts if part.names]
    os.makedirs(out, exist_ok=True)
    write_command(out)
    rng = np.random.default_rng(a.seed)
    defaults = {k: v for k, v in default_params().items() if k != "seed"}

    # 2. analyse every unit
    todo = units(all_parts)
    say(f"[2/3] analysing {len(todo)} condition(s) and replicate(s), {len(keep)} methods, "
        f"{a.boot} bootstrap replicates each")
    us, small = [], []
    for k, (key, parts) in enumerate(todo, 1):
        t = time.time()
        n_e = len(parts[0].elections)
        if n_e < a.min_elections:
            small.append((describe(dict(zip(PARAMS, key)), defaults), n_e, sorted({p.file.split("/")[0] for p in parts})))
            say(f"  {k}/{len(todo)} {label_of(small[-1][0])}: only {n_e} elections; left out")
            continue
        res = analyse_unit(parts, a.boot, rng)
        if res is None:
            say(f"  {k}/{len(todo)}: no elections all methods share; left out")
            continue
        res.update(key=key, diff=describe(dict(zip(PARAMS, key)), defaults))
        us.append(res)
        say(f"  {k}/{len(todo)} {label_of(res['diff'])}: {res['elections']} elections, {len(res['methods'])} "
            f"methods, {time.time() - t:.1f} s")
    if not us:
        raise SystemExit("no conditions with the selected methods")
    say("[3/3] comparing across conditions and writing the report")
    # weights: each condition counts once for every method (and pair) run in it, shared equally among
    # the replicates of the condition that contain the method (the pair)
    reps = {}
    for u in us:
        reps.setdefault(u["key"], []).append(u)

    def weight(u, *ms):
        k = sum(all(m in v["methods"] for m in ms) for v in reps[u["key"]])
        return 1.0 / k if k and all(m in u["methods"] for m in ms) else 0.0
    n_rep = 0
    for i, u in enumerate(us, 1):
        u["tag"] = f"c{i}"
        u["label"] = label_of(u["diff"])
        if len(reps[u["key"]]) > 1:
            n_rep += 1
            u["label"] += f" (replicate: {u['elections']} elections, seed {', '.join(u['seeds'])})"
    n_cond = len(reps)

    # per method
    meth = OrderedDict()
    for m in sorted(keep):
        rec = {"n": 0.0, "units": 0, "und": 0.0, "par": 0.0, "near": {d: 0.0 for d in NEAR},
               "sum": {x: 0.0 for x in AXES_SHOWN + ["VSE", "WME"]}, "worst": {}}
        for u in us:
            if m not in u["methods"]:
                continue
            j = u["methods"].index(m)
            wgt = weight(u, m)
            rec["n"] += wgt
            rec["units"] += 1
            rec["und"] += wgt * (not u["dom"][j].any())
            rec["par"] += wgt * (j in u["pareto"])
            for d in NEAR:
                rec["near"][d] += wgt * bool(u["near"][d][j])
            for x in rec["sum"]:
                rec["sum"][x] += wgt * u["vals"][x][j]
            for x in ("VSE*", "WME*"):
                val = u["vals"][x][j]
                if x not in rec["worst"] or val < rec["worst"][x][0]:
                    rec["worst"][x] = (val, u["tag"])
        if rec["n"] > 0:
            rec["mean"] = {x: s / rec["n"] for x, s in rec["sum"].items()}
            meth[m] = rec
    min_n = a.min_conditions if a.min_conditions is not None else int(np.ceil(n_cond / 2))
    rank = lambda m: (-meth[m]["und"] / meth[m]["n"], -meth[m]["near"][NEAR[0]] / meth[m]["n"], -meth[m]["mean"]["VSE*"])
    order = sorted([m for m in meth if meth[m]["n"] >= min_n - 1e-9], key=rank)
    few = sorted([m for m in meth if meth[m]["n"] < min_n - 1e-9], key=rank)

    # pairs: for ordered (A, B): weighted counts of shared conditions, A clearly better per axis, A dominates
    def pair(A, B):
        rec = {"shared": 0.0, "dom": 0.0, "dom_tags": [], **{x: 0.0 for x in FRONT}, **{x + "_tags": [] for x in FRONT}}
        for u in us:
            if A not in u["methods"] or B not in u["methods"]:
                continue
            i, j = u["methods"].index(B), u["methods"].index(A)       # better[i, j, k]: j better than i
            wgt = weight(u, A, B)
            rec["shared"] += wgt
            if u["dom"][i, j]:
                rec["dom"] += wgt
                rec["dom_tags"].append(u)
            for k, x in enumerate(FRONT):
                if u["better"][i, j, k]:
                    rec[x] += wgt
                    rec[x + "_tags"].append(u)
        return rec

    names = list(meth)
    P = {(A, B): pair(A, B) for A in names for B in names if A != B}

    # ---- outputs ----
    with open(os.path.join(out, "conditions.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["condition", "label", "replicates", "elections", "methods", "seeds", "inputs"] + PARAMS)
        for u in us:
            wr.writerow([u["tag"], u["label"], len(reps[u["key"]]), u["elections"], len(u["methods"]),
                         " ".join(u["seeds"]), " ".join(u["files"])] + list(u["key"]))
    with open(os.path.join(out, "methods.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "group", "conditions", "analysed", "undominated", "pareto_front"]
                    + [f"near_front_{frac(d)}" for d in NEAR]
                    + [f"mean_{x}" for x in AXES_SHOWN + ["VSE", "WME"]]
                    + ["worst_VSE*", "worst_VSE*_condition", "worst_WME*", "worst_WME*_condition"])
        for m in order + few:
            r = meth[m]
            wr.writerow([m, group_of(m), f"{r['n']:.3f}", r["units"], f"{r['und'] / r['n']:.4f}",
                         f"{r['par'] / r['n']:.4f}"] + [f"{r['near'][d] / r['n']:.4f}" for d in NEAR]
                        + [f"{r['mean'][x]:.5f}" for x in AXES_SHOWN + ["VSE", "WME"]]
                        + [f"{r['worst']['VSE*'][0]:.5f}", r["worst"]["VSE*"][1],
                           f"{r['worst']['WME*'][0]:.5f}", r["worst"]["WME*"][1]])
    with open(os.path.join(out, "pairs.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["A", "B", "shared_conditions", "A_dominates_B", "A_dominates_B_in"]
                    + [f"A_clearly_better_{x}" for x in FRONT] + [f"A_clearly_better_{x}_in" for x in FRONT])
        for (A, B), r in P.items():
            if r["shared"] == 0:
                continue
            wr.writerow([A, B, f"{r['shared']:.3f}", f"{r['dom']:.3f}", " ".join(u["tag"] for u in r["dom_tags"])]
                        + [f"{r[x]:.3f}" for x in FRONT] + [" ".join(u["tag"] for u in r[x + "_tags"]) for x in FRONT])

    L = ["# Ballot Front Finder Lab — comparison across runs", ""]
    if notes:
        L += ["Older inputs:", ""] + [f"- {len(fs)} file(s) {msg}: {', '.join(fs[:4])}{' …' if len(fs) > 4 else ''}"
                                      for msg, fs in notes.items()] + [""]
    if small:
        L += [f"Left out, fewer than {a.min_elections} elections (--min-elections): "
              + "; ".join(f"{label_of(d)} ({e} elections, {', '.join(fs)})" for d, e, fs in small) + ".", ""]
    if forced_fp:
        L += [f"**Runs of different code, compared anyway (--force-versions):** {len(forced_fp)} method(s) give "
              f"different results in different runs ({', '.join(forced_fp[:8])}{' …' if len(forced_fp) > 8 else ''}).", ""]
    if forced:
        L += [f"**Mixed simulator versions, compared anyway (--force-versions): {', '.join(forced)}.** "
              "Results of different versions may not be comparable; see the list of inputs below.", ""]
    L += [f"Folder: {a.folder}. {len(folders)} run or sweep folder(s), {len(paths)} results file(s); simulator "
          f"version {', '.join(all_v)}; analysis version {VERSION}. {n_cond} distinct condition(s)"
          + (f" in {len(us)} analysed parts (results files with other elections are replicates of their "
             "condition and share its weight)" if len(us) > n_cond else "") + f"; {len(meth)} methods. Bootstrap: {a.boot} paired replicates per "
          "condition.", "",
          "How to read this:", "",
          "- **Every condition counts once**, whatever the number of runs or elections behind it. Averages and "
          "shares are over the conditions listed below, and only over those a method was run in (column "
          "*conditions*); they describe this collection of runs, not all possible electorates.",
          "- **Clearly better** and **dominates** are as in analyze.py's reports: the paired 95% interval of the "
          "difference within the condition excludes 0 (sincerity: and the difference is at least 1 point); "
          "dominance on VSE\\*, WME\\* and sincerity. **Near the front (1 / 2)**: no method in the condition is "
          "better by 1 (2) points on VSE\\* and WME\\* at once (point estimates), the guide lines of the arrow plots.",
          "- Pair counts are over the conditions both methods were run in.", ""]

    L += ["## Conditions", "", "| Condition | Differs from the defaults | Elections | Methods | Inputs |", "|---|---|---|---|---|"]
    for u in us:
        L.append(f"| {u['tag']} | {u['label']} | {u['elections']} | {len(u['methods'])} | {', '.join(u['files'])} |")
    L.append("")
    covered = [u for u in us if u["cover"]]
    if covered:
        L += ["Conditions where not all methods had the same elections (only the common ones are used): "
              + "; ".join(f"{u['tag']}: {u['cover']}" for u in covered), ""]

    head = ["| Method | Group | Conditions | Undominated | Pareto front | Near front (1 / 2) | Mean VSE\\* | Mean WME\\* "
            "| Mean sincerity | Worst VSE\\* | Worst WME\\* |", "|---|---|---|---|---|---|---|---|---|---|---|"]

    def method_row(m):
        r = meth[m]
        n = r["n"]
        return (f"| {m} | {group_of(m)} | {frac(n)} | {r['und'] / n:.0%} | {r['par'] / n:.0%} | "
                f"{r['near'][NEAR[0]] / n:.0%} / {r['near'][NEAR[1]] / n:.0%} | {100 * r['mean']['VSE*']:.1f} | "
                f"{100 * r['mean']['WME*']:.1f} | {100 * r['mean']['sincerity']:.1f} | "
                f"{100 * r['worst']['VSE*'][0]:.1f} ({r['worst']['VSE*'][1]}) | "
                f"{100 * r['worst']['WME*'][0]:.1f} ({r['worst']['WME*'][1]}) |")
    L += ["## Methods", "",
          f"Methods run in at least {min_n} of the {n_cond} conditions, sorted by how often undominated, then near "
          "the front. Means are weighted by condition; worst: the lowest value over the conditions, with the "
          "condition.", ""] + head + [method_row(m) for m in order] + [""]
    if few:
        L += [f"**Run in fewer conditions** (shares over few conditions say little; not in the pair matrix):", ""] \
            + head + [method_row(m) for m in few] + [""]

    top = order[:a.top]
    extra = [(A, B) for A, B in requested if A in meth and B in meth]
    if len(top) > 1:
        L += [f"## Who dominates whom (the {len(top)} leading methods)", "",
              "Row dominates column in this many of the conditions both were run in (shared conditions after the slash).", "",
              "| | " + " | ".join(top) + " |", "|---|" + "---|" * len(top)]
        for A in top:
            cells = []
            for B in top:
                if A == B:
                    cells.append("—")
                    continue
                r = P[(A, B)]
                cells.append(f"{frac(r['dom'])}/{frac(r['shared'])}" if r["shared"] else "")
            L.append(f"| **{A}** | " + " | ".join(cells) + " |")
        L.append("")

    # exceptions: where the usual outcome flips
    def sentences(A, B):
        """One line per measure (dominance, VSE*, WME*): who is ahead how often, and where the other is."""
        r_ab, r_ba = P[(A, B)], P[(B, A)]
        n = r_ab["shared"]
        if n == 0:
            return []
        out_l = []
        for x, what in (("dom", "dominates"), ("VSE*", "VSE\\*"), ("WME*", "WME\\*")):
            ka, kb = r_ab[x], r_ba[x]
            if ka == 0 and kb == 0:
                continue
            hi, lo, kh, kl = (A, B, ka, kb) if ka >= kb else (B, A, kb, ka)
            tl = r_ba if hi == A else r_ab
            tags_l = tl["dom_tags"] if x == "dom" else tl[x + "_tags"]
            if x == "dom":
                s = f"{hi} dominates {lo} in {frac(kh)} of {frac(n)} conditions"
                other = f"{lo} dominates {hi}"
            else:
                s = f"{what}: {hi} clearly better than {lo} in {frac(kh)} of {frac(n)} conditions"
                other = f"{lo} better"
            if kl > 0:
                common = common_setting([u["diff"] for u in tags_l])
                def tag(u):
                    w = weight(u, A, B)
                    return f"{u['tag']} ({u['label']}" + (f"; counts {w:.2g}" if w < 1 - 1e-9 else "") + ")"
                s += (f"; {other} in {frac(kl)}: " + ", ".join(tag(u) for u in tags_l)
                      + (f", all with {common}" if common and len(tags_l) > 1 else ""))
            out_l.append(s + ".")
        return out_l

    if requested:
        L += ["## Requested pairs", "", "The pairs asked for with --pairs, in that order: how often each method is "
              "ahead, and where the other one is.", ""]
        for A, B in requested:
            L.append(f"**{A} — {B}**")
            L.append("")
            missing = [x for x in (A, B) if x not in meth]
            if missing:
                L.append(f"- Not compared: {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not among the "
                         "compared methods (not in these runs, left out by --methods / --exclude, or a reference or "
                         "inverse method, which need --methods).")
            elif P[(A, B)]["shared"] == 0:
                L.append("- Never run in the same condition.")
            else:
                sn = sentences(A, B)
                L += [f"- {x}" for x in sn] if sn else [
                    f"- No clear difference on any of the front axes in any of the {frac(P[(A, B)]['shared'])} "
                    "conditions both were run in."]
            L.append("")
    seen = {frozenset(p) for p in requested}
    blocks = []
    for A, B in [(A, B) for i, A in enumerate(top) for B in top[i + 1:]]:
        if frozenset((A, B)) in seen:
            continue
        seen.add(frozenset((A, B)))
        r_ab, r_ba = P[(A, B)], P[(B, A)]
        if any(min(r_ab[x], r_ba[x]) > 0 for x in ("dom", "VSE*", "WME*")):
            sn = sentences(A, B)
            if sn:
                blocks.append((A, B, sn))
    if blocks:
        L += ["## Exceptions", "",
              "Pairs of leading methods where each is ahead somewhere: how often each is ahead, and the conditions "
              "where the less usual outcome happens, with the settings they share. Pairs asked for with --pairs "
              "are above, not repeated here.", ""]
        for A, B, sn in blocks:
            L.append(f"**{A} — {B}**")
            L.append("")
            L += [f"- {x}" for x in sn]
            L.append("")
    elif len(top) > 1:
        L += ["## Exceptions", "", "Among the leading methods no pair reverses: where one is ahead, it is ahead in "
              "every shared condition where there is a clear difference.", ""]

    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(an.md_stars("\n".join(L)) + "\n")
    print(f"wrote {out}/: report.md, methods.csv, pairs.csv, conditions.csv, command.txt "
          f"({n_cond} conditions, {len(meth)} methods)")


if __name__ == "__main__":
    main()
