"""Run honest and strategic elections for all methods on one scenario and write CSV.

Usage:
    python run.py --elections 200                      # defaults: see docs/running.md, Parameters

Each electorate is tallied twice per method: with sincere ballots, and after the groups
(voters with the same sincere ranking) have answered polls strategically (see strategy.py: poll
rounds by default, convergent learning with --converge). The "learned" rows hold those ballots.

Axes of the Pareto front (0 = random ballot, 100 = best candidate; see metrics.py):
    VSE, WME     voter satisfaction and worst-margin efficiency, sincere ballots
    VSE*, WME*   the same with learned ballots
CE, CE* (Condorcet efficiency) are reported alongside, as are the sincerity of the learned
ballots (see strategy.sincerity) and the share of elections where the strategic ballots settled
(poll rounds: the last round repeated the one before).

Partial strategy: --strategic p makes a random share p of the voters always strategic (the same
voters for every method). --adaptive a makes a further share a adaptive: each poll round they vote
strategically if the race between the two leaders matters enough to them. The rest always vote
sincerely. Default: no one always strategic, everyone adaptive (--adaptive left out: everyone not
in --strategic; with --converge nobody, as adaptive voters need poll rounds).

Outputs, in one folder per run (--out; a name without a directory goes into output/; default
run-<date>-<time>). `python analyze.py <folder>` puts its analysis into <folder>/analysis/.
    elections.csv   one row per (election, method, ballots), with the run's parameters as columns,
                    so several runs can be combined (see analyze.py)
    summary.csv     one row per method, with on_front for the VSE*/WME*/sincerity front
    table.txt       the printed table
    command.txt     the command, simulator version and start time (a resumed run adds its own)
Every election is saved as soon as it is done; after an interruption the same command with
--resume continues where it stopped (see docs/running.md, Progress, saving and resuming).
"""
import argparse
import contextlib
import csv
import fnmatch
import hashlib
import os
import shlex
import sys
import time
import zlib
from multiprocessing import Pool
import numpy as np

from scenario import Scenario, electorate_positions
from models import draw_electorate, model_text, MODELS
from nomination import nominate, clone_setup, utilities, to_original
from methods import METHODS, run_selection, set_max_score, set_lazy, set_voter_lazy, set_voter_approval, sincerity_defined
from strategy import learn, respond, strategy_types, default_poll_size, fidelity
from metrics import election_row, summarize, pareto, worst_margins, electorate_stats, ballot_condorcet, ties, FRONT_TOLERANCE
import participation as part
import manipulation as manip


def lazy_cuts(rng, n, lazy):
    """Each voter's laziness cut, uniform between 0 and `lazy` (--lazy); None without laziness."""
    return rng.random(n) * lazy if lazy > 0 else None


def personal_thresholds(draw, share, adapt):
    """Adaptive voters: those whose draw lies in [share, share + adapt) (after the always-strategic share).
    Their position in that band, rescaled to 0..1, is their personal threshold: they vote strategically
    in a round when their stake in the leading race exceeds it (strategy.py). nan = not adaptive;
    None without adaptive voters."""
    if adapt <= 0:
        return None
    inside = (draw >= share) & (draw < share + adapt)
    return np.where(inside, (draw - share) / adapt, np.nan)


def one_election(job):
    """One electorate, every method in job["names"]: sincere and learned rows for each number of poll
    rounds. job is a dict:
        sc            the Scenario (electorate generator)
        p             the run parameters (keys as in PARAMS; "rounds" is not used here)
        e             the election number (with p["seed"], it fixes every random stream)
        rounds        list of numbers of poll rounds to report
        names         the methods
        participation also run the no-show check (participation.py)
        manipulation  also run the coalition manipulation test (manipulation.py) on the sincere ballots
        dummy         --start previous: the previous election's ballots per method, else None
    A new run parameter only needs to be in PARAMS and be read from p here."""
    sc, p, e, rounds, names = job["sc"], job["p"], job["e"], job["rounds"], job["names"]
    participation, dummy = job.get("participation", False), job.get("dummy")
    manipulating = job.get("manipulation", False)
    seed, samples, group_size, share = p["seed"], p["samples"], p["group_size"], p["strategic"]
    adapt = p.get("adaptive", 0.0)
    exits, clones, poll_size, poll_error = p["exits"], p["clones"], p["poll_size"], p["poll_error"]
    search, converge, aggr, start = p["search"], p["converge"], p["aggressive"], p["start"]
    set_max_score(p["max_score"])                   # per process: workers do not share module state
    set_lazy(p.get("lazy", 0.0))                    # laziness (methods.set_lazy)
    rng = np.random.default_rng([seed, e])
    ele = draw_electorate(sc, rng)                  # the scenario's model (models.py)
    u, voters, cands = ele.u, ele.voters, ele.cands
    C = u.shape[1]
    # laziness: each voter's own cut, uniform between 0 and --lazy (own stream, the same for every method)
    set_voter_lazy(lazy_cuts(np.random.default_rng([seed, e, zlib.crc32(b"lazy")]), len(u), p.get("lazy", 0.0)))
    # honest approval: each voter's draw for their threshold between mean and favourite (ballots.approves)
    set_voter_approval(np.random.default_rng([seed, e, zlib.crc32(b"approval")]).random(len(u)))
    nominating = bool(exits or clones)
    if nominating:   # clones' positions: own random stream, the same for every method
        u_clone, cu = clone_setup(voters, cands, sc.width,
                                  np.random.default_rng([seed, e, zlib.crc32(b"clones")]),
                                  ele.voter_weights, ele.cand_weights, getattr(sc, "utility", "exp"))
    m_cand = worst_margins(u)                       # each candidate's worst head-to-head margin
    estats = electorate_stats(u)                    # properties of the electorate, for the report
    # every candidate's total utility and worst margin, for the frontier of what any rule could reach
    estats["U_cands"] = ";".join(f"{x:.6g}" for x in u.sum(axis=0))
    estats["WM_cands"] = ";".join(f"{x:.6g}" for x in m_cand)
    # every candidate's total of the voters' normalized utilities (each voter stretched to 0..1, least to
    # most liked), for the frontier of normalized ballots (analyze.normalized_frontier)
    mn, mx = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    estats["Un_cands"] = ";".join(f"{x:.6g}" for x in ((u - mn) / np.where(mx - mn > 1e-12, mx - mn, 1)).sum(axis=0))
    # strategic voters: own random stream, the same voters for every method
    # adaptive voters: the next share of the same draw; their place in it is their personal threshold (0..1)
    draw = np.random.default_rng([seed, e, zlib.crc32(b"strategic")]).random(len(u))
    strategic = draw < share
    adaptive = personal_thresholds(draw, share, adapt)
    # aggressive strategic voters (exaggerate when it cannot hurt) vs kind ones: own stream, same for every method
    aggressive = np.random.default_rng([seed, e, zlib.crc32(b"aggressive")]).random(len(u)) < aggr
    out = []
    for name in names:
        m = METHODS[name]
        with m.applied():                           # the method's own settings, if any (methods.SETTINGS)
            # Strategic nomination (nomination.py): the field the voters face. Without it, the original field.
            # Results are always judged on the original field (a clone's win counts as its original's).
            uf, back = u, (lambda P: P)
            nstats = {"nom_exits": 0, "nom_clones": 0, "nom_settled": 1, "nom_moves": 0, "nom_changed": 0}
            if nominating:
                # the candidates' order of moves is a property of the election: the same stream for every
                # method, so methods that agree on every field (e.g. IRV and ER-IRV on strict ballots) agree
                nrng = np.random.default_rng([seed, e, zlib.crc32(b"nomination")])
                field, st = nominate(m, u, u_clone, cu, nrng, exits, clones)
                ents = field.entries()
                uf, back = utilities(ents, u, u_clone), (lambda P, ents=ents: to_original(P, ents, C))
            P_h = back(m.honest(uf))
            if nominating:
                nstats = {**st, "nom_changed": int(np.abs(P_h - m.honest(u)).sum() > 1e-9)}
            check = (lambda B: part.no_show(m, B, uf, u, back)) if participation else (lambda B: part.NAN)
            defined = sincerity_defined(name)          # blank for TRASH and the Inverse-* methods
            sincere_row = {**election_row(u, P_h, m_cand), **estats, "sincerity": 1.0 if defined else "",
                       "fidelity": float(fidelity(m, m.sincere(uf), uf).mean()) if defined else "", "settled": 1,
                           "adapted": "", "compromise": 0.0, "pushover": 0.0, "burial": 0.0, **nstats, **check(m.sincere(uf))}
            # Each method learns with a fresh copy of the same stream (common random numbers: the same
            # polls for every method as long as they draw alike), so results stay the same when methods
            # are added or reordered, and paired comparisons are not blurred by different poll draws.
            mrng = np.random.default_rng([seed, e, zlib.crc32(b"learning")])
            # --start previous: the first poll shows the previous election's result (the same dummy
            # election for every election), its candidates relabelled at random as this election's
            B0 = None
            if start == "previous":
                perm = np.random.default_rng([seed, e, zlib.crc32(b"relabel")]).permutation(C)
                B0 = np.empty_like(dummy[name])
                B0[:, perm] = dummy[name]
            act = {}
            if converge > 0:     # convergent learning does not use rounds: one result for every rounds value
                res = [learn(m, uf, mrng, converge, samples, group_cap(group_size, len(u)), strategic)] * len(rounds)
            else:                # all requested numbers of poll rounds from one run (see strategy.respond)
                act = {}
                res = respond(m, uf, mrng, samples=samples, poll_size=poll_size, search=search, strategic=strategic,
                              poll_error=poll_error, report=rounds, aggressive=aggressive, start=B0,
                              adaptive=adaptive, activated=act)
            sincere_row.update(ballot_condorcet(m, m.sincere(uf), u, back, P_h))
            sincere_row.update(ties(m, m.sincere(uf), P_h))
            sincere_row.update(manip.manipulation(m, m.sincere(uf), uf, u, back) if manipulating else manip.NAN)
            for r, (B, settled, sin) in zip(rounds, res):
                P_l = back(m.tally(B))
                out.append({"rounds": r, "election": e, "method": name, "ballots": "sincere", **sincere_row})
                out.append({"rounds": r, "election": e, "method": name, "ballots": "learned",
                            **election_row(u, P_l, m_cand), **estats, **ballot_condorcet(m, B, u, back, P_l),
                            **ties(m, B, P_l),
                            "sincerity": sin if defined else "",
                            "fidelity": float(fidelity(m, B, uf).mean()) if defined else "",
                            "settled": int(settled), "adapted": act[r] if adaptive is not None else "",
                            **strategy_types(B, m.pref(uf)), **nstats, **check(B), **manip.NAN})
    set_voter_lazy(None)
    set_voter_approval(None)
    return out


OUTPUT_DIR = "output"
VERSION = "1.1.0"      # written into every row (column "version") and shown in the analysis report


def out_path(name):
    """Output prefix or directory: a bare name goes into OUTPUT_DIR (created if needed)."""
    path = name if os.path.dirname(name) else os.path.join(OUTPUT_DIR, name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


# What the lab writes into its folders. An existing folder given as an output must hold only these
# (and hidden files, e.g. .DS_Store), so that a mistyped --out (a home or project folder) is refused
# instead of having files written into it, or overwritten.
RUN_FILES = ["elections.csv", "elections.csv.tmp", "summary.csv", "table.txt", "command.txt"]
SWEEP_FILES = ["command.txt", "elections.csv"]          # elections.csv: sweeps before runs/
ANALYSIS_FILES = ["report.md", "summary.csv", "command.txt", "*.png"]
OWN_DIRS = ["analysis", "analysis-vs-*", "runs"]


def check_folder(path, files, dirs=(), hint="choose another --out"):
    """Refuse an existing folder that holds anything the lab does not write there (files: names or
    patterns; dirs: names or patterns of subfolders)."""
    if not os.path.exists(path):
        return
    if not os.path.isdir(path):
        raise SystemExit(f"{path} exists and is not a folder; {hint}")
    foreign = []
    for n in sorted(os.listdir(path)):
        if n.startswith("."):
            continue
        pats = dirs if os.path.isdir(os.path.join(path, n)) else files
        if not any(fnmatch.fnmatch(n, p) for p in pats):
            foreign.append(n)
    if foreign:
        raise SystemExit(f"{path} holds things the lab did not write there ({', '.join(foreign[:6])}"
                         f"{' …' if len(foreign) > 6 else ''}); {hint}")


def run_dir(name, kind, overwrite=False, resume=False):
    """Folder for one run or sweep, which also takes its analysis: output/<name>/ (a name with a
    directory is used as given). Without a name: <kind>-<date>-<time>, so nothing is overwritten.
    An existing run is only replaced with overwrite, or continued with resume. Checked before
    simulating."""
    if resume and not name:
        raise SystemExit("--resume needs --out: the folder of the interrupted run")
    if not name:
        name = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}"
    path = name if os.path.dirname(name) else os.path.join(OUTPUT_DIR, name)
    runs = os.path.join(path, "runs")
    has = os.path.exists(os.path.join(path, "elections.csv")) or (os.path.isdir(runs) and bool(os.listdir(runs)))
    if resume and not has:
        raise SystemExit(f"{path} holds no run to resume")
    if has and not overwrite and not resume:
        raise SystemExit(f"{path} already holds a run; choose another --out, add --overwrite, "
                         "or --resume to continue it")
    check_folder(path, RUN_FILES if kind == "run" else SWEEP_FILES, OWN_DIRS)
    if has and overwrite:
        old = [d for d in sorted(os.listdir(path)) if d.startswith("analysis") and os.path.isdir(os.path.join(path, d))]
        if old:
            print(f"note: {', '.join(old)} in {path} belong to the run being replaced; analyse again")
    os.makedirs(path, exist_ok=True)
    return path


def write_command(path, resume=False):
    """command.txt: the exact command, the simulator version and the start time, for reproducing.
    A resumed run adds its command below the first."""
    with open(os.path.join(path, "command.txt"), "a" if resume else "w") as f:
        f.write(f"python {' '.join(shlex.quote(x) for x in sys.argv)}\n")
        f.write(f"simulator version {VERSION}, started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def resolve_salience(p):
    """The clusters model's salience follows --cluster-salience-exponent: 0 means every issue weighed
    alike ("none", the same electorates as weights of 1), above 0 salience per cluster ("cluster").
    Kept as a column so older files stay readable."""
    return {**p, "salience": "cluster" if p.get("salience_exponent", 0) > 0 else "none"}


def resolve_adaptive(p):
    """--adaptive left out (None): every voter who is not always strategic is adaptive (1 - --strategic),
    or nobody with --converge (adaptive voters need poll rounds). Returns p with a number."""
    if p.get("adaptive") is None:
        p = {**p, "adaptive": 0.0 if p.get("converge", 0) > 0 else round(1.0 - p["strategic"], 12)}
    return p


def valid_combination(p):
    """A condition the simulation can run: shares of strategic and adaptive voters within 1, and no
    adaptive voters with convergent learning (sweep.py skips the others)."""
    return p["strategic"] + p["adaptive"] <= 1 + 1e-9 and not (p["adaptive"] > 0 and p["converge"] > 0)


def check_params(a, error):
    """Reject values the simulation cannot use (a: parsed arguments; lists allowed, as in sweep.py)."""
    vals = lambda k: getattr(a, k) if isinstance(getattr(a, k), list) else [getattr(a, k)]
    for k, lo in (("rounds", 0), ("samples", 1), ("search", 0), ("poll_size", 0), ("converge", 0), ("clones", 0)):
        if any(v < lo for v in vals(k)):
            error(f"--{k.replace('_', '-')} must be at least {lo}")
    for k in ("strategic", "adaptive", "aggressive", "lazy"):
        if any(v is not None and not 0 <= v <= 1 for v in vals(k)):
            error(f"--{k} must be between 0 and 1")
    # (sweep.py skips the combinations that break these two; here only all of them breaking is an error;
    # an --adaptive left out (None) adapts to the others, see resolve_adaptive)
    if min(vals("converge")) > 0 and max(vals("strategic")) == 0:
        error("--converge needs a --strategic share (adaptive voters, the default, need poll rounds)")
    given = [v for v in vals("adaptive") if v is not None]
    if given and len(given) == len(vals("adaptive")):
        if min(vals("strategic")) + min(given) > 1 + 1e-9:
            error("--strategic and --adaptive together must be at most 1 (shares of the voters)")
        if min(given) > 0 and min(vals("converge")) > 0:
            error("--adaptive is for poll rounds: not with --converge")
    if any(v <= 0 for v in vals("field_kappa")):
        error("--field-kappa must be above 0")
    if any(v < 0 for v in vals("noise_strength")):
        error("--noise-strength must be 0 or more")
    if any(v < 0 for v in vals("field_salience")):
        error("--field-salience must be 0 or more")
    if any(v < 0 for v in vals("salience_exponent")):
        error("--cluster-salience-exponent must be 0 or more")
    if any(v <= 0 for v in vals("ks_alpha")):
        error("--ks-alpha must be above 0")
    if any(not 0 < v < 1 for v in vals("ks_cut")):
        error("--ks-cut must be between 0 and 1")
    if any(v < 0 for v in vals("poll_error")):
        error("--poll-error must be 0 or more")
    if "previous" in vals("start") and (any(vals("exits")) or any(vals("clones"))):
        error("--start previous needs the original field: not with --exits or --clones")
    if "previous" in vals("start") and any(vals("converge")):
        error("--start previous is for poll rounds, not --converge")


def cand_text(disp, shift):
    """', candidates: ...' when they are not drawn like the voters, else ''."""
    parts = ([f"dispersion {disp}"] if float(disp) != 1.0 else []) + ([f"shift {shift}"] if float(shift) != 0.0 else [])
    return f", candidates: {', '.join(parts)}" if parts else ""


def width_text(p):
    """The utility fall-off of a run: "width 0.7" (exp), or "linear utility" / "quadratic utility" (width
    only a scale then)."""
    g = (lambda k: p.get(k, "exp")) if isinstance(p, dict) else (lambda k: getattr(p, k, "exp"))
    return f"{g('utility')} utility" if g("utility") in ("linear", "quadratic") else f"width {g('width')}"


def strategy_text(p, n):
    """One-line description of the strategy model of a run (p: namespace or dict of parameters)."""
    g = (lambda k: p[k]) if isinstance(p, dict) else (lambda k: getattr(p, k))
    if int(g("converge")) > 0:
        return (f"convergent learning {g('converge')} rounds x {g('samples')} poll samples, "
                f"group size {group_text(g('group_size'))}")
    ps = int(g("poll_size")) or default_poll_size(int(n))
    return (f"{g('rounds')} poll round(s), {g('samples')} polls of {ps} voters, poll error {g('poll_error')}x, "
            f"search {g('search')}, aggressive share {g('aggressive')}"
            + (f", laziness {g('lazy')}" if float(g("lazy")) > 0 else "")
            + ("; first poll from a previous election" if g("start") == "previous" else ""))


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
PARAMS = ["voters", "cands", "model", "dims", "zipf", "clusters", "spread", "salience", "salience_exponent", "fixed_variance", "field_kappa", "field_salience", "noise_strength", "ks_alpha", "ks_cut", "ks_shared", "width", "utility", "cand_dispersion", "cand_shift", "rounds", "samples", "group_size",
          "strategic", "adaptive", "aggressive", "lazy", "max_score", "exits", "clones", "poll_size", "poll_error", "search", "converge",
          "start", "seed"]


def previous_election(args):
    """The dummy election for --start previous: an extra electorate (its own random stream) under the
    same settings, through the same number of poll rounds; returns its final ballots."""
    sc, p, rounds, name = args
    set_max_score(p["max_score"])
    set_lazy(p.get("lazy", 0.0))
    rng = np.random.default_rng([p["seed"], zlib.crc32(b"previous")])
    u = electorate_positions(sc, rng)[0]
    set_voter_lazy(lazy_cuts(np.random.default_rng([p["seed"], zlib.crc32(b"previous-lazy")]), len(u), p.get("lazy", 0.0)))
    set_voter_approval(np.random.default_rng([p["seed"], zlib.crc32(b"previous-approval")]).random(len(u)))
    draw = rng.random(len(u))
    strategic = draw < p["strategic"]
    adaptive = personal_thresholds(draw, p["strategic"], p.get("adaptive", 0.0))
    aggressive = rng.random(len(u)) < p["aggressive"]
    m = METHODS[name]
    with m.applied():
        B, _, _ = respond(m, u, np.random.default_rng([p["seed"], zlib.crc32(b"previous-learning")]), rounds=rounds,
                          samples=p["samples"], poll_size=p["poll_size"], search=p["search"], strategic=strategic,
                          poll_error=p["poll_error"], aggressive=aggressive, adaptive=adaptive)
    set_voter_lazy(None)
    set_voter_approval(None)
    return B


def _fmt_time(t):
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


class Progress:
    """Elections done, time used and an estimate of the time left, on one line (stderr): in a
    terminal redrawn in place after every election; otherwise (output to a file) a new line at
    most `updates` times per run (default 10, one per tenth). before: elections finished earlier
    (--resume), counted in the numbers shown but not in the time estimate."""

    def __init__(self, total, label="", updates=None, before=0):
        self.total, self.label, self.done, self.t0 = total, label, 0, time.time()
        self.before = before
        self.tty = sys.stderr.isatty()
        self.updates = self.total if self.tty else (updates or 10)
        self.next_mark = 1

    def step(self):
        self.done += 1
        if self.done * self.updates < self.next_mark * self.total and self.done < self.total:
            return
        while self.next_mark * self.total <= self.done * self.updates:
            self.next_mark += 1
        el = time.time() - self.t0
        left = el / self.done * (self.total - self.done)
        text = (f"{self.label}{self.done + self.before}/{self.total + self.before}, {_fmt_time(el)} so far, "
                f"~{_fmt_time(left)} left")
        if self.tty:     # \r and clear to the end of the line: redraw in place
            print("\r\033[K" + text, end="", file=sys.stderr, flush=True)
        else:
            print(text, file=sys.stderr, flush=True)

    def close(self):
        if self.tty and self.done:
            print(file=sys.stderr, flush=True)


def _ignore_sigint():     # workers: Ctrl-C goes to the main process only, which stops the pool
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def simulate(p, elections, workers=None, names=None, participation=False, sink=None, skip=(), label="",
             manipulation=False, fingerprint=None, progress=True):
    """Run one condition. p: dict with the keys in PARAMS; names: methods (default all);
    participation: also run the no-show check (participation.py) on sincere and learned ballots;
    manipulation: also run the coalition manipulation test (manipulation.py) on the sincere ballots.
    p["rounds"] may be a list of numbers of poll rounds: they all come from one run (the rounds
    up to the highest are computed once), one condition per value, in that order.
    sink: called with each election's rows as soon as it is done (in election order), so a
    caller can save as it goes; skip: election numbers already done (resuming), not run again.
    Every election depends only on its seed and number, so skipped and new ones fit together.
    fingerprint: {method: fingerprint} (see fingerprints), written into each row as a column.
    workers: processes (None: one per core); 0: no worker processes, everything in this one.
    Shows progress (see Progress) unless progress is False. Returns the new rows with p as columns."""
    names = names or list(METHODS)
    rounds = list(p["rounds"]) if isinstance(p["rounds"], (list, tuple)) else [p["rounds"]]
    sc = Scenario(p["voters"], p["cands"], p["dims"], p["clusters"], p["spread"], p["width"], p["zipf"],
                  p["cand_dispersion"], p["cand_shift"], p.get("model", "clusters"), p.get("ks_alpha", 1.5),
                  p.get("ks_cut", 0.2), p.get("ks_shared", 0), salience=p.get("salience", "none"),
                  salience_exponent=p.get("salience_exponent", 1.0), fixed_variance=p.get("fixed_variance", 0),
                  field_kappa=p.get("field_kappa", 1.5), field_salience=p.get("field_salience", 0.0),
                  noise_strength=p.get("noise_strength", 3.0), utility=p.get("utility", "exp"))
    todo = [e for e in range(elections) if e not in set(skip)]
    jobs = [{"sc": sc, "p": p, "e": e, "rounds": rounds, "names": names, "participation": participation,
             "manipulation": manipulation} for e in todo]
    meta = {k: p[k] for k in PARAMS if k != "rounds"} | {"version": VERSION}
    rows = []
    if not todo:
        return rows
    prog = Progress(len(todo), label, before=elections - len(todo)) if progress else None
    # workers=0: everything in this process, without a pool of worker processes (for callers that are
    # not a script with a __main__ guard, such as test_methods.py: with the "spawn" and "forkserver"
    # start methods, the default on macOS, Windows and from Python 3.14 on Linux, each worker imports
    # the main script again). The settings one_election puts into methods.py are restored afterwards.
    import methods as _m
    saved = (_m.MAXS, _m.LAZY, _m.VOTER_LAZY, _m.VOTER_APPROVAL) if workers == 0 else None
    try:
        with (Pool(workers, initializer=_ignore_sigint) if workers != 0 else contextlib.nullcontext()) as pool:
            pmap = pool.map if pool else (lambda f, xs: list(map(f, xs)))
            pimap = (lambda f, xs: pool.imap(f, xs, chunksize=1)) if pool else map
            dummy = None
            if p["start"] == "previous":          # one previous election for the whole batch, per method
                dummy = dict(zip(names, pmap(previous_election, [(sc, p, max(rounds), n) for n in names])))
            jobs = [{**j, "dummy": dummy} for j in jobs]
            for batch in pimap(one_election, jobs):   # in order: the file stays reproducible
                batch = [{**meta, **r} for r in batch]
                batch = [{k: r[k] for k in PARAMS} | r for r in batch]      # parameters first, as columns
                if fingerprint is not None:
                    for r in batch:
                        r["fingerprint"] = fingerprint.get(r["method"], "")
                if sink:
                    sink(batch)
                rows += batch
                if prog:
                    prog.step()
    finally:
        if prog:
            prog.close()
        if saved is not None:
            set_max_score(saved[0])
            set_lazy(saved[1])
            set_voter_lazy(saved[2])
            set_voter_approval(saved[3])
    rows.sort(key=lambda r: rounds.index(r["rounds"]))          # stable: one condition after another
    return rows


def numeric(row):
    """A row read back from elections.csv with its numbers as floats (for summarize)."""
    out = {}
    for k, v in row.items():
        if k in ("method", "ballots"):       # names, even when they look like numbers (321)
            out[k] = v
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def run_key(row, params):
    """Identifies a run (everything but rounds and the election) in rows as written to CSV."""
    return tuple(str(row[k]) for k in params if k != "rounds")


# Fingerprints. The version number does not change while the model and the methods are developed, so
# each run also records, per method, a fingerprint of what the code computes: the method's results in
# a small fixed test condition (6 elections, 31 voters, 6 candidates, some voters strategic and some
# adaptive), hashed. Same fingerprint: the method (and the model behind it) gives the same results as when the
# other run was made; a different one: the code changed in between, and results should not be pooled.
# Since 2026-10-26 a second small condition is added, made for ties (FINGERPRINT_TIES: 20 voters, an
# even number, 5 candidates, laziness 0.5): every method on ordered ballots meets pairwise ties there,
# and several results are split, so a change in how a method handles ties is more likely seen (the
# seed is chosen so that the BTR tie rule of 2026-10-26 changes the fingerprints of both BTR-score and
# BTR-Borda; with the first condition alone BTR-Borda's did not change). Seen only if a tie decides
# one of these few elections, so not guaranteed for every tie rule.
# It costs a few seconds per run (about 6 s for the standard set). Methods that behave alike in these
# elections share a fingerprint, so a change that only shows in other elections can be missed. Changes that only show under other settings (another electorate
# model, nomination, the linear utility) are not seen by it.
FINGERPRINT_ELECTIONS = 6
FINGERPRINT_PARAMS = {"voters": 31, "cands": 6, "model": "clusters", "dims": 3, "zipf": 1, "clusters": 3,
                      "spread": 0.5, "salience": "none", "salience_exponent": 1.0, "fixed_variance": 0,
                      "field_kappa": 1.5, "field_salience": 0.0, "noise_strength": 3.0, "ks_alpha": 1.5,
                      "ks_cut": 0.2, "ks_shared": 0, "width": 1.0, "utility": "exp", "cand_dispersion": 1.0,
                      "cand_shift": 0.0, "rounds": 2, "samples": 3, "group_size": 10, "strategic": 0.3,
                      "adaptive": 0.5, "aggressive": 0.5, "lazy": 0.0, "max_score": 5, "exits": 0, "clones": 0,
                      "poll_size": 0, "poll_error": 2.0, "search": 4, "converge": 0, "start": "honest", "seed": 7}
FINGERPRINT_TIES = {**FINGERPRINT_PARAMS, "voters": 20, "cands": 5, "lazy": 0.5, "seed": 9}
FINGERPRINT_COLS = ["U_method", "WM", "sincerity", "fidelity", "compromise", "burial", "pushover",
                    "U_best", "U_rb", "WM_best", "WM_rb"]


def fingerprints(names, workers=None):
    """{method: 10 hex digits} from each method's results in FINGERPRINT_PARAMS and FINGERPRINT_TIES
    (8 significant digits, so that rounding in the last digits between machines does not change it)."""
    by = {}
    for c, params in enumerate((FINGERPRINT_PARAMS, FINGERPRINT_TIES)):
        for r in simulate(params, FINGERPRINT_ELECTIONS, workers, names, progress=False):
            by.setdefault(r["method"], []).append((c, r))
    out = {}
    for m, crs in by.items():
        crs.sort(key=lambda cr: (cr[0], int(cr[1]["election"]), cr[1]["ballots"]))
        rs = [r for _, r in crs]
        text = "|".join(",".join(f"{float(r[c]):.8g}" if r.get(c) not in ("", None) else "" for c in FINGERPRINT_COLS)
                        for r in rs)
        out[m] = hashlib.sha256(text.encode()).hexdigest()[:10]
    return out


def file_fingerprints(path):
    """{method: set of fingerprints} recorded in a results file (empty sets for files from before)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["method"], set()).update([r["fingerprint"]] if r.get("fingerprint") else [])
    return out


def check_resume_fingerprints(path, fp):
    """--resume: refuse if a method's code changed since the run started (its fingerprint differs)."""
    changed = sorted(m for m, fs in file_fingerprints(path).items() if fs and m in fp and fs != {fp[m]})
    if changed:
        raise SystemExit(f"{path}: {', '.join(changed[:6])}{' …' if len(changed) > 6 else ''} give(s) other results "
                         "now than when this run started (fingerprint changed): the code changed in between. "
                         "Start a new run instead of resuming")


def resume_state(path, names, rounds_per_run):
    """Read an interrupted elections.csv for --resume: keep the elections that are complete (every
    method, both ballot sets, every requested number of rounds), rewrite the file with only those,
    and return (done, header): done maps a run key (see run_key) to the set of finished election
    numbers. rounds_per_run: run key -> the poll rounds values written per election (the runs of
    this command). Stops if the file holds other methods, other parameters or another simulator
    version."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    if not rows:
        return {}, header
    missing = [k for k in PARAMS if k not in header]
    if missing:
        raise SystemExit(f"{path} predates the column(s) {', '.join(missing)}: it cannot be resumed; rerun it "
                         "(it can still be analysed)")
    if {r["method"] for r in rows} - set(names) or {r.get("version") for r in rows} != {VERSION}:
        raise SystemExit(f"{path} was made with other methods or another simulator version; "
                         "--resume needs the same command")
    new = sorted(set(names) - {r["method"] for r in rows})
    if new:          # otherwise no election would count as complete and the run would start over
        raise SystemExit(f"{path} lacks the method(s) {', '.join(new)} (added to the selection since, e.g. a new "
                         "add-on with --methods all); --resume needs the same methods. Leave them out with "
                         "--exclude, and run them separately later to combine (docs/running.md, Combining runs)")
    if any(run_key(r, PARAMS) not in rounds_per_run
           or r["rounds"] not in {str(x) for x in rounds_per_run[run_key(r, PARAMS)]} for r in rows):
        raise SystemExit(f"{path} was made with other parameters; --resume needs the same command")
    count = {}
    for r in rows:
        key = (run_key(r, PARAMS), int(r["election"]))
        count[key] = count.get(key, 0) + 1
    complete = {k for k, c in count.items() if c == len(names) * 2 * len(rounds_per_run[k[0]])}
    kept = [r for r in rows if (run_key(r, PARAMS), int(r["election"])) in complete]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(kept)
    os.replace(tmp, path)
    done = {}
    for k, e in complete:
        done.setdefault(k, set()).add(e)
    return done, header


class CsvSink:
    """Appends rows to elections.csv as elections finish (flushed each time), writing the header
    first unless the file already has one (resuming)."""

    def __init__(self, path, header=None):
        self.f = open(path, "a" if header else "w", newline="")
        # resuming: the file's own columns; a column added to the lab since the run started is left out
        self.w = csv.DictWriter(self.f, fieldnames=header, extrasaction="ignore") if header else None

    def __call__(self, rows):
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(rows[0]))
            self.w.writeheader()
        self.w.writerows(rows)
        self.f.flush()

    def close(self):
        self.f.close()


def interrupted(out, n, where="elections.csv"):
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    if "--resume" not in sys.argv:
        cmd += " --resume" + ("" if "--out" in sys.argv else f" --out {shlex.quote(out)}")
    raise SystemExit(f"\ninterrupted; {n} complete election(s) are saved in {out}/{where}. "
                     f"Continue with:\n  python {cmd}")


def parser():
    """run.py's command-line options (also read by compare.py for the defaults)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--elections", type=int, default=200)
    ap.add_argument("--voters", type=int, default=99)
    ap.add_argument("--cands", type=int, default=6)
    ap.add_argument("--model", default="noise", choices=list(MODELS),
                    help="electorate model (models.py); the --cluster-* options are for 'clusters'")
    ap.add_argument("--dims", type=int, default=6)
    ap.add_argument("--cluster-zipf", dest="zipf", type=int, default=0, choices=[0, 1],
                    help="1: dimension i has scale 1/i (Zipf dimensions, see scenario.py)")
    ap.add_argument("--cluster-count", dest="clusters", metavar="CLUSTER_COUNT", type=int, default=20)
    ap.add_argument("--cluster-spread", dest="spread", metavar="CLUSTER_SPREAD", type=float, default=0.5)
    ap.add_argument("--cluster-salience-exponent", dest="salience_exponent", metavar="CLUSTER_SALIENCE_EXPONENT", type=float, default=0.5,
                    help="model clusters: each cluster weighs the issues 1/i^s in its own random order (1 = Zipf; "
                         "0 = every issue weighed alike, no salience; see models.py)")
    ap.add_argument("--cluster-fixed-variance", dest="fixed_variance", type=int, default=1, choices=[0, 1],
                    help="model clusters: 1 = scale positions so the dimensions' total variance is pi^2/6 whatever "
                         "their number (more dimensions spread the same disagreement over more issues)")
    ap.add_argument("--field-kappa", type=float, default=1.5,
                    help="model field: saturation of opinions (small: smooth, nearly Gaussian; large: two camps per issue)")
    ap.add_argument("--field-salience", type=float, default=0.0,
                    help="model field: strength of the salience field (0 = every issue weighed alike)")
    ap.add_argument("--noise-strength", type=float, default=5.0,
                    help="model noise: how strongly voters gather where the fractal noise is high "
                         "(0 = a Gaussian blob; larger: sharper clumps)")
    ap.add_argument("--ks-alpha", type=float, default=1.5,
                    help="model ks: faction alpha (higher: more, smaller factions)")
    ap.add_argument("--ks-cut", type=float, default=0.2,
                    help="model ks: issue importance below which no further bundles or dimensions are drawn "
                         "(lower: more dimensions)")
    ap.add_argument("--ks-shared", type=int, default=0, choices=[0, 1],
                    help="model ks: 1 = the same factions in every issue bundle (no cross-cutting cleavages)")
    ap.add_argument("--width", type=float, default=0.7)
    ap.add_argument("--utility", default="exp", choices=["exp", "linear", "quadratic"],
                    help="utility of distance: exp = exp(-distance / width), far voters nearly indifferent; "
                         "linear = -distance / width (as vse-sim), every unit of distance counts the same; "
                         "quadratic = -(distance / width)^2, each further unit costs more")
    ap.add_argument("--cand-dispersion", type=float, default=1.0,
                    help="candidates' distance from the electorate's centre x this (< 1: clustered in the centre, "
                         "> 1: spread out; Merrill 1984); voters unchanged")
    ap.add_argument("--cand-shift", type=float, default=0.0,
                    help="candidates moved along the first (main) dimension by this many voter SDs; voters unchanged")
    ap.add_argument("--rounds", type=int, default=3,
                    help="poll rounds before the vote (0 = everyone votes sincerely); see strategy.py")
    ap.add_argument("--samples", type=int, default=5,
                    help="poll rounds: polls per round (they stand for the uncertainty; 10 until 2026-10-26); "
                         "--converge: bootstrap resamples per round")
    ap.add_argument("--poll-size", type=int, default=0,
                    help="poll rounds only: voters per poll (0 = half the electorate, rounded to an odd number)")
    ap.add_argument("--search", type=int, default=10,
                    help="poll rounds only: random ballot changes a group tries per round, after the naive "
                         "exaggerations (at most; it stops after 3 in a row without gain)")
    ap.add_argument("--aggressive", type=float, default=0.5,
                    help="share of strategic voters who exaggerate whenever it cannot hurt; the others (kind) "
                         "only when the polls show a gain (poll rounds only)")
    ap.add_argument("--lazy", type=float, default=0.0,
                    help="laziness of honest ballots, 0..1: each voter has a cut drawn between 0 and this, and ranks, "
                         "approves or scores only candidates whose utility, stretched to 0..1 between least and most "
                         "liked, is at least their cut (0 = full ballots); strategic voters may deviate "
                         "(see docs/model.md, Truncation and laziness)"),
    ap.add_argument("--poll-error", type=float, default=2.0,
                    help="poll rounds only: voters treat polls as this many times less precise than their "
                         "sampling error")
    ap.add_argument("--converge", type=int, default=0,
                    help="N > 0: instead of poll rounds, the convergent learning model with N rounds "
                         "(groups see each other's moves; uses --group-size)")
    ap.add_argument("--group-size", type=group_size_arg, default=10,
                    help="--converge only (poll rounds use whole ranking groups): max voters coordinating as one "
                         "group: a number of voters (1 = individual), a share of the electorate between 0 and 1 "
                         "(0.05 = 5%%), or 0 = whole ranking group")
    ap.add_argument("--strategic", type=float, default=0.0,
                    help="share of voters who always vote strategically (random, the same for every method)")
    ap.add_argument("--adaptive", type=float, default=None,
                    help="share of voters who decide each poll round whether to vote strategically, by how much "
                         "the race between the two leaders matters to them (see docs/model.md, Partial strategy); "
                         "the rest always vote sincerely. Default: everyone not in --strategic (with --converge: 0)")
    ap.add_argument("--max-score", type=int, default=5,
                    help="score ballots range over 0..max (Score, STAR, MJ, ARA, ...; MCA and 321 keep 3 grades)")
    ap.add_argument("--exits", type=int, default=0, choices=[0, 1],
                    help="1: candidates may withdraw and re-enter strategically (see nomination.py)")
    ap.add_argument("--clones", type=int, default=0,
                    help="most clones in the race at once, fielded strategically (0 = none; see nomination.py)")
    ap.add_argument("--start", default="honest", choices=["honest", "previous"],
                    help="what the first poll shows: honest ballots, or the result of a previous election (one "
                         "dummy election under the same settings, its candidates relabelled at random)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None, help="run only these methods or groups (start from none); \"all\" for every method")
    ap.add_argument("--exclude", nargs="+", default=None, help="leave out these methods or groups (from the standard set, or from --methods)")
    ap.add_argument("--participation", action="store_true",
                    help="also check participation (no-show paradox) on sincere and learned ballots; slower")
    ap.add_argument("--manipulation", action="store_true",
                    help="also test coalition manipulation from honest ballots, one- and two-sided (manipulation.py)")
    ap.add_argument("--out", default=None,
                    help="folder for this run (in output/ unless it has a directory); default run-<date>-<time>")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run in --out")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in --out (same command): finished elections are kept")
    return ap


def default_params():
    """The run parameters of a plain `python run.py` (the defaults), as written to the results."""
    a = parser().parse_args([])
    a.salience = resolve_salience(vars(a))["salience"]
    return resolve_salience(resolve_adaptive({k: getattr(a, k) for k in PARAMS}))


def main():
    ap = parser()
    a = ap.parse_args()

    check_params(a, ap.error)
    out = run_dir(a.out, "run", a.overwrite, a.resume)
    write_command(out, a.resume)
    a.salience = resolve_salience(vars(a))["salience"]   # from the exponent (resolve_salience)
    p = resolve_salience(resolve_adaptive({k: getattr(a, k) for k in PARAMS}))
    names = run_selection(a.methods, a.exclude)
    path = os.path.join(out, "elections.csv")
    key = run_key(p, PARAMS)
    t = time.time()
    fp = fingerprints(names, a.workers)
    print(f"fingerprints of {len(fp)} method(s): {time.time() - t:.0f} s")
    if a.resume:
        check_resume_fingerprints(path, fp)
    done, header = resume_state(path, names, {key: [a.rounds]}) if a.resume else ({}, None)
    if a.resume:
        print(f"resuming: {len(done.get(key, ()))} of {a.elections} elections already done")
    sink = CsvSink(path, header)
    try:
        simulate(p, a.elections, a.workers, names, a.participation, sink=sink, skip=done.get(key, set()),
                 manipulation=a.manipulation, fingerprint=fp)
    except KeyboardInterrupt:
        sink.close()
        with open(path, newline="") as f:
            interrupted(out, len({r["election"] for r in csv.DictReader(f)}))
    sink.close()
    with open(path, newline="") as f:     # all elections, including those of an earlier, interrupted start
        results = [numeric(r) for r in csv.DictReader(f)]

    summ = {}
    for m in names:
        sinc = [r for r in results if r["method"] == m and r["ballots"] == "sincere"]
        lrn = [r for r in results if r["method"] == m and r["ballots"] == "learned"]
        s, l = summarize(sinc), summarize(lrn)
        summ[m] = {"VSE": s["VSE"], "WME": s["WME"], "VSE_learned": l["VSE"], "WME_learned": l["WME"],
                   "CE": s["CE"], "CE_learned": l["CE"],
                   "sincerity": float(np.mean([r["sincerity"] for r in lrn])) if sincerity_defined(m) else float("nan"),
                   "settled": float(np.mean([r["settled"] for r in lrn]))}
    axes = ["VSE_learned", "WME_learned", "sincerity"]
    # undefined sincerity (TRASH, Inverse-*): no advantage on that axis in this quick check
    front = set(pareto([[summ[m][x] if not np.isnan(summ[m][x]) else -np.inf for x in axes] for m in names],
                       tol=[FRONT_TOLERANCE["VSE*"], FRONT_TOLERANCE["WME*"], FRONT_TOLERANCE["sincerity"]]))

    cols = ["VSE", "WME", "VSE_learned", "WME_learned", "CE", "CE_learned", "sincerity", "settled"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols + ["on_front"])
        for i, m in enumerate(names):
            w.writerow([m] + [f"{summ[m][c]:.4f}" for c in cols] + [int(i in front)])

    lines = []
    lines.append(f"{a.elections} elections, {a.voters} voters, {a.cands} candidates, {model_text(a)}, "
          f"{width_text(a)}{cand_text(a.cand_dispersion, a.cand_shift)}; {strategy_text(a, a.voters)}, "
          f"strategic share {a.strategic}, adaptive share {p['adaptive']}, scores 0..{a.max_score}"
          f"{f', nomination: exits {a.exits}, clones {a.clones}' if a.exits or a.clones else ''}; seed {a.seed}")
    lines.append(f"{'method':<26}{'VSE':>7}{'WME':>7}{'VSE*':>7}{'WME*':>7}{'CE':>7}{'CE*':>7}{'sincerity':>11}{'settled':>9}  front")
    for m in sorted(names, key=lambda m: -summ[m]["VSE_learned"]):
        s = summ[m]
        lines.append(f"{m:<26}{s['VSE']*100:7.1f}{s['WME']*100:7.1f}{s['VSE_learned']*100:7.1f}{s['WME_learned']*100:7.1f}"
              f"{s['CE']*100:7.1f}{s['CE_learned']*100:7.1f}"
              + (f"{s['sincerity']*100:10.1f}%" if not np.isnan(s['sincerity']) else f"{'—':>11}")
              + f"{s['settled']*100:8.0f}%"
              f"  {'*' if names.index(m) in front else ''}")
    lines.append("0 = random ballot, 100 = best candidate; *: with learned ballots; front on VSE*, WME*, sincerity")
    with open(os.path.join(out, "table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}/: elections.csv, summary.csv, table.txt, command.txt; analyse with: python analyze.py {out}")


if __name__ == "__main__":
    main()
