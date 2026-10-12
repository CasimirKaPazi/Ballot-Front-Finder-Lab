"""Run honest and strategic elections for all methods on one scenario and write CSV.

Usage:
    python run.py --elections 200                      # defaults: see README, Parameters

Each electorate is tallied twice per method: with sincere ballots, and after the groups
(voters with the same sincere ranking) have answered polls strategically (see strategy.py: poll
rounds by default, convergent learning with --converge). The "learned" rows hold those ballots.

Axes of the Pareto front (0 = random ballot, 100 = best candidate; see metrics.py):
    VSE, WME     voter satisfaction and worst-margin efficiency, sincere ballots
    VSE*, WME*   the same with learned ballots
CE, CE* (Condorcet efficiency) are reported alongside, as are the sincerity of the learned
ballots (see strategy.sincerity) and the share of elections where the strategic ballots settled
(poll rounds: the last round repeated the one before).

Partial strategy: --strategic p makes a random share p of the voters strategic (the same voters
for every method); the rest always vote sincerely.

Outputs, in one folder per run (--out; a name without a directory goes into output/; default
run-<date>-<time>). `python analyze.py <folder>` puts its analysis into <folder>/analysis/.
    elections.csv   one row per (election, method, ballots), with the run's parameters as columns,
                    so several runs can be combined (see analyze.py)
    summary.csv     one row per method, with on_front for the VSE*/WME*/sincerity front
    table.txt       the printed table
    command.txt     the command, simulator version and start time (a resumed run adds its own)
Every election is saved as soon as it is done; after an interruption the same command with
--resume continues where it stopped (see README, Progress, saving and resuming).
"""
import argparse
import csv
import os
import shlex
import sys
import time
import zlib
from multiprocessing import Pool
import numpy as np

from scenario import Scenario, electorate_positions
from nomination import nominate, clone_setup, utilities, to_original
from methods import METHODS, select, set_max_score, sincerity_defined
from strategy import learn, respond, strategy_types, default_poll_size, fidelity
from metrics import election_row, summarize, pareto, worst_margins, electorate_stats, ballot_condorcet
import participation as part


def one_election(args):
    (sc, seed, e, rounds, samples, group_size, share, max_score, exits, clones, poll_size, poll_error, search,
     converge, aggr, names, participation, start, dummy) = args
    set_max_score(max_score)                        # per process: workers do not share module state
    rng = np.random.default_rng([seed, e])
    u, voters, cands = electorate_positions(sc, rng)
    C = u.shape[1]
    nominating = bool(exits or clones)
    if nominating:   # clones' positions: own random stream, the same for every method
        u_clone, cu = clone_setup(voters, cands, sc.width,
                                            np.random.default_rng([seed, e, zlib.crc32(b"clones")]))
    m_cand = worst_margins(u)                       # each candidate's worst head-to-head margin
    estats = electorate_stats(u)                    # properties of the electorate, for the report
    # every candidate's total utility and worst margin, for the frontier of what any rule could reach
    estats["U_cands"] = ";".join(f"{x:.6g}" for x in u.sum(axis=0))
    estats["WM_cands"] = ";".join(f"{x:.6g}" for x in m_cand)
    # strategic voters: own random stream, the same voters for every method
    strategic = np.random.default_rng([seed, e, zlib.crc32(b"strategic")]).random(len(u)) < share
    # aggressive strategic voters (exaggerate when it cannot hurt) vs kind ones: own stream, same for every method
    aggressive = np.random.default_rng([seed, e, zlib.crc32(b"aggressive")]).random(len(u)) < aggr
    out = []
    for name in names:
        m = METHODS[name]
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
                       "compromise": 0.0, "pushover": 0.0, "burial": 0.0, **nstats, **check(m.sincere(uf))}
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
        if converge > 0:     # convergent learning does not use rounds: one result for every rounds value
            res = [learn(m, uf, mrng, converge, samples, group_cap(group_size, len(u)), strategic)] * len(rounds)
        else:                # all requested numbers of poll rounds from one run (see strategy.respond)
            res = respond(m, uf, mrng, samples=samples, poll_size=poll_size, search=search, strategic=strategic,
                          poll_error=poll_error, report=rounds, aggressive=aggressive, start=B0)
        sincere_row.update(ballot_condorcet(m, m.sincere(uf), u, back, P_h))
        for r, (B, settled, sin) in zip(rounds, res):
            P_l = back(m.tally(B))
            out.append({"rounds": r, "election": e, "method": name, "ballots": "sincere", **sincere_row})
            out.append({"rounds": r, "election": e, "method": name, "ballots": "learned",
                        **election_row(u, P_l, m_cand), **estats, **ballot_condorcet(m, B, u, back, P_l),
                        "sincerity": sin if defined else "",
                        "fidelity": float(fidelity(m, B, uf).mean()) if defined else "",
                        "settled": int(settled), **strategy_types(B, m.pref(uf)), **nstats, **check(B)})
    return out


OUTPUT_DIR = "output"
VERSION = "1.0.0"      # written into every row (column "version") and shown in the analysis report


def out_path(name):
    """Output prefix or directory: a bare name goes into OUTPUT_DIR (created if needed)."""
    path = name if os.path.dirname(name) else os.path.join(OUTPUT_DIR, name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


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
    has = os.path.exists(os.path.join(path, "elections.csv"))
    if resume and not has:
        raise SystemExit(f"{path} holds no run to resume")
    if has and not overwrite and not resume:
        raise SystemExit(f"{path} already holds a run; choose another --out, add --overwrite, "
                         "or --resume to continue it")
    os.makedirs(path, exist_ok=True)
    return path


def write_command(path, resume=False):
    """command.txt: the exact command, the simulator version and the start time, for reproducing.
    A resumed run adds its command below the first."""
    with open(os.path.join(path, "command.txt"), "a" if resume else "w") as f:
        f.write(f"python {' '.join(shlex.quote(x) for x in sys.argv)}\n")
        f.write(f"simulator version {VERSION}, started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def check_params(a, error):
    """Reject values the simulation cannot use (a: parsed arguments; lists allowed, as in sweep.py)."""
    vals = lambda k: getattr(a, k) if isinstance(getattr(a, k), list) else [getattr(a, k)]
    for k, lo in (("rounds", 0), ("samples", 1), ("search", 0), ("poll_size", 0), ("converge", 0), ("clones", 0)):
        if any(v < lo for v in vals(k)):
            error(f"--{k.replace('_', '-')} must be at least {lo}")
    for k in ("strategic", "aggressive"):
        if any(not 0 <= v <= 1 for v in vals(k)):
            error(f"--{k} must be between 0 and 1")
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


def strategy_text(p, n):
    """One-line description of the strategy model of a run (p: namespace or dict of parameters)."""
    g = (lambda k: p[k]) if isinstance(p, dict) else (lambda k: getattr(p, k))
    if int(g("converge")) > 0:
        return (f"convergent learning {g('converge')} rounds x {g('samples')} poll samples, "
                f"group size {group_text(g('group_size'))}")
    ps = int(g("poll_size")) or default_poll_size(int(n))
    return (f"{g('rounds')} poll round(s), {g('samples')} polls of {ps} voters, poll error {g('poll_error')}x, "
            f"search {g('search')}, aggressive share {g('aggressive')}"
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
PARAMS = ["voters", "cands", "dims", "zipf", "clusters", "spread", "width", "cand_dispersion", "cand_shift", "rounds", "samples", "group_size",
          "strategic", "aggressive", "max_score", "exits", "clones", "poll_size", "poll_error", "search", "converge",
          "start", "seed"]


def previous_election(args):
    """The dummy election for --start previous: an extra electorate (its own random stream) under the
    same settings, through the same number of poll rounds; returns its final ballots."""
    sc, p, rounds, name = args
    set_max_score(p["max_score"])
    rng = np.random.default_rng([p["seed"], zlib.crc32(b"previous")])
    u = electorate_positions(sc, rng)[0]
    strategic = rng.random(len(u)) < p["strategic"]
    aggressive = rng.random(len(u)) < p["aggressive"]
    m = METHODS[name]
    B, _, _ = respond(m, u, np.random.default_rng([p["seed"], zlib.crc32(b"previous-learning")]), rounds=rounds,
                      samples=p["samples"], poll_size=p["poll_size"], search=p["search"], strategic=strategic,
                      poll_error=p["poll_error"], aggressive=aggressive)
    return B


def _fmt_time(t):
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


class Progress:
    """Elections done, time used and an estimate of the time left, on one line (stderr): in a
    terminal redrawn in place after every election; otherwise (output to a file) a new line at
    most `updates` times per run (default 10, one per tenth)."""

    def __init__(self, total, label="", updates=None):
        self.total, self.label, self.done, self.t0 = total, label, 0, time.time()
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
        text = f"{self.label}{self.done}/{self.total}, {_fmt_time(el)} so far, ~{_fmt_time(left)} left"
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


def simulate(p, elections, workers=None, names=None, participation=False, sink=None, skip=(), label=""):
    """Run one condition. p: dict with the keys in PARAMS; names: methods (default all);
    participation: also run the no-show check (participation.py) on sincere and learned ballots.
    p["rounds"] may be a list of numbers of poll rounds: they all come from one run (the rounds
    up to the highest are computed once), one condition per value, in that order.
    sink: called with each election's rows as soon as it is done (in election order), so a
    caller can save as it goes; skip: election numbers already done (resuming), not run again.
    Every election depends only on its seed and number, so skipped and new ones fit together.
    Shows progress (see Progress). Returns the new rows with p as columns."""
    names = names or list(METHODS)
    rounds = list(p["rounds"]) if isinstance(p["rounds"], (list, tuple)) else [p["rounds"]]
    sc = Scenario(p["voters"], p["cands"], p["dims"], p["clusters"], p["spread"], p["width"], p["zipf"],
                  p["cand_dispersion"], p["cand_shift"])
    todo = [e for e in range(elections) if e not in set(skip)]
    jobs = [(sc, p["seed"], e, rounds, p["samples"], p["group_size"], p["strategic"], p["max_score"],
             p["exits"], p["clones"], p["poll_size"], p["poll_error"], p["search"], p["converge"], p["aggressive"],
             names, participation) for e in todo]
    meta = {k: p[k] for k in PARAMS if k != "rounds"} | {"version": VERSION}
    rows = []
    if not todo:
        return rows
    prog = Progress(len(todo), label)
    try:
        with Pool(workers, initializer=_ignore_sigint) as pool:
            dummy = None
            if p["start"] == "previous":          # one previous election for the whole batch, per method
                dummy = dict(zip(names, pool.map(previous_election, [(sc, p, max(rounds), n) for n in names])))
            jobs = [j + (p["start"], dummy) for j in jobs]
            for batch in pool.imap(one_election, jobs, chunksize=1):   # in order: the file stays reproducible
                batch = [{**meta, **r} for r in batch]
                batch = [{k: r[k] for k in PARAMS} | r for r in batch]      # parameters first, as columns
                if sink:
                    sink(batch)
                rows += batch
                prog.step()
    finally:
        prog.close()
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
    if {r["method"] for r in rows} - set(names) or {r.get("version") for r in rows} != {VERSION}:
        raise SystemExit(f"{path} was made with other methods or another simulator version; "
                         "--resume needs the same command")
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
        self.w = csv.DictWriter(self.f, fieldnames=header) if header else None

    def __call__(self, rows):
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(rows[0]))
            self.w.writeheader()
        self.w.writerows(rows)
        self.f.flush()

    def close(self):
        self.f.close()


def interrupted(out, n):
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    if "--resume" not in sys.argv:
        cmd += " --resume" + ("" if "--out" in sys.argv else f" --out {shlex.quote(out)}")
    raise SystemExit(f"\ninterrupted; {n} complete election(s) are saved in {out}/elections.csv. "
                     f"Continue with:\n  python {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elections", type=int, default=200)
    ap.add_argument("--voters", type=int, default=99)
    ap.add_argument("--cands", type=int, default=6)
    ap.add_argument("--dims", type=int, default=10)
    ap.add_argument("--zipf", type=int, default=1, choices=[0, 1],
                    help="1: dimension i has scale 1/i (Zipf dimensions, see scenario.py)")
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--cand-dispersion", type=float, default=1.0,
                    help="candidates' distance from the electorate's centre x this (< 1: clustered in the centre, "
                         "> 1: spread out; Merrill 1984); voters unchanged")
    ap.add_argument("--cand-shift", type=float, default=0.0,
                    help="candidates moved along the first (main) dimension by this many voter SDs; voters unchanged")
    ap.add_argument("--rounds", type=int, default=3,
                    help="poll rounds before the vote (0 = everyone votes sincerely); see strategy.py")
    ap.add_argument("--samples", type=int, default=10,
                    help="poll rounds: polls per round (they stand for the uncertainty); --converge: bootstrap "
                         "resamples per round")
    ap.add_argument("--poll-size", type=int, default=0,
                    help="poll rounds only: voters per poll (0 = half the electorate, rounded to an odd number)")
    ap.add_argument("--search", type=int, default=10,
                    help="poll rounds only: random ballot changes a group tries per round, after the naive "
                         "exaggerations (at most; it stops after 3 in a row without gain)")
    ap.add_argument("--aggressive", type=float, default=0.5,
                    help="share of strategic voters who exaggerate whenever it cannot hurt; the others (kind) "
                         "only when the polls show a gain (poll rounds only)")
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
    ap.add_argument("--strategic", type=float, default=0.5,
                    help="share of voters who vote strategically (random, the same for every method)")
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
    ap.add_argument("--methods", nargs="+", default=None, help="run only these methods (start from none)")
    ap.add_argument("--exclude", nargs="+", default=None, help="run all methods except these")
    ap.add_argument("--participation", action="store_true",
                    help="also check participation (no-show paradox) on sincere and learned ballots; slower")
    ap.add_argument("--out", default=None,
                    help="folder for this run (in output/ unless it has a directory); default run-<date>-<time>")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run in --out")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in --out (same command): finished elections are kept")
    a = ap.parse_args()

    check_params(a, ap.error)
    out = run_dir(a.out, "run", a.overwrite, a.resume)
    write_command(out, a.resume)
    p = {k: getattr(a, k) for k in PARAMS}
    names = select(a.methods, a.exclude)
    path = os.path.join(out, "elections.csv")
    key = run_key(p, PARAMS)
    done, header = resume_state(path, names, {key: [a.rounds]}) if a.resume else ({}, None)
    if a.resume:
        print(f"resuming: {len(done.get(key, ()))} of {a.elections} elections already done")
    sink = CsvSink(path, header)
    try:
        simulate(p, a.elections, a.workers, names, a.participation, sink=sink, skip=done.get(key, set()))
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
    front = set(pareto([[summ[m][x] if not np.isnan(summ[m][x]) else -np.inf for x in axes] for m in names]))

    cols = ["VSE", "WME", "VSE_learned", "WME_learned", "CE", "CE_learned", "sincerity", "settled"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols + ["on_front"])
        for i, m in enumerate(names):
            w.writerow([m] + [f"{summ[m][c]:.4f}" for c in cols] + [int(i in front)])

    lines = []
    lines.append(f"{a.elections} elections, {a.voters} voters, {a.cands} candidates, {a.dims} dimensions"
          f"{' (Zipf)' if a.zipf else ''}, {a.clusters} clusters, "
          f"spread {a.spread}, width {a.width}{cand_text(a.cand_dispersion, a.cand_shift)}; {strategy_text(a, a.voters)}, "
          f"strategic share {a.strategic}, scores 0..{a.max_score}"
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
