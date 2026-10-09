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

Outputs (prefix set by --out; a name without a directory goes into output/):
    <out>-elections.csv   one row per (election, method, ballots), with the run's parameters
                          as columns, so several runs can be combined (see analyze.py)
    <out>-summary.csv     one row per method, with on_front for the VSE/WME/VSE*/WME* front
"""
import argparse
import csv
import os
import zlib
from multiprocessing import Pool
import numpy as np

from scenario import Scenario, electorate_positions
from nomination import nominate, clone_setup, utilities, to_original
from methods import METHODS, select, set_max_score
from strategy import learn, respond, strategy_types, default_poll_size
from metrics import election_row, summarize, pareto, worst_margins, electorate_stats


def one_election(args):
    (sc, seed, e, rounds, samples, group_size, share, max_score, exits, clones, poll_size, poll_error, search,
     converge, aggr, names) = args
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
            nrng = np.random.default_rng([seed, e, zlib.crc32(name.encode()), zlib.crc32(b"nomination")])
            field, st = nominate(m, u, u_clone, cu, nrng, exits, clones)
            ents = field.entries()
            uf, back = utilities(ents, u, u_clone), (lambda P, ents=ents: to_original(P, ents, C))
        P_h = back(m.honest(uf))
        if nominating:
            nstats = {**st, "nom_changed": int(np.abs(P_h - m.honest(u)).sum() > 1e-9)}
        sincere_row = {**election_row(u, P_h, m_cand), **estats, "sincerity": 1.0, "settled": 1,
                       "compromise": 0.0, "pushover": 0.0, "burial": 0.0, **nstats}
        # Each method learns with its own random stream, keyed by name, so results stay
        # the same when methods are added or reordered.
        mrng = np.random.default_rng([seed, e, zlib.crc32(name.encode())])
        if converge > 0:     # convergent learning does not use rounds: one result for every rounds value
            res = [learn(m, uf, mrng, converge, samples, group_cap(group_size, len(u)), strategic)] * len(rounds)
        else:                # all requested numbers of poll rounds from one run (see strategy.respond)
            res = respond(m, uf, mrng, samples=samples, poll_size=poll_size, search=search, strategic=strategic,
                          poll_error=poll_error, report=rounds, aggressive=aggressive)
        for r, (B, settled, sin) in zip(rounds, res):
            out.append({"rounds": r, "election": e, "method": name, "ballots": "sincere", **sincere_row})
            out.append({"rounds": r, "election": e, "method": name, "ballots": "learned",
                        **election_row(u, back(m.tally(B)), m_cand), **estats, "sincerity": sin,
                        "settled": int(settled), **strategy_types(B, uf), **nstats})
    return out


OUTPUT_DIR = "output"


def out_path(name):
    """Output prefix or directory: a bare name goes into OUTPUT_DIR (created if needed)."""
    path = name if os.path.dirname(name) else os.path.join(OUTPUT_DIR, name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


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


def strategy_text(p, n):
    """One-line description of the strategy model of a run (p: namespace or dict of parameters)."""
    g = (lambda k: p[k]) if isinstance(p, dict) else (lambda k: getattr(p, k))
    if int(g("converge")) > 0:
        return (f"convergent learning {g('converge')} rounds x {g('samples')} poll samples, "
                f"group size {group_text(g('group_size'))}")
    ps = int(g("poll_size")) or default_poll_size(int(n))
    return (f"{g('rounds')} poll round(s), {g('samples')} polls of {ps} voters, poll error {g('poll_error')}x, "
            f"search {g('search')}, aggressive share {g('aggressive')}")


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
PARAMS = ["voters", "cands", "dims", "zipf", "clusters", "spread", "width", "rounds", "samples", "group_size",
          "strategic", "aggressive", "max_score", "exits", "clones", "poll_size", "poll_error", "search", "converge",
          "seed"]


def simulate(p, elections, workers=None, names=None):
    """Run one condition. p: dict with the keys in PARAMS; names: methods (default all).
    p["rounds"] may be a list of numbers of poll rounds: they all come from one run (the rounds
    up to the highest are computed once), one condition per value, in that order.
    Returns rows with p as columns."""
    names = names or list(METHODS)
    rounds = list(p["rounds"]) if isinstance(p["rounds"], (list, tuple)) else [p["rounds"]]
    sc = Scenario(p["voters"], p["cands"], p["dims"], p["clusters"], p["spread"], p["width"], p["zipf"])
    jobs = [(sc, p["seed"], e, rounds, p["samples"], p["group_size"], p["strategic"], p["max_score"],
             p["exits"], p["clones"], p["poll_size"], p["poll_error"], p["search"], p["converge"], p["aggressive"],
             names) for e in range(elections)]
    with Pool(workers) as pool:
        rows = [r for batch in pool.imap(one_election, jobs, chunksize=4) for r in batch]
    meta = {k: p[k] for k in PARAMS if k != "rounds"}
    rows = [{**meta, **r} for r in rows]
    rows.sort(key=lambda r: rounds.index(r["rounds"]))          # stable: one condition after another
    return [{k: r[k] for k in PARAMS} | r for r in rows]


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
    ap.add_argument("--rounds", type=int, default=1,
                    help="poll rounds before the vote (0 = everyone votes sincerely); see strategy.py")
    ap.add_argument("--samples", type=int, default=30,
                    help="poll rounds: polls per round (they stand for the uncertainty); --converge: bootstrap "
                         "resamples per round (results up to 2026-10-08 used 10: add --samples 10 to reproduce them)")
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
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None, help="run only these methods (start from none)")
    ap.add_argument("--exclude", nargs="+", default=None, help="run all methods except these")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    check_params(a, ap.error)
    p = {k: getattr(a, k) for k in PARAMS}
    names = select(a.methods, a.exclude)
    results = simulate(p, a.elections, a.workers, names)

    out = out_path(a.out)
    with open(f"{out}-elections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    summ = {}
    for m in names:
        sinc = [r for r in results if r["method"] == m and r["ballots"] == "sincere"]
        lrn = [r for r in results if r["method"] == m and r["ballots"] == "learned"]
        s, l = summarize(sinc), summarize(lrn)
        summ[m] = {"VSE": s["VSE"], "WME": s["WME"], "VSE_learned": l["VSE"], "WME_learned": l["WME"],
                   "CE": s["CE"], "CE_learned": l["CE"],
                   "sincerity": float(np.mean([r["sincerity"] for r in lrn])),
                   "settled": float(np.mean([r["settled"] for r in lrn]))}
    axes = ["VSE", "WME", "VSE_learned", "WME_learned"]
    front = set(pareto([[summ[m][x] for x in axes] for m in names]))

    cols = ["VSE", "WME", "VSE_learned", "WME_learned", "CE", "CE_learned", "sincerity", "settled"]
    with open(f"{out}-summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols + ["on_front"])
        for i, m in enumerate(names):
            w.writerow([m] + [f"{summ[m][c]:.4f}" for c in cols] + [int(i in front)])

    print(f"{a.elections} elections, {a.voters} voters, {a.cands} candidates, {a.dims} dimensions"
          f"{' (Zipf)' if a.zipf else ''}, {a.clusters} clusters, "
          f"spread {a.spread}, width {a.width}; {strategy_text(a, a.voters)}, "
          f"strategic share {a.strategic}, scores 0..{a.max_score}"
          f"{f', nomination: exits {a.exits}, clones {a.clones}' if a.exits or a.clones else ''}; seed {a.seed}")
    print(f"{'method':<26}{'VSE':>7}{'WME':>7}{'VSE*':>7}{'WME*':>7}{'CE':>7}{'CE*':>7}{'sincerity':>11}{'settled':>9}  front")
    for m in sorted(names, key=lambda m: -summ[m]["VSE"]):
        s = summ[m]
        print(f"{m:<26}{s['VSE']*100:7.1f}{s['WME']*100:7.1f}{s['VSE_learned']*100:7.1f}{s['WME_learned']*100:7.1f}"
              f"{s['CE']*100:7.1f}{s['CE_learned']*100:7.1f}"
              f"{s['sincerity']*100:10.1f}%{s['settled']*100:8.0f}%"
              f"  {'*' if names.index(m) in front else ''}")
    print("0 = random ballot, 100 = best candidate; *: with learned ballots; front on VSE, WME, VSE*, WME*")


if __name__ == "__main__":
    main()
