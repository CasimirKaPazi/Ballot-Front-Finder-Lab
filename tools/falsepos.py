"""False positives of strategy: moves that buy nothing against what the others cast (a diagnostic, not
part of the model).

For every method, the learned ballots of the last poll round are judged by strategy.measures.
false_positives: the voters who changed their ballot are grouped into coalitions (their sincere order of
the three leading places, and what their ballot does among these three), each coalition goes home in
turn, and each member's gain from the cast ballots over that is counted, by polls of the cast ballots
(deviating as much as the voters assume, --poll-error) and by the true count. A false positive is a voter whose coalition's move gains nothing or loses in
the polls. The measure reads only ballots, so it is the same for every engine, ballot format and move.

Reported per method, over all elections (voters who changed pooled):
    changed    share of all voters whose learned ballot differs from their sincere one
    fp         share of those whose move gains nothing or loses in the polls (false positives)
    none       ... gains exactly nothing (it changes no poll's result)
    hurt       ... loses (it backfires against what the others cast)
    gain       their mean gain in the polls, in % of their utility range
    help/hurt (exact)   shares with a gain / loss in the true count of all ballots
    coal       coalitions per election (mean)
Then per method the kinds of move by what the ballot does to the voter's own three leading places (1 =
the one they like most; "1>3>2": burial of their second, "2>1>3": compromise, "1=2>3": levelling,
"1>2>3": a change elsewhere), each with its share of the voters who changed, its false positives and
mean gain (kinds of at least 5%). The share with no gain needs many polls to settle (a small gain
shows in few of them): with 20 polls Score's was 24%, with 600 under 1%; hence 500 by default.

Usage (from the lab's folder; flags after -- go to run.py's parser, so any condition can be checked):
    python tools/falsepos.py --elections 300
    python tools/falsepos.py --elections 300 --methods Score Borda -- --strategic 1 --engine arithmetic
    python tools/falsepos.py --elections 300 --measure-error 1      # polls with their sampling error only
    python tools/falsepos.py --elections 300 --share engine         # a quarter of each coalition goes home
    python tools/falsepos.py --elections 300 --view engine          # polls as the engines read them
    python tools/falsepos.py --elections 300 --against seen         # against the ballots the voters' polls were of
Writes nothing; prints the table.
"""
import argparse
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the lab's modules
import run  # noqa: E402
from scenario import Scenario  # noqa: E402
from strategy.falsepos import false_positives  # noqa: E402

DEFAULT = ["Plurality", "IRV", "Approval", "Score", "STAR", "321", "MJ", "Borda", "Score runoff", "Schulze",
           "Minimax", "Soft Minimax-D"]
KEYS = ["fp", "fp_none", "fp_hurt", "gain", "exact_help", "exact_hurt"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--elections", type=int, default=300)
    ap.add_argument("--first", type=int, default=0, help="the number of the first election")
    ap.add_argument("--methods", nargs="+", default=DEFAULT)
    ap.add_argument("--polls", type=int, default=500, help="polls of the cast ballots per coalition (500)")
    ap.add_argument("--share", default="1",
                    help="the share of a coalition that goes home (1: all of it; 'engine': the run's --coordination)")
    ap.add_argument("--view", choices=["count", "engine"], default="count",
                    help="polls counted by the method itself (count) or as the engines read them (engine: the poll view)")
    ap.add_argument("--against", choices=["cast", "seen"], default="cast",
                    help="the others' ballots a move is judged against: as cast (cast), or those of the round "
                         "before, which the voters' polls were of (seen)")
    ap.add_argument("--measure-error", type=float, default=None,
                    help="how far the measure's polls deviate, as --poll-error (default: the run's --poll-error)")
    argv = sys.argv[1:]
    own, rest = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    a = ap.parse_args(own)
    ra = run.parser().parse_args(rest)
    ra.salience = run.resolve_salience(vars(ra))["salience"]
    p = run.resolve({k: getattr(ra, k) for k in run.PARAMS})
    sc = Scenario(p["voters"], p["cands"], p["dims"], p["clusters"], p["spread"], p["width"], p["zipf"],
                  p["cand_dispersion"], p["cand_shift"], p.get("model", "clusters"), p.get("ks_alpha", 1.5),
                  p.get("ks_cut", 0.2), p.get("ks_shared", 0), salience=p.get("salience", "none"),
                  salience_exponent=p.get("salience_exponent", 1.0), fixed_variance=p.get("fixed_variance", 0),
                  field_kappa=p.get("field_kappa", 1.5), field_salience=p.get("field_salience", 0.0),
                  noise_strength=p.get("noise_strength", 3.0), utility=p.get("utility", "exp"),
                  cands_vote=int(p.get("cands_vote", 0)))      # as run.simulate
    rounds = int(p.get("rounds", 3))
    err = p["poll_error"] if a.measure_error is None else a.measure_error
    share = float(p["coordination"]) if a.share == "engine" else float(a.share)
    orig = run.fidelity
    acc, now = {}, {}
    run.ensure(a.methods)                          # compositions registered
    names = {id(run.METHODS[x]): x for x in a.methods}

    seen = {}

    def fidelity(m, B, uf, honest):            # run's fidelity, called once per learned row: measure there
        name = names[id(m)]
        if a.against == "seen" and rounds > 1 and name not in seen:
            seen[name] = B.copy()              # the round before: what the last round's polls were of
            return orig(m, B, uf, honest)
        base = None if a.against == "cast" else seen.pop(name, honest)
        r = false_positives(m, B, honest, uf, np.random.default_rng([p["seed"], now["e"], zlib.crc32(name.encode())]),
                            samples=a.polls, poll_size=p["poll_size"], poll_error=err, share=share, view=a.view,
                            base=base)
        acc.setdefault(name, []).append(r)
        return orig(m, B, uf, honest)
    run.fidelity = fidelity
    t0 = time.time()
    for k, e in enumerate(range(a.first, a.first + a.elections)):
        now["e"] = e
        seen.clear()
        run.one_election({"sc": sc, "p": p, "e": e, "rounds": [rounds - 1, rounds] if a.against == "seen" and rounds > 1
                          else [rounds], "names": a.methods,
                          "participation": False, "manipulation": False, "dummy": None})
        print(f"\r{k + 1}/{a.elections} elections, {time.time() - t0:.0f} s", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    print(f"False positives of strategy: elections {a.first}..{a.first + a.elections - 1}, {a.polls} polls per "
          f"coalition (poll error {err:g}), {share:g} of each coalition home, polls read by {a.view}, against the others' ballots {'as cast' if a.against == 'cast' else 'of the round before'}, run.py {' '.join(rest) or 'defaults'} (engine {p['engine']}, {rounds} rounds). "
          f"Shares of the voters who changed their ballot, pooled over the elections; changed: of all voters.")
    print(f"{'method':<18}{'changed':>8}{'fp':>7}{'none':>7}{'hurt':>7}{'gain':>8}{'help':>8}{'hurt':>7}{'coal':>6}")
    print(f"{'':<18}{'':>8}{'(polls)':>21}{'':>8}{'(exact)':>15}")
    for name in a.methods:
        rs = acc.get(name, [])
        if not rs:
            print(f"{name:<18}  (no sincerity defined: not measured)")
            continue
        w = np.array([r["changed"] for r in rs])
        tot = w.sum()
        cell = {}
        for key in KEYS:
            x = np.array([r[key] if r["changed"] > 0 else 0.0 for r in rs])
            cell[key] = float((w * x).sum() / tot) if tot > 0 else float("nan")
        print(f"{name:<18}{w.mean() * 100:7.1f}%{cell['fp'] * 100:6.1f}%{cell['fp_none'] * 100:6.1f}%"
              f"{cell['fp_hurt'] * 100:6.1f}%{cell['gain'] * 100:7.2f}%{cell['exact_help'] * 100:7.1f}%"
              f"{cell['exact_hurt'] * 100:6.1f}%{np.mean([r['coalitions'] for r in rs]):6.1f}")
    print("\nKinds of move (1, 2, 3: the voter's own order of the three leading places): share of the voters who "
          "changed, false positives among them, mean gain in the polls.")
    for name in a.methods:
        pat = {}
        for r in acc.get(name, []):
            for k, (c, f, g) in r["patterns"].items():
                c0, f0, g0 = pat.get(k, (0, 0, 0.0))
                pat[k] = (c0 + c, f0 + f, g0 + g)
        tot = sum(c for c, _, _ in pat.values())
        if not tot:
            continue
        cells = [f"{k} {c / tot * 100:.0f}% fp {f / c * 100:.0f}% {g / c * 100:+.1f}%"
                 for k, (c, f, g) in sorted(pat.items(), key=lambda x: -x[1][0]) if c / tot >= 0.05]
        print(f"{name:<18}" + "; ".join(cells))


if __name__ == "__main__":
    main()
