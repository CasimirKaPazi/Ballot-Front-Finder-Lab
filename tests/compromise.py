"""A compromise at the maximal lottery's mean position (a theoretical diagnostic, not part of the model).

Where the sincere preferences have no Condorcet winner, the maximal lottery mixes the candidates of the
cycle. A candidate placed at the lottery's mean position (its candidates' positions weighted by their
chances) is a natural compromise. With utilities concave in position, Jensen's inequality says every
voter likes such a candidate at least as much as the lottery itself; with the lab's default utility,
exp(-distance / width), which is convex in distance, that need not hold. This script checks it, and
then repeats the step: the compromise joins the field, the maximal lottery of the new field is found,
and if it is still a lottery a new compromise is added at its mean, up to --rounds times.

Per election with a cycle (honest preferences, no strategy):
    prefer       the share of voters who like the compromise better than the lottery (strictly), and
                 the share who like it less; unanimous: no voter likes it less
    CW after 1   the compromise is a Condorcet winner of the new field (the cycle is gone in one step)
    rounds       how many compromises until the maximal lottery picks one candidate (> --rounds: never)
    spread       the cycle's size: the expected distance between two candidates drawn independently from
                 the lottery (positions, unweighted), after the rounds against before
    cost         the lottery's loss against the best candidate of the same field, in total utility, as a
                 share of the best candidate's lead over the average candidate of the original field;
                 before and after the rounds

With --challengers, positions that beat the field also come in (the McKelvey chase), from a pool: the
voters' own positions (every voter could stand). Each round: the maximal lottery of the field; if it is
a lottery, a compromise at its mean joins the field; if it picks one candidate, the pool position that
beats that candidate by the largest majority joins (none: the winner is stable against the pool, and
the chase ends). With --pool N, N random positions drawn fresh each round join the pool, half spread as the voters
are (their mean and covariance), half around the current winner at scales from the voters' spread down
to a hundredth of it, so that a challenger can come from (nearly) anywhere, as in McKelvey's
theorem; a winner is then stable only against that round's pool. Per election, from the field as drawn (with or without a Condorcet winner):
    stable       the share of elections that reach a winner no voter's position beats, and in how
                 many rounds (at most --rounds)
    cycles       how many rounds had a lottery (a cycle in the field)
    spread       the largest spread of a lottery along the way
    region       where the chase ends up: the last 10 results' average distance from their mean, and how
                 far their mean lies from the voters' mean (shares of the voters' average distance from it)
    centre       the winner's distance from the voters' mean position, at the start and at the end
                 (the last field's maximal lottery, its mean), as a share of the voters' average
                 distance from that point
    U            the total utility of the end result against the field as drawn's winner (or lottery),
                 as a share of the best voter position's lead over the average candidate

Usage (from the lab's folder):
    python tools/compromise.py --elections 5000
    python tools/compromise.py --elections 5000 --utility quadratic
    python tools/compromise.py --elections 500 --challengers --rounds 30
Electorates as run.py's defaults otherwise (--voters, --cands, --seed, --utility).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the lab's modules
import run  # noqa: E402
from metrics import maximal_lottery  # noqa: E402
from models import draw_electorate, distances, utility  # noqa: E402
from scenario import Scenario  # noqa: E402


def lottery(M):
    """The maximal lottery of margins M, by linear programming (metrics.maximal_lottery enumerates
    supports, too slow for the growing fields of the chase): p >= 0, sum 1, p @ M >= 0; with ties among
    several, the one the solver finds."""
    if len(M) <= 8:
        return maximal_lottery(M)
    from scipy.optimize import linprog
    C = len(M)
    res = linprog(np.zeros(C), A_ub=-M.T, b_ub=np.zeros(C), A_eq=np.ones((1, C)), b_eq=[1.0], bounds=(0, None),
                  method="highs")
    p = np.clip(res.x, 0, None)
    p[p < 1e-9] = 0.0
    return p / p.sum()


def margins(u):
    gt = (u[:, :, None] > u[:, None, :]).sum(axis=0)
    return (gt - gt.T).astype(float)


def spread(p, X):
    """Expected distance between two independent draws from the lottery p over positions X."""
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return float(p @ D @ p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--elections", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--voters", type=int, default=99)
    ap.add_argument("--cands", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--utility", default="exp", choices=["exp", "linear", "quadratic"])
    ap.add_argument("--challengers", action="store_true", help="the chase with challengers from the voters' positions")
    ap.add_argument("--pool", type=int, default=0,
                    help="with --challengers: also this many random positions per round, drawn as the voters are "
                         "spread (their mean and covariance), so challengers can come from anywhere")
    a = ap.parse_args()
    p = run.default_params()
    p.update(voters=a.voters, cands=a.cands, seed=a.seed, utility=a.utility)
    sc = Scenario(p["voters"], p["cands"], p["dims"], p["clusters"], p["spread"], p["width"], p["zipf"],
                  p["cand_dispersion"], p["cand_shift"], p.get("model", "clusters"), p.get("ks_alpha", 1.5),
                  p.get("ks_cut", 0.2), p.get("ks_shared", 0), salience=p.get("salience", "none"),
                  salience_exponent=p.get("salience_exponent", 1.0), fixed_variance=p.get("fixed_variance", 0),
                  field_kappa=p.get("field_kappa", 1.5), field_salience=p.get("field_salience", 0.0),
                  noise_strength=p.get("noise_strength", 3.0), utility=p.get("utility", "exp"),
                  cands_vote=int(p.get("cands_vote", 0)))      # as run.simulate
    if a.challengers:
        return chase(a, p, sc)
    rec = []
    for e in range(a.elections):
        ele = draw_electorate(sc, np.random.default_rng([p["seed"], e]))
        u, V, X, vw = ele.u, ele.voters, ele.cands.copy(), ele.voter_weights
        lot = maximal_lottery(margins(u))
        if lot.max() > 1 - 1e-9:
            continue                                    # a Condorcet winner: nothing to do
        U = u.sum(axis=0)
        scale = U.max() - U.mean()
        r = {"spread0": spread(lot, X), "cost0": (U.max() - lot @ U) / scale}
        uu, XX, pl = u, X, lot
        rounds = None
        for k in range(1, a.rounds + 1):
            m = pl @ XX                                 # the lottery's mean position
            um = utility(distances(V, m[None], vw), sc.width, sc.utility)[:, 0]
            if k == 1:
                ul = u @ lot
                tol = 1e-12 * np.ptp(u)
                r["better"] = float((um > ul + tol).mean())
                r["worse"] = float((um < ul - tol).mean())
                r["gain"] = float((um - ul).sum() / scale)
            uu, XX = np.hstack([uu, um[:, None]]), np.vstack([XX, m])
            pl = maximal_lottery(margins(uu))
            if k == 1:
                r["cw1"] = bool(pl.max() > 1 - 1e-9 and pl.argmax() == len(pl) - 1)
            if pl.max() > 1 - 1e-9:
                rounds = k
                break
        Uu = uu.sum(axis=0)
        r["rounds"] = rounds
        r["spread1"] = spread(pl, XX)
        r["cost1"] = (Uu.max() - pl @ Uu) / scale
        r["best_new"] = bool(Uu.argmax() >= u.shape[1])
        rec.append(r)
        print(f"\r{e + 1}/{a.elections} elections, {len(rec)} with a cycle", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    n = len(rec)
    print(f"Compromise at the maximal lottery's mean: {a.elections} elections, {n} without a Condorcet winner "
          f"({n / a.elections:.1%}); utility {a.utility}, {a.voters} voters, {a.cands} candidates, seed {a.seed}.")
    if not n:
        return
    g = lambda k: np.array([x[k] for x in rec], float)
    rounds = np.array([x["rounds"] if x["rounds"] is not None else np.inf for x in rec])
    print(f"  voters liking the compromise better than the lottery: {g('better').mean():.1%} on average "
          f"(median {np.median(g('better')):.1%}), less: {g('worse').mean():.1%}; "
          f"unanimous (nobody less) in {(g('worse') == 0).mean():.0%} of the cycles")
    print(f"  total utility of the compromise against the lottery: {g('gain').mean():+.2f} of the best "
          f"candidate's lead over the average (better in {(g('gain') > 0).mean():.0%})")
    print(f"  the compromise is the Condorcet winner of the new field at once: {g('cw1').mean():.0%}")
    print("  a Condorcet winner after " + ", ".join(f"{k}: {(rounds <= k).mean():.0%}" for k in range(1, a.rounds + 1)
                                                 if k <= 3 or k == a.rounds) + f" compromises; never: {np.isinf(rounds).mean():.0%}")
    print(f"  spread of the lottery (expected distance between two draws): {g('spread0').mean():.3f} before, "
          f"{g('spread1').mean():.3f} after (0 once a single candidate wins)")
    print(f"  cost of the lottery against the best candidate of the field: {g('cost0').mean():.3f} before, "
          f"{g('cost1').mean():.3f} after; the field's best candidate is a compromise in "
          f"{g('best_new').mean():.0%}")


def chase(a, p, sc):
    """Compromises and challengers in turn (see the module text)."""
    rec = []
    for e in range(a.elections):
        ele = draw_electorate(sc, np.random.default_rng([p["seed"], e]))
        u, V, vw = ele.u, ele.voters, ele.voter_weights
        up0 = utility(distances(V, V, vw), sc.width, sc.utility)           # every voter's position as a candidate
        prng = np.random.default_rng([p["seed"], e, 7])
        cov = np.cov(V.T)
        centre = V.mean(axis=0)
        dscale = float(np.linalg.norm(V - centre, axis=1).mean())
        scale = up0.sum(axis=0).max() - u.sum(axis=0).mean()
        uu, XX = u, ele.cands.copy()
        pl = lottery(margins(uu))
        start_cycle = pl.max() < 1 - 1e-9
        r = {"cycle0": start_cycle, "d0": float(np.linalg.norm(pl @ XX - centre)) / dscale,
             "U0": float(pl @ uu.sum(axis=0))}
        stable, cycles, spr = None, int(start_cycle), spread(pl, XX)
        path = [pl @ XX]                                # the result's (mean) position after each round
        for k in range(1, a.rounds + 1):
            if pl.max() < 1 - 1e-9:                     # a cycle: the compromise at the lottery's mean
                m = pl @ XX
                col = utility(distances(V, m[None], vw), sc.width, sc.utility)
                uu, XX = np.hstack([uu, col]), np.vstack([XX, m])
            else:                                       # a winner: the strongest challenger from the pool
                if a.pool:                              # half spread as the voters, half around the winner at scales
                    x_w = XX[int(pl.argmax())]          # from 1 down to 0.01 of the voters' spread (log-uniform)
                    h = a.pool // 2
                    sc_ = 10 ** prng.uniform(-2, 0, a.pool - h)[:, None]
                    loc = x_w + sc_ * prng.multivariate_normal(np.zeros(len(centre)), cov, a.pool - h)
                    Xp = np.vstack([V, prng.multivariate_normal(centre, cov, h), loc])
                else:
                    Xp = V
                up = up0 if not a.pool else np.hstack([up0, utility(distances(V, Xp[len(V):], vw), sc.width, sc.utility)])
                w = uu[:, int(pl.argmax())]
                mg = (up > w[:, None]).sum(axis=0) - (up < w[:, None]).sum(axis=0)
                j = int(mg.argmax())
                if mg[j] <= 0:
                    stable = k - 1
                    break
                uu, XX = np.hstack([uu, up[:, j:j + 1]]), np.vstack([XX, Xp[j]])
            pl = lottery(margins(uu))
            path.append(pl @ XX)
            if pl.max() < 1 - 1e-9:
                cycles += 1
                spr = max(spr, spread(pl, XX))
        else:                                           # out of rounds: is the last winner stable (voters' positions)?
            if pl.max() > 1 - 1e-9:
                w = uu[:, int(pl.argmax())]
                if ((up0 > w[:, None]).sum(axis=0) - (up0 < w[:, None]).sum(axis=0)).max() <= 0 and not a.pool:
                    stable = a.rounds
        tail = np.array(path[-10:])                     # where the chase wanders: the last 10 results
        r["region"] = float(np.linalg.norm(tail - tail.mean(axis=0), axis=1).mean()) / dscale
        r["region_d"] = float(np.linalg.norm(tail.mean(axis=0) - centre)) / dscale
        r.update(stable=stable, cycles=cycles, spread=spr, d1=float(np.linalg.norm(pl @ XX - centre)) / dscale,
                 dU=float(pl @ uu.sum(axis=0) - r["U0"]) / scale, n_added=uu.shape[1] - u.shape[1])
        rec.append(r)
        print(f"\r{e + 1}/{a.elections} elections", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    print(f"The chase (compromises and challengers from the voters' positions"
          + (f" and {a.pool} random positions per round" if a.pool else "") + f"): {a.elections} elections, "
          f"utility {a.utility}, {a.voters} voters, {a.cands} candidates, at most {a.rounds} rounds, seed {a.seed}.")
    for label, sel in (("all elections", lambda x: True), ("starting from a cycle", lambda x: x["cycle0"]),
                       ("starting from a Condorcet winner", lambda x: not x["cycle0"])):
        R = [x for x in rec if sel(x)]
        if not R:
            continue
        g = lambda k: np.array([x[k] for x in R], float)
        st = np.array([x["stable"] if x["stable"] is not None else np.inf for x in R])
        fin = np.isfinite(st)
        print(f"  {label} ({len(R)}):")
        print(f"    stable against the pool: {fin.mean():.0%}"
              + (f", after {np.median(st[fin]):.0f} rounds (median; at most {st[fin].max():.0f})" if fin.any() else "")
              + f"; already at the start: {(st == 0).mean():.0%}")
        print(f"    rounds with a cycle: {g('cycles').mean():.2f} on average, at least one in {(g('cycles') > 0).mean():.0%}; "
              f"largest spread of a lottery {g('spread').mean():.3f}")
        print(f"    distance of the result from the voters' mean (share of their average distance): "
              f"{g('d0').mean():.2f} at the start, {g('d1').mean():.2f} at the end")
        print(f"    the last 10 results: spread around their mean {g('region').mean():.3f}, their mean "
              f"{g('region_d').mean():.2f} from the voters' mean (shares of the voters' average distance)")
        print(f"    total utility, end against start: {g('dU').mean():+.3f} of the best voter position's lead "
              f"over the average candidate; candidates added {g('n_added').mean():.1f}")


if __name__ == "__main__":
    main()
