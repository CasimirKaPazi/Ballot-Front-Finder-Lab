"""Per-election quantities and scenario-level axes.

Both axes are efficiencies with the same two reference points, as ratios of sums over elections:
    E = sum(X_method - X_rb) / sum(X_best - X_rb)
0 = random ballot (the winner is a random voter's favourite), 100% = the best candidate of each
election. Values above 100% cannot be reached; values below 0 are worse than random ballot.

VSE (voter satisfaction efficiency): X = total utility of the winner, U(c) = sum over voters of
u_v(c); best = the candidate with the highest total utility. The usual VSE puts the zero at a
random candidate; here it is the random ballot, which needs no assumption about who stands.
With candidates drawn like voters, the two differ only in scale.

WME (worst-margin efficiency), a refined Condorcet measure: X = the winner's worst margin, its
smallest head-to-head margin against any other candidate, by sincere preferences, as a share of
voters (positive: the Condorcet winner, by that much; negative: a majority prefers some rival).
Best = the candidate with the best worst margin (the Minimax winner, by construction), so with
honest ballots Minimax reaches 100%, as Condorcet methods do on CE.

Condorcet efficiency (CE), reported alongside: among elections with a sincere Condorcet winner,
the average probability that the method elects it.
"""
import numpy as np
from ballots import BALLOTS
from outcomes import value

def worst_margins(u):
    """m[c]: candidate c's smallest pairwise margin against any other, as a share of voters."""
    n, C = u.shape
    gt = (u[:, :, None] > u[:, None, :]).sum(axis=0)
    M = (gt - gt.T) / n
    np.fill_diagonal(M, np.inf)
    return M.min(axis=1)


def electorate_stats(u):
    """Properties of the electorate itself (the same for all methods):
    cw_is_util  1 if the sincere Condorcet winner also has the highest total utility, 0 if not,
                empty if there is no Condorcet winner
    U_cw        total utility of the Condorcet winner (empty if none)
    fav_ratio   median over voters of u(favourite) / u(least liked): how steep preferences are (only
                for positive utilities, --utility exp; empty for linear)"""
    n, C = u.shape
    gt = (u[:, :, None] > u[:, None, :]).sum(axis=0)
    cw = np.flatnonzero((gt > gt.T).sum(axis=1) == C - 1)
    U = u.sum(axis=0)
    has = len(cw) == 1
    return {"cw_is_util": int(cw[0] == U.argmax()) if has else "",
            "U_cw": float(U[cw[0]]) if has else "",
            "fav_ratio": float(np.median(u.max(axis=1) / u.min(axis=1))) if u.min() > 0 else ""}


def maximal_lottery(M, tol=1e-9):
    """The maximal lottery for pairwise margins M (M[a, b]: voters preferring a to b minus the reverse):
    a lottery p over the candidates that no candidate beats in expectation (p @ M >= 0 in every column),
    the optimal mixed strategy of the symmetric zero-sum game M (Fishburn 1984). With a Condorcet
    winner it elects that candidate; in a cycle it mixes the candidates of the cycle. Found by support
    enumeration (smallest supports first): on the support S the lottery makes every candidate of S
    tie with it (M[S, S]^T p = 0, sum 1), and no candidate outside beats it. With an odd number of voters
    and strict preferences it is unique; with tied margins the first one found is used."""
    from itertools import combinations
    C = len(M)
    for k in range(1, C + 1):
        for S in combinations(range(C), k):
            S = list(S)
            A = np.vstack([M[np.ix_(S, S)].T, np.ones(k)])
            b = np.r_[np.zeros(k), 1.0]
            q, *_ = np.linalg.lstsq(A, b, rcond=None)
            if np.abs(A @ q - b).max() > 1e-7 or q.min() < -tol:
                continue
            p = np.zeros(C)
            p[S] = np.clip(q, 0, None)
            p /= p.sum()
            if (p @ M).min() >= -1e-7 * max(1.0, np.abs(M).max()):
                return p
    raise ValueError("no maximal lottery found")      # cannot happen: a symmetric zero-sum game has a solution


def election_row(u, P, m):
    """Quantities for one election: u utilities, P method's winner probabilities,
    m worst margins from worst_margins (the same for all methods)."""
    U = u.sum(axis=0)
    n, C = u.shape
    rb = np.bincount(u.argmax(axis=1), minlength=C) / n          # random ballot lottery
    gt = (u[:, :, None] > u[:, None, :]).sum(axis=0)
    beats = gt > gt.T
    cw = np.flatnonzero(beats.sum(axis=1) == C - 1)
    return {
        "U_method": float(value(P, U)),
        "U_best": float(U.max()),
        "U_rb": float(value(rb, U)),
        "U_rand": float(U.mean()),                                 # random candidate (classical VSE zero)
        "has_cw": int(len(cw) == 1),
        "p_cw": float(P[cw[0]]) if len(cw) == 1 else float("nan"),
        "WM": float(value(P, m)),
        "WM_best": float(m.max()),
        "WM_rb": float(value(rb, m)),
    }




def condorcet_winner(gt):
    """Index of the Condorcet winner for pairwise counts gt[a, b] (voters putting a above b), or None."""
    C = gt.shape[0]
    cw = np.flatnonzero((gt > gt.T).sum(axis=1) == C - 1)
    return int(cw[0]) if len(cw) == 1 else None


def ballot_condorcet(method, B, u, back, P):
    """Does the Condorcet winner as the cast ballots show it match the real one (sincere preferences)?
    This is what a study of real ballot data sees. Pairwise counts come from the ballots themselves
    (a voter counts for a over b if the ballot puts a higher; equal = no preference), only for ballots
    that record an order (rankings, scores, grades; not plurality, approval, veto or two-round marks,
    whose pairwise counts mostly reflect the marks). For the Inverse-* methods the ballot expresses
    reversed preferences, so it is read reversed. Ballot candidates (nomination: the field) are mapped
    back to the original candidates. P: the method's winner lottery on the original field.
    bcw: same (the real Condorcet winner), false (another candidate), none (no Condorcet winner in the
    ballots although there is a real one), spurious (one in the ballots, none in reality), neither;
    empty for other ballot types. p_bcw: probability that the method elects the ballots' Condorcet
    winner (empty without one): its Condorcet efficiency as ballot data would show it."""
    if not BALLOTS[method.ballot].ordered:              # ballots that record an order (ballots.py)
        return {"bcw": "", "p_bcw": ""}
    X = -B if method.inverse else B
    gt_b = (X[:, :, None] > X[:, None, :]).sum(axis=0)
    cb = condorcet_winner(gt_b)
    if cb is not None:
        onehot = np.zeros(B.shape[1])
        onehot[cb] = 1
        cb = int(np.argmax(back(onehot)))          # clones count as their original
    cr = condorcet_winner((u[:, :, None] > u[:, None, :]).sum(axis=0))
    if cr is not None:
        kind = "same" if cb == cr else ("none" if cb is None else "false")
    else:
        kind = "neither" if cb is None else "spurious"
    return {"bcw": kind, "p_bcw": float(P[cb]) if cb is not None else ""}


def ties(method, B, P):
    """Ties in one result (columns since 2026-10-26):
    split         1 if the winner is drawn by lot among several candidates of the original field (the
                  method's result is a lottery: a tie broken at random, equal margins locked in every
                  order, or a method that draws, as Condorcet-random-drop), else 0
    zero_margins  share of candidate pairs that the cast ballots leave tied (as many voters put a above
                  b as b above a; pairwise counts as in ballot_condorcet, on the field the voters face);
                  only for ballots that record an order, else empty"""
    split = int(np.max(P) < 1 - 1e-9)
    if not BALLOTS[method.ballot].ordered:
        return {"split": split, "zero_margins": ""}
    gt = (B[:, :, None] > B[:, None, :]).sum(axis=0)
    iu = np.triu_indices(B.shape[1], 1)
    return {"split": split, "zero_margins": float(np.mean(gt[iu] == gt.T[iu]))}


def efficiency(rows, base):
    num = sum(r["U_method"] - r[base] for r in rows)
    den = sum(r["U_best"] - r[base] for r in rows)
    return num / den


def summarize(rows):
    cw = [r["p_cw"] for r in rows if r["has_cw"]]
    return {
        "VSE": efficiency(rows, "U_rb"),
        "WME": sum(r["WM"] - r["WM_rb"] for r in rows) / sum(r["WM_best"] - r["WM_rb"] for r in rows),
        "CE": float(np.mean(cw)) if cw else float("nan"),
        "elections_with_cw": len(cw),
    }


# Smallest difference that counts on each front axis (fractions, as the values): sincerity differences
# below 1 point are ties. They are measured precisely (the same elections for every method), but at
# 99.7 against 99.9 no voter is noticeably more honest; VSE* and WME* count any difference.
FRONT_TOLERANCE = {"VSE*": 0.0, "WME*": 0.0, "sincerity": 0.01}


def pareto(points, tol=0.0):
    """Indices of non-dominated points (all axes: higher is better). tol: per axis (or one number), the
    smallest difference that counts; smaller ones are ties."""
    pts = np.asarray(points)
    tol = np.broadcast_to(np.asarray(tol, float), pts.shape[1:])
    keep = []
    for i, p in enumerate(pts):
        dominated = any(np.all(q >= p - tol) and np.any(q > p + tol) for j, q in enumerate(pts) if j != i)
        if not dominated:
            keep.append(i)
    return keep
