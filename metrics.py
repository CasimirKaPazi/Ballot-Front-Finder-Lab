"""Per-election quantities and scenario-level axes.

Efficiency against a baseline lottery q:
    E = sum(U_method - U_base) / sum(U_best - U_base)      (ratio of sums over elections)
with U(c) = total utility of candidate c, U_best = max_c U(c).

Baselines, from each voter's personal lottery p_v(c) proportional to u_v(c)^tau, averaged over voters:
    tau = 0    random candidate          -> VSE  (classical)
    tau = 1    utility-proportional      -> PE   (proportional efficiency, the headline axis)
    tau = inf  random ballot             -> RBE  (random-ballot efficiency)

Condorcet efficiency (CE): among elections with a sincere Condorcet winner,
the average probability that the method elects it.
"""
import numpy as np


def election_row(u, P):
    """Quantities for one election: u utilities, P method's winner probabilities."""
    U = u.sum(axis=0)
    n, C = u.shape
    q0 = np.full(C, 1.0 / C)
    q1 = (u / u.sum(axis=1, keepdims=True)).mean(axis=0)
    qinf = np.bincount(u.argmax(axis=1), minlength=C) / n
    gt = (u[:, :, None] > u[:, None, :]).sum(axis=0)
    beats = gt > gt.T
    cw = np.flatnonzero(beats.sum(axis=1) == C - 1)
    return {
        "U_method": float(P @ U),
        "U_best": float(U.max()),
        "U_tau0": float(q0 @ U),
        "U_tau1": float(q1 @ U),
        "U_tauinf": float(qinf @ U),
        "has_cw": int(len(cw) == 1),
        "p_cw": float(P[cw[0]]) if len(cw) == 1 else float("nan"),
    }


def efficiency(rows, base):
    num = sum(r["U_method"] - r[base] for r in rows)
    den = sum(r["U_best"] - r[base] for r in rows)
    return num / den


def summarize(rows):
    cw = [r["p_cw"] for r in rows if r["has_cw"]]
    return {
        "PE": efficiency(rows, "U_tau1"),
        "VSE": efficiency(rows, "U_tau0"),
        "RBE": efficiency(rows, "U_tauinf"),
        "CE": float(np.mean(cw)) if cw else float("nan"),
        "elections_with_cw": len(cw),
    }


def pareto(points):
    """Indices of non-dominated points (all axes: higher is better)."""
    pts = np.asarray(points)
    keep = []
    for i, p in enumerate(pts):
        dominated = any(np.all(q >= p) and np.any(q > p) for j, q in enumerate(pts) if j != i)
        if not dominated:
            keep.append(i)
    return keep
