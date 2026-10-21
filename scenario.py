"""Scenario: the settings of one condition's electorates (which model, its parameters, and those
common to all models). The models themselves are in models.py.
"""
from dataclasses import dataclass
import numpy as np

from models import draw_electorate


@dataclass
class Scenario:
    n_voters: int = 99      # odd, so pairwise majorities rarely tie
    n_cands: int = 6
    dims: int = 6           # issue-space dimensions
    n_clusters: int = 20    # voter clusters
    spread: float = 0.5     # standard deviation of voters around their cluster centre
    width: float = 0.7      # distance scale of the utility fall-off
    zipf: int = 0           # 1: dimension i has scale 1/i (see models.py); 0: all dimensions equal
    cand_dispersion: float = 1.0   # candidates' distance from the electorate's centre, x this (< 1: clustered)
    cand_shift: float = 0.0        # candidates moved along the first dimension, in voter SDs on it
    model: str = "noise"           # electorate model (models.MODELS); dims, n_clusters, spread, zipf: "clusters"
    ks_alpha: float = 1.5          # "ks": faction alpha (Chinese restaurant process)
    ks_cut: float = 0.2            # "ks": importance below which no further bundles / dimensions
    ks_shared: int = 0             # "ks": 1 = the same factions in every bundle (no cross-cutting)
    salience: str = "cluster"      # "clusters": "cluster" = each cluster weighs the issues in its own Zipf order
    salience_exponent: float = 0.5 # "clusters", salience "cluster": weights 1/i^s
    field_kappa: float = 1.5       # "field": saturation of opinions (small: smooth; large: two camps per issue)
    field_salience: float = 0.0    # "field": strength of the salience field (0 = every issue weighed alike)
    fixed_variance: int = 1        # "clusters": 1 = total variance of the dimensions fixed (pi^2/6), whatever their number
    noise_strength: float = 5.0    # "noise": how strongly voters gather where the noise is high (0 = a Gaussian blob)
    utility: str = "exp"           # utility of distance: "exp" exp(-d / width), "linear" -d / width, "quadratic" -(d / width)^2 (models.utility)


def electorate(sc: Scenario, rng: np.random.Generator):
    """Return utilities u[v, c] > 0 for one election."""
    return draw_electorate(sc, rng).u


def electorate_positions(sc: Scenario, rng: np.random.Generator):
    """Return (u, voter positions, candidate positions) for one election (same draws as electorate).
    For the dimension weights as well, use models.draw_electorate."""
    e = draw_electorate(sc, rng)
    return e.u, e.voters, e.cands
