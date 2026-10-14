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
    dims: int = 10          # issue-space dimensions
    n_clusters: int = 5     # voter clusters
    spread: float = 0.5     # standard deviation of voters around their cluster centre
    width: float = 1.0      # distance scale of the utility fall-off
    zipf: int = 1           # 1: dimension i has scale 1/i (see models.py); 0: all dimensions equal
    cand_dispersion: float = 1.0   # candidates' distance from the electorate's centre, x this (< 1: clustered)
    cand_shift: float = 0.0        # candidates moved along the first dimension, in voter SDs on it
    model: str = "clusters"        # electorate model (models.MODELS); dims, n_clusters, spread, zipf: "clusters"


def electorate(sc: Scenario, rng: np.random.Generator):
    """Return utilities u[v, c] > 0 for one election."""
    return draw_electorate(sc, rng).u


def electorate_positions(sc: Scenario, rng: np.random.Generator):
    """Return (u, voter positions, candidate positions) for one election (same draws as electorate).
    For the dimension weights as well, use models.draw_electorate."""
    e = draw_electorate(sc, rng)
    return e.u, e.voters, e.cands
