"""Electorate generator: spatial model with voter clusters and absolute utilities.

Utilities are not normalised. A voter's utility for a candidate is exp(-distance / width):
1 at zero distance, falling toward 0 far away. Zero therefore means "infinitely far",
which gives the proportional baseline (tau = 1) a meaning.

Zipf dimensions (zipf=1), after Clay Smith's Mixture of Zipf Gaussians
(https://github.com/cdsmith/spatial-voting, src/Model.hs): dimension i (1, 2, ...) has standard
deviation 1/i, for cluster centres and candidates' and voters' spread alike, so the first issue
dominates and later ones matter less and less (total variance converges to pi^2/6, so beyond a
few dimensions their number hardly matters). Each cluster applies the 1/i scaling to its own,
slightly shuffled order of dimensions (each rank perturbed by normal noise with standard
deviation 0.1 x dims, Smith's ratio of 10 for 100 dimensions), so camps differ in which issues
they vary on most. Cluster centres keep the global order.

Candidates are drawn like voters (same clusters and weights), then optionally transformed relative to
the electorate: cand_dispersion scales their distances from the voters' mean position (Merrill 1984's
relative candidate dispersion: < 1 clusters them toward the centre, > 1 spreads them out), and
cand_shift moves them all along the first, most important dimension by that many standard deviations
of the voters on it (a systematic bias, e.g. candidates more often from one social group). Both act
after all random draws, so the voters (and the candidates' draws) are the same for every setting.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Scenario:
    n_voters: int = 99      # odd, so pairwise majorities rarely tie
    n_cands: int = 6
    dims: int = 10          # issue-space dimensions
    n_clusters: int = 5     # voter clusters
    spread: float = 0.5     # standard deviation of voters around their cluster centre
    width: float = 1.0      # distance scale of the utility fall-off
    zipf: int = 1           # 1: dimension i has scale 1/i (see module docstring); 0: all dimensions equal
    cand_dispersion: float = 1.0   # candidates' distance from the electorate's centre, x this (< 1: clustered)
    cand_shift: float = 0.0        # candidates moved along the first dimension, in voter SDs on it


def electorate(sc: Scenario, rng: np.random.Generator):
    """Return utilities u[v, c] > 0 for one election."""
    return electorate_positions(sc, rng)[0]


def electorate_positions(sc: Scenario, rng: np.random.Generator):
    """Return (u, voter positions, candidate positions) for one election (same draws as electorate)."""
    centres = rng.normal(0.0, 1.0, (sc.n_clusters, sc.dims))
    # Cluster sizes vary: Dirichlet weights, so some elections have a dominant cluster
    weights = rng.dirichlet(np.ones(sc.n_clusters))
    vc = rng.choice(sc.n_clusters, sc.n_voters, p=weights)
    voters = centres[vc] + rng.normal(0.0, sc.spread, (sc.n_voters, sc.dims))
    # Candidates are drawn from the same distribution as voters
    cc = rng.choice(sc.n_clusters, sc.n_cands, p=weights)
    cands = centres[cc] + rng.normal(0.0, sc.spread, (sc.n_cands, sc.dims))
    if sc.zipf:
        rank = np.arange(1, sc.dims + 1)
        # each cluster: its own slightly shuffled order of dimensions, then scale 1/rank
        order = np.argsort(rank[None, :] + rng.normal(0.0, 0.1 * sc.dims, (sc.n_clusters, sc.dims)), axis=1)
        own = np.empty((sc.n_clusters, sc.dims))
        own[np.arange(sc.n_clusters)[:, None], order] = 1.0 / rank[None, :]
        centres_z = centres / rank                                  # centres: global order
        voters = centres_z[vc] + (voters - centres[vc]) * own[vc]
        cands = centres_z[cc] + (cands - centres[cc]) * own[cc]
    if sc.cand_dispersion != 1.0 or sc.cand_shift != 0.0:
        centre = voters.mean(axis=0)
        cands = centre + sc.cand_dispersion * (cands - centre)
        cands[:, 0] += sc.cand_shift * voters[:, 0].std()
    d = np.linalg.norm(voters[:, None, :] - cands[None, :, :], axis=2)
    return np.exp(-d / sc.width), voters, cands
