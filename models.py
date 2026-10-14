"""Electorate models: how voters' and candidates' positions in the issue space are drawn.

A model is one entry in MODELS (name -> Model), chosen with --model. For one election it draws the
positions of the voters and of the candidates (candidates are drawn like voters), and optionally
per-person weights of the dimensions (how much each voter or candidate cares about each issue;
None = every dimension counts alike). A model's own parameters are fields of the Scenario
(scenario.py); each model reads the ones it uses. Everything after the draw is common to all models:

    distance    |(x - p) * w|: Euclidean, each dimension scaled by the weight w of the person at x
    utility     exp(-distance / width): 1 at zero distance, falling toward 0 far away. Utilities are
                not normalised; zero means "infinitely far".
    candidates  optionally transformed relative to the electorate: cand_dispersion scales their
                distances from the voters' mean position (Merrill 1984's relative candidate
                dispersion: < 1 clusters them toward the centre, > 1 spreads them out), and
                cand_shift moves them all along the first (most important) dimension by that many
                standard deviations of the voters on it (a systematic bias, e.g. candidates more
                often from one social group). Both act after all random draws, so the voters (and
                the candidates' draws) are the same for every setting.

MODEL "clusters" (default): voter clusters in a space of `dims` dimensions.
    Cluster centres are drawn N(0, 1) per dimension; cluster sizes vary (Dirichlet weights, so some
    elections have a dominant cluster); voters and candidates are drawn around their cluster's
    centre with standard deviation `spread`. A voter belongs to one cluster on all dimensions at once.
    Zipf dimensions (zipf=1), after Clay Smith's Mixture of Zipf Gaussians
    (https://github.com/cdsmith/spatial-voting, src/Model.hs): dimension i (1, 2, ...) has standard
    deviation 1/i, for cluster centres and candidates' and voters' spread alike, so the first issue
    dominates and later ones matter less and less (total variance converges to pi^2/6, so beyond a
    few dimensions their number hardly matters). Each cluster applies the 1/i scaling to its own,
    slightly shuffled order of dimensions (each rank perturbed by normal noise with standard
    deviation 0.1 x dims, Smith's ratio of 10 for 100 dimensions), so camps differ in which issues
    they vary on most. Cluster centres keep the global order.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Model:
    """One electorate model: draw(sc, rng) -> (voters, cands, voter_weights, cand_weights), positions
    as arrays (n, dims) and (C, dims), weights of the same shapes or None (all dimensions alike)."""
    name: str
    draw: Callable


def _draw_clusters(sc, rng):
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
    return voters, cands, None, None


MODELS = {m.name: m for m in [
    Model("clusters", _draw_clusters),
]}


# ---------- common to all models ----------
def distances(X, P, w=None):
    """d[i, j]: distance from person i (position X[i], dimension weights w[i] or None) to point P[j]."""
    D = X[:, None, :] - P[None, :, :]
    if w is not None:
        D = D * w[:, None, :]
    return np.linalg.norm(D, axis=2)


def utility(d, width):
    """Utility of a candidate at distance d: exp(-d / width)."""
    return np.exp(-d / width)


@dataclass
class Electorate:
    """One election's electorate: utilities u[v, c], positions and dimension weights (None: all alike)."""
    u: np.ndarray
    voters: np.ndarray
    cands: np.ndarray
    voter_weights: np.ndarray = None
    cand_weights: np.ndarray = None


def draw_electorate(sc, rng):
    """The electorate of one election under the scenario's model (sc.model, default "clusters")."""
    voters, cands, vw, cw = MODELS[getattr(sc, "model", "clusters")].draw(sc, rng)
    if sc.cand_dispersion != 1.0 or sc.cand_shift != 0.0:
        centre = voters.mean(axis=0)
        cands = centre + sc.cand_dispersion * (cands - centre)
        cands[:, 0] += sc.cand_shift * voters[:, 0].std()
    u = utility(distances(voters, cands, vw), sc.width)
    return Electorate(u, voters, cands, vw, cw)
