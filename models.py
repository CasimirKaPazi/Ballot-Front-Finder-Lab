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

MODEL "ks": Jameson Quinn's hierarchical ("kitchen sink") model, from vse-sim (voter_models.py,
KSModel), with the parameters of its published results. Written anew from the algorithm, with one
change (4. below). Parameters: ks_alpha (faction alpha), ks_cut (where importance stops); --dims,
--clusters, --spread and --zipf do not apply.
    1. Issue bundles: the first bundle weighs 1, each further one a Beta(1, 3) fraction of the one
       before, until the weight falls below ks_cut (0.2). Within a bundle, its first dimension has
       the bundle's weight, each further one a Beta(1.5, 3) fraction of the one before, again down to
       ks_cut. So the number of bundles and dimensions varies by election; few issues matter much.
    2. Factions, per bundle and independently across bundles (cross-cutting cleavages): a Chinese
       restaurant process over voters and candidates; person i starts a new faction with probability
       alpha / (i + alpha), else joins the faction of a random earlier person (ks_alpha = 1.5). A few
       large factions and a tail of small ones; their number is not fixed.
    3. Each faction, per bundle: a position, N(0, c) per dimension with c ~ Beta(3, 1.5), and a caring
       level ~ Beta(3, 1.5).
    4. Each person, per dimension: the faction's position plus N(0, 1) x sqrt(1 - caring): the more a
       faction cares, the tighter it is. Change: an own offset per dimension; vse-sim's code uses one
       offset for all dimensions of a bundle, which looks unintended.
    5. Weights: a person weighs each dimension by its importance x the caring of the person's faction
       on that bundle, normalised (distance = sqrt(sum((x - p)^2 (w c)^2) / sum((w c)^2)), as in
       vse-sim). Utilities then use exp(-distance / width) like every model here (vse-sim: -distance).
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
    describe: Callable          # g -> text of its parameters for reports; g(name) reads a run parameter


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


# MODEL "ks": Quinn's hierarchical "kitchen sink" model (vse-sim, voter_models.py, KSModel)
KS_BUNDLE_DECAY = (1.0, 3.0)     # Beta parameters: each issue bundle weighs this fraction of the one before
KS_DIM_DECAY = (1.5, 3.0)        # Beta parameters: each dimension in a bundle weighs this fraction of the one before
KS_CARING = (3.0, 1.5)           # Beta parameters of a faction's caring (and of the spread of faction positions)


def _draw_ks(sc, rng):
    """Quinn's hierarchical model, written from the description of vse-sim's KSModel (published
    parameters; see the README, Electorates). Persons = voters, then candidates, drawn alike."""
    n = sc.n_voters + sc.n_cands
    # 1. Issue bundles and their dimensions, with decaying importance
    bundles, dim_w = [], []
    bw = 1.0
    while bw > sc.ks_cut:
        w, k = bw, 0
        while w > sc.ks_cut:
            dim_w.append(w)
            k += 1
            w *= rng.beta(*KS_DIM_DECAY)
        bundles.append(k)
        bw *= rng.beta(*KS_BUNDLE_DECAY)
    dim_w = np.array(dim_w)
    D = len(dim_w)
    pos = np.empty((n, D))
    care = np.empty((n, D))
    start = 0
    for k in bundles:
        # 2. Factions in this bundle, independently of other bundles (Chinese restaurant process:
        #    each person starts a new faction with probability alpha / (i + alpha), else joins the
        #    faction of a random earlier person)
        fac = np.empty(n, dtype=int)
        nf = 0
        for i in range(n):
            r = (i + sc.ks_alpha) * rng.random()
            if r > i:
                fac[i] = nf
                nf += 1
            else:
                fac[i] = fac[int(r)]
        # 3. Each faction: a position (spread sqrt(c) for a random c) and a caring level
        means = np.empty((nf, k))
        caring = np.empty(nf)
        for f in range(nf):
            c = rng.beta(*KS_CARING)
            means[f] = rng.normal(0.0, np.sqrt(c), k)
            caring[f] = rng.beta(*KS_CARING)
        # 4. Persons: around their faction's position, the tighter the more the faction cares
        #    (an own offset per dimension; vse-sim uses one offset for all dimensions of a bundle)
        noise = rng.normal(0.0, 1.0, (n, k))
        pos[:, start:start + k] = means[fac] + noise * np.sqrt(1 - caring[fac])[:, None]
        care[:, start:start + k] = caring[fac][:, None]
        start += k
    # 5. Weights: dimension importance x the person's caring, normalised per person, so a distance is
    #    sqrt(sum((x - p)^2 (w c)^2) / sum((w c)^2)) as in vse-sim
    W = dim_w[None, :] * care
    W = W / np.sqrt((W ** 2).sum(axis=1, keepdims=True))
    nv = sc.n_voters
    return pos[:nv], pos[nv:], W[:nv], W[nv:]


MODELS = {m.name: m for m in [
    Model("clusters", _draw_clusters,
          lambda g: f"{g('dims')} dimensions{' (Zipf)' if int(g('zipf')) else ''}, {g('clusters')} clusters, "
                    f"spread {g('spread')}"),
    Model("ks", _draw_ks,
          lambda g: f"Quinn's hierarchical issue bundles and factions, faction alpha {g('ks_alpha')}, "
                    f"cut {g('ks_cut')}"),
]}


def model_text(p):
    """One-line description of a run's electorate model with its own parameters (p: namespace or dict
    of run parameters, values as numbers or text)."""
    g = (lambda k: p[k]) if isinstance(p, dict) else (lambda k: getattr(p, k))
    name = g("model") if (isinstance(p, dict) and "model" in p) or hasattr(p, "model") else "clusters"
    return f"model {name}: {MODELS[name].describe(g)}"


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
