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

MODEL "clusters": voter clusters in a space of `dims` dimensions.
    Cluster centres are drawn N(0, 1) per dimension; cluster sizes vary (Dirichlet weights, so some
    elections have a dominant cluster); voters and candidates are drawn around their cluster's
    centre with standard deviation `spread`. A voter belongs to one cluster on all dimensions at once.
    Zipf dimensions (zipf=1), after Chris Smith's Mixture of Zipf Gaussians
    (https://github.com/cdsmith/spatial-voting, src/Model.hs): dimension i (1, 2, ...) has standard
    deviation 1/i, for cluster centres and candidates' and voters' spread alike, so the first issue
    dominates and later ones matter less and less (total variance converges to pi^2/6, so beyond a
    few dimensions their number hardly matters). Each cluster applies the 1/i scaling to its own,
    slightly shuffled order of dimensions (each rank perturbed by normal noise with standard
    deviation 0.1 x dims, Smith's ratio of 10 for 100 dimensions), so camps differ in which issues
    they vary on most. Cluster centres keep the global order.
    Fixed total variance (fixed_variance=1): all positions are scaled by one factor, so that the total
    variance of the dimension scales is pi^2/6 (the limit of Zipf dimensions) whatever their number.
    More dimensions then spread the same disagreement over more issues instead of making every distance
    longer (which would make preferences steeper at the same width). With zipf=1 the factor is close to 1.
    Salience per cluster (salience="cluster"): positions say where people stand; salience says what they
    care about. Each cluster weighs the issues 1, 1/2, 1/3, ... (Zipf, as Smith's dimensions; more
    generally 1/i^s with s = salience_exponent, default 1; 0 = all alike) in its own random order, in its members' distances (and its candidates'), scaled so that the mean squared weight
    is 1. So society can disagree on many issues while each group mainly cares about a few, and groups
    differ in which. "none" (default): every voter weighs every dimension alike.

MODEL "ks": Jameson Quinn's hierarchical ("kitchen sink") model, from vse-sim (voter_models.py,
KSModel), with the parameters of its published results. Written anew from the algorithm, with one
change (4. below). This is the version that may evolve: changes come as new parameters whose default
keeps the earlier behaviour (so older runs stay comparable), or under a new model name.

MODEL "ks-quinn": the same, kept as close to vse-sim's code as this lab allows, and not to be changed:
it keeps vse-sim's one offset per person and bundle (4. below), and has no ks_shared. What still
differs from vse-sim is common to all models here: utilities exp(-distance / width) instead of
-distance, and the candidate parameters (cand_dispersion, cand_shift). Parameters: ks_alpha (faction alpha), ks_cut (where importance stops), ks_shared
(1: the same factions in every bundle, a test variant without cross-cutting cleavages); --dims,
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

MODEL "field": opinions as scale-free fields over a social space, built from what is known about how
opinions arise and spread, rather than from imposed clusters or decay laws (README, Electorates):
    - People live on a two-dimensional social space (a 64 x 64 torus: who meets whom), placed by a
      Levy flight, so they are clustered at all scales (towns, neighbourhoods, circles).
    - Each issue is a Gaussian random field over that space with power spectrum k^-2: correlations that
      decay with the logarithm of distance, the signature found in real vote statistics (Borghesi &
      Bouchaud 2010), i.e. structure at every scale without a characteristic one.
    - A person's opinion on an issue is the field where they live (a share FIELD_SHARED = 0.4 of the
      variance; Borghesi & Bouchaud find a third to a half shared) plus an own part, passed through
      tanh(field_kappa x .): a saturating response, as a spin in the Ising model. Small field_kappa gives
      smooth, nearly Gaussian opinions; large field_kappa pushes each issue towards two camps.
    - Many issues (dims) with a fixed total variance pi^2/6, as fixed_variance does for clusters.
    - field_salience > 0: what people care about is a field too (weights exp(field_salience x another
      field), mean square 1 per person), so neighbours share their priorities partly.
    Candidates are drawn like voters.

MODEL "noise" (default): voters are dense where a fractal noise is high (as Perlin noise with octaves), so they
form clumps within clumps at every scale, without a number of clusters, a Zipf law or salience:
    - The noise, new for every election: NOISE_OCTAVES octaves of NOISE_WAVES plane waves each, in
      random directions in the issue space, the frequency doubling and the amplitude falling by
      2^-NOISE_H from one octave to the next (random Fourier features: fractal noise in any number of
      dimensions, where Perlin's lattice would be impractical), scaled to variance 1.
    - People: drawn from the density Gaussian(0, 1 per dimension) x exp(noise_strength x noise).
      0 = a plain Gaussian blob; larger values = sharper clumps (a Condorcet winner rarer and costlier).
      Sampling (annealed resampling with local moves): NOISE_PARTICLES points start from the
      Gaussian; the strength is raised in NOISE_STEPS equal steps, each time the points are
      resampled by their gain in weight and then take NOISE_MOVES Metropolis steps (random-walk,
      size NOISE_STEP), so that clumps get their real width instead of repeated copies of a few
      points. The voters and candidates are distinct points drawn at random from the result.
    - Scaled so the dimensions' total variance is pi^2/6 (over all points), as fixed_variance does
      for clusters.
    Candidates are drawn like voters.
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
    if getattr(sc, "fixed_variance", 0):
        # one scale for all dimensions, so that their total variance is pi^2/6 (the limit of Zipf
        # dimensions) whatever their number: more dimensions spread the same variance over more issues
        rank = np.arange(1, sc.dims + 1)
        per_dim = (1.0 / rank ** 2) if sc.zipf else np.ones(sc.dims)
        f = np.sqrt((np.pi ** 2 / 6) / per_dim.sum())
        voters, cands = voters * f, cands * f
    vw = cw = None
    if getattr(sc, "salience", "none") == "cluster":
        # each cluster weighs the issues 1, 1/2, 1/3, ... in its own random order (drawn after all other
        # draws, so positions are the same with and without salience); scaled so that the mean squared
        # weight is 1, which leaves the expected distance in an isotropic space unchanged
        sal = np.empty((sc.n_clusters, sc.dims))
        for k in range(sc.n_clusters):
            sal[k, rng.permutation(sc.dims)] = 1.0 / np.arange(1, sc.dims + 1) ** getattr(sc, "salience_exponent", 1.0)
        sal = sal / np.sqrt((sal ** 2).mean(axis=1, keepdims=True))
        vw, cw = sal[vc], sal[cc]
    return voters, cands, vw, cw


# MODEL "ks": Quinn's hierarchical "kitchen sink" model (vse-sim, voter_models.py, KSModel)
KS_BUNDLE_DECAY = (1.0, 3.0)     # Beta parameters: each issue bundle weighs this fraction of the one before
KS_DIM_DECAY = (1.5, 3.0)        # Beta parameters: each dimension in a bundle weighs this fraction of the one before
KS_CARING = (3.0, 1.5)           # Beta parameters of a faction's caring (and of the spread of faction positions)


def _draw_ks(sc, rng):
    """Quinn's hierarchical model, written from the description of vse-sim's KSModel (published
    parameters; see the README, Electorates). Persons = voters, then candidates, drawn alike."""
    pos, W = _ks_structure(sc, rng)[:2]
    nv = sc.n_voters
    return pos[:nv], pos[nv:], W[:nv], W[nv:]


def _draw_ks_quinn(sc, rng):
    """As close to vse-sim's KSModel as this lab allows: as ks, but with one random offset per person
    and bundle, shared by all dimensions of the bundle (as vse-sim's code does), and without ks_shared."""
    pos, W = _ks_structure(sc, rng, quinn=True)[:2]
    nv = sc.n_voters
    return pos[:nv], pos[nv:], W[:nv], W[nv:]


def _ks_structure(sc, rng, quinn=False):
    """The ks draw: positions (n, D) and weights of all persons (voters, then candidates), and each
    bundle's faction of every person (list of arrays) and the number of dimensions of each bundle
    (for the tests and show_electorates.py). quinn=True: the ks-quinn
    variant (one offset per person and bundle; ks_shared ignored)."""
    n = sc.n_voters + sc.n_cands
    shared = None                          # ks_shared: the factions of the first bundle, for all bundles
    factions = []
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
        #    faction of a random earlier person). ks_shared = 1: the same factions in every bundle
        #    (no cross-cutting; positions and caring are still drawn per bundle)
        if shared is not None:
            fac, nf = shared
        else:
            fac = np.empty(n, dtype=int)
            nf = 0
            for i in range(n):
                r = (i + sc.ks_alpha) * rng.random()
                if r > i:
                    fac[i] = nf
                    nf += 1
                else:
                    fac[i] = fac[int(r)]
            if getattr(sc, "ks_shared", 0) and not quinn:
                shared = (fac, nf)
        factions.append(fac)
        # 3. Each faction: a position (spread sqrt(c) for a random c) and a caring level
        means = np.empty((nf, k))
        caring = np.empty(nf)
        for f in range(nf):
            c = rng.beta(*KS_CARING)
            means[f] = rng.normal(0.0, np.sqrt(c), k)
            caring[f] = rng.beta(*KS_CARING)
        # 4. Persons: around their faction's position, the tighter the more the faction cares
        #    (an own offset per dimension; vse-sim uses one offset for all dimensions of a bundle)
        if quinn:   # vse-sim: the offset is read by bundle number, so one per person for all of its dimensions
            noise = np.repeat(rng.normal(0.0, 1.0, (n, 1)), k, axis=1)
        else:
            noise = rng.normal(0.0, 1.0, (n, k))
        pos[:, start:start + k] = means[fac] + noise * np.sqrt(1 - caring[fac])[:, None]
        care[:, start:start + k] = caring[fac][:, None]
        start += k
    # 5. Weights: dimension importance x the person's caring, normalised per person, so a distance is
    #    sqrt(sum((x - p)^2 (w c)^2) / sum((w c)^2)) as in vse-sim
    W = dim_w[None, :] * care
    W = W / np.sqrt((W ** 2).sum(axis=1, keepdims=True))
    return pos, W, factions, bundles


# MODEL "noise": voters dense where a fractal noise is high (see the module docstring)
NOISE_OCTAVES = 5        # octaves of the noise
NOISE_WAVES = 8          # plane waves per octave
NOISE_K0 = 0.7           # frequency of the coarsest octave (wavelength about 9 voter SDs: a broad tilt)
NOISE_H = 0.5            # amplitude falls by 2^-NOISE_H per octave (frequency doubles)
NOISE_PARTICLES = 2000   # sampling: points that are moved toward the density
NOISE_STEPS = 10         # sampling: the strength is raised in this many steps
NOISE_MOVES = 5          # sampling: Metropolis steps per strength step
NOISE_STEP = 0.15        # sampling: Metropolis step size per dimension (the finest waves have length ~0.6)


def _draw_noise(sc, rng):
    """Voters and candidates where a fresh fractal noise is high (see the module docstring)."""
    n, D = sc.n_voters + sc.n_cands, sc.dims
    k, phase, amp = [], [], []
    for o in range(NOISE_OCTAVES):
        d = rng.normal(size=(NOISE_WAVES, D))
        k.append(d / np.linalg.norm(d, axis=1, keepdims=True) * NOISE_K0 * 2 ** o)
        phase.append(rng.random(NOISE_WAVES) * 2 * np.pi)
        amp.append(np.full(NOISE_WAVES, 2.0 ** (-NOISE_H * o)))
    k, phase, amp = np.vstack(k), np.concatenate(phase), np.concatenate(amp)
    amp = amp / np.sqrt((amp ** 2).sum() / 2)                               # noise variance 1

    def noise(Y):
        return (amp * np.cos(Y @ k.T + phase)).sum(axis=1)

    M = max(NOISE_PARTICLES, 2 * n)
    Y = rng.normal(size=(M, D))
    f = noise(Y)
    for step in range(1, NOISE_STEPS + 1):
        beta = sc.noise_strength * step / NOISE_STEPS
        gain = np.exp((sc.noise_strength / NOISE_STEPS) * (f - f.max()))    # resample by the gain in weight
        keep = rng.choice(M, M, p=gain / gain.sum())
        Y, f = Y[keep], f[keep]
        for _ in range(NOISE_MOVES):                                         # local moves: Metropolis
            Z = Y + NOISE_STEP * rng.normal(size=Y.shape)
            fz = noise(Z)
            log_ratio = beta * (fz - f) - 0.5 * ((Z ** 2).sum(axis=1) - (Y ** 2).sum(axis=1))
            ok = np.log(rng.random(M)) < log_ratio
            Y[ok], f[ok] = Z[ok], fz[ok]
    Y = Y - Y.mean(axis=0)
    Y = Y * np.sqrt((np.pi ** 2 / 6) / (Y ** 2).sum(axis=1).mean())          # total variance pi^2/6
    X = Y[rng.choice(M, n, replace=False)]
    return X[:sc.n_voters], X[sc.n_voters:], None, None


# MODEL "field": opinions as scale-free fields over a social space (see the module docstring)
FIELD_GRID = 64          # the social space: a 64 x 64 torus
FIELD_EXPONENT = 2.0     # power spectrum k^-2 in two dimensions: logarithmically decaying correlations
FIELD_SHARED = 0.4       # share of each opinion's variance that is shared with one's surroundings
FIELD_LEVY = 1.0         # where people live: a Levy flight with P(step > x) ~ 1/x (clustered at all scales)


def _fields(rng, n):
    """n independent Gaussian random fields on the FIELD_GRID torus with power spectrum k^-FIELD_EXPONENT,
    each with mean 0 and variance 1."""
    kx, ky = np.fft.fftfreq(FIELD_GRID)[:, None], np.fft.fftfreq(FIELD_GRID)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = np.inf
    z = rng.normal(size=(n, FIELD_GRID, FIELD_GRID)) + 1j * rng.normal(size=(n, FIELD_GRID, FIELD_GRID))
    f = np.fft.ifft2(z * k ** (-FIELD_EXPONENT / 2), axes=(1, 2)).real
    f -= f.mean(axis=(1, 2), keepdims=True)
    return f / f.std(axis=(1, 2), keepdims=True)


def _levy_cells(rng, n):
    """Grid cells of n persons placed by a Levy flight on the torus (step lengths with P(l > x) = x^-FIELD_LEVY,
    at least one cell, at most the grid; random directions), in random order."""
    length = np.minimum(rng.random(n) ** (-1.0 / FIELD_LEVY), FIELD_GRID)
    angle = rng.random(n) * 2 * np.pi
    pos = np.cumsum(np.c_[length * np.cos(angle), length * np.sin(angle)], axis=0)
    return (np.floor(pos[rng.permutation(n)]) % FIELD_GRID).astype(int)


def _draw_field(sc, rng):
    """Opinions as scale-free fields over a social space (see the module docstring)."""
    n, D = sc.n_voters + sc.n_cands, sc.dims
    cell = _levy_cells(rng, n)
    F = _fields(rng, D)[:, cell[:, 0], cell[:, 1]].T                       # (n, D): the surroundings' view
    Z = np.sqrt(FIELD_SHARED) * F + np.sqrt(1 - FIELD_SHARED) * rng.normal(size=(n, D))
    X = np.tanh(sc.field_kappa * Z)                                         # saturating, as a spin
    X = X / np.sqrt((X ** 2).mean()) * np.sqrt((np.pi ** 2 / 6) / D)        # fixed total variance pi^2/6
    W = None
    if sc.field_salience > 0:                                               # what people care about: a field too
        H = _fields(rng, D)[:, cell[:, 0], cell[:, 1]].T
        W = np.exp(sc.field_salience * H)
        W = W / np.sqrt((W ** 2).mean(axis=1, keepdims=True))
    nv = sc.n_voters
    return X[:nv], X[nv:], (None if W is None else W[:nv]), (None if W is None else W[nv:])


MODELS = {m.name: m for m in [
    Model("clusters", _draw_clusters,
          lambda g: f"{g('dims')} dimensions{' (Zipf)' if int(g('zipf')) else ''}"
                    f"{', fixed total variance' if int(g('fixed_variance')) else ''}, {g('clusters')} clusters, "
                    f"spread {g('spread')}" + (f", salience per cluster (exponent {g('salience_exponent')})"
                                               if g("salience") == "cluster" else "")),
    Model("ks", _draw_ks,
          lambda g: f"Quinn's hierarchical issue bundles and factions, faction alpha {g('ks_alpha')}, "
                    f"cut {g('ks_cut')}" + (", factions shared across bundles" if int(g("ks_shared")) else "")),
    Model("field", _draw_field,
          lambda g: f"{g('dims')} issues as scale-free fields over a social space, saturation {g('field_kappa')}"
                    + (f", salience field {g('field_salience')}" if float(g("field_salience")) > 0 else "")),
    Model("ks-quinn", _draw_ks_quinn,
          lambda g: f"Quinn's hierarchical model as in vse-sim's code (one offset per bundle), faction alpha "
                    f"{g('ks_alpha')}, cut {g('ks_cut')}"),
    Model("noise", _draw_noise,
          lambda g: f"{g('dims')} dimensions, voters dense where a fractal noise is high, strength {g('noise_strength')}"),
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
    """The electorate of one election under the scenario's model (sc.model, default "noise")."""
    voters, cands, vw, cw = MODELS[getattr(sc, "model", "clusters")].draw(sc, rng)
    if sc.cand_dispersion != 1.0 or sc.cand_shift != 0.0:
        centre = voters.mean(axis=0)
        cands = centre + sc.cand_dispersion * (cands - centre)
        cands[:, 0] += sc.cand_shift * voters[:, 0].std()
    u = utility(distances(voters, cands, vw), sc.width)
    return Electorate(u, voters, cands, vw, cw)
