"""Look at the electorates a model draws, to see by eye whether something is off.

    python show_electorates.py --model clusters ks ks-quinn --out look
    python show_electorates.py --model ks --width 0.5 --ks-alpha 5 --out look-ks

Takes the electorate parameters of run.py (one value each, except --model, which may list several
models to compare side by side) and writes into output/<out>/ (or the folder given):

    examples.png   a few elections per model (rows): voters as dots coloured by their favourite,
                   candidates as lettered circles, on the voters' first two principal components
                   (the plane in which their positions vary most). The Condorcet winner has a thick
                   ring; the candidate with the highest total utility is underlined in the title.
    summary.png    distributions over many elections, one colour per model:
                   - variance explained by the first k principal components (median over elections),
                     with the 70% line (a fit often taken as good, e.g. in Alós-Ferrer & Granić 2015)
                   - how many components reach 70% of the variance, per election
                   - favourite / least liked utility, per voter (how steep preferences are)
                   - all utilities u(voter, candidate)
                   - VSE lost by electing the Condorcet winner, per election
                   - distances from voters to candidates
    summary.txt    the same as numbers, plus how often a Condorcet winner exists and is the
                   utility winner, and the number of dimensions the model drew

Principal components are of the voters' positions. In the ks models every person weighs the
dimensions differently; the positions are scaled by the voters' average weights first, so the plot
shows the space roughly as an average voter sees it. Only the electorates are drawn: no methods, no
strategy, so this is fast.
"""
import argparse
import os

import numpy as np

import plots   # noqa: F401  (the lab's plot style)
import matplotlib.pyplot as plt

from models import MODELS, draw_electorate
from run import out_path
from scenario import Scenario

CATS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK2, MUTED = plots.INK2, plots.MUTED


def scaled_positions(e):
    """Voter and candidate positions in a common space: scaled by the voters' average dimension
    weights where the model has weights (ks), else as drawn."""
    if e.voter_weights is None:
        return e.voters, e.cands
    w = e.voter_weights.mean(axis=0)
    return e.voters * w, e.cands * w


def components(X):
    """Principal components of the rows of X: (explained variance shares, projection on the first two)."""
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    share = var / var.sum() if var.sum() > 0 else np.ones_like(var)
    return share, vt[:2]


def stats(e):
    """Per-election properties (see the module docstring)."""
    u = e.u
    n, C = u.shape
    V, K = scaled_positions(e)
    share, _ = components(V)
    gt = (u[:, :, None] > u[:, None, :]).sum(axis=0)
    cw = np.flatnonzero((gt > gt.T).sum(axis=1) == C - 1)
    U = u.sum(axis=0)
    rb = np.bincount(u.argmax(axis=1), minlength=C) / n
    d = np.linalg.norm(V[:, None, :] - K[None, :, :], axis=2)
    return {"dims": V.shape[1], "share": share, "k70": int(np.searchsorted(np.cumsum(share), 0.7 - 1e-12) + 1),
            "fav_ratio": u.max(axis=1) / u.min(axis=1), "u": u.ravel(), "dist": d.ravel(),
            "has_cw": len(cw) == 1, "cw_is_util": len(cw) == 1 and cw[0] == U.argmax(),
            "cw_loss": (U.max() - U[cw[0]], U.max() - rb @ U) if len(cw) == 1 else None}


def draw_examples(models, scs, examples, seed, path):
    fig, axs = plt.subplots(len(models), examples, figsize=(3.1 * examples, 3.1 * len(models)), squeeze=False)
    for r, (name, sc) in enumerate(zip(models, scs)):
        for c in range(examples):
            ax = axs[r, c]
            e = draw_electorate(sc, np.random.default_rng([seed, c]))
            V, K = scaled_positions(e)
            share, comp = components(V)
            mean = V.mean(axis=0)
            pv, pk = (V - mean) @ comp.T, (K - mean) @ comp.T
            if pv.shape[1] < 2:                                    # one dimension: spread on y = 0
                pv, pk = np.c_[pv, np.zeros(len(pv))], np.c_[pk, np.zeros(len(pk))]
            u = e.u
            fav = u.argmax(axis=1)
            colours = [CATS[f % len(CATS)] for f in fav]
            ax.scatter(pv[:, 0], pv[:, 1], s=9, c=colours, alpha=0.75, linewidths=0, zorder=2)
            C = u.shape[1]
            gt = (u[:, :, None] > u[:, None, :]).sum(axis=0)
            cw = np.flatnonzero((gt > gt.T).sum(axis=1) == C - 1)
            best = int(u.sum(axis=0).argmax())
            for k in range(C):
                ax.scatter(pk[k, 0], pk[k, 1], s=150, color=CATS[k % len(CATS)], edgecolors="#0b0b0b",
                           linewidths=2.2 if k in cw else 0.6, zorder=3)
                ax.text(pk[k, 0], pk[k, 1], "ABCDEFGHIJ"[k], ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold", zorder=4)
            cw_txt = f"CW {'ABCDEFGHIJ'[cw[0]]}" if len(cw) == 1 else "no CW"
            ax.set_title(f"{cw_txt}, utility winner {'ABCDEFGHIJ'[best]}\n{V.shape[1]} dims, "
                         f"PC1+2 {share[:2].sum():.0%}", fontsize=8, fontweight="normal", color=INK2)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            if c == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold", color="#0b0b0b")
    fig.suptitle("Example electorates: voters coloured by favourite, candidates lettered "
                 "(thick ring: Condorcet winner); first two principal components", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_summary(models, S, path):
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.6))
    ax = axs[0, 0]
    for i, name in enumerate(models):
        kmax = max(len(s["share"]) for s in S[name])
        cum = np.array([np.r_[np.cumsum(s["share"]), np.ones(kmax - len(s["share"]))] for s in S[name]])
        med = np.median(cum, axis=0)[:10]
        ax.plot(np.arange(1, len(med) + 1), med * 100, marker="o", ms=4, lw=2, color=MODEL_COLOURS[i], label=name)
    ax.axhline(70, color=MUTED, lw=1, ls="--")
    ax.set(xlabel="principal components", ylabel="variance explained (%, median)", ylim=(0, 102),
           title="Variance explained by the first k components")
    ax.legend(frameon=False)

    def hist(ax, key, title, xlabel, per_election=False, log=False, rng=None, bins=30):
        data = {n: (np.array([s[key] for s in S[n]]) if per_election else np.concatenate([s[key] for s in S[n]]))
                for n in models}
        allv = np.concatenate(list(data.values()))
        lo, hi = rng or (np.percentile(allv, 0.5), np.percentile(allv, 99.5))
        edges = np.geomspace(max(lo, 1e-3), hi, bins + 1) if log else np.linspace(lo, hi, bins + 1)
        for i, n in enumerate(models):          # values beyond the range are left out, not piled at the edge
            ax.hist(data[n], bins=edges, density=True, histtype="step", lw=2, color=MODEL_COLOURS[i], label=n)
        if log:
            ax.set_xscale("log")
            ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20, 50, 100) if edges[0] <= t <= edges[-1]]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set(title=title, xlabel=xlabel, ylabel="density")

    k70 = {n: np.array([s["k70"] for s in S[n]]) for n in models}
    ax = axs[0, 1]
    top = max(int(v.max()) for v in k70.values())
    width = 0.8 / len(models)
    for i, n in enumerate(models):
        counts = np.bincount(k70[n], minlength=top + 1)[1:] / len(k70[n]) * 100
        ax.bar(np.arange(1, top + 1) + (i - (len(models) - 1) / 2) * width, counts, width=width * 0.92,
               color=MODEL_COLOURS[i], label=n)
    ax.set(title="Components needed for 70% of the variance", xlabel="components", ylabel="elections (%)",
           xticks=range(1, top + 1))
    hist(axs[0, 2], "fav_ratio", "Favourite ÷ least liked, per voter", "ratio (log scale)", log=True)
    hist(axs[1, 0], "u", "Utilities u(voter, candidate)", "utility", rng=(0, 1))
    # VSE lost by electing the Condorcet winner, in the elections where it is not the utility winner
    ax = axs[1, 1]
    edges = np.linspace(0, 100, 26)
    for i, n in enumerate(models):
        with_cw = [s for s in S[n] if s["cw_loss"] and s["cw_loss"][1] > 0]
        loss = np.array([s["cw_loss"][0] / s["cw_loss"][1] * 100 for s in with_cw])
        part = loss[loss > 1e-9]
        share = len(part) / max(len(loss), 1)
        ax.hist(np.minimum(part, 100), bins=edges, weights=np.full(len(part), 100 / max(len(loss), 1)),
                histtype="step", lw=2, color=MODEL_COLOURS[i], label=f"{n}: {share:.0%} of elections")
    ax.set(title="VSE lost electing the Condorcet winner,\nwhere it is not the utility winner",
           xlabel="% of (best − random ballot), per election (100+ in the last bin)", ylabel="elections with a CW (%)")
    ax.legend(frameon=False, fontsize=8)
    hist(axs[1, 2], "dist", "Distances voter – candidate", "distance (common space)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_text(models, scs, S, elections, seed):
    lines = [f"{elections} electorates per model, seed {seed}", ""]
    for name, sc in zip(models, scs):
        s = S[name]
        dims = np.array([x["dims"] for x in s])
        k70 = np.array([x["k70"] for x in s])
        cum2 = np.median([np.cumsum(x["share"])[min(1, len(x["share"]) - 1)] for x in s])
        cw = np.mean([x["has_cw"] for x in s])
        with_cw = [x for x in s if x["has_cw"]]
        cwu = np.mean([x["cw_is_util"] for x in with_cw]) if with_cw else float("nan")
        num = sum(x["cw_loss"][0] for x in with_cw)
        den = sum(x["cw_loss"][1] for x in with_cw)
        fr = np.median(np.concatenate([x["fav_ratio"] for x in s]))
        lines += [f"{name}:",
                  f"  dimensions drawn          median {np.median(dims):.0f} (range {dims.min()}–{dims.max()})",
                  f"  components for 70%        median {np.median(k70):.0f} (range {k70.min()}–{k70.max()}); "
                  f"first two explain {cum2:.0%} (median)",
                  f"  Condorcet winner          {cw:.1%} of elections; also the utility winner in {cwu:.1%} of those",
                  f"  VSE lost electing it      {num / den:.1%} (ratio of sums, as VSE)" if den > 0 else "",
                  f"  favourite ÷ least liked   {fr:.2f} (median voter)", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Plots of the electorates a model draws (no methods).")
    ap.add_argument("--model", nargs="+", default=["clusters"], choices=list(MODELS))
    ap.add_argument("--elections", type=int, default=300, help="electorates per model for the distributions")
    ap.add_argument("--examples", type=int, default=4, help="example electorates per model")
    ap.add_argument("--voters", type=int, default=99)
    ap.add_argument("--cands", type=int, default=6)
    ap.add_argument("--dims", type=int, default=10)
    ap.add_argument("--zipf", type=int, default=1, choices=[0, 1])
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--ks-alpha", type=float, default=1.5)
    ap.add_argument("--ks-cut", type=float, default=0.2)
    ap.add_argument("--ks-shared", type=int, default=0, choices=[0, 1])
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--cand-dispersion", type=float, default=1.0)
    ap.add_argument("--cand-shift", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="electorates", help="folder (in output/ unless it has a directory)")
    a = ap.parse_args()

    out = out_path(a.out)
    os.makedirs(out, exist_ok=True)
    scs = [Scenario(a.voters, a.cands, a.dims, a.clusters, a.spread, a.width, a.zipf, a.cand_dispersion,
                    a.cand_shift, m, a.ks_alpha, a.ks_cut, a.ks_shared) for m in a.model]
    # the same random streams as run.py: election e of seed s is the electorate run.py uses
    S = {m: [stats(draw_electorate(sc, np.random.default_rng([a.seed, e]))) for e in range(a.elections)]
         for m, sc in zip(a.model, scs)}
    draw_examples(a.model, scs, a.examples, a.seed, os.path.join(out, "examples.png"))
    draw_summary(a.model, S, os.path.join(out, "summary.png"))
    text = summary_text(a.model, scs, S, a.elections, a.seed)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {out}/: examples.png, summary.png, summary.txt")


if __name__ == "__main__":
    main()
