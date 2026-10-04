"""Static plots for analyze.py (matplotlib, PNG).

Colour encodes one thing only: front membership probability, on a single-hue blue ramp
(light = rarely on the front, dark = always). Method identity is carried by direct labels,
never by colour, since there are far more methods than distinguishable hues.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8b8a85"
GRID = "#e6e5e1"
# Sequential blue, steps 150 ... 700 of the reference ramp (lightest kept visible on the surface)
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("front", RAMP)
HEAT = LinearSegmentedColormap.from_list("heat", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _repel(ax, texts, anchors, iterations=300):
    """Nudge labels apart vertically until their boxes no longer overlap (simple, deterministic).
    Labels that end up away from their point get a thin leader line back to it."""
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for _ in range(iterations):
        moved = False
        boxes = [t.get_window_extent().expanded(1.05, 1.15) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if boxes[i].overlaps(boxes[j]):
                    # push the lower one down and the upper one up, by a few pixels
                    a, b = (i, j) if boxes[i].y0 <= boxes[j].y0 else (j, i)
                    for k, dy in ((a, -2.0), (b, 2.0)):
                        x, y = texts[k].get_position()
                        px, py = ax.transData.transform((x, y))
                        texts[k].set_position(inv.transform((px, py + dy)))
                    moved = True
        if not moved:
            break
        fig.canvas.draw()
    for t, (x, y) in zip(texts, anchors):
        tx, ty = t.get_position()
        if abs(ax.transData.transform((tx, ty))[1] - ax.transData.transform((x, y))[1]) > 4:
            ax.plot([x, tx], [y, ty], color="#c4c3be", lw=0.8, zorder=1)


def _colorbar(fig, ax):
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("front probability", color=INK2)
    cb.outline.set_visible(False)


def _visible(r, hide):
    return [j for j, m in enumerate(r["methods"]) if m not in hide]


def arrows(r, path, hide):
    """Dumbbell chart: per method, honest (open circle) -> learned ballots (filled), for PE and CE.
    Rows sorted by front probability; filled marks coloured by it."""
    idx = sorted(_visible(r, hide), key=lambda j: r["front_prob"][j])
    names = [r["methods"][j] for j in idx]
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.9 + 0.32 * len(idx)), sharey=True)
    for ax, (a0, a1, title) in zip(axs, [("PE", "PE*", "PE: proportional efficiency (%)"),
                                         ("CE", "CE*", "CE: Condorcet efficiency (%)")]):
        for y, j in enumerate(idx):
            x0, x1 = r[a0][j] * 100, r[a1][j] * 100
            if abs(x1 - x0) > 0.15:
                ax.annotate("", xy=(x1, y), xytext=(x0, y),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=4, shrinkB=5,
                                            mutation_scale=8))
            ax.scatter([x0], [y], s=34, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.2, zorder=3)
            ax.scatter([x1], [y], s=58, color=CMAP(r["front_prob"][j]), edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.set_xlabel(title)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(idx)), names)
    axs[0].set_title(f"What strategy does: honest (open) → learned ballots (filled)\n{r['label']}", loc="left")
    _colorbar(fig, axs[1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dissonance(r, path, hide):
    """PE* and CE* against dissonance, side by side. No marginal intervals: they show how much
    single values vary between elections, not how surely methods differ (front probability does)."""
    idx = _visible(r, hide)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True)
    for ax, axis, name in zip(axs, ["PE*", "CE*"], ["PE* (learned ballots, %)", "CE* (learned ballots, %)"]):
        texts, anchors = [], []
        for j in idx:
            x, y = r["dissonance"][j] * 100, r[axis][j] * 100
            anchors.append((x, y))
            ax.scatter([x], [y], s=60, color=CMAP(r["front_prob"][j]), edgecolors=SURFACE, linewidths=1.5, zorder=3)
            texts.append(ax.text(x + 0.12, y, r["methods"][j], fontsize=7.5, color=INK2, va="center", zorder=4))
        ax.set_xlabel("cognitive dissonance of learned ballots (%)")
        ax.set_ylabel(name)
        ax.set_xlim(left=0)
        _repel(ax, texts, anchors)
    axs[0].set_title(f"Quality under strategy against dissonance (lower right is worse)\n{r['label']}", loc="left")
    _colorbar(fig, axs[1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _row_order(results):
    methods = results[0]["methods"]
    mean_fp = np.mean([[r["front_prob"][r["methods"].index(m)] for m in methods] for r in results], axis=0)
    return [methods[j] for j in np.argsort(-mean_fp, kind="stable")]


def _heat(ax, data, rows, cols, fmt, vmin, vmax, cmap):
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(len(rows)):
        for k in range(len(cols)):
            v = data[i, k]
            dark = (v - vmin) / (vmax - vmin + 1e-12) > 0.55
            ax.text(k, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if dark else INK)


def front_heatmap(results, path):
    rows = _row_order(results)
    cols = [r["label"] for r in results]
    data = np.array([[r["front_prob"][r["methods"].index(m)] for r in results] for m in rows])
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(cols), 0.9 + 0.3 * len(rows)))
    _heat(ax, data, rows, cols, "{:.2f}", 0, 1, HEAT)
    ax.set_title("Front probability by condition", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def axes_heatmap(results, axes, path):
    rows = _row_order(results)
    cols = [r["label"] for r in results]
    fig, axs = plt.subplots(1, len(axes), figsize=(2.2 + 1.0 * len(cols) * len(axes), 0.9 + 0.3 * len(rows)),
                            sharey=True)
    for ax, axis in zip(axs, axes):
        data = np.array([[r[axis][r["methods"].index(m)] * 100 for r in results] for m in rows])
        # colour range from the bulk of methods, so far-off baselines do not wash out the rest
        lo, hi = np.percentile(data, 15), data.max()
        _heat(ax, data, rows, cols, "{:.1f}", lo, hi, HEAT)
        ax.set_title(f"{axis} (%)", loc="left")
    fig.suptitle("Axis values by condition", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
