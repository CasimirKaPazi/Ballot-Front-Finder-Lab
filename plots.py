"""Static plots for analyze.py (matplotlib, PNG).

Method identity is carried by direct labels, never by colour, since there are far more methods
than distinguishable hues. In the arrow plot, the learned-ballot points are coloured by sincerity
on a single-hue red ramp (light = honest, dark = much reversed preference). Heatmaps use blue.
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
POINT = "#2a78d6"   # learned-ballot points
# Sequential red, light to dark; used reversed for sincerity (dark = less honest)
REDS = LinearSegmentedColormap.from_list("insincerity", ["#f7c4bf", "#ef8f86", "#e34948", "#b8302f", "#7f1d1d"])
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


def _visible(r, hide):
    return [j for j, m in enumerate(r["methods"]) if m not in hide]


def arrows(r, path, hide):
    """Honest (open circle) -> learned ballots (filled) in the PE / WM plane, all methods together.
    Filled points are coloured by the sincerity of the learned ballots (darker = less honest)."""
    idx = _visible(r, hide)
    # cap the colour scale at the 10th percentile, so one extreme method does not wash out the rest
    vals = np.array([r["sincerity"][j] * 100 for j in idx])
    smin = min(99.5, float(np.percentile(vals, 10)))
    norm = Normalize(smin, 100, clip=True)
    cmap = REDS.reversed()
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    texts, anchors = [], []
    for j in idx:
        x0, y0 = r["PE"][j] * 100, r["WM"][j] * 100
        x1, y1 = r["PE*"][j] * 100, r["WM*"][j] * 100
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, shrinkA=4, shrinkB=5, mutation_scale=9))
        ax.scatter([x0], [y0], s=34, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.2, zorder=3)
        ax.scatter([x1], [y1], s=64, color=cmap(norm(r["sincerity"][j] * 100)), edgecolors=MUTED,
                   linewidths=0.6, zorder=4)
        anchors.append((x1, y1))
        texts.append(ax.text(x1 + 0.12, y1, r["methods"][j], fontsize=7.5, color=INK2, va="center", zorder=5))
    ax.set_xlabel("PE, proportional efficiency (%)")
    ax.set_ylabel("WM, winner's worst margin (% of voters)")
    ax.set_title(f"What strategy does: honest (open) → learned ballots (filled, labelled)\n{r['label']}", loc="left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.02,
                      extend="min" if vals.min() < smin else "neither")
    cb.set_label("sincerity of learned ballots (%)", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    _repel(ax, texts, anchors)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shift(r, path, hide):
    """The same shifts as a dumbbell chart: one row per method, PE and WM panels. Rows sorted by PE."""
    idx = sorted(_visible(r, hide), key=lambda j: r["PE"][j])
    names = [r["methods"][j] for j in idx]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 0.9 + 0.32 * len(idx)), sharey=True)
    for ax, (a0, a1, title) in zip(axs, [("PE", "PE*", "PE: proportional efficiency (%)"),
                                         ("WM", "WM*", "WM: winner's worst margin (% of voters)")]):
        for y, j in enumerate(idx):
            x0, x1 = r[a0][j] * 100, r[a1][j] * 100
            if abs(x1 - x0) > 0.15:
                ax.annotate("", xy=(x1, y), xytext=(x0, y),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=4, shrinkB=5,
                                            mutation_scale=8))
            ax.scatter([x0], [y], s=34, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.2, zorder=3)
            ax.scatter([x1], [y], s=58, color=POINT, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.set_xlabel(title)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(idx)), names)
    axs[0].set_title(f"What strategy does: honest (open) → learned ballots (filled)\n{r['label']}", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sincerity(r, path, hide):
    """PE* and WM* against sincerity, side by side. No marginal intervals: they show how much
    single values vary between elections, not how surely methods differ (paired differences do)."""
    idx = _visible(r, hide)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True)
    for ax, axis, name in zip(axs, ["PE*", "WM*"], ["PE* (learned ballots, %)", "WM* (learned ballots, % of voters)"]):
        texts, anchors = [], []
        for j in idx:
            x, y = r["sincerity"][j] * 100, r[axis][j] * 100
            anchors.append((x, y))
            ax.scatter([x], [y], s=60, color=POINT, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            texts.append(ax.text(x + 0.12, y, r["methods"][j], fontsize=7.5, color=INK2, va="center", zorder=4))
        ax.set_xlabel("sincerity of learned ballots (%)")
        ax.set_ylabel(name)
        _repel(ax, texts, anchors)
    axs[0].set_title(f"Quality under strategy against sincerity (upper right is best)\n{r['label']}", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _row_order(results):
    """Methods sorted by honest PE, averaged over conditions."""
    methods = results[0]["methods"]
    mean_pe = np.mean([[r["PE"][r["methods"].index(m)] for m in methods] for r in results], axis=0)
    return [methods[j] for j in np.argsort(-mean_pe, kind="stable")]


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
