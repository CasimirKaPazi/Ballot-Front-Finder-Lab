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
    # labels start just right of their point, by a fixed share of the final axis width
    # (a fixed offset in data units pushes them out of view when the axis is zoomed in)
    x_lo, x_hi = ax.get_xlim()
    dx = 0.008 * (x_hi - x_lo)
    for t, (x, y) in zip(texts, anchors):
        t.set_position((x + dx, y))
    fig.canvas.draw()
    inv = ax.transData.inverted()
    y_lo, y_hi = ax.get_ylim()
    pad = 0.015 * (y_hi - y_lo)
    y_lo, y_hi = y_lo + pad, y_hi - pad
    renderer = fig.canvas.get_renderer()       # text extents follow set_position without a redraw
    for _ in range(iterations):
        moved = False
        boxes = [t.get_window_extent(renderer).expanded(1.05, 1.15) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if boxes[i].overlaps(boxes[j]):
                    # push the lower one down and the upper one up, by a few pixels
                    a, b = (i, j) if boxes[i].y0 <= boxes[j].y0 else (j, i)
                    for k, dy in ((a, -2.0), (b, 2.0)):
                        x, y = texts[k].get_position()
                        px, py = ax.transData.transform((x, y))
                        nx, ny = inv.transform((px, py + dy))
                        texts[k].set_position((nx, min(max(ny, y_lo), y_hi)))   # stay inside the axes
                    moved = True
        if not moved:
            break
    for t, (x, y) in zip(texts, anchors):
        tx, ty = t.get_position()
        if abs(ax.transData.transform((tx, ty))[1] - ax.transData.transform((x, y))[1]) > 4:
            ax.plot([x, tx], [y, ty], color="#c4c3be", lw=0.8, zorder=1)


VSE_LABEL = "VSE (%; 0 = random ballot, 100 = best candidate)"
WME_LABEL = "WME (%; 0 = random ballot, 100 = best worst margin)"
UNREACHABLE = "#ecebe7"
BOUNDARY = "#a8a7a2"       # line along the edge of every unreachable area


def _unreachable(ax, which, extend=True, lines=True):
    """Grey out values above 100 on the x and/or y axis (efficiencies cannot exceed 100). With
    extend, the axis is widened so the boundary is in view; without, only a visible part is shaded."""
    for w in which:
        lo, hi = ax.get_xlim() if w == "x" else ax.get_ylim()
        if not extend and hi <= 100:
            continue
        hi = max(hi, 100 + 0.04 * (hi - lo))
        if w == "x":
            ax.set_xlim(lo, hi)
            ax.axvspan(100, hi, color=UNREACHABLE, zorder=0, lw=0)
            if lines:
                ax.axvline(100, color=BOUNDARY, lw=1.0, zorder=1)
        else:
            ax.set_ylim(lo, hi)
            ax.axhspan(100, hi, color=UNREACHABLE, zorder=0, lw=0)
            if lines:
                ax.axhline(100, color=BOUNDARY, lw=1.0, zorder=1)


def _frontier(ax, F):
    """Draw the reachable frontier (points (VSE, WME) as fractions, best WME first) and grey out the
    region beyond it, which no rule can reach in these elections."""
    if F is None or len(F) == 0:
        return
    x, y = F[:, 0] * 100, F[:, 1] * 100
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    hx, hy = max(x1, x.max() + 0.5), max(y1, y.max() + 0.5)
    # region beyond: above the curve and to its right
    px = np.concatenate([[x[0], x[0]], x, [hx, hx]])
    py = np.concatenate([[hy, y[0]], y, [y[-1], hy]])
    ax.fill(px, py, color=UNREACHABLE, zorder=0, lw=0)
    # one boundary line: WME = 100 up to the frontier, the frontier, then VSE = 100 below it
    ax.plot(np.concatenate([[min(x0, x[0]), x[0]], x, [100, 100]]),
            np.concatenate([[100, 100], y, [y[-1], min(y0, y[-1])]]), color=BOUNDARY, lw=1.0, zorder=1)
    ax.set_xlim(x0, hx)
    ax.set_ylim(y0, hy)


def _visible(r, hide):
    return [j for j, m in enumerate(r["methods"]) if m not in hide]


def arrows(r, path, hide):
    """Honest (open circle) -> learned ballots (filled) in the VSE / WME plane, all methods together.
    The grey area beyond the curve is out of reach for any rule in these elections (see analyze.frontier).
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
        x0, y0 = r["VSE"][j] * 100, r["WME"][j] * 100
        x1, y1 = r["VSE*"][j] * 100, r["WME*"][j] * 100
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, shrinkA=4, shrinkB=5, mutation_scale=9))
        ax.scatter([x0], [y0], s=34, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.2, zorder=3)
        ax.scatter([x1], [y1], s=64, color=cmap(norm(r["sincerity"][j] * 100)), edgecolors=MUTED,
                   linewidths=0.6, zorder=4)
        anchors.append((x1, y1))
        texts.append(ax.text(x1 + 0.12, y1, r["methods"][j], fontsize=7.5, color=INK2, va="center", zorder=5))
    ax.set_xlabel(VSE_LABEL)
    ax.set_ylabel(WME_LABEL)
    _unreachable(ax, "xy", lines=r.get("frontier") is None)
    _frontier(ax, r.get("frontier"))
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
    """The same shifts as a dumbbell chart: one row per method, VSE and WME panels. Rows sorted by VSE."""
    idx = sorted(_visible(r, hide), key=lambda j: r["VSE"][j])
    names = [r["methods"][j] for j in idx]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 0.9 + 0.32 * len(idx)), sharey=True)
    for ax, (a0, a1, title) in zip(axs, [("VSE", "VSE*", VSE_LABEL), ("WME", "WME*", WME_LABEL)]):
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
        _unreachable(ax, "x")
    axs[0].set_yticks(range(len(idx)), names)
    axs[0].set_title(f"What strategy does: honest (open) → learned ballots (filled)\n{r['label']}", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sincerity(r, path, hide):
    """VSE* and WME* against sincerity, side by side. No marginal intervals: they show how much
    single values vary between elections, not how surely methods differ (paired differences do)."""
    idx = _visible(r, hide)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True)
    for ax, axis, name in zip(axs, ["VSE*", "WME*"], ["VSE* (learned ballots, %)", "WME* (learned ballots, %)"]):
        texts, anchors = [], []
        for j in idx:
            x, y = r["sincerity"][j] * 100, r[axis][j] * 100
            anchors.append((x, y))
            ax.scatter([x], [y], s=60, color=POINT, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            texts.append(ax.text(x + 0.12, y, r["methods"][j], fontsize=7.5, color=INK2, va="center", zorder=4))
        ax.set_xlabel("sincerity of learned ballots (%)")
        ax.set_ylabel(name)
        _unreachable(ax, "y", extend=False)
        _repel(ax, texts, anchors)
    axs[0].set_title(f"Quality under strategy against sincerity (upper right is best)\n{r['label']}", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


CHAIN = LinearSegmentedColormap.from_list("chain", ["#bcd6f5", "#6aa3e6", "#2a78d6", "#1c5cab", "#0d366b"])


def chain(results, param, values, path, hide, learning_only, groups, same_electorates=None):
    """One parameter swept: for each method, a chain of arrows through the conditions in the order
    of the parameter's values, in the VSE/WME plane with learned ballots. Point shade follows the
    value (light = first, dark = last). If the parameter only affects learning (the electorates are
    the same), every chain starts at the method's honest point (open circle); otherwise the honest
    points form their own chain (open circles, dashed, labelled in grey italics). One panel per method group (groups: name ->
    methods; methods in no group go to "other"), all panels on the same scale. If the electorates
    are the same in every condition (same_electorates; default: learning_only), the reachable
    frontier is drawn."""
    base = results[0]
    names = [m for m in base["methods"] if m not in hide and all(m in r["methods"] for r in results)]
    panels = [(g, [m for m in names if m in ms]) for g, ms in groups.items()]
    grouped = {m for _, ms in panels for m in ms}
    panels.append(("other", [m for m in names if m not in grouped]))
    panels = [(g, ms) for g, ms in panels if ms]
    k = len(results)
    shades = [CHAIN(0.15 + 0.85 * i / max(1, k - 1)) for i in range(k)]
    ncols = min(3, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows + 0.8), sharex=True, sharey=True,
                            squeeze=False)

    def series(m):
        js = [r["methods"].index(m) for r in results]
        return ([r["VSE*"][j] * 100 for r, j in zip(results, js)], [r["WME*"][j] * 100 for r, j in zip(results, js)],
                [r["VSE"][j] * 100 for r, j in zip(results, js)], [r["WME"][j] * 100 for r, j in zip(results, js)])

    labelled = []
    for ax, (g, ms) in zip(axs.flat, panels):
        texts, anchors = [], []
        for m in ms:
            lx, ly, hx, hy = series(m)
            if learning_only:
                pts = [(hx[0], hy[0])] + list(zip(lx, ly))
                ax.scatter([hx[0]], [hy[0]], s=28, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.1, zorder=3)
            else:
                pts = list(zip(lx, ly))
                ax.plot(hx, hy, ls="--", lw=0.8, color="#c4c3be", zorder=1)
                for i in range(k):
                    ax.scatter([hx[i]], [hy[i]], s=24, facecolors=SURFACE, edgecolors=shades[i], linewidths=1.1,
                               zorder=3)
                # label the honest chain too (grey italics), at its last point, so honest and learned
                # chains can be told apart when they lie far from each other
                # (skipped when the honest end lies on the learned end: one label is enough there)
                if abs(hx[-1] - lx[-1]) + abs(hy[-1] - ly[-1]) > 0.3:
                    anchors.append((hx[-1], hy[-1]))
                    texts.append(ax.text(hx[-1], hy[-1], m, fontsize=6.5, color=MUTED, style="italic",
                                         va="center", zorder=5))
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=3, shrinkB=4,
                                            mutation_scale=7))
            for i in range(k):
                ax.scatter([lx[i]], [ly[i]], s=40, color=shades[i], edgecolors=SURFACE, linewidths=0.8, zorder=4)
            anchors.append((lx[-1], ly[-1]))
            texts.append(ax.text(lx[-1], ly[-1], m, fontsize=7.5, color=INK2, va="center", zorder=5))
        ax.set_title(g, loc="left", fontsize=10)
        labelled.append((ax, texts, anchors))
    for ax in axs.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel(WME_LABEL.replace("WME", "WME*", 1), fontsize=8)
    for ax in axs[-1, :]:
        ax.set_xlabel(VSE_LABEL.replace("VSE", "VSE*", 1), fontsize=8)
    if same_electorates is None:
        same_electorates = learning_only
    F = base.get("frontier") if same_electorates else None   # same electorates: one frontier for all conditions
    _unreachable(axs.flat[0], "xy", lines=F is None)       # shared axes: limits apply to all panels
    _frontier(axs.flat[0], F)
    for ax in list(axs.flat)[1:len(panels)]:
        lo, hi = ax.get_xlim()
        ax.axvspan(100, hi, color=UNREACHABLE, zorder=0, lw=0)
        lo, hi = ax.get_ylim()
        ax.axhspan(100, hi, color=UNREACHABLE, zorder=0, lw=0)
        if F is None:
            ax.axvline(100, color=BOUNDARY, lw=1.0, zorder=1)
            ax.axhline(100, color=BOUNDARY, lw=1.0, zorder=1)
        _frontier(ax, F)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=shades[i], label=str(v)) for i, v in enumerate(values)]
    handles.append(plt.Line2D([], [], ls="--" if not learning_only else "", lw=0.8, color="#c4c3be", marker="o", ms=6,
                              mfc=SURFACE, mec=MUTED,
                              label="honest ballots" + ("" if learning_only else " (grey italic labels)")))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=8,
               title=f"{param} (light → dark)", title_fontsize=8)
    fig.suptitle(f"Sweep over {param}: learned ballots", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ax, texts, anchors in labelled:
        _repel(ax, texts, anchors)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _row_order(results):
    """Methods sorted by honest VSE, averaged over conditions."""
    methods = results[0]["methods"]
    mean_pe = np.mean([[r["VSE"][r["methods"].index(m)] for m in methods] for r in results], axis=0)
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
