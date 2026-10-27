"""Static plots for analyze.py (matplotlib, PNG).

Method identity is carried by direct labels, never by colour, since there are far more methods
than distinguishable hues. In the arrow plot, the learned-ballot points are coloured by sincerity
from blue (sincere) through purple to dark red (much reversed preference). Heatmaps use blue.
"""
import textwrap
import zlib
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path
from matplotlib.patches import PathPatch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8b8a85"
GRID = "#e6e5e1"
POINT = "#2a78d6"   # learned-ballot points
# Sincerity: blue (fully sincere) through purple to dark red (much reversed preference)
SINC = LinearSegmentedColormap.from_list("sincerity", ["#7f1d1d", "#c0392b", "#9b5aa0", "#4f7cc4", "#2f6db5"])
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


REACH = 0.0     # a boundary (100, the frontier) is brought into view only if it lies within this share
                # of the data's span beyond the data; further away it would squeeze the methods together
                # (0 since 2026-10-27: the plots zoom on the methods, boundaries show where they fall in view)
DOT = 0.81      # dot areas x this (0.9 of the earlier diameter, 2026-10-27)
FORMATS = ["png"]   # file formats of the plots, the first linked in the reports (analyze.py / compare.py --format)
DPI = 150           # resolution of raster formats (--dpi)
# what the VSE/WME plots draw besides the methods (analyze.py --overlay): "frontier" (the reachable region
# and the 100 lines), "guides" (−1 … −8 lines), "normalized" (the normalized-ballots line), "points"
# (reference points; POINTS: which, None = all)
OVERLAY = {"frontier", "guides", "normalized", "rounded", "points"}
POINTS = None
SHORT = {"maximal lottery": "max. lottery", "Condorcet winner, else random ballot": "CW//random ballot",
         "Condorcet winner, else utility winner": "CW//utility winner", "random pair, majority": "random pair",
         "nearest the median voter": "median voter", "nearest the mean voter": "mean voter"}   # labels in the plots


def save(fig, path):
    """Save a figure in every format of FORMATS: path's extension is replaced by each."""
    base = os.path.splitext(path)[0]
    for f in FORMATS:
        fig.savefig(f"{base}.{f}", dpi=DPI)


def ext():
    """The file extension of the plots the reports link to."""
    return FORMATS[0]


def _pad(ax, frac=0.08, min_span=1.0):
    """Widen the current (data) limits by frac of the span on each side, and to at least min_span
    points, so markers and labels at the edge are not cut off and tight clusters are not blown up."""
    for get, set_ in ((ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim)):
        lo, hi = get()
        span = max(hi - lo, min_span)
        mid = (lo + hi) / 2
        lo, hi = min(lo, mid - span / 2), max(hi, mid + span / 2)
        set_(lo - frac * (hi - lo), hi + frac * (hi - lo))


def _unreachable(ax, which, extend=True, lines=True):
    """Grey out values above 100 on the x and/or y axis (efficiencies cannot exceed 100). With
    extend, the axis is widened so the boundary is in view; without, only a visible part is shaded."""
    if "frontier" not in OVERLAY:
        return
    for w in which:
        lo, hi = ax.get_xlim() if w == "x" else ax.get_ylim()
        if hi <= 100 and (not extend or 100 - hi > REACH * (hi - lo)):
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


GUIDES = (1, 2, 4, 8)      # distances (points) of the faint lines below the frontier
GUIDE = "#cfcec8"


def _crossing(A, B, a):
    """Value of B where the polyline (A, B) first passes A = a (linear between vertices), or None."""
    for k in range(len(A) - 1):
        lo, hi = sorted((A[k], A[k + 1]))
        if lo <= a <= hi:
            t = 0.0 if A[k + 1] == A[k] else (a - A[k]) / (A[k + 1] - A[k])
            return B[k] + t * (B[k + 1] - B[k])
    return None


def _guides(ax, bx, by):
    """Faint copies of the boundary polyline (bx, by) shifted by d points down and left on both axes,
    for d in GUIDES: a method on the line marked -d could gain d points of VSE and of WME at once
    before it hits the frontier. Separates moving along the trade-off from getting worse on both."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for d in GUIDES:
        X, Y = bx - d, by - d
        ax.plot(X, Y, color=GUIDE, lw=0.7, ls=(0, (4, 2)), zorder=0.5)
        # label where the line leaves the plot at the bottom (away from the methods), or else at the left
        yl = y0 + 0.01 * (y1 - y0)
        xl = _crossing(Y, X, yl)
        if xl is not None and x0 < xl < x1:
            ax.text(xl, yl, f"−{d}", fontsize=6, color=MUTED, ha="center", va="bottom", zorder=0.6, clip_on=True)
        else:
            yc = _crossing(X, Y, x0)
            if yc is not None and y0 < yc < y1:
                ax.text(x0 + 0.005 * (x1 - x0), yc, f"−{d}", fontsize=6, color=MUTED, ha="left", va="bottom",
                        zorder=0.6, clip_on=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _normalized(ax, N, R=None, label="normalized ballots", style=(0, (1, 2)), color=None):
    """The frontier of normalized ballots (analyze.normalized_frontier): a faint dotted line, labelled at
    its best-VSE end, and from there straight down: with normalized ballots, giving up more WME buys no
    more VSE, so the line is a wall on the right. Beyond it lies what needs more than each voter's own
    scale: how much one voter cares compared with another. With R (the reachable region) the line is
    clipped to it, as the guides are."""
    if N is None or len(N) == 0:
        return
    x, y = N[:, 0] * 100, N[:, 1] * 100
    x, y = np.append(x, x[-1]), np.append(y, -1000.0)      # down to below the axes (clipped there)
    ln, = ax.plot(x, y, color=color or BOUNDARY, lw=0.9, ls=style, zorder=1)
    if R is not None and len(R) >= 3:
        clip = PathPatch(Path(R * 100, closed=True), transform=ax.transData, facecolor="none", lw=0)
        ax.add_patch(clip)
        ln.set_clip_path(clip)
    ax.text(x[-2], y[-2], " " + label, fontsize=6, color=MUTED, ha="left", va="center", zorder=1.1,
            clip_on=True)


def _frontier(ax, F, R=None, refs=None, N=None, NR=None):
    """Draw the reachable frontier (points (VSE, WME) as fractions, best WME first) and grey out the
    region beyond it, which no rule can reach in these elections. With R (the whole reachable region,
    see analyze.region) everything outside that polygon is greyed out: also the parts below the ends of
    the trade-off (always electing the best-WME candidate pins VSE as well). Faint guide lines mark
    1, 2, 4, 8 points below the frontier (see _guides); without a frontier they follow the 100 / 100
    corner instead. refs: reference points {name: (VSE, WME)} as fractions, drawn as small crosses
    where they fall inside the plot. N: the frontier of normalized ballots, a dotted line (_normalized).
    NR: the same with ballots rounded to the score range, dash-dotted. What is drawn follows OVERLAY and POINTS."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    shown = {k: v for k, v in (refs or {}).items() if POINTS is None or k in POINTS} if "points" in OVERLAY else {}
    if "normalized" not in OVERLAY:
        N = None
    if "rounded" not in OVERLAY:
        NR = None
    if "frontier" not in OVERLAY:
        _normalized(ax, N)
        _normalized(ax, NR, label="rounded ballots", style=(0, (4, 1.5, 1, 1.5)))
        F = None
    for name, (rx_, ry_) in shown.items():
        ax.scatter([rx_ * 100], [ry_ * 100], marker="x", s=DOT * 30, color=INK2, linewidths=1.1, zorder=2.5, clip_on=True)
        ax.annotate(SHORT.get(name, name), (rx_ * 100, ry_ * 100), xytext=(4, -4), textcoords="offset points", fontsize=7,
                    color=MUTED, ha="left", va="top", zorder=2.5, annotation_clip=True)
    if F is None or len(F) == 0:
        far = -1000.0          # far outside the axes; the lines are clipped
        if "guides" in OVERLAY:
            _guides(ax, np.array([far, 100, 100]), np.array([100, 100, far]))
        return
    _normalized(ax, N, R)
    _normalized(ax, NR, R, label="rounded ballots", style=(0, (4, 1.5, 1, 1.5)))
    x, y = F[:, 0] * 100, F[:, 1] * 100
    # bring the frontier's far ends into view only if they are near the data (see REACH)
    hx = max(x1, x.max() + 0.05 * (x1 - x0)) if x.max() - x1 <= REACH * (x1 - x0) else x1
    hy = max(y1, y.max() + 0.05 * (y1 - y0)) if y.max() - y1 <= REACH * (y1 - y0) else y1
    if R is not None and len(R) >= 3:
        rx, ry = R[:, 0] * 100, R[:, 1] * 100
        lx, ly = min(x0, rx.min()) - 1, min(y0, ry.min()) - 1
        outer = [(lx, ly), (hx + 1, ly), (hx + 1, hy + 1), (lx, hy + 1), (lx, ly)]
        inner = list(zip(rx[::-1], ry[::-1])) + [(rx[-1], ry[-1])]      # clockwise: a hole
        verts = outer + inner
        codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY] \
            + [Path.MOVETO] + [Path.LINETO] * (len(inner) - 2) + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=UNREACHABLE, lw=0, zorder=0))
        ax.plot(np.append(rx, rx[0]), np.append(ry, ry[0]), color=BOUNDARY, lw=1.0, zorder=1)
        # guides: the frontier itself shifted by d points down and left, inside the region only
        # (beyond the frontier's ends the region bends away, so straight extensions would mislead)
        inside = PathPatch(Path(np.column_stack([rx, ry]), closed=False), transform=ax.transData,
                           facecolor="none", lw=0)
        ax.add_patch(inside)
        for d in (GUIDES if "guides" in OVERLAY else ()):
            ln, = ax.plot(x - d, y - d, color=GUIDE, lw=0.7, ls=(0, (4, 2)), zorder=0.5)
            ln.set_clip_path(inside)
            ax.text(x[-1] - d, y[-1] - d, f"−{d}", fontsize=6, color=MUTED, ha="center", va="top", zorder=0.6,
                    clip_on=True)
        ax.set_xlim(x0, hx)
        ax.set_ylim(y0, hy)
        return
    else:
        # region beyond: above the curve and to its right
        px = np.concatenate([[x[0], x[0]], x, [hx, hx]])
        py = np.concatenate([[hy, y[0]], y, [y[-1], hy]])
        ax.fill(px, py, color=UNREACHABLE, zorder=0, lw=0)
    # boundary for the guides: WME = 100 up to the frontier, the frontier, then VSE = 100 below it
    bx = np.concatenate([[min(x0, x[0]), x[0]], x, [100, 100]])
    by = np.concatenate([[100, 100], y, [y[-1], min(y0, y[-1])]])
    ax.plot(bx, by, color=BOUNDARY, lw=1.0, zorder=1)
    ax.set_xlim(x0, hx)
    ax.set_ylim(y0, hy)
    far = -1000.0
    if "guides" in OVERLAY:
        _guides(ax, np.concatenate([[far], bx[1:-1], [100]]), np.concatenate([[100], by[1:-1], [far]]))


def _visible(r, hide):
    return [j for j, m in enumerate(r["methods"]) if m not in hide]


def relative_intervals(r, idx, axis):
    """95% intervals of each shown method's value relative to the average of the shown methods (per
    bootstrap replicate: value minus the replicates' mean over methods), placed around the method's
    point. This removes the variation all methods share (some elections are harder for everyone), so
    clearly non-overlapping bars suggest a real difference; an approximation of the paired comparison.
    Returns (lo, hi) in percent."""
    bs = r["boot"][axis][:, idx] * 100
    dev = bs - bs.mean(axis=1, keepdims=True)
    pt = np.asarray(r[axis])[idx] * 100
    lo, hi = np.nanpercentile(dev, [2.5, 97.5], axis=0)
    return pt + lo - (pt - pt.mean()), pt + hi - (pt - pt.mean())


def _errorbars(ax, r, idx, xa, ya, color, alpha):
    (xl, xh), (yl, yh) = relative_intervals(r, idx, xa), relative_intervals(r, idx, ya)
    x, y = np.asarray(r[xa])[idx] * 100, np.asarray(r[ya])[idx] * 100
    ax.errorbar(x, y, xerr=[np.clip(x - xl, 0, None), np.clip(xh - x, 0, None)],
                yerr=[np.clip(y - yl, 0, None), np.clip(yh - y, 0, None)],
                fmt="none", ecolor=color, elinewidth=0.8, alpha=alpha, zorder=2, capsize=0)


def arrows(r, path, hide, errorbars=False):
    """Honest (open circle) -> learned ballots (filled) in the VSE / WME plane, all methods together.
    The grey area beyond the curve is out of reach for any rule in these elections (see analyze.frontier).
    Filled points are coloured by the sincerity of the learned ballots (blue = sincere, red = less honest).
    errorbars: relative 95% intervals (see relative_intervals) on both points of every method."""
    idx = _visible(r, hide)
    # cap the colour scale at the 10th percentile, so one extreme method does not wash out the rest
    vals = np.array([r["sincerity"][j] * 100 for j in idx])
    vals = vals[~np.isnan(vals)]           # sincerity undefined for TRASH and the Inverse-* methods
    smin = min(99.5, float(np.percentile(vals, 10))) if len(vals) else 99.5
    norm = Normalize(smin, 100, clip=True)
    cmap = SINC
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    texts, anchors = [], []
    for j in idx:
        x0, y0 = r["VSE"][j] * 100, r["WME"][j] * 100
        x1, y1 = r["VSE*"][j] * 100, r["WME*"][j] * 100
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, shrinkA=4, shrinkB=5, mutation_scale=9))
        ax.scatter([x0], [y0], s=DOT * 34, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.2, zorder=3)
        sj = r["sincerity"][j]
        ax.scatter([x1], [y1], s=DOT * 64, color=MUTED if np.isnan(sj) else cmap(norm(sj * 100)), edgecolors=MUTED,
                   linewidths=0.6, zorder=4)
        anchors.append((x1, y1))
        texts.append(ax.text(x1 + 0.12, y1, r["methods"][j], fontsize=7.5, color=INK2, va="center", zorder=5))
    if errorbars:
        _errorbars(ax, r, idx, "VSE", "WME", MUTED, 0.35)
        _errorbars(ax, r, idx, "VSE*", "WME*", INK2, 0.7)
    ax.set_xlabel(VSE_LABEL)
    ax.set_ylabel(WME_LABEL)
    _pad(ax)
    _unreachable(ax, "xy", lines=r.get("frontier") is None)
    _frontier(ax, r.get("frontier"), r.get("region"), r.get("references"), r.get("nfrontier"), r.get("rfrontier"))
    ax.set_title(f"What strategy does: honest (open) → learned ballots (filled, labelled)\n{r['label']}"
                 + ("; bars: 95% intervals relative to the average of the methods shown" if errorbars else ""),
                 loc="left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.02,
                      extend="min" if len(vals) and vals.min() < smin else "neither")
    cb.set_label("sincerity of learned ballots (%)", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    _repel(ax, texts, anchors)
    save(fig, path)
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
            ax.scatter([x0], [y], s=DOT * 34, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.2, zorder=3)
            ax.scatter([x1], [y], s=DOT * 58, color=POINT, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.set_xlabel(title)
        ax.grid(axis="y", visible=False)
        _unreachable(ax, "x")
    axs[0].set_yticks(range(len(idx)), names)
    axs[0].set_title(f"What strategy does: honest (open) → learned ballots (filled)\n{r['label']}", loc="left")
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def sincerity(r, path, hide):
    """VSE* and WME* against sincerity, side by side. No marginal intervals: they show how much
    single values vary between elections, not how surely methods differ (paired differences do)."""
    idx = [j for j in _visible(r, hide) if not np.isnan(r["sincerity"][j])]   # undefined: TRASH, Inverse-*
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True)
    for ax, axis, name in zip(axs, ["VSE*", "WME*"], ["VSE* (learned ballots, %)", "WME* (learned ballots, %)"]):
        texts, anchors = [], []
        for j in idx:
            x, y = r["sincerity"][j] * 100, r[axis][j] * 100
            anchors.append((x, y))
            ax.scatter([x], [y], s=DOT * 60, color=POINT, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            texts.append(ax.text(x + 0.12, y, r["methods"][j], fontsize=7.5, color=INK2, va="center", zorder=4))
        ax.set_xlabel("sincerity of learned ballots (%)")
        ax.set_ylabel(name)
        _unreachable(ax, "y", extend=False)
        _repel(ax, texts, anchors)
    axs[0].set_title(f"Quality under strategy against sincerity (upper right is best)\n{r['label']}", loc="left")
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def fidelity(r, path, hide):
    """What strategy does to the ballots: honest (open) -> learned (filled) in the plane of fidelity
    (x: how well the ballots reproduce the utility differences) and sincerity (y: how far they go
    against the voters' own preferences). Honest ballots sit at sincerity 100, at the fidelity the
    format allows; going left is information lost to strategy (exaggeration, levelling), going down
    is preferences reversed. Returns False when the input has no fidelity (older CSVs)."""
    idx = [j for j in _visible(r, hide) if not np.isnan(r["fidelity"][j])]
    if not idx:
        return False
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    texts, anchors = [], []
    for j in idx:
        x0, y0 = r["fidelity_honest"][j] * 100, 100.0
        x1, y1 = r["fidelity"][j] * 100, r["sincerity"][j] * 100
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, shrinkA=4, shrinkB=5, mutation_scale=9))
        ax.scatter([x0], [y0], s=DOT * 30, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.1, zorder=3)
        ax.scatter([x1], [y1], s=DOT * 60, color=POINT, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        anchors.append((x1, y1))
        texts.append(ax.text(x1 + 0.12, y1, r["methods"][j], fontsize=7.5, color=INK2, va="center", zorder=5))
    ax.set_xlabel("fidelity: how well the ballots reproduce the utility differences (%)")
    ax.set_ylabel("sincerity: no preference reversed = 100 (%)")
    _pad(ax)
    _unreachable(ax, "xy", extend=False)
    ax.set_title("What strategy does to the ballots: honest (open, at sincerity 100) → learned (filled)\n"
                 f"left: information lost, down: preferences reversed; {r['label']}", loc="left")
    fig.tight_layout()
    _repel(ax, texts, anchors)
    save(fig, path)
    plt.close(fig)
    return True


TINTS = ["#dbe7f7", "#f6e3cf", "#dcefe0", "#efdcef", "#f5f0cc", "#d6eef0", "#f3d9d9", "#e4e2f5",
         "#e6efd2", "#f7dfe9", "#dfe3e8", "#f2e6d8"]   # region fills


def method_tint(name):
    """A method's own region colour: chosen from its name alone, so a method keeps its colour across
    runs whatever other methods are present."""
    return zlib.crc32(name.encode()) % len(TINTS)


def _region_colours(lab, keys, names):
    """Colour per region: each method's own tint (method_tint), except where two neighbouring regions
    would share one; then the smaller region takes the next tint none of its neighbours has."""
    adj = {k: set() for k in keys}
    for a, b in ((lab[:, 1:], lab[:, :-1]), (lab[1:, :], lab[:-1, :])):
        m = (a != b) & (a >= 0) & (b >= 0)
        for x, y in set(zip(a[m].tolist(), b[m].tolist())):
            adj[x].add(y)
            adj[y].add(x)
    size = {k: int((lab == k).sum()) for k in keys}
    out = {}
    for k in sorted(keys, key=lambda k: -size[k]):           # larger regions keep their own tint first
        taken = {out[n] for n in adj[k] if n in out}
        t = method_tint(names[k])
        while t in taken and len(taken) < len(TINTS):
            t = (t + 1) % len(TINTS)
        out[k] = t
    return {k: TINTS[t] for k, t in out.items()}


def tradeoff(r, path, axes=("VSE*", "WME*", "sincerity"), hide=()):
    """Ternary map of the Pareto front: each point of the triangle is a weighting of the three axes
    (corners: all weight on one axis), coloured by the Pareto method with the highest weighted sum.
    Each axis is first stretched to 0..1 over the Pareto methods (worst to best among them), so a
    weight means "share of the range the front spans on that axis". Methods on the Pareto front that
    are best for no weighting (in a dent of the front) are listed below the triangle. Methods in
    `hide` are left out before the front is found, so it is the Pareto front of the methods shown.
    Returns False with fewer than two Pareto methods."""
    vis = [j for j in _visible(r, hide) if all(not np.isnan(r[a][j]) for a in axes)]
    P = np.array([[r[a][j] for a in axes] for j in vis]) if vis else np.zeros((0, len(axes)))
    idx = [vis[i] for i in range(len(vis))
           if not any(np.all(P[k] >= P[i]) and np.any(P[k] > P[i]) for k in range(len(vis)) if k != i)]
    if len(idx) < 2:
        return False
    X = np.array([[r[a][j] for a in axes] for j in idx])
    lo, hi = X.min(axis=0), X.max(axis=0)
    Z = (X - lo) / np.where(hi - lo > 1e-12, hi - lo, 1)
    h = np.sqrt(3) / 2
    n = 400
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, h, int(n * h)))
    w3 = gy / h                                  # top corner: third axis
    w2 = gx - w3 / 2                             # right corner: second axis
    w1 = 1 - w2 - w3                             # left corner: first axis
    inside = (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)
    W = np.stack([w1, w2, w3], axis=-1)
    best = np.argmax(W @ Z.T, axis=-1)
    lab = np.where(inside, best, -1)
    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    shown = sorted(set(lab[inside].ravel()))
    colour = _region_colours(lab, shown, {k: r["methods"][idx[k]] for k in shown})
    img = np.ones(lab.shape + (4,))
    img[..., 3] = 0
    for k in shown:
        rgb = matplotlib.colors.to_rgb(colour[k])
        img[lab == k, :3] = rgb
        img[lab == k, 3] = 1
    edge = np.zeros(lab.shape, bool)               # region borders
    edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] >= 0) & (lab[:, :-1] >= 0)
    edge[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] >= 0) & (lab[:-1, :] >= 0)
    img[edge] = (*matplotlib.colors.to_rgb(MUTED), 1)
    ax.imshow(img, origin="lower", extent=(0, 1, 0, h), interpolation="nearest", zorder=1)
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color=INK2, lw=1.0, zorder=2)
    for k in shown:
        yy, xx = np.nonzero(lab == k)
        cx, cy = gx[0, xx].mean(), gy[yy, 0].mean()
        j = idx[k]
        small = len(yy) < 0.02 * inside.sum()        # small region: name only, smaller
        ax.text(cx, cy, r["methods"][j] + ("" if small else "\n" + " / ".join(f"{r[a][j] * 100:.1f}" for a in axes)),
                ha="center", va="center", fontsize=6.5 if small else 7.5, color=INK, zorder=3)
    names = [f"{a} ({lo[i] * 100:.1f}–{hi[i] * 100:.1f})" for i, a in enumerate(axes)]
    ax.text(-0.02, -0.03, names[0], ha="left", va="top", fontsize=9, color=INK)
    ax.text(1.02, -0.03, names[1], ha="right", va="top", fontsize=9, color=INK)
    ax.text(0.5, h + 0.025, names[2], ha="center", va="bottom", fontsize=9, color=INK)
    never = [r["methods"][idx[k]] for k in range(len(idx)) if k not in shown]
    if never:
        ax.text(0.5, -0.09, textwrap.fill("Also on the Pareto front, best for no weighting: " + ", ".join(never), 120),
                ha="center", va="top", fontsize=7.5, color=INK2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.16, h + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Pareto front: the best method for each weighting of the three axes",
                 x=0.02, y=0.985, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.02, 0.952, textwrap.fill("corner = all weight on that axis; axes stretched to the range of the front "
                                        f"(in brackets); {r['label']}", 110),
             ha="left", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, path)
    plt.close(fig)
    return True


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
                ax.scatter([hx[0]], [hy[0]], s=DOT * 28, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.1, zorder=3)
            else:
                pts = list(zip(lx, ly))
                ax.plot(hx, hy, ls="--", lw=0.8, color="#c4c3be", zorder=1)
                for i in range(k):
                    ax.scatter([hx[i]], [hy[i]], s=DOT * 24, facecolors=SURFACE, edgecolors=shades[i], linewidths=1.1,
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
                ax.scatter([lx[i]], [ly[i]], s=DOT * 40, color=shades[i], edgecolors=SURFACE, linewidths=0.8, zorder=4)
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
    R = base.get("region") if same_electorates else None
    refs = base.get("references") if same_electorates else None
    NF = base.get("nfrontier") if same_electorates else None
    NR = base.get("rfrontier") if same_electorates else None
    _pad(axs.flat[0])
    _unreachable(axs.flat[0], "xy", lines=F is None)       # shared axes: limits apply to all panels
    _frontier(axs.flat[0], F, R, refs, NF, NR)
    for ax in list(axs.flat)[1:len(panels)]:
        if "frontier" in OVERLAY:
            lo, hi = ax.get_xlim()
            ax.axvspan(100, hi, color=UNREACHABLE, zorder=0, lw=0)
            lo, hi = ax.get_ylim()
            ax.axhspan(100, hi, color=UNREACHABLE, zorder=0, lw=0)
            if F is None:
                ax.axvline(100, color=BOUNDARY, lw=1.0, zorder=1)
                ax.axhline(100, color=BOUNDARY, lw=1.0, zorder=1)
        _frontier(ax, F, R, refs, NF, NR)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=shades[i], label=str(v)) for i, v in enumerate(values)]
    handles.append(plt.Line2D([], [], ls="--" if not learning_only else "", lw=0.8, color="#c4c3be", marker="o", ms=6,
                              mfc=SURFACE, mec=MUTED,
                              label="honest ballots" + ("" if learning_only else " (grey italic labels)")))
    fig.suptitle(f"Sweep over {param}: learned ballots", x=0.01, ha="left", fontweight="bold")
    H = fig.get_size_inches()[1]
    if ncols > 1:      # wide figure: legend to the right of the title
        fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=8,
                   title=f"{param} (light → dark)", title_fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:              # one panel: no room beside the title, so the legend goes below it, wrapped
        per_row = len(handles) if len(handles) <= 6 else 4
        rows = -(-len(handles) // per_row)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.4 / H), ncol=per_row,
                   frameon=False, fontsize=8, title=f"{param} (light → dark)", title_fontsize=8,
                   alignment="left")
        fig.tight_layout(rect=(0, 0, 1, 1 - (0.55 + 0.22 * rows) / H))
    for ax, texts, anchors in labelled:
        _repel(ax, texts, anchors)
    save(fig, path)
    plt.close(fig)


def track(conds, method, varying, names, path, same_electorates, errorbars=False):
    """One method across all conditions of a sweep, in the VSE*/WME* plane (learned ballots; honest
    values as open circles). conds: list of (result, params dict) in condition order; varying: the
    parameters that differ between conditions; names: display names of parameters. With one varying
    parameter the points form a chain in its order; with two, lines connect the points along the
    parameter with more values and colour shows the other; with more, points carry condition tags.
    errorbars: each point's own 95% interval (raw: includes the variation all methods share)."""
    pts = [(r, p, r["methods"].index(method)) for r, p in conds if method in r["methods"]]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    def num(v):                    # sort key: numbers by value, words (e.g. start: honest, previous) by name
        try:
            return (0, float(v), "")
        except ValueError:
            return (1, 0.0, v)
    lab = lambda k: names.get(k, k)
    texts, anchors = [], []

    def xy(r, j, star=True):
        s = "*" if star else ""
        return r["VSE" + s][j] * 100, r["WME" + s][j] * 100

    # honest points: one if the electorates are the same in every condition, else one per condition
    hon = pts[:1] if same_electorates else pts
    for r, p, j in hon:
        ax.scatter(*xy(r, j, False), s=DOT * 34, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.2, zorder=3)
    if len(varying) == 2:
        k1, k2 = sorted(varying, key=lambda k: -len({p[k] for _, p, _ in pts}))   # lines along k1, colour by k2
        v2 = sorted({p[k2] for _, p, _ in pts}, key=num)
        shades = {v: CHAIN(0.2 + 0.8 * i / max(1, len(v2) - 1)) for i, v in enumerate(v2)}
        for v in v2:
            line = sorted([(r, p, j) for r, p, j in pts if p[k2] == v], key=lambda t: num(t[1][k1]))
            X = [xy(r, j) for r, p, j in line]
            ax.plot([x for x, _ in X], [y for _, y in X], color=shades[v], lw=1.4, zorder=3,
                    label=f"{lab(k2)} {v}")
            ax.scatter([x for x, _ in X], [y for _, y in X], s=DOT * 36, color=shades[v], edgecolors=SURFACE,
                       linewidths=0.7, zorder=4)
            for (x, y), (r, p, j) in zip(X, line):
                anchors.append((x, y)); texts.append(ax.text(x, y, p[k1], fontsize=7, color=INK2, va="center", zorder=5))
        ax.legend(frameon=False, fontsize=8, loc="lower left", title=f"labels: {lab(k1)}", title_fontsize=8)
        title = f"{method}: {lab(k1)} (labels, lines) × {lab(k2)} (colour)"
    else:
        order = sorted(pts, key=lambda t: num(t[1][varying[0]])) if len(varying) == 1 else pts
        X = [xy(r, j) for r, p, j in order]
        if len(varying) == 1:
            ax.plot([x for x, _ in X], [y for _, y in X], color=MUTED, lw=1.0, zorder=2)
        for i, ((x, y), (r, p, j)) in enumerate(zip(X, order)):
            ax.scatter([x], [y], s=DOT * 40, color=CHAIN(0.2 + 0.8 * i / max(1, len(X) - 1)), edgecolors=SURFACE,
                       linewidths=0.7, zorder=4)
            t = p[varying[0]] if len(varying) == 1 else r["tag"]
            anchors.append((x, y)); texts.append(ax.text(x, y, t, fontsize=7, color=INK2, va="center", zorder=5))
        title = f"{method}: " + (f"{lab(varying[0])} (labels, light → dark)" if len(varying) == 1 else "conditions (tags)")
    if errorbars:
        for r, p, j in pts:
            x, y = xy(r, j)
            ax.errorbar([x], [y], xerr=[[x - r["VSE*_lo"][j] * 100], [r["VSE*_hi"][j] * 100 - x]],
                        yerr=[[y - r["WME*_lo"][j] * 100], [r["WME*_hi"][j] * 100 - y]],
                        fmt="none", ecolor=MUTED, elinewidth=0.7, alpha=0.5, zorder=2)
    ax.set_xlabel(VSE_LABEL.replace("VSE", "VSE*", 1))
    ax.set_ylabel(WME_LABEL.replace("WME", "WME*", 1))
    F = pts[0][0].get("frontier") if same_electorates else None
    R = pts[0][0].get("region") if same_electorates else None
    refs = pts[0][0].get("references") if same_electorates else None
    NF = pts[0][0].get("nfrontier") if same_electorates else None
    NR = pts[0][0].get("rfrontier") if same_electorates else None
    _pad(ax)
    _unreachable(ax, "xy", lines=F is None)
    _frontier(ax, F, R, refs, NF, NR)
    ax.set_title(title + "\nlearned ballots (filled); honest (open)"
                 + ("; bars: each point's own 95% interval" if errorbars else ""), loc="left")
    fig.tight_layout()
    _repel(ax, texts, anchors)
    save(fig, path)
    plt.close(fig)
    return True


def _row_order(results):
    """Methods present in every condition, sorted by VSE*, averaged over conditions."""
    methods = [m for m in results[0]["methods"] if all(m in r["methods"] for r in results)]
    mean = np.mean([[r["VSE*"][r["methods"].index(m)] for m in methods] for r in results], axis=0)
    return [methods[j] for j in np.argsort(-mean, kind="stable")]


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
    save(fig, path)
    plt.close(fig)


def beatpath(nodes, edges, path, groups, share):
    """Who clearly dominates whom across conditions (compare.py): methods as boxes in layers, an arrow from
    A down to B when A dominates B in at least `share` of the conditions both were run in (and more often
    than B dominates A), labelled with that share. edges: (A, B, share) after the arrows implied by a chain
    were dropped. The top layer holds the methods nobody dominates at this level; every other method sits
    one layer below the lowest method that dominates it. groups: method -> group, for the box colours."""
    pred = {n: [] for n in nodes}
    for a, b, _ in edges:
        pred[b].append(a)
    layer = {}
    left = list(nodes)
    while left:                                   # layers by longest path from the top (Kahn's order)
        ready = [n for n in left if all(p in layer for p in pred[n])] or left[:1]   # a cycle: break it
        for n in ready:
            layer[n] = 1 + max((layer[p] for p in pred[n] if p in layer), default=-1)
            left.remove(n)
    depth = max(layer.values()) + 1
    rows = [[n for n in nodes if layer[n] == k] for k in range(depth)]
    WIDE = 8                                      # a layer with more methods is drawn as two staggered rows
    x, y = {}, {}
    for k, row in enumerate(rows):                # order within a layer: by the mean position of the
        if k:                                     # methods above that dominate it, to limit crossings
            row.sort(key=lambda n: (np.mean([x[p] for p in pred[n]]) if pred[n] else 0, nodes.index(n)))
        step = 0.55 if len(row) > WIDE else 1.0
        for i, n in enumerate(row):
            x[n] = (i - (len(row) - 1) / 2) * step
            y[n] = -1.4 * k - (0.5 if len(row) > WIDE and i % 2 else 0)
    width = max(len(r) * (0.55 if len(r) > WIDE else 1.0) for r in rows)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * width + 1), 1.4 * depth + 1.6))
    ax.set_axis_off()
    labels = len(edges) <= 40                     # with more arrows the shares would hide the graph
    for a, b, f in edges:
        ax.annotate("", xy=(x[b], y[b]), xytext=(x[a], y[a]),
                    arrowprops=dict(arrowstyle="-|>", color="#a3a29b", lw=0.7, shrinkA=10, shrinkB=10,
                                    mutation_scale=7, alpha=0.9), zorder=1)
        if labels:
            mx, my = x[a] + 0.35 * (x[b] - x[a]), y[a] + 0.35 * (y[b] - y[a])
            ax.text(mx, my, f"{f:.0%}", fontsize=5.5, color=MUTED, ha="center", va="center", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none"))
    names = list(dict.fromkeys(groups.get(n, "") for n in nodes))
    tint = {g: TINTS[i % len(TINTS)] for i, g in enumerate(names)}
    for n in nodes:
        ax.text(x[n], y[n], n, fontsize=7, ha="center", va="center", zorder=3, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=tint[groups.get(n, "")], edgecolor=GRID, lw=0.8))
    ax.set_xlim(-width / 2 - 0.4, width / 2 + 0.4)
    ax.set_ylim(min(y.values()) - 0.5, 0.9)
    ax.set_title("Clear dominance across conditions", loc="left")
    fig.text(0.01, 0.005, f"Arrow A → B: A dominates B (clearly better on a front axis, clearly worse on none) in at least "
             f"{share:.0%} of the conditions both were run in, and more often than B dominates A"
             + ("; the label gives that share" if labels else " (the shares are in beatpath.csv)")
             + ". Arrows implied by a chain of others are left out.", fontsize=6.5, color=MUTED, wrap=True)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=tint[g], edgecolor=GRID) for g in names]
    ax.legend(handles, names, loc="upper right", fontsize=6, frameon=False, ncol=min(3, len(names)))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, path)
    plt.close(fig)


GROUP_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417f", "#eda100", "#008300", "#e34948",
                 "#5f6b7a", "#8a5a2b", "#0097a7", "#9c27b0"]


def method_map(names, H, L, groups, path, explained=float("nan")):
    """The map of the methods (compare.py): each method at its position honest (H, open circle) and with
    learned ballots (L, filled, labelled), an arrow between them, coloured by group. Positions from
    multidimensional scaling of how often two methods elect different winners."""
    gs = list(dict.fromkeys(groups[m] for m in names))
    col = {g: GROUP_COLOURS[i % len(GROUP_COLOURS)] for i, g in enumerate(gs)}
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    texts, anchors = [], []
    for i, m in enumerate(names):
        c = col[groups[m]]
        (x0, y0), (x1, y1) = H[i], L[i]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, alpha=0.55, lw=0.9, shrinkA=3, shrinkB=4))
        ax.scatter([x0], [y0], s=DOT * 30, facecolors=SURFACE, edgecolors=c, linewidths=1.1, zorder=3)
        ax.scatter([x1], [y1], s=DOT * 55, color=c, edgecolors=SURFACE, linewidths=1.2, zorder=4)
        texts.append(ax.text(x1, y1, m, fontsize=7.5, color=INK2, va="center", zorder=5))
        anchors.append((x1, y1))
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=col[g], label=g) for g in gs]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=6, mfc=SURFACE, mec=MUTED, label="honest ballots"),
                plt.Line2D([], [], ls="", marker="o", ms=7, color=MUTED, label="learned ballots")]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Map of the methods: close together = the same winner in most elections"
                 + (f"\n(distances shown {explained:.0%} faithfully in two dimensions)" if explained == explained else ""),
                 loc="left")
    fig.tight_layout()
    _repel(ax, texts, anchors)
    save(fig, path)
    plt.close(fig)
