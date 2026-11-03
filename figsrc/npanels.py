"""Plot elements for the revised data panels (Claude, 2026-11-03): bounded kernel densities, horizontal rainclouds,
marginal densities, category tints, small 2 x 2 factorial tiles and value pills.  Everything is drawn from the values
passed in; nothing is smoothed into the data except the kernel densities, whose bandwidth rule is stated here
(Silverman's rule, reflected at the bounds of the quantity).  House style: figstyle.py."""
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path
from figstyle import *

# tick and cross as vector markers: the text font (Arial/Liberation Sans) has no check-mark glyphs (2026-11-03)
CHECK = Path([(-0.50, 0.02), (-0.14, -0.38), (0.52, 0.42)], [Path.MOVETO, Path.LINETO, Path.LINETO])
CROSS = Path([(-0.40, -0.40), (0.40, 0.40), (-0.40, 0.40), (0.40, -0.40)],
             [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])

# decision categories of a rule checked against FE-verified envelopes (Fig. 6c, Fig. 8e)
CAT = {"agree": "#D9D9D9", "permissive": "#D6604D", "strict": "#4393C3"}
CAT_TINT = {"agree": "#FFFFFF", "permissive": "#FBE7E3", "strict": "#E7F0F8"}
ZONE_IN, ZONE_OUT = "#E5F3EC", "#FBE7E3"          # within / beyond an envelope (green and red tints)
GREEN_D = "#1B7F5B"                               # text on the within tint
RED_D = "#B03A2E"                                 # text on the beyond tint
BLUES = LinearSegmentedColormap.from_list("tiles", ["#F4F8FB", "#C9DDEE", "#8DB7DA", "#4A8BC2", "#1D5F96", "#0B3C66"])


def kde1d(x, grid, lo=None, hi=None, bw=None):
    """Gaussian kernel density (Silverman's rule of thumb), reflected at lo/hi when given; zero outside them."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.zeros_like(grid)
    sd = np.std(x, ddof=1) if n > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25]); s = min(sd, (q75 - q25) / 1.34) if q75 > q25 else sd
    h = bw if bw else 0.9 * max(s, 1e-9) * n ** (-0.2)
    if h <= 1e-9:
        h = 0.02 * (np.ptp(grid) or 1.0)
    g = grid[:, None]
    d = np.exp(-0.5 * ((g - x[None]) / h) ** 2).sum(1)
    if lo is not None:
        d += np.exp(-0.5 * ((g - (2 * lo - x[None])) / h) ** 2).sum(1)
    if hi is not None:
        d += np.exp(-0.5 * ((g - (2 * hi - x[None])) / h) ** 2).sum(1)
    d /= n * h * np.sqrt(2 * np.pi)
    if lo is not None:
        d[grid < lo] = 0
    if hi is not None:
        d[grid > hi] = 0
    return d


def raincloud_h(ax, pos, vals, color, lo=None, hi=None, grid=None, height=0.36, seed=0, ms=6.0, edge="white",
                box=True, alpha_cloud=0.30, jitter=0.11, gap=0.06, zorder=3, mark_median=True):
    """Horizontal raincloud on row `pos` (rows increase downwards): half-violin above, jittered points below and a
    median/IQR box between them."""
    vals = np.asarray(vals, float)
    if grid is None:
        a, b = (lo if lo is not None else vals.min()), (hi if hi is not None else vals.max())
        pad = 0.05 * (b - a or 1.0); grid = np.linspace(a - (0 if lo is not None else pad), b + (0 if hi is not None else pad), 400)
    d = kde1d(vals, grid, lo, hi)
    if d.max() > 0:
        d = d / d.max() * height
    top = pos - gap
    rng_x = np.ptp(grid) or 1.0
    on = (grid >= vals.min() - 0.03 * rng_x) & (grid <= vals.max() + 0.03 * rng_x)
    ax.fill_between(grid, top, top - d, where=on, color=color, alpha=alpha_cloud, lw=0, zorder=zorder - 1)
    dd = np.where(on, top - d, np.nan)
    ax.plot(grid, dd, color=color, lw=0.6, zorder=zorder - 0.5)
    rng = np.random.default_rng(seed)
    y = pos + gap + 0.04 + rng.uniform(0, jitter, len(vals))
    ax.scatter(vals, y, s=ms, color=color, edgecolors=edge, linewidths=0.3, zorder=zorder + 1)
    if box:
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([q1, q3], [pos, pos], color=INK, lw=1.6, solid_capstyle="butt", zorder=zorder + 2)
        if mark_median:
            ax.scatter([med], [pos], s=11, color="white", edgecolors=INK, linewidths=0.7, zorder=zorder + 3)
    return d


def marginal(ax, vals_list, colors, grid, orient="x", lo=None, hi=None, fill=True, lw=0.7, alpha=0.25, scale=None):
    """Kernel densities of several groups on a marginal axis (orient 'x': along x; 'y': along y)."""
    ds = [kde1d(v, grid, lo, hi) for v in vals_list]
    m = max([d.max() for d in ds] + [1e-12]) if scale is None else scale
    for d, c in zip(ds, colors):
        d = d / m
        if orient == "x":
            if fill:
                ax.fill_between(grid, 0, d, color=c, alpha=alpha, lw=0)
            ax.plot(grid, d, color=c, lw=lw)
        else:
            if fill:
                ax.fill_betweenx(grid, 0, d, color=c, alpha=alpha, lw=0)
            ax.plot(d, grid, color=c, lw=lw)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    if orient == "x":
        ax.set_ylim(0, 1.08)
    else:
        ax.set_xlim(0, 1.08)
    return ax


def pill(cv, x, y, s, fc, color="white", fs=FS_SMALL, pad=0.45, w=None, h=2.3, weight="normal", ec="none", lw=0.4):
    """Rounded label box centred at (x, y) mm on a Canvas."""
    tw = w if w is not None else 0.62 * fs * 0.3528 * len(s) + 2 * pad
    r = FancyBboxPatch((cv.fx(x - tw / 2), cv.fy(y + h / 2)), tw / cv.w, h / cv.h, transform=cv.fig.transFigure,
                       boxstyle="round,pad=0,rounding_size=%.4f" % (0.5 * h / cv.h), fc=fc, ec=ec, lw=lw)
    cv.fig.add_artist(r)
    cv.text(x, y + 0.72, s, ha="center", fontsize=fs, color=color, fontweight=weight)
    return tw


def tiles2x2(ax, M, cmap, norm, fmt, sub=None, hi_cell=(0, 0), better="high"):
    """2 x 2 factorial tiles: M[i][j] (rows: first factor present/absent; columns: second factor present/absent).
    Values printed in each cell (white on dark cells); `sub` adds a second, smaller line; hi_cell is outlined."""
    for i in range(2):
        for j in range(2):
            v = M[i][j]
            c = cmap(norm(v))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=c, ec="white", lw=1.6, zorder=1))
            r, g, b = c[:3]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            tc = "white" if lum < 0.5 else INK
            dy = -0.10 if sub and sub[i][j] else 0.0
            ax.text(j, i + dy, fmt(v), ha="center", va="center", fontsize=FS_LABEL, color=tc, zorder=3)
            if sub and sub[i][j]:
                ax.text(j, i + 0.24, sub[i][j], ha="center", va="center", fontsize=FS_SMALL - 0.5 + 0.5, color=tc,
                        zorder=3)
    i, j = hi_cell
    ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fc="none", ec=INK, lw=0.9, zorder=4))
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def zone_y(ax, y0, y1, color, x0=None, x1=None, z=0):
    """Horizontal tinted zone between y0 and y1 (whole width unless x0/x1)."""
    if x0 is None:
        return ax.axhspan(y0, y1, color=color, lw=0, zorder=z)
    return ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=color, lw=0, zorder=z))


def iso_f1(ax, levels=(0.6, 0.7, 0.8, 0.9), xlim=(0.5, 1.0), color=GRID, label_color=MUTED, fs=FS_SMALL):
    """Iso-F1 curves in a recall (x) - precision (y) plane."""
    r = np.linspace(0.01, 1.0, 400)
    for f in levels:
        p = f * r / (2 * r - f)
        ok = (p > 0) & (p <= 1.0) & (r >= xlim[0]) & (r <= xlim[1])
        ax.plot(r[ok], p[ok], color=color, lw=0.6, zorder=0)
        if ok.any():
            k = np.nonzero(ok)[0][0]
            ax.text(r[k] + 0.004, min(p[k], 0.985), f"F₁ = {f:.1f}", fontsize=fs, color=label_color, ha="left",
                    va="top", rotation=0, zorder=0)
