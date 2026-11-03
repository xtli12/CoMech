"""House style for the CoMech figures, version 2 (Nature-portfolio print specifications).

Print rules (Nature research figure guide):
  * width 183 mm (two columns) or 89 mm (one column), height at most 170 mm; supplementary figures 164 mm;
  * Arial/Helvetica (Liberation Sans is metric-identical to Arial), TrueType embedded, editable text;
  * text 5-7 pt (here: 6 pt ticks/legends, 6.5 pt axis labels and plot headers, 5.5 pt minimum), panel letters
    8 pt bold lowercase; no bold sentence titles inside the figure -- the legend carries the explanation;
  * hairline axes (0.5 pt), outward ticks, axes detached from the data area, no grids or frames, no colour in text;
  * colour-blind-safe Okabe-Ito categorical colours (Wong, Nat. Methods 8, 441; 2011) assigned by meaning and used
    identically in every figure; one accent colour for CoMech, greys for comparators wherever the axis already names
    them; secondary encoding (marker shape, fill, line style) wherever two colours meet;
  * lightness-monotonic sequential maps for magnitudes, a blue-grey-red diverging map only for signed changes.
Layout: figures are composed on a millimetre grid (Canvas) so that panels, letters and axes align across columns.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, ListedColormap as _LCM

MM = 1 / 25.4
FIG_W = 183            # mm, two-column width
FIG_W1 = 89            # mm, one-column width
FIG_W_SUPP = 164       # mm, text width of the Supplementary Information: no rescaling
FS = 6.0               # tick labels, legends, annotations (pt)
FS_LABEL = 6.5         # axis labels and plot headers (pt)
FS_SMALL = 5.5         # smallest text used anywhere (pt)
FS_TITLE = 6.5         # kept for older scripts: headers are no longer bold sentences
FS_PANEL = 8.0         # panel letters (pt)
FS_MAX = 7.0           # largest text other than the panel letters (Nature final artwork: 5-7 pt; Claude, 2026-11-03)

# ---------------------------------------------------------------- neutrals
INK = "#1A1A1A"        # text, axes
INK2 = "#4D4D4D"       # secondary text
MUTED = "#8C8C8C"      # reference lines
BASE = "#D2D2D2"       # identity lines, separators
GRID = "#ECECEC"       # very light reference band
PAPER = "#F5F5F4"      # very light fill
BONE_BG = "#F2EFEA"    # background tint behind renderings

# ---------------------------------------------------------------- Okabe-Ito categorical colours
BLUE = "#0072B2"       # CoMech; type I; middle facet; anatomy-based classification
VERM = "#D55E00"       # vermillion: type II; posterior facet; action-concatenated surrogate
GREEN = "#009E73"      # bluish green: anterior facet; DeepONet; GINO
ORANGE = "#E69F00"     # orange: type III; talonavicular joint; geometry->field; Geo-FNO
PINK = "#CC79A7"       # reddish purple: type IV; FNO
SKY = "#56B4E9"        # sky blue: light companion of BLUE
GREY = "#9A9A9A"       # neutral category (indeterminate type, Rozansky classification)
GREY_D = "#5E5E5E"     # comparators on a labelled categorical axis
FE = "#1A1A1A"         # finite element ground truth
COAL = "#B64342"       # the coalition in drawings and renderings (base colour of the engine renders)
BLUE_L = "#9CC7E4"     # light tint of BLUE
VERM_L = "#F2B48D"     # light tint of VERM

# backwards-compatible names used by older scripts
BLUE2, BLUE3, RED, RED2, TEAL, AMBER, VIOLET = BLUE, BLUE_L, VERM, VERM_L, GREEN, ORANGE, PINK
C1, C2, C3, C4, C5 = BLUE, VERM, GREEN, ORANGE, PINK
PENDING_RED = VERM

MODEL = {"comech": BLUE, "actcat": VERM, "deeponet": GREEN, "geo": ORANGE, "fno": PINK,
         "gino": GREEN, "geofno": ORANGE}
TYPE = {"I": BLUE, "II": VERM, "III": ORANGE, "IV": PINK, "?": GREY}
TYPE_MARKER = {"I": "o", "II": "s", "III": "^", "IV": "D", "?": "o"}
FACET = {"mid": BLUE, "post": VERM, "ant": GREEN, "tn": ORANGE}
TAXO = {"anatomy": BLUE, "rozansky": GREY}
# anatomical label maps (voxel sections, drawings): bones in warm greys, subtalar cartilage in three blues (anterior
# light, middle medium, posterior dark), TN/CC cartilage in greens, the coalition in the render colour
ANAT = {"talus": "#CDBFA6", "calc": "#B49F7E", "nav": "#C2BCA6", "cub": "#B0A992", "ant": BLUE_L, "mid": SKY,
        "post": BLUE, "tn": "#7FC8B3", "cc": "#A8DCCB", "coal": COAL, "fib": "#E9A6A1", "interp": ORANGE,
        "fuse": "#5A5A5A", "lig": "#B9A6CF"}

# ---------------------------------------------------------------- colour maps
# von Mises stress (>= 0): ColorBrewer YlOrRd without its near-white end
STRESS = _LCM(plt.get_cmap("YlOrRd")(_np.linspace(0.10, 1.0, 256)), name="stress_ylorrd")
# displacement magnitude (>= 0): light blue -> navy
DISP = LinearSegmentedColormap.from_list(
    "disp_seq", ["#F3F7FB", "#CFE0F1", "#9DBFE3", "#5D93CC", "#3775BA", "#0F4D92", "#082A52"])
# absolute error (>= 0): light grey -> near black
ERR = LinearSegmentedColormap.from_list("err_seq", ["#F7F7F7", "#D9D9D9", "#A6A6A6", "#6E6E6E", "#3B3B3B", "#141414"])
# signed change: ColorBrewer RdBu (blue = decrease, red = increase) with a neutral grey-white midpoint
DIV = LinearSegmentedColormap.from_list(
    "div", ["#053061", "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D",
            "#B2182B", "#67001F"])

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    "mathtext.sf": "Liberation Sans",
    "font.size": FS,
    "axes.titlesize": FS_LABEL,
    "axes.titleweight": "normal",
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "legend.frameon": False,
    "legend.handlelength": 1.3,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2,
    "legend.borderpad": 0.2,
    "legend.labelspacing": 0.28,
    "legend.columnspacing": 0.9,
    "axes.linewidth": 0.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.labelpad": 2.0,
    "axes.titlepad": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": True,
    "axes.xmargin": 0.04, "axes.ymargin": 0.04,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.6, "ytick.major.pad": 1.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 0.9,
    "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.5,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.4,
    "errorbar.capsize": 0,
    "scatter.edgecolors": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.dpi": 150,
})


# ================================================================ layout on a millimetre grid
class Canvas:
    """A figure laid out in millimetres from its top-left corner."""

    def __init__(self, w=FIG_W, h=100.0):
        self.w, self.h = float(w), float(h)
        self.fig = plt.figure(figsize=(self.w * MM, self.h * MM))

    def fx(self, x):
        return x / self.w

    def fy(self, y):
        return 1 - y / self.h

    def ax(self, x, y, w, h, **kw):
        """Axes whose data area has its top-left corner at (x, y) mm and is w x h mm."""
        return self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h], **kw)

    def letter(self, x, y, s):
        """Panel letter (8 pt bold lowercase); (x, y) is the top-left of the glyph box."""
        return self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=FS_PANEL, fontweight="bold", color=INK,
                             va="top", ha="left")

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS); kw.setdefault("color", INK)
        kw.setdefault("va", "baseline"); kw.setdefault("ha", "left")
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)

    def header(self, x, y, s, **kw):
        """Plain 6.5-pt header (sentence case, regular weight); (x, y) is its baseline-left point."""
        kw.setdefault("fontsize", FS_LABEL)
        return self.text(x, y, s, **kw)

    def line(self, xs, ys, **kw):
        from matplotlib.lines import Line2D
        kw.setdefault("color", INK); kw.setdefault("lw", 0.5)
        ln = Line2D([x / self.w for x in xs], [1 - y / self.h for y in ys], transform=self.fig.transFigure, **kw)
        self.fig.add_artist(ln)
        return ln

    def arrow(self, x0, y0, x1, y1, **kw):
        from matplotlib.patches import FancyArrowPatch
        kw.setdefault("color", INK); kw.setdefault("lw", 0.6)
        kw.setdefault("arrowstyle", "-|>,head_length=2.6,head_width=1.3"); kw.setdefault("mutation_scale", 1.0)
        a = FancyArrowPatch((x0 / self.w, 1 - y0 / self.h), (x1 / self.w, 1 - y1 / self.h),
                            transform=self.fig.transFigure, shrinkA=0, shrinkB=0, **kw)
        self.fig.add_artist(a)
        return a

    def image(self, x, y, w, h, img, **kw):
        """An image placed in a w x h mm box (aspect preserved, centred), without axes."""
        ax = self.ax(x, y, w, h)
        kw.setdefault("interpolation", "none")
        ax.imshow(img, **kw)
        ax.set_axis_off()
        return ax

    def save(self, path_noext, png=True):
        save_fig(self.fig, path_noext, png=png)
        plt.close(self.fig)


# ================================================================ axes helpers
def style(ax, offset=2.0, left=True, bottom=True):
    """Hairline axes detached from the data area by `offset` points."""
    for side in ("left", "bottom"):
        sp = ax.spines[side]
        sp.set_position(("outward", offset))
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.tick_params(axis="x", length=0)
    ax.tick_params(which="both", direction="out")
    ax.set_axisbelow(True)
    return ax


def tidy(ax, grid_axis=None):
    """Compatibility wrapper for older scripts."""
    style(ax, offset=0)


def hide(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


despine_all = hide


def ref_line(ax, v, axis="y", **kw):
    kw.setdefault("color", BASE); kw.setdefault("lw", 0.5); kw.setdefault("zorder", 0)
    return (ax.axhline if axis == "y" else ax.axvline)(v, **kw)


def strip_points(ax, pos, vals, color, horizontal=True, width=0.26, size=5.0, seed=0, filled=True, alpha=0.85,
                 zorder=3, marker="o"):
    """Individual values as a jittered strip (deterministic jitter)."""
    vals = _np.asarray(vals, float)
    rng = _np.random.default_rng(seed)
    j = (rng.random(len(vals)) - 0.5) * 2 * width
    kw = dict(s=size, marker=marker, zorder=zorder, alpha=alpha, linewidths=0.4)
    if filled:
        kw.update(color=color, edgecolors="white")
    else:
        kw.update(facecolors="white", edgecolors=color)
    if horizontal:
        return ax.scatter(vals, pos + j, **kw)
    return ax.scatter(pos + j, vals, **kw)


def summary_bar(ax, pos, med, lo, hi, color=INK, horizontal=True, half=0.32, lw=1.1, zorder=4):
    """Median tick with an interquartile (or other) interval line."""
    if horizontal:
        ax.plot([lo, hi], [pos, pos], color=color, lw=0.6, zorder=zorder, solid_capstyle="butt")
        ax.plot([med, med], [pos - half, pos + half], color=color, lw=lw, zorder=zorder + 1, solid_capstyle="butt")
    else:
        ax.plot([pos, pos], [lo, hi], color=color, lw=0.6, zorder=zorder, solid_capstyle="butt")
        ax.plot([pos - half, pos + half], [med, med], color=color, lw=lw, zorder=zorder + 1, solid_capstyle="butt")


def heatmap(ax, M, cmap, norm, fmt="{:+.0f}", gap=0.9, text=True, fs=FS, nan_color=PAPER, light_text_above=0.62,
            na="–"):
    """Cells separated by white gaps; values printed in ink or white depending on cell lightness."""
    from matplotlib.patches import Rectangle
    M = _np.asarray(M, float)
    nr, nc = M.shape
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            c = nan_color if not _np.isfinite(v) else cmap(norm(v))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, edgecolor="white", lw=gap))
            if text:
                if _np.isfinite(v):
                    r, g, b = c[:3]
                    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    ax.text(j, i, fmt.format(v).replace("-", "−"), ha="center", va="center", fontsize=fs,
                            color="white" if lum < 1 - light_text_above + 0.1 else INK)
                else:
                    ax.text(j, i, na, ha="center", va="center", fontsize=fs, color=MUTED)
    ax.set_xlim(-0.5, nc - 0.5); ax.set_ylim(nr - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return ax


def colorbar(fig, mappable, rect, label=None, ticks=None, fmt=None, orientation="horizontal", labelpad=1.5):
    """Thin colour bar (rect in figure fractions)."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, ticks=ticks, format=fmt)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(INK2)
    cb.ax.tick_params(labelsize=FS, length=1.8, width=0.4, pad=1.2)
    if label:
        cb.set_label(label, fontsize=FS, labelpad=labelpad)
    return cb


def colorbar_h(fig, mappable, rect, label=None, ticks=None, fmt=None):
    return colorbar(fig, mappable, rect, label, ticks, fmt, "horizontal")


def colorbar_v(fig, mappable, rect, label=None, ticks=None, fmt=None):
    return colorbar(fig, mappable, rect, label, ticks, fmt, "vertical", labelpad=2.0)


def cbar_mm(cv, mappable, x, y, w, h, label=None, ticks=None, fmt=None, orientation="horizontal"):
    """Colour bar placed in millimetres on a Canvas."""
    return colorbar(cv.fig, mappable, [x / cv.w, 1 - (y + h) / cv.h, w / cv.w, h / cv.h], label, ticks, fmt,
                    orientation, labelpad=1.5 if orientation == "horizontal" else 2.0)


def panel_label(fig, x, y, s):
    """Older interface: panel letter at figure fractions."""
    fig.text(x, y, s, fontsize=FS_PANEL, fontweight="bold", color=INK, va="top", ha="left")


def panel_title(fig, x, y, s, **kw):
    """Older interface: a plain 6.5-pt header (no longer bold)."""
    kw.setdefault("fontsize", FS_LABEL); kw.setdefault("color", INK)
    kw.setdefault("va", "top"); kw.setdefault("ha", "left")
    fig.text(x, y, s, **kw)


def pending(ax, text, xlabel=None, ylabel=None, title=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(BASE); s.set_linewidth(0.5)
    ax.text(0.5, 0.5, "PENDING\n" + text, transform=ax.transAxes, ha="center", va="center", fontsize=FS, color=INK2)


def _bold(t):
    w = t.get_fontweight()
    return (w if isinstance(w, (int, float)) else {"bold": 700, "heavy": 800, "black": 900, "semibold": 600,
                                                    "demibold": 600, "extra bold": 800}.get(str(w), 400)) >= 600


def _is_panel_letter(t):
    s = t.get_text()
    return len(s) == 1 and s.isalpha() and s.islower() and t.get_fontsize() >= FS_PANEL - 1e-6 and _bold(t)


def nature_text(fig):
    """Nature final-artwork rules on every text artist: bold only for the 8-pt panel letters, all other text at most
    7 pt (Claude, 2026-11-03).  Italics are set where they are written (variables only)."""
    import matplotlib.text as mtext
    for t in fig.findobj(mtext.Text):
        if not t.get_text().strip() or _is_panel_letter(t):
            continue
        if _bold(t):
            t.set_fontweight("normal")
        if t.get_fontsize() > FS_MAX + 1e-6:
            t.set_fontsize(FS_MAX)


_TEXT_CMAP = None


def _text_cmap():
    """Code points of the text font (Liberation Sans, metric-identical to Arial)."""
    global _TEXT_CMAP
    if _TEXT_CMAP is None:
        from matplotlib import font_manager as fm
        from matplotlib.ft2font import FT2Font
        _TEXT_CMAP = set(FT2Font(fm.findfont(fm.FontProperties(family="Liberation Sans"))).get_charmap())
    return _TEXT_CMAP


def check_text(fig, min_size=FS_SMALL - 1e-6, max_size=FS_MAX + 1e-6):
    """Exact check of every text artist (ticks, legends, labels): sizes 5.5-7 pt, panel letters 8 pt bold, no other
    bold, one typeface (glyphs missing from Liberation Sans would fall back to another font); returns problems."""
    import os
    import matplotlib.text as mtext
    from matplotlib import font_manager as fm
    fig.canvas.draw()
    bad = []
    cmap = _text_cmap()
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if not t.get_visible() or not s.strip():
            continue
        fsz = t.get_fontsize()
        if _is_panel_letter(t):
            if abs(fsz - FS_PANEL) > 1e-6:
                bad.append(f"panel letter at {fsz:.2f} pt: {s!r}")
            continue
        if fsz < min_size or fsz > max_size:
            bad.append(f"{fsz:.2f} pt: {s[:60]!r}")
        if _bold(t):
            bad.append(f"bold text: {s[:60]!r}")
        if "$" in s:
            if "_" in s or "^" in s:
                bad.append(f"math sub/superscript (rendered below {fsz:.1f} pt): {s[:60]!r}")
            continue
        fam = os.path.basename(fm.findfont(t.get_fontproperties()))
        if "Liberation" not in fam:
            bad.append(f"font {fam}: {s[:60]!r}")
        else:
            miss = sorted({c for c in s if ord(c) not in cmap and not c.isspace()})
            if miss:
                bad.append(f"glyphs not in the text font {''.join(miss)!r}: {s[:60]!r}")
    return bad


def save_fig(fig, path_noext, png=True, check=True):
    """Vector PDF (fonts embedded) plus a PNG preview; applies the Nature text rules and reports any text that still
    breaks them."""
    nature_text(fig)
    if check:
        for b in check_text(fig):
            print("   TEXT SIZE:", b)
    fig.savefig(path_noext + ".pdf")
    if png:
        fig.savefig(path_noext + ".png", dpi=300)
