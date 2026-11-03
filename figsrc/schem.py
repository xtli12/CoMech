"""Drawing kit for the schematic figures (Fig. 1, Fig. 4): everything is placed in millimetres on one board
(y downwards), in the house style of figstyle.py. Shapes and icons are plain vector graphics."""
import numpy as np
from figstyle import *
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle, Ellipse, FancyArrowPatch, PathPatch
from matplotlib.path import Path


class Board(Canvas):
    def __init__(self, w=FIG_W, h=100.0):
        super().__init__(w, h)
        u = self.fig.add_axes([0, 0, 1, 1])
        u.set_xlim(0, self.w); u.set_ylim(self.h, 0); u.set_aspect("equal"); u.axis("off")
        u.set_zorder(-5); u.patch.set_alpha(0)
        b = self.fig.add_axes([0, 0, 1, 1])
        b.set_xlim(0, self.w); b.set_ylim(self.h, 0); b.set_aspect("equal"); b.axis("off")
        b.set_zorder(5); b.patch.set_alpha(0)
        self.b, self.u = b, u

    def _ax(self, under):
        return self.u if under else self.b

    # ------------------------------------------------------------- primitives (mm, y downwards)
    def box(self, x, y, w, h, fc="white", ec=INK2, lw=0.5, ls="-", r=1.0, z=2, alpha=1.0, under=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                           zorder=z, alpha=alpha, mutation_aspect=1)
        self._ax(under).add_patch(p)
        return p

    def rect(self, x, y, w, h, fc="white", ec="none", lw=0.0, z=2, alpha=1.0, ls="-", under=False):
        p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha, ls=ls)
        self._ax(under).add_patch(p)
        return p

    def poly(self, pts, fc="white", ec=INK2, lw=0.5, z=2, ls="-", alpha=1.0, closed=True):
        p = Polygon(pts, closed=closed, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls, alpha=alpha, joinstyle="round")
        self.b.add_patch(p)
        return p

    def trap(self, x, y, w, h_in, h_out, fc="white", ec=INK2, lw=0.5, z=2, narrowing=True):
        """Trapezoid centred vertically on y: left height h_in, right height h_out."""
        pts = [(x, y - h_in / 2), (x + w, y - h_out / 2), (x + w, y + h_out / 2), (x, y + h_in / 2)]
        return self.poly(pts, fc=fc, ec=ec, lw=lw, z=z)

    def circ(self, x, y, r, fc="white", ec=INK2, lw=0.5, z=2, ls="-"):
        p = Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
        self.b.add_patch(p)
        return p

    def ell(self, x, y, w, h, fc="white", ec=INK2, lw=0.5, z=2, angle=0, ls="-", alpha=1.0):
        p = Ellipse((x, y), w, h, angle=angle, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls, alpha=alpha)
        self.b.add_patch(p)
        return p

    def seg(self, xs, ys, color=INK, lw=0.5, ls="-", z=3, **kw):
        return self.b.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt", **kw)[0]

    def arr(self, x0, y0, x1, y1, color=INK, lw=0.6, ls="-", z=4, head=(2.4, 1.2), rad=0.0, both=False,
            shrinkA=0, shrinkB=0):
        style = ("<|-|>" if both else "-|>") + f",head_length={head[0]},head_width={head[1]}"
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=1.0, color=color, lw=lw, ls=ls,
                            zorder=z, connectionstyle=f"arc3,rad={rad}", shrinkA=shrinkA, shrinkB=shrinkB,
                            joinstyle="miter", capstyle="butt")
        self.b.add_patch(a)
        return a

    def elbow(self, pts, color=INK, lw=0.6, ls="-", z=4, head=(2.4, 1.2)):
        """Orthogonal connector through pts, arrowhead at the end."""
        xs, ys = zip(*pts)
        self.b.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls, zorder=z,
                    solid_capstyle="butt", solid_joinstyle="miter")
        self.arr(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, ls=ls, z=z, head=head)

    def t(self, x, y, s, size=FS, color=INK, ha="left", va="baseline", weight="normal", style="normal", z=6, **kw):
        if not (len(s) == 1 and s.isalpha() and s.islower() and size >= FS_PANEL - 1e-6 and weight == "bold"):
            weight, size = "normal", min(size, FS_MAX)   # Nature: bold only for panel letters, text <= 7 pt (2026-11-03)
        return self.b.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, fontstyle=style,
                           zorder=z, **kw)

    def tw(self, s, size=FS, **kw):
        """Width of a text string in mm (measured with the figure renderer)."""
        r = self.fig.canvas.get_renderer()
        t = self.b.text(0, 0, s, fontsize=size, **kw)
        bb = t.get_window_extent(renderer=r)
        t.remove()
        return bb.width / self.fig.dpi * 25.4

    def tsub(self, x, y, base, sub, size=FS, color=INK, ha="left", style="italic", sub_style="normal", z=6,
             base_kw=None, gap=0.15):
        """Base symbol with a lowered subscript (both real text at >= 5.5 pt); returns the end x."""
        base_kw = base_kw or {}
        wb = self.tw(base, size, fontstyle=style, **base_kw)
        ws = self.tw(sub, max(size - 0.8, FS_SMALL), fontstyle=sub_style)
        x0 = x - (wb + ws + gap) / 2 if ha == "center" else x
        self.t(x0, y, base, size=size, color=color, style=style, z=z, **base_kw)
        self.t(x0 + wb + gap, y + 0.75, sub, size=max(size - 0.8, FS_SMALL), color=color, style=sub_style, z=z)
        return x0 + wb + gap + ws

    def tsup(self, x, y, base, sup, size=FS, color=INK, ha="left", style="italic", sup_style="normal", z=6,
             gap=0.1, rise=0.75):
        """Base symbol with a raised superscript, both in the text font at the same size (>= 5.5 pt), e.g. the a
        with a superscript minus of the action-sensitivity term (no glyph from another font; Claude, 2026-11-03)."""
        wb = self.tw(base, size, fontstyle=style)
        ws = self.tw(sup, size, fontstyle=sup_style)
        x0 = x - (wb + ws + gap) / 2 if ha == "center" else x
        self.t(x0, y, base, size=size, color=color, style=style, z=z)
        self.t(x0 + wb + gap, y - rise, sup, size=size, color=color, style=sup_style, z=z)
        return x0 + wb + gap + ws

    def img(self, x, y, w, h, im, z=1, fit=True, alpha=None, under=False):
        """Image fitted into the box (x, y, w, h), aspect preserved and centred; returns the used extent."""
        im = np.asarray(im)
        ih, iw = im.shape[:2]
        if fit:
            s = min(w / iw, h / ih)
            ww, hh = iw * s, ih * s
            x0, y0 = x + (w - ww) / 2, y + (h - hh) / 2
        else:
            ww, hh, x0, y0 = w, h, x, y
        self._ax(under).imshow(im, extent=(x0, x0 + ww, y0 + hh, y0), interpolation="none", zorder=z, alpha=alpha)
        self._ax(under).set_xlim(0, self.w); self._ax(under).set_ylim(self.h, 0)
        return (x0, y0, ww, hh)

    def tokens(self, x, y, nx, ny, size=1.5, gap=0.35, cmap=None, seed=0, lo=0.25, hi=0.85, ec="white", z=3):
        """A block of latent tokens (schematic): nx x ny squares, top-left at (x, y)."""
        cmap = cmap or LinearSegmentedColormap.from_list("tok", ["#DCEBF6", BLUE_L, SKY, BLUE, "#004A75"])
        rng = np.random.default_rng(seed)
        for i in range(nx):
            for j in range(ny):
                c = cmap(lo + (hi - lo) * rng.random())
                self.rect(x + i * (size + gap), y + j * (size + gap), size, size, fc=c, ec=ec, lw=0.25, z=z)
        return x + nx * (size + gap) - gap, y + ny * (size + gap) - gap


# ------------------------------------------------------------- images
def rgba_render(Z, crop=True, pad=4):
    """RGBA image of a render (white background made transparent), cropped to its silhouette."""
    img = np.clip(Z["img"], 0, 1); al = np.clip(Z["alpha"], 0, 1)
    rgba = np.dstack([img, al])
    if crop:
        m = al > 0.02
        rr, cc = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        rgba = rgba[max(rr[0] - pad, 0):rr[-1] + pad, max(cc[0] - pad, 0):cc[-1] + pad]
    return rgba


def cut_dark_base(rgba, thr=0.4):
    """Remove the shaded side wall of a superior-view render (rows below the last coloured row that are dark):
    it carries no field data."""
    im = rgba[..., :3]; al = rgba[..., 3]
    sat = (im.max(2) - im.min(2)) > 0.3 * im.max(2)
    rows = np.nonzero(sat.any(1))[0]
    if not len(rows):
        return rgba
    last = rows[-1]
    L = im.mean(2)
    for r in range(last, len(L)):
        m = al[r] > 0.02
        if m.any() and L[r][m].mean() < thr:
            return rgba[:max(r - 2, last + 1)]
    return rgba


def rounded_mask(im, r_frac=0.08):
    """Give a rectangular RGB(A) image rounded corners (alpha)."""
    im = np.asarray(im).astype(float)
    if im.max() > 1.5:
        im = im / 255.0
    h, w = im.shape[:2]
    if im.ndim == 2:
        im = np.dstack([im] * 3)
    if im.shape[2] == 3:
        im = np.dstack([im, np.ones((h, w))])
    r = r_frac * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = np.clip(xx, r, w - 1 - r); cy = np.clip(yy, r, h - 1 - r)
    d = np.hypot(xx - cx, yy - cy)
    im[..., 3] *= np.clip(r - d + 0.5, 0, 1)
    return im


# ------------------------------------------------------------- icons (drawn in a s x s mm box, top-left x, y)
BONE_F = "#E9E2D6"; BONE_E = "#9C9283"; GRAFT = "#F2D06B"; METAL = "#6E7B87"


def _bones(bd, x, y, s, gap=0.30, split=None, shift=0.0):
    """Talus (upper) and calcaneus (lower) blocks of a simple subtalar icon."""
    tw, th = 0.62 * s, 0.30 * s
    bd.box(x + 0.19 * s, y + 0.06 * s, tw, th, fc=BONE_F, ec=BONE_E, lw=0.45, r=0.12 * s, z=3)
    cy = y + 0.06 * s + th + gap * s * 0.35
    if split is None:
        bd.box(x + 0.10 * s, cy, 0.80 * s, 0.34 * s, fc=BONE_F, ec=BONE_E, lw=0.45, r=0.10 * s, z=3)
    else:
        # oblique osteotomy: anterior part fixed, posterior part shifted medially (drawn as a downward shift)
        xa, xb = x + 0.10 * s, x + 0.90 * s
        xm = x + split * s
        yt, yb = cy, cy + 0.34 * s
        bd.poly([(xm + 0.05 * s, yt), (xb, yt), (xb, yb), (xm - 0.05 * s, yb)], fc=BONE_F, ec=BONE_E, lw=0.45, z=3)
        bd.poly([(xa, yt + shift), (xm - 0.05 * s, yt + shift), (xm - 0.15 * s, yb + shift), (xa, yb + shift)],
                fc=BONE_F, ec=BONE_E, lw=0.45, z=3)
    return cy


def icon_resection(bd, x, y, s):
    cy = _bones(bd, x, y, s)
    gap_top = y + 0.06 * s + 0.30 * s
    bd.rect(x + 0.36 * s, gap_top - 0.02 * s, 0.28 * s, cy - gap_top + 0.04 * s, fc="white", ec=COAL, lw=0.5,
            ls=(0, (1.2, 0.8)), z=4)


def icon_osteotomy(bd, x, y, s):
    """Calcaneal osteotomy: the posterior fragment is cut and translated (arrow)."""
    tw, th = 0.62 * s, 0.30 * s
    bd.box(x + 0.19 * s, y + 0.06 * s, tw, th, fc=BONE_F, ec=BONE_E, lw=0.45, r=0.12 * s, z=3)
    cy = y + 0.06 * s + th + 0.105 * s
    bd.box(x + 0.10 * s, cy, 0.42 * s, 0.34 * s, fc=BONE_F, ec=BONE_E, lw=0.45, r=0.08 * s, z=3)
    bd.box(x + 0.60 * s, cy, 0.40 * s, 0.34 * s, fc=BONE_F, ec=BONE_E, lw=0.45, r=0.08 * s, z=3)
    bd.seg([x + 0.56 * s, x + 0.56 * s], [cy - 0.04 * s, cy + 0.40 * s], color=COAL, lw=0.5, ls=(0, (1.2, 0.8)), z=4)
    bd.arr(x + 0.62 * s, cy + 0.52 * s, x + 0.98 * s, cy + 0.52 * s, color=INK, lw=0.5, head=(1.4, 0.8))


def icon_interposition(bd, x, y, s):
    cy = _bones(bd, x, y, s)
    gap_top = y + 0.06 * s + 0.30 * s
    bd.ell(x + 0.5 * s, (gap_top + cy) / 2, 0.34 * s, max(cy - gap_top, 0.08 * s) * 1.25, fc=GRAFT, ec="#B8952E",
           lw=0.4, z=4)


def icon_fusion(bd, x, y, s):
    cy = _bones(bd, x, y, s, gap=0.0)
    # screw across the joint
    x0, y0, x1, y1 = x + 0.58 * s, y + 0.08 * s, x + 0.46 * s, cy + 0.30 * s
    bd.seg([x0, x1], [y0, y1], color=METAL, lw=1.0, z=5)
    bd.seg([x0 - 0.07 * s, x0 + 0.07 * s], [y0 - 0.01 * s, y0 + 0.01 * s], color=METAL, lw=1.4, z=5)


def icon_film(bd, x, y, s, n=3):
    """Stacked video frames with pixel blocks (image/video world models)."""
    rng = np.random.default_rng(4)
    for k in range(n)[::-1]:
        ox, oy = x + k * 0.12 * s, y + k * 0.10 * s
        bd.box(ox, oy, 0.66 * s, 0.52 * s, fc="white", ec=INK2, lw=0.45, r=0.05 * s, z=3 + (n - k))
        if k == 0:
            for i in range(5):
                for j in range(4):
                    v = 0.55 + 0.4 * rng.random()
                    bd.rect(ox + 0.06 * s + i * 0.11 * s, oy + 0.05 * s + j * 0.105 * s, 0.10 * s, 0.095 * s,
                            fc=plt.cm.Greys(v * 0.6), ec="none", z=8)


def icon_load(bd, x, y, s):
    """Body-weight load on a block."""
    bd.box(x + 0.15 * s, y + 0.55 * s, 0.70 * s, 0.32 * s, fc=BONE_F, ec=BONE_E, lw=0.45, r=0.08 * s, z=3)
    for dx in (0.32, 0.5, 0.68):
        bd.arr(x + dx * s, y + 0.05 * s, x + dx * s, y + 0.50 * s, color=INK, lw=0.5, head=(1.3, 0.75))


def icon_funnel(bd, x, y, s, color=INK2):
    bd.poly([(x, y), (x + s, y), (x + 0.6 * s, y + 0.55 * s), (x + 0.6 * s, y + s), (x + 0.4 * s, y + 0.9 * s),
             (x + 0.4 * s, y + 0.55 * s)], fc="white", ec=color, lw=0.5, z=4)


def icon_check(bd, x, y, s, color=INK):
    bd.seg([x, x + 0.35 * s, x + s], [y + 0.55 * s, y + 0.9 * s, y + 0.1 * s], color=color, lw=1.0, z=5)


def icon_hourglass(bd, x, y, s, color=INK2):
    bd.poly([(x, y), (x + s, y), (x + 0.56 * s, y + 0.5 * s), (x + s, y + s), (x, y + s), (x + 0.44 * s, y + 0.5 * s)],
            fc="white", ec=color, lw=0.45, z=4)
    bd.poly([(x + 0.2 * s, y + 0.9 * s), (x + 0.8 * s, y + 0.9 * s), (x + 0.5 * s, y + 0.62 * s)], fc=GREY, ec="none",
            z=5)


# =====================================================================================================================
# Pipeline style (v3, 2026-10-28): the visual language of the authors' own pipeline figures (tinted panels with side
# tabs, pale chevrons between stages, gradient trapezoid encoders, a glossy sphere for the latent transition, round
# swap/sensitivity badges, check/cross marks). Every element below is drawn procedurally by this code; no image
# generator is involved, and the only pictures in the figures are results of this study.
# =====================================================================================================================
from matplotlib.patches import Arc, Wedge
from scipy import ndimage as _ndi

PAL = dict(
    navy="#22325A", navy2="#3A5C9C",
    blue_fill="#EEF3FA", blue_edge="#3E63A8", blue_tab="#DCE6F5", blue_chev="#BCCBDE",
    green_fill="#F0F7EB", green_edge="#5A9A3A", green_tab="#DDEED2", green_chev="#BEDDAA",
    teal_fill="#ECF6F5", teal_edge="#2A8A86", teal_tab="#D3ECEA", teal_chev="#B4DAD6",
    warm_fill="#FBF3F0", warm_edge="#B5604E", warm_tab="#F3DDD7", warm_chev="#E8C6BD",
    sim="#1A9893", clin="#DD7E35", act="#7556A8", act_l="#DCD1EF", seal="#D3CCF2",
    loss="#B64342", loss_fill="#FBEDEC", check="#4FA23A", cross="#C23B30",
    enc0="#F9DCC3", enc1="#D3E3F6", sph0="#1E2A6E", sph1="#6C86E0")
KIND = {"blue": ("blue_fill", "blue_edge", "blue_tab"), "green": ("green_fill", "green_edge", "green_tab"),
        "teal": ("teal_fill", "teal_edge", "teal_tab"), "warm": ("warm_fill", "warm_edge", "warm_tab")}


def _hex(c):
    from matplotlib.colors import to_rgb
    return np.array(to_rgb(c))


def _rr_mask(h, w, r):
    """Anti-aliased rounded-rectangle mask (h x w px, radius r px)."""
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    cx = np.clip(xx, r, w - r); cy = np.clip(yy, r, h - r)
    d = np.hypot(xx - cx, yy - cy)
    return np.clip(r - d + 0.5, 0, 1)


def feather(rgba, frac=0.12, power=1.6):
    """Fade the borders of an image to transparent (the soft-edged collages of the pipeline style)."""
    im = np.asarray(rgba, float)
    if im.max() > 1.5:
        im = im / 255.0
    if im.shape[2] == 3:
        im = np.dstack([im, np.ones(im.shape[:2])])
    h, w = im.shape[:2]
    fy = np.clip(np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5) / (frac * h), 0, 1) ** power
    fx = np.clip(np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5) / (frac * w), 0, 1) ** power
    im = im.copy(); im[..., 3] *= fy[:, None] * fx[None, :]
    return im


def _board_methods():
    B = Board

    def panel(self, x, y, w, h, kind="blue", tab=None, r=2.2, lw=0.75, tab_w=4.2, z=-3, letter=None):
        """Tinted panel with a coloured border; optional vertical side tab carrying the panel letter (top) and a
        rotated label."""
        f, e, t = (PAL[k] for k in KIND[kind])
        self.box(x, y, w, h, fc=f, ec=e, lw=lw, r=r, z=z, under=True)
        if letter:
            self.u.text(x + tab_w / 2 + 0.1, y + 1.0, letter, ha="center", va="top", fontsize=FS_PANEL,
                        fontweight="bold", color=INK, zorder=z + 4)
        if tab:
            # tab = the left strip of the panel, clipped to its rounded outline
            p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec="none",
                               zorder=z + 1, transform=self.u.transData)
            self.u.add_patch(p)
            rr = Rectangle((x, y), tab_w, h, fc=t, ec="none", zorder=z + 1)
            self.u.add_patch(rr); rr.set_clip_path(p)
            self.u.plot([x + tab_w, x + tab_w], [y, y + h], color=e, lw=lw * 0.8, zorder=z + 2,
                        solid_capstyle="butt")
            yc = y + h / 2 if not letter else y + 5.5 + (h - 5.5) / 2
            self.u.text(x + tab_w / 2 + 0.15, yc, tab, rotation=90, ha="center", va="center",
                        fontsize=FS_LABEL, fontweight="normal", color=e, zorder=z + 3)
        return x + (tab_w if tab else 0.0)

    def chevron(self, x, y, w=4.6, h=5.4, fc=None, z=3, body=0.52):
        """Pale block arrow pointing right; (x, y) = left end of the arrow, vertically centred."""
        fc = fc or PAL["blue_chev"]
        bh, hx = h * body, x + w * 0.52
        pts = [(x, y - bh / 2), (hx, y - bh / 2), (hx, y - h / 2), (x + w, y), (hx, y + h / 2), (hx, y + bh / 2),
               (x, y + bh / 2)]
        p = Polygon(pts, closed=True, fc=fc, ec=fc, lw=1.4, joinstyle="round", zorder=z)
        self.b.add_patch(p)
        return p

    def gimg(self, extent, c0, c1, angle=-35.0, n=160, clip=None, z=2, under=False, alpha=1.0):
        """Linear colour gradient filling extent=(x, y, w, h), clipped to a patch."""
        x, y, w, h = extent
        a = np.deg2rad(angle)
        yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
        t = (xx * np.cos(a) - yy * np.sin(a))
        t = (t - t.min()) / (t.max() - t.min())
        img = (1 - t)[..., None] * _hex(c0) + t[..., None] * _hex(c1)
        img = np.dstack([img, np.full((n, n), alpha)])
        ax = self._ax(under)
        im = ax.imshow(img, extent=(x, x + w, y + h, y), interpolation="bilinear", zorder=z)
        ax.set_xlim(0, self.w); ax.set_ylim(self.h, 0)
        if clip is not None:
            im.set_clip_path(clip)
        return im

    def gtrap(self, x, y, w, h_in, h_out, c0=None, c1=None, ec=None, lw=0.6, z=3, label=None, sub=None,
              fs=FS_LABEL):
        """Trapezoid encoder/decoder with a gradient fill (peach to light blue) and a bold label inside."""
        c0 = c0 or PAL["enc0"]; c1 = c1 or PAL["enc1"]; ec = ec or PAL["navy"]
        pts = [(x, y - h_in / 2), (x + w, y - h_out / 2), (x + w, y + h_out / 2), (x, y + h_in / 2)]
        clip = Polygon(pts, closed=True, fc="none", ec="none", transform=self.b.transData)
        self.b.add_patch(clip)
        hh = max(h_in, h_out)
        self.gimg((x, y - hh / 2, w, hh), c0, c1, clip=clip, z=z)
        self.poly(pts, fc="none", ec=ec, lw=lw, z=z + 1)
        if label:
            ls = label.split("\n")
            y0 = y - (len(ls) - 1) * 1.2 + 0.9
            for k, l in enumerate(ls):
                self.t(x + w / 2, y0 + 2.4 * k, l, size=fs, weight="bold", ha="center", color=PAL["navy"], z=z + 2)
        return pts

    def shadow(self, x, y, w, h, r=1.0, off=(0.35, 0.45), blur=0.55, alpha=0.22, z=1, under=False, ppm=12):
        """Soft drop shadow under a card (raster, blurred rounded rectangle)."""
        pad = 3 * blur
        H, W = int((h + 2 * pad) * ppm), int((w + 2 * pad) * ppm)
        m = np.zeros((H, W))
        hh, ww = int(h * ppm), int(w * ppm)
        y0, x0 = int(pad * ppm), int(pad * ppm)
        m[y0:y0 + hh, x0:x0 + ww] = _rr_mask(hh, ww, r * ppm)
        m = _ndi.gaussian_filter(m, blur * ppm)
        img = np.zeros((H, W, 4)); img[..., :3] = 0.12; img[..., 3] = m * alpha
        ax = self._ax(under)
        ax.imshow(img, extent=(x - pad + off[0], x + w + pad + off[0], y + h + pad + off[1], y - pad + off[1]),
                  interpolation="bilinear", zorder=z)
        ax.set_xlim(0, self.w); ax.set_ylim(self.h, 0)

    def card(self, x, y, w, h, im=None, r=0.9, ec=None, lw=0.45, pad=0.45, z=3, shadow=True, fc="white", ls="-",
             fit=True, under=False):
        """White rounded card with a soft shadow and an image inset (under=True: drawn below axes placed on it)."""
        if shadow:
            self.shadow(x, y, w, h, r=r, z=z - 1, under=under)
        self.box(x, y, w, h, fc=fc, ec=ec or "#B8C2CF", lw=lw, r=r, z=z, ls=ls, under=under)
        if im is not None:
            return self.img(x + pad, y + pad, w - 2 * pad, h - 2 * pad, im, z=z + 1, fit=fit, under=under)

    def sphere(self, x, y, r, c0=None, c1=None, z=4, ring=False, ring_color=None, label=None, fs=FS_MAX,
               dots=0, dot_colors=("#7FE3F0", "#F6A5C8", "#FFFFFF"), seed=2, ppm=40):
        """Glossy sphere (procedural shading: diffuse, specular highlight, rim), optional orbit ring and particles."""
        c0 = _hex(c0 or PAL["sph0"]); c1 = _hex(c1 or PAL["sph1"])
        n = int(2 * r * ppm)
        v, u = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1
        rr = np.hypot(u, v)
        inside = rr < 1
        nz = np.sqrt(np.clip(1 - rr ** 2, 0, 1))
        Lv = np.array([-0.45, -0.55, 0.70]); Lv = Lv / np.linalg.norm(Lv)
        dif = np.clip(u * Lv[0] + v * Lv[1] + nz * Lv[2], 0, 1)
        k = 0.18 + 0.82 * dif ** 1.35
        col = (1 - k)[..., None] * c0 + k[..., None] * c1
        Hv = Lv + np.array([0, 0, 1.0]); Hv = Hv / np.linalg.norm(Hv)
        spec = np.clip(u * Hv[0] + v * Hv[1] + nz * Hv[2], 0, 1) ** 38
        col = col + 0.75 * spec[..., None] * (1 - col)
        rim = np.clip((rr - 0.82) / 0.18, 0, 1) ** 2
        col = col * (1 - 0.35 * rim[..., None]) + 0.18 * rim[..., None] * _hex("#9FB4FF")
        alpha = np.clip((1 - rr) * n / 2 + 0.5, 0, 1)
        img = np.dstack([np.clip(col, 0, 1), alpha])
        ext = (x - r, x + r, y + r, y - r)
        if ring:
            rc = ring_color or PAL["navy"]
            self.b.add_patch(Arc((x, y), 3.0 * r, 1.05 * r, angle=-24, theta1=0, theta2=180, color=rc, lw=0.9,
                                 zorder=z - 1))
        self.b.imshow(img, extent=ext, interpolation="bilinear", zorder=z)
        self.b.set_xlim(0, self.w); self.b.set_ylim(self.h, 0)
        if dots:
            rng = np.random.default_rng(seed)
            clip = Circle((x, y), r * 0.98, transform=self.b.transData, fc="none", ec="none")
            self.b.add_patch(clip)
            th = rng.uniform(0, 2 * np.pi, dots); rad = r * np.sqrt(rng.uniform(0.02, 0.92, dots))
            px, py = x + rad * np.cos(th), y + rad * np.sin(th)
            cs = [dot_colors[i % len(dot_colors)] for i in range(dots)]
            sc = self.b.scatter(px, py, s=rng.uniform(0.6, 2.2, dots), c=cs, linewidths=0, zorder=z + 0.5,
                                alpha=0.95)
            sc.set_clip_path(clip)
        if ring:
            self.b.add_patch(Arc((x, y), 3.0 * r, 1.05 * r, angle=-24, theta1=180, theta2=360, color=rc, lw=0.9,
                                 zorder=z + 1))
            for ang in (200.0,):
                a = np.deg2rad(ang)
                cxr, cyr = 1.5 * r * np.cos(a), 0.525 * r * np.sin(a)
                rot = np.deg2rad(-24)
                self.circ(x + cxr * np.cos(rot) - cyr * np.sin(rot), y + cxr * np.sin(rot) + cyr * np.cos(rot),
                          0.32 * r / 1.0 * 0.45, fc=rc, ec="none", z=z + 1.2)
        if label:
            self.t(x, y + 0.36 * fs / 2.2 * 0.95, label, size=fs, color="white", ha="center", style="italic",
                   z=z + 2)
        return ext

    def badge(self, x, y, r=1.6, kind="check", z=8):
        """Filled round badge with a white check or cross."""
        col = PAL["check"] if kind == "check" else PAL["cross"]
        self.circ(x, y, r, fc=col, ec="white", lw=0.5, z=z)
        if kind == "check":
            self.b.plot([x - 0.52 * r, x - 0.12 * r, x + 0.55 * r], [y + 0.02 * r, y + 0.42 * r, y - 0.40 * r],
                        color="white", lw=max(0.9, 0.65 * r), solid_capstyle="round", solid_joinstyle="round",
                        zorder=z + 1)
        else:
            for s in (1, -1):
                self.b.plot([x - 0.42 * r, x + 0.42 * r], [y - 0.42 * r * s, y + 0.42 * r * s], color="white",
                            lw=max(0.9, 0.6 * r), solid_capstyle="round", zorder=z + 1)

    def icon_swap(self, x, y, r=2.6, z=6):
        """Effect-swap badge: two exchanged arrows in a blue disc (drawn after the authors' pipeline icon)."""
        self.circ(x, y, r, fc="#BFE2F4", ec="none", z=z)
        self.circ(x, y, r * 0.82, fc="#2F6FB5", ec="none", z=z + 0.1)
        self.arr(x - 0.50 * r, y - 0.22 * r, x + 0.52 * r, y - 0.22 * r, color="#F39A2D", lw=1.4 * r / 2.6,
                 head=(0.55 * r, 0.42 * r), z=z + 0.2)
        self.arr(x + 0.50 * r, y + 0.24 * r, x - 0.52 * r, y + 0.24 * r, color="#8FD9F2", lw=1.4 * r / 2.6,
                 head=(0.55 * r, 0.42 * r), z=z + 0.2)

    def icon_sens(self, x, y, r=2.6, z=6):
        """Action-sensitivity badge: a forking arrow (neighbouring actions, diverging effects)."""
        self.circ(x, y, r, fc="#E3CBEA", ec="none", z=z)
        self.circ(x, y, r * 0.82, fc="#2F6FB5", ec="none", z=z + 0.1)
        self.b.plot([x, x], [y + 0.55 * r, y + 0.05 * r], color="white", lw=1.3 * r / 2.6, zorder=z + 0.2,
                    solid_capstyle="butt")
        self.arr(x, y + 0.08 * r, x - 0.48 * r, y - 0.50 * r, color="#D9D9D9", lw=1.3 * r / 2.6,
                 head=(0.5 * r, 0.40 * r), z=z + 0.2)
        self.arr(x, y + 0.08 * r, x + 0.48 * r, y - 0.50 * r, color="#F2795A", lw=1.3 * r / 2.6,
                 head=(0.5 * r, 0.40 * r), z=z + 0.2)

    def seal(self, x, y, r=2.2, text="δ", fc=None, ec=None, n=10, z=6, fs=FS_LABEL + 0.5):
        """Scalloped badge (the effect descriptor tag of the pipeline style)."""
        fc = fc or PAL["seal"]; ec = ec or PAL["navy"]
        th = np.linspace(0, 2 * np.pi, 361)
        rad = r * (1 + 0.07 * np.cos(n * th))
        self.poly(np.c_[x + rad * np.cos(th), y + rad * np.sin(th)], fc=fc, ec=ec, lw=0.55, z=z)
        self.t(x, y + 0.36 * fs / 2.6, text, size=fs, ha="center", style="italic", color=PAL["navy"], z=z + 1)

    def cluster(self, x, y, rx, ry, color, n=5, seed=0, z=5, fill=None, alpha=0.35):
        """Soft elliptical blob with dots (a cluster of latent actions)."""
        from matplotlib.colors import to_rgba
        self.ell(x, y, 2 * rx, 2 * ry, fc=to_rgba(fill or color, alpha), ec="none", z=z)
        rng = np.random.default_rng(seed)
        th = rng.uniform(0, 2 * np.pi, n); rad = np.sqrt(rng.uniform(0.05, 0.6, n))
        self.b.scatter(x + rx * rad * np.cos(th), y + ry * rad * np.sin(th), s=2.0, color=color,
                       linewidths=0, zorder=z + 0.2)

    def label_box(self, x, y, text, fs=FS_LABEL, pad=(1.1, 0.8), z=9, ec=None, fc=(1, 1, 1, 0.86), ha="center",
                  weight="normal", color=None, lw=0.45, ls="-"):
        """Semi-transparent white label box (the stream labels of the pipeline style); (x, y) = centre."""
        w = self.tw(text, fs, fontweight=weight) + 2 * pad[0]
        h = fs * 0.3528 * 1.05 + 2 * pad[1]
        x0 = x - w / 2 if ha == "center" else x
        self.box(x0, y - h / 2, w, h, fc=fc, ec=ec or PAL["navy"], lw=lw, r=0.5, z=z, ls=ls)
        self.t(x0 + w / 2, y + fs * 0.3528 * 0.36, text, size=fs, weight=weight, ha="center",
               color=color or PAL["navy"], z=z + 0.5)
        return x0, w

    def film(self, x, y, w, h, frames, z=4, gap=0.5):
        """Film strip (dark band with sprocket holes) holding frames (list of images)."""
        bh = 0.17 * h
        self.box(x, y, w, h, fc="#1C1C1C", ec="none", r=0.4, z=z)
        nh = int(w / 1.55)
        for i in range(nh):
            hx = x + 0.55 + i * (w - 1.1) / max(nh - 1, 1) - 0.3
            for yy in (y + bh * 0.28, y + h - bh * 0.72):
                self.rect(hx, yy, 0.6, bh * 0.44, fc="white", z=z + 0.2)
        n = len(frames)
        fw = (w - (n + 1) * gap) / n
        for k, f in enumerate(frames):
            self.img(x + gap + k * (fw + gap), y + bh, fw, h - 2 * bh, f, z=z + 0.3, fit=False)

    for nm, fn in list(locals().items()):
        if callable(fn) and nm != "B":
            setattr(B, nm, fn)


_board_methods()


def tokens_round(bd, x, y, nx, ny, size=1.5, gap=0.38, cmap=None, seed=0, lo=0.25, hi=0.85, z=3, ec="white"):
    """Latent tokens as small rounded squares."""
    cmap = cmap or LinearSegmentedColormap.from_list("tok", ["#DCEBF6", BLUE_L, SKY, BLUE, "#004A75"])
    rng = np.random.default_rng(seed)
    for i in range(nx):
        for j in range(ny):
            c = cmap(lo + (hi - lo) * rng.random())
            bd.box(x + i * (size + gap), y + j * (size + gap), size, size, fc=c, ec=ec, lw=0.3, r=0.28 * size, z=z)
    return x + nx * (size + gap) - gap, y + ny * (size + gap) - gap


# ------------------------------------------------------------- refined surgical icons (side view of the subtalar
# joint: talus above, calcaneus below; drawn in an s x s mm box, top-left x, y)
BONE_F2 = "#EFE7DA"; BONE_E2 = "#8F826F"


def _talus(bd, x, y, s, z=3):
    th = np.linspace(np.pi, 2 * np.pi, 40)
    cx, cy, a, b = x + 0.47 * s, y + 0.36 * s, 0.30 * s, 0.27 * s
    top = np.c_[cx + a * np.cos(th), cy + b * np.sin(th)]
    neck = [(cx + a, cy), (x + 0.86 * s, y + 0.30 * s), (x + 0.90 * s, y + 0.36 * s), (x + 0.86 * s, y + 0.42 * s),
            (cx + 0.6 * a, cy + 0.06 * s), (cx - a, cy + 0.06 * s)]
    bd.poly(np.r_[top, neck], fc=BONE_F2, ec=BONE_E2, lw=0.45, z=z)
    return cy + 0.06 * s


def _calc(bd, x, y, s, z=3, dy=0.0):
    pts = [(x + 0.04 * s, y + 0.56 * s + dy), (x + 0.30 * s, y + 0.50 * s + dy), (x + 0.62 * s, y + 0.50 * s + dy),
           (x + 0.78 * s, y + 0.47 * s + dy), (x + 0.96 * s, y + 0.53 * s + dy), (x + 0.96 * s, y + 0.70 * s + dy),
           (x + 0.70 * s, y + 0.78 * s + dy), (x + 0.30 * s, y + 0.84 * s + dy), (x + 0.06 * s, y + 0.86 * s + dy),
           (x + 0.00 * s, y + 0.72 * s + dy)]
    return bd.poly(pts, fc=BONE_F2, ec=BONE_E2, lw=0.45, z=z)


def icon2_resection(bd, x, y, s, z=3):
    _talus(bd, x, y, s, z); _calc(bd, x, y, s, z)
    bd.rect(x + 0.50 * s, y + 0.41 * s, 0.22 * s, 0.10 * s, fc="white", ec=COAL, lw=0.55, ls=(0, (1.1, 0.7)),
            z=z + 1)


def icon2_osteotomy(bd, x, y, s, z=3):
    _talus(bd, x, y, s, z)
    # anterior calcaneus (fixed) and tuberosity fragment shifted by t
    pa = [(x + 0.40 * s, y + 0.50 * s), (x + 0.62 * s, y + 0.50 * s), (x + 0.78 * s, y + 0.47 * s),
          (x + 0.96 * s, y + 0.53 * s), (x + 0.96 * s, y + 0.70 * s), (x + 0.70 * s, y + 0.78 * s),
          (x + 0.32 * s, y + 0.84 * s)]
    bd.poly(pa, fc=BONE_F2, ec=BONE_E2, lw=0.45, z=z)
    d = 0.07 * s
    pp = [(x + 0.04 * s - d, y + 0.56 * s + d), (x + 0.36 * s - d, y + 0.50 * s + d), (x + 0.28 * s - d, y + 0.84 * s + d),
          (x + 0.06 * s - d, y + 0.86 * s + d), (x + 0.00 * s - d, y + 0.72 * s + d)]
    bd.poly(pp, fc=BONE_F2, ec=BONE_E2, lw=0.45, z=z)
    bd.seg([x + 0.40 * s, x + 0.31 * s], [y + 0.47 * s, y + 0.90 * s], color=COAL, lw=0.55, ls=(0, (1.1, 0.7)),
           z=z + 1)
    bd.arr(x + 0.12 * s, y + 0.98 * s, x - 0.10 * s, y + 0.98 * s, color=INK, lw=0.5, head=(0.16 * s, 0.10 * s),
           z=z + 1)


def icon2_interposition(bd, x, y, s, z=3):
    _talus(bd, x, y, s, z); _calc(bd, x, y, s, z)
    bd.ell(x + 0.61 * s, y + 0.46 * s, 0.24 * s, 0.10 * s, fc="#F5CE55", ec="#B8922B", lw=0.4, z=z + 1)


def icon2_fusion(bd, x, y, s, z=3):
    _talus(bd, x, y, s, z); _calc(bd, x, y, s, z, dy=-0.04 * s)
    bd.rect(x + 0.22 * s, y + 0.40 * s, 0.50 * s, 0.08 * s, fc=BONE_F2, ec="none", z=z + 0.5)
    bd.seg([x + 0.56 * s, x + 0.44 * s], [y + 0.12 * s, y + 0.74 * s], color="#5E6B78", lw=1.0, z=z + 1)
    bd.seg([x + 0.50 * s, x + 0.62 * s], [y + 0.11 * s, y + 0.13 * s], color="#5E6B78", lw=1.4, z=z + 1)


def icon2_load(bd, x, y, s, z=3):
    _talus(bd, x, y, s, z); _calc(bd, x, y, s, z)
    for dx in (0.32, 0.50, 0.68):
        bd.arr(x + dx * s, y - 0.12 * s, x + dx * s, y + 0.06 * s, color=INK, lw=0.5, head=(0.13 * s, 0.09 * s),
               z=z + 1)


# ------------------------------------------------------------- hindfoot pictograms, lateral view (anterior to the
# right): smooth outlines of the talus and calcaneus through hand-placed control points (schematic icons)
_TAL = [(0.22, 0.47), (0.17, 0.40), (0.22, 0.24), (0.36, 0.13), (0.52, 0.12), (0.64, 0.19), (0.73, 0.25),
        (0.83, 0.23), (0.91, 0.30), (0.87, 0.39), (0.75, 0.40), (0.60, 0.44), (0.42, 0.47), (0.30, 0.48)]
_CAL = [(0.09, 0.52), (0.24, 0.53), (0.37, 0.52), (0.50, 0.51), (0.60, 0.56), (0.74, 0.53), (0.87, 0.54),
        (0.93, 0.60), (0.91, 0.69), (0.70, 0.73), (0.45, 0.80), (0.20, 0.86), (0.07, 0.85), (0.02, 0.74),
        (0.04, 0.60)]


def _smooth_closed(pts, n=24):
    """Closed Catmull-Rom spline through the control points."""
    P = np.asarray(pts, float); m = len(P)
    out = []
    for i in range(m):
        p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def _bone(bd, x, y, s, pts, dx=0.0, dy=0.0, z=3, clip=None):
    P = _smooth_closed(pts)
    P = np.c_[x + (P[:, 0] + dx) * s, y + (P[:, 1] + dy) * s]
    p = Polygon(P, closed=True, fc=BONE_F2, ec=BONE_E2, lw=0.45, zorder=z, joinstyle="round")
    bd.b.add_patch(p)
    if clip is not None:
        cp = Polygon(clip, closed=True, fc="none", ec="none", transform=bd.b.transData)
        bd.b.add_patch(cp); p.set_clip_path(cp)
    return p


def icon3_base(bd, x, y, s, z=3, gap=0.0):
    _bone(bd, x, y, s, _CAL, dy=-gap, z=z)
    _bone(bd, x, y, s, _TAL, z=z + 0.1)


def icon3_resection(bd, x, y, s, z=3):
    icon3_base(bd, x, y, s, z)
    th = np.linspace(0, 2 * np.pi, 60)
    bd.poly(np.c_[x + (0.665 + 0.085 * np.cos(th)) * s, y + (0.475 + 0.045 * np.sin(th)) * s], fc="white", ec=COAL,
            lw=0.55, ls=(0, (1.1, 0.7)), z=z + 1)


def icon3_interposition(bd, x, y, s, z=3):
    icon3_base(bd, x, y, s, z)
    th = np.linspace(0, 2 * np.pi, 60)
    bd.poly(np.c_[x + (0.665 + 0.085 * np.cos(th)) * s, y + (0.475 + 0.045 * np.sin(th)) * s], fc="#F5CE55",
            ec="#B8922B", lw=0.4, z=z + 1)


def icon3_osteotomy(bd, x, y, s, z=3, d=0.07):
    """Oblique calcaneal osteotomy: the tuberosity fragment behind the cut is translated (arrow)."""
    c0, c1 = (0.40, 0.46), (0.30, 0.92)                       # cut line (top, bottom) in icon units
    big = 2.0
    def side(sign, shift):
        a = np.array(c0); b = np.array(c1); dvec = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([dvec[1], -dvec[0]]) * sign
        q = [a - dvec * big, b + dvec * big, b + dvec * big + nrm * big, a - dvec * big + nrm * big]
        return [(x + (u + shift[0]) * s, y + (v + shift[1]) * s) for u, v in q]
    _bone(bd, x, y, s, _CAL, z=z, clip=side(+1, (0, 0)))                      # anterior part, fixed
    _bone(bd, x, y, s, _CAL, dx=-d, dy=0.35 * d, z=z, clip=side(-1, (-d, 0.35 * d)))
    _bone(bd, x, y, s, _TAL, z=z + 0.1)
    bd.seg([x + c0[0] * s, x + c1[0] * s], [y + c0[1] * s, y + c1[1] * s], color=COAL, lw=0.55,
           ls=(0, (1.1, 0.7)), z=z + 1)
    bd.arr(x + 0.16 * s, y + 0.99 * s, x - 0.06 * s, y + 0.99 * s, color=INK, lw=0.5, head=(0.15 * s, 0.10 * s),
           z=z + 1)


def icon3_fusion(bd, x, y, s, z=3):
    icon3_base(bd, x, y, s, z, gap=0.035)
    bd.seg([x + 0.60 * s, x + 0.47 * s], [y + 0.14 * s, y + 0.74 * s], color="#5E6B78", lw=1.0, z=z + 1)
    bd.seg([x + 0.545 * s, x + 0.655 * s], [y + 0.125 * s, y + 0.155 * s], color="#5E6B78", lw=1.4, z=z + 1)


def icon3_load(bd, x, y, s, z=3):
    icon3_base(bd, x, y, s, z)
    for dx in (0.32, 0.45, 0.58):
        bd.arr(x + dx * s, y - 0.16 * s, x + dx * s, y + 0.06 * s, color=INK, lw=0.5, head=(0.13 * s, 0.09 * s),
               z=z + 1)
