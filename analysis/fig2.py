"""Figure 2: the finite element atlas (template, coalition locations, talocalcaneal stress, taxonomies, fingerprints).
Values are transcribed from the FE reports (fe_data.py); the atlas pictures in a and c are the solver's own renderings
(cropped, pixels unchanged); the coalition locations in b are drawn on the calcaneus of the engine (with the real-bone
engine, the atlas foot itself: FIG2_RB=1; Claude, 2026-11-03)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figsrc"))   # figstyle, fe_data
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import *
from fe_data import *
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D

CROPS = os.environ.get("FIG2_CROPS", os.path.join(HERE, "..", "crops"))
RENDERS = os.environ.get("FIG2_RENDERS", os.path.join(HERE, "..", "comech", "results", "renders"))
# superior views of render_icons.superior: camera centre and half-width of the parametric engine, or of the real-bone
# engine (figsrc/rbview.py: centre (-43, 2.5) mm, half-width 31 x 0.92); crop to the subtalar joint region; facet letters
RB = os.environ.get("FIG2_RB", "0") == "1"
if RB:
    CAM = (-43.0, 2.5, 31.0 * 0.92); CROP = (-70.0, -16.0, -16.0, 22.0)
    LETTERS = [(-52.8, -1.1, "P"), (-39.1, 16.4, "M"), (-24.8, 6.0, "A")]
else:
    CAM = (45.0, 3.0, 31.0); CROP = (18.0, 72.0, -15.5, 22.0)
    LETTERS = [(33.0, -2.0, "P"), (50.5, 11.0, "M"), (61.0, 9.0, "A")]
OUT = sys.argv[1] if len(sys.argv) > 1 else "fig2"


def crop_white(name, thr=246, pad=4):
    im = np.asarray(Image.open(os.path.join(CROPS, name + ".png")).convert("RGB"))
    m = (im < thr).any(2)
    rr, cc = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    return im[max(rr[0] - pad, 0):rr[-1] + pad, max(cc[0] - pad, 0):cc[-1] + pad]


def fit(cv, x, y, w, h, im, valign="bottom"):
    """Image fitted into a w x h mm box (aspect kept); returns (x0, y0, w0, h0) in mm."""
    ih, iw = im.shape[:2]
    s = min(w / iw, h / ih); ww, hh = iw * s, ih * s
    x0 = x + (w - ww) / 2
    y0 = y + (h - hh) if valign == "bottom" else y + (h - hh) / 2
    ax = cv.ax(x0, y0, ww, hh)
    ax.imshow(im, interpolation="none"); ax.set_axis_off()
    return x0, y0, ww, hh


def superior(cv, x, y, w, name, letters=False):
    """Superior-view rendering of the engine calcaneus (talus removed), world extent known."""
    Z = np.load(os.path.join(RENDERS, name + ".npz"))
    img = np.clip(Z["img"], 0, 1); al = np.clip(Z["alpha"], 0, 1)
    H, W = img.shape[:2]
    cx, cy, o = CAM
    ext = [cx - o, cx + o, cy - o * H / W, cy + o * H / W]          # camera of render_icons.superior
    # keep the joint region (drop the heel and the shaded base)
    x0, x1, y0, y1 = CROP
    h = w * (y1 - y0) / (x1 - x0)
    ax = cv.ax(x, y, w, h)
    ax.imshow(np.dstack([img, al]), extent=ext, origin="upper", interpolation="none")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); hide(ax)
    if letters:
        for (px, py, t) in LETTERS:
            ax.text(px, py, t, ha="center", va="center", fontsize=FS, color=INK)   # no bold (Nature; 2026-11-03)
    return h


def fig2():
    cv = Canvas(FIG_W, 121)
    # ---------------- a: FE template ----------------
    cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, "Finite element template")
    for j, l in enumerate(["Adult foot–ankle CT: bones, cartilage, plantar fascia,",
                           "ligament springs, soft-tissue envelope, ground plate",
                           "Load: tibia 300 N, fibula 60 N (red); ground plate fixed (green)"]):
        cv.text(1.0, 10.2 + j * 2.4, l, fontsize=FS_SMALL, color=INK2)
    for k, (n, lab) in enumerate([("r1_031", "Geometry"), ("r1_071", "Loading"), ("r1_061", "Mesh")]):
        x0, y0, w0, h0 = fit(cv, 1.0 + k * 19.4, 18.0, 18.2, 28.0, crop_white(n))
        cv.text(1.0 + k * 19.4 + 9.1, 50.0, lab, ha="center", fontsize=FS_LABEL)

    # ---------------- b: coalition locations ----------------
    xb = 61.5
    cv.letter(xb, 0.5, "b"); cv.header(xb + 4.8, 3.4, "Coalition location (superior view)")
    items = [("icon_coal_none", ["Intact"], True), ("icon_coal_I", ["Type I", "middle"], False),
             ("icon_coal_II", ["Type II", "posterior"], False), ("icon_coal_III", ["Type III", "extra-articular"], False),
             ("icon_coal_IV", ["Type IV", "complex"], False)]
    pos = [(xb + 0.5, 8.6), (xb + 20.2, 8.6), (xb + 39.9, 8.6), (xb + 0.5, 31.0), (xb + 20.2, 31.0)]
    for (n, lab, let), (x, y) in zip(items, pos):
        h = superior(cv, x, y, 18.6, n, letters=let)
        for j, l in enumerate(lab):
            cv.text(x + 9.3, y + h + 2.6 + j * 2.4, l, ha="center", fontsize=FS)
    # key
    cv.fig.add_artist(mpl.patches.Rectangle((cv.fx(xb + 41.0), cv.fy(36.4)), 3.0 / cv.w, 2.0 / cv.h, color=COAL,
                                            transform=cv.fig.transFigure))
    cv.text(xb + 45.0, 36.2, "Coalition", fontsize=FS)
    cv.fig.add_artist(mpl.patches.Rectangle((cv.fx(xb + 41.0), cv.fy(40.4)), 3.0 / cv.w, 2.0 / cv.h, color=BLUE_L,
                                            transform=cv.fig.transFigure))
    cv.text(xb + 45.0, 40.2, "Articular facet", fontsize=FS)
    cv.text(xb + 41.0, 44.6, "A, M, P: anterior,", fontsize=FS_SMALL, color=INK2)
    cv.text(xb + 41.0, 46.9, "middle, posterior", fontsize=FS_SMALL, color=INK2)

    # ---------------- c: coalition stress maps (solver renderings) ----------------
    xc = 124.0
    cv.letter(xc, 0.5, "c"); cv.header(xc + 4.8, 3.4, "Talocalcaneal stress (solver output)")
    maps = [("r1_115", "Type I", 66.76), ("r1_121", "Type II", 61.18), ("r1_127", "Type III", 36.36),
            ("r1_133", "Type IV", 45.27)]
    for i, (f, lab, pk) in enumerate(maps):
        r, c = divmod(i, 2)
        x, y = xc + 1.0 + c * 29.4, 7.4 + r * 21.4
        fit(cv, x, y, 27.6, 15.6, crop_white(f), valign="center")
        cv.text(x + 13.8, y + 18.0, f"{lab}: peak {pk:.2f} MPa", ha="center", fontsize=FS)
    ax = cv.ax(xc + 9.0, 51.6, 42.0, 1.5)
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto",
              cmap=LinearSegmentedColormap.from_list("ans", ["#0000ff", "#00bcd4", "#00e676", "#ffeb3b", "#ff9800",
                                                             "#ff0000"]))
    ax.set_xticks([0, 255]); ax.set_xticklabels(["Low", "High"]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, pad=1.5)
    cv.text(xc + 30.0, 50.6, "Solver scale, set per map", ha="center", fontsize=FS_SMALL, color=INK2)

    # ---------------- d: peak stress at the bridge by taxonomy ----------------
    cv.letter(0, 60.5, "d"); cv.header(4.8, 63.4, "Peak talocalcaneal stress at the bridge")
    ax = cv.ax(22.0, 70.0, 62.0, 39.0)
    rows = [("Rozansky I", 1), ("Rozansky II", 2), ("Rozansky III", 3), ("Rozansky IV", 4), ("Rozansky V", 5),
            (None, None), ("Type I", 6), ("Type II", 7), ("Type III", 8), ("Type IV", 9)]
    ylab, yt = [], []
    for k, (nm, i) in enumerate(rows):
        if nm is None:
            continue
        v = CLS["tc_max"][i]
        col = TAXO["rozansky"] if i <= 5 else TAXO["anatomy"]
        ax.plot([0, v], [k, k], color=GRID, lw=0.6, zorder=1, solid_capstyle="butt")
        ax.scatter([v], [k], s=20, color=col, edgecolors="white", linewidths=0.5, zorder=3)
        ylab.append(nm); yt.append(k)
    ax.axvline(CLS["tc_max"][0], color=INK2, lw=0.6, zorder=2)
    ax.axvline(CLS["tc_max"][10], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.text(CLS["tc_max"][0] - 1.2, -0.75, f"Intact\n{CLS['tc_max'][0]:.2f}", ha="right", va="bottom", fontsize=FS_SMALL,
            color=INK2, linespacing=1.15)
    ax.text(CLS["tc_max"][10] + 1.2, -0.75, f"Complete fusion\n{CLS['tc_max'][10]:.2f}", ha="left", va="bottom",
            fontsize=FS_SMALL, color=INK2, linespacing=1.15)

    def gap(v0, v1, y, txt):
        ax.plot([v0, v0, v1, v1], [y + 0.32, y + 0.45, y + 0.45, y + 0.32], color=INK2, lw=0.5, zorder=4)
        ax.text((v0 + v1) / 2, y + 0.55, txt, ha="center", va="top", fontsize=FS_SMALL, color=INK2)

    gap(CLS["tc_max"][3], CLS["tc_max"][4], 3.0, "")
    ax.annotate("Smallest gap\n2.86 MPa", xy=(CLS["tc_max"][4] + 0.5, 3.45), xytext=(80, 3.2), fontsize=FS_SMALL,
                color=INK2, va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2))
    gap(CLS["tc_max"][7], CLS["tc_max"][6], 7.0, "")
    ax.annotate("Smallest gap\n5.58 MPa", xy=(CLS["tc_max"][6] + 0.5, 7.45), xytext=(80, 7.6), fontsize=FS_SMALL,
                color=INK2, va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2))
    ax.set_yticks(yt); ax.set_yticklabels(ylab); ax.set_ylim(9.6, -1.0)
    ax.set_xlim(0, 112); ax.set_xticks([0, 20, 40, 60, 80, 100])
    style(ax, left=False)
    ax.set_xlabel("Peak von Mises stress (MPa)")
    for t in ax.get_yticklabels()[5:]:
        t.set_color(BLUE)
    cv.text(1.0, 118.4, "CV of articular peaks: anatomy-based I–IV 21–53%; Rozansky I–IV 5–26%", fontsize=FS_SMALL,
            color=INK2)

    # ---------------- e: articular fingerprints ----------------
    cv.letter(95.0, 60.5, "e"); cv.header(99.8, 63.4, "Articular fingerprints (peak stress, % of intact)")
    ax = cv.ax(120.0, 76.0, 46.0, 41.0)
    rows = list(range(1, 11))
    keys = ["ant_max", "mid_max", "post_max", "tn_max", "cc_max"]
    M = np.full((len(rows), len(keys)), np.nan)
    for r, i in enumerate(rows):
        for c, k in enumerate(keys):
            v = CLS[k][i]
            if v is not None:
                M[r, c] = 100 * v / CLS[k][0]
    norm = TwoSlopeNorm(vmin=20, vcenter=100, vmax=180)
    heatmap(ax, M, DIV, norm, fmt="{:.0f}", gap=1.0, fs=FS, na="n/a")
    ax.set_xticks(range(5)); ax.set_xticklabels(["Anterior\nfacet", "Middle\nfacet", "Posterior\nfacet", "Talo-\nnavicular",
                                                 "Calcaneo-\ncuboid"], linespacing=1.1)
    ax.xaxis.tick_top()
    ylabs = ["Rozansky I", "Rozansky II", "Rozansky III", "Rozansky IV", "Rozansky V",
             "Type I", "Type II", "Type III", "Type IV", "Complete fusion"]
    ax.set_yticks(range(10)); ax.set_yticklabels(ylabs)
    ax.axhline(4.5, color="white", lw=2.4); ax.axhline(8.5, color="white", lw=2.4)
    from matplotlib.cm import ScalarMappable
    cbar_mm(cv, ScalarMappable(norm, DIV), 169.5, 76.0, 1.6, 41.0, "Peak stress (% of intact)",
            ticks=[20, 60, 100, 140, 180], orientation="vertical")
    cv.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    fig2()
