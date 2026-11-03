"""Supplementary Figs 1-5: galleries of the FE reports' own renderings (geometry, stress maps, validation), arranged on
a millimetre grid at the 164-mm text width of the Supplementary Information (house style v2). The images themselves are
unchanged crops of the reports; renderings of one series share one pixel scale, so their sizes stay comparable."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figsrc"))   # figstyle, fe_data
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import *
from fe_data import *
from PIL import Image
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

CROPS = os.environ.get("ATLAS_CROPS", os.path.join(os.path.dirname(__file__), "..", "crops"))   # ANSYS screenshots of the atlas (not distributed)
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
JET = plt.get_cmap("jet")
GAP, GGAP = 1.2, 3.2          # gap between tiles and between groups (mm)


def load(name):
    return np.asarray(Image.open(os.path.join(CROPS, name + ".png")).convert("RGB"))


def tile(cv, x, y, im, s, interp="none"):
    """Image at its pixel size times s (mm per pixel), top-left at (x, y); returns (w, h) in mm."""
    w, h = im.shape[1] * s, im.shape[0] * s
    ax = cv.ax(x, y, w, h)
    ax.imshow(im, interpolation=interp); ax.set_axis_off()
    return w, h


def group(cv, x0, x1, y, s):
    """Group header with a hairline rule underneath, spanning x0..x1 (mm)."""
    cv.text(x0, y, s, fontsize=FS)
    cv.line([x0, x1], [y + 1.2, y + 1.2], color=INK2, lw=0.4)


def solver_key(cv, x, y, w=22.0, note="Solver color scale, set per map"):
    cbar_mm(cv, ScalarMappable(Normalize(0, 1), JET), x, y, w, 1.2, None, ticks=[])
    cv.text(x - 1.0, y + 1.1, "Low", fontsize=FS_SMALL, ha="right")
    cv.text(x + w + 1.0, y + 1.1, "High", fontsize=FS_SMALL)
    cv.text(x + w + 8.0, y + 1.1, note, fontsize=FS_SMALL, color=INK2)


def row(cv, y, groups, s, label_dy=3.0, interp="none"):
    """groups: list of (header or None, [(crop, label lines), ...]); tiles of a row share the scale s and the top y;
    labels share one baseline under the tallest tile. Returns the bottom of the labels."""
    hmax = max(load(n).shape[0] * s for _, items in groups for n, _ in items)
    x = 0.0; bottom = y
    for hdr, items in groups:
        gx0 = x
        for n, lab in items:
            w, h = tile(cv, x, y, load(n), s, interp)
            for k, ln in enumerate(lab):
                cv.text(x + w / 2, y + hmax + label_dy + 2.5 * k, ln, ha="center", fontsize=FS if k == 0 else FS_SMALL,
                        color=INK if k == 0 else INK2)
            bottom = max(bottom, y + hmax + label_dy + 2.5 * (len(lab) - 1))
            x += w + GAP
        if hdr:
            group(cv, gx0, x - GAP, y - 2.6, hdr)
        x += GGAP - GAP
    return bottom


def width(names, s, groups):
    return sum(load(n).shape[1] * s for n in names) + GAP * (len(names) - groups) + GGAP * (groups - 1)


# ---------------------------------------------------------------- S1 geometry gallery
def supp_fig1():
    cv = Canvas(FIG_W_SUPP, 136)
    s = 0.0500
    roz = ["r1_036", "r1_039", "r1_042", "r1_045", "r1_048"]
    typ = ["r1_051", "r1_054", "r1_057", "r1_060"]
    b = row(cv, 6.0, [(None, [("r1_033", ["Intact"])]),
                      ("Rozansky types", [(n, [t]) for n, t in zip(roz, ["I", "II", "III", "IV", "V"])])], s)
    y2 = b + 9.0
    x = 0.0
    for n, t in zip(typ, ["I", "II", "III", "IV"]):
        w, h = tile(cv, x, y2, load(n), s)
        cv.text(x + w / 2, y2 + h + 3.0, t, ha="center")
        x += w + GAP
    group(cv, 0.0, x - GAP, y2 - 2.6, "Anatomy-based types")
    x += GGAP - GAP
    fu = load("r3_057"); sf = h / fu.shape[0]
    w, _ = tile(cv, x, y2, fu, sf)
    group(cv, x, x + w, y2 - 2.6, "Complete fusion (medial view)")
    y3 = y2 + h + 12.0
    al = [load(n) for n in ["r3_032", "r3_034", "r3_036"]]
    sa = 36.0 / max(a.shape[0] for a in al)
    x = 0.0
    for a, t in zip(al, ["Neutral", "10° varus", "10° valgus"]):
        w, h3 = tile(cv, x, y3, a, sa)
        cv.text(x + w / 2, y3 + h3 + 3.0, t, ha="center")
        x += w + GAP
    group(cv, 0.0, x - GAP, y3 - 2.6, "Type I, hindfoot alignment (posterior view)")
    cv.save(os.path.join(OUTDIR, "supp_fig1"))
    print("saved supp_fig1")


# ---------------------------------------------------------------- S2 talocalcaneal stress maps
def supp_fig2():
    cv = Canvas(FIG_W_SUPP, 82)
    s = 0.0595
    v = lambda i: f"{CLS['tc_max'][i]:.2f} MPa"
    roz = ["r1_085", "r1_091", "r1_097", "r1_103", "r1_109"]
    typ = ["r1_115", "r1_121", "r1_127", "r1_133"]
    b = row(cv, 6.0, [(None, [("r2_079", ["Intact", v(0)])]),
                      ("Rozansky types", [(n, [t, v(i + 1)]) for i, (n, t) in enumerate(zip(roz, ["I", "II", "III", "IV", "V"]))])], s)
    b = row(cv, b + 9.5, [("Anatomy-based types", [(n, [t, v(i + 6)]) for i, (n, t) in enumerate(zip(typ, ["I", "II", "III", "IV"]))]),
                          (None, [("r3_148", ["Complete fusion", v(10)])])], s)
    solver_key(cv, 8.0, b + 4.5, note="Solver color scale, set per map; labels give peak von Mises stress")
    cv.save(os.path.join(OUTDIR, "supp_fig2"))
    print("saved supp_fig2")


# ---------------------------------------------------------------- S3 alignment
def supp_fig3():
    starts = {"Type I": 74, "Type II": 92, "Type III": 110, "Type IV": 128}
    co = {t: [load(f"r3_{t0 + d + 5:03d}") for d in (0, 6, 12)] for t, t0 in starts.items()}
    po = {t: [load(f"r3_{t0 + d + 2:03d}") for d in (0, 6, 12)] for t, t0 in starts.items()}
    Hr = 25.5
    sc = Hr / max(im.shape[0] for v in co.values() for im in v)
    sp = Hr / max(im.shape[0] for v in po.values() for im in v)
    wc = max(im.shape[1] for v in co.values() for im in v) * sc
    wp = max(im.shape[1] for v in po.values() for im in v) * sp
    X0 = 13.0
    xs = [X0 + k * (wc + GAP) for k in range(3)]
    xs += [X0 + 3 * (wc + GAP) - GAP + GGAP + k * (wp + GAP) for k in range(3)]
    cv = Canvas(FIG_W_SUPP, 143)
    Y0 = 11.5; RH = Hr + 5.6
    group(cv, xs[0], xs[2] + wc, 3.0, "Talocalcaneal (coalition) stress")
    group(cv, xs[3], xs[5] + wp, 3.0, "Posterior-facet stress")
    for k, x in enumerate(xs):
        cv.text(x + (wc if k < 3 else wp) / 2, 8.6, ["Neutral", "10° varus", "10° valgus"][k % 3], ha="center",
                fontsize=FS_SMALL, color=INK2)
    for r, t in enumerate(starts):
        y = Y0 + r * RH
        cv.text(0.0, y + Hr / 2 + 1.0, t, fontsize=FS)
        for k in range(6):
            im = co[t][k] if k < 3 else po[t][k - 3]
            s = sc if k < 3 else sp
            cw = wc if k < 3 else wp
            w, h = im.shape[1] * s, im.shape[0] * s
            tile(cv, xs[k] + (cw - w) / 2, y + (Hr - h), im, s)
            val = ALIGN[t]["tc_max" if k < 3 else "post_max"][k % 3]
            bad = t in ("Type III", "Type IV") and k % 3 == 2      # valgus models with input errors (2026-10-31; Methods)
            cv.text(xs[k] + cw / 2, y + Hr + 2.8, f"{val:.2f} MPa" + ("\u2020" if bad else ""), ha="center", fontsize=FS_SMALL)
    solver_key(cv, 21.0, Y0 + 4 * RH + 2.0, note="Solver color scale, set per map; labels give peak von Mises stress. "
                                                   "\u2020Input error in the model; not analyzed (Methods)")
    cv.save(os.path.join(OUTDIR, "supp_fig3"))
    print("saved supp_fig3")


# ---------------------------------------------------------------- S4 facet-defect series
def supp_fig4():
    mid = ["r2_075", "r2_081", "r2_087", "r2_093", "r2_099", "r2_105", "r2_111", "r2_117"]
    post = ["r2_076", "r2_124", "r2_130", "r2_136", "r2_142", "r2_148", "r2_154", "r2_160"]
    ims = {n: load(n) for n in mid + post + ["r2_074", "r2_098", "r2_116", "r2_077", "r2_161"]}
    X0 = 15.0
    cw = (FIG_W_SUPP - X0 - 7 * GAP) / 8
    s = cw / max(ims[n].shape[1] for n in mid + post + ["r2_074", "r2_098", "r2_116"])
    Hr = max(ims[n].shape[0] for n in mid + post) * s
    cv = Canvas(FIG_W_SUPP, 11.0 + 2 * (Hr + 6.0) + 6.0 + Hr + 15.0)
    group(cv, X0, FIG_W_SUPP, 3.0, "Defect left by complete resection (% of the facet)")
    for k, d in enumerate(DEFECTS):
        cv.text(X0 + k * (cw + GAP) + cw / 2, 8.6, f"{d}%", ha="center", fontsize=FS_SMALL, color=INK2)
    Y0 = 11.0; RH = Hr + 6.0
    for r, (nm, seq, D, key) in enumerate([("Middle\nfacet", mid, MID, "mid_max"), ("Posterior\nfacet", post, POST, "post_max")]):
        y = Y0 + r * RH
        cv.text(0.0, y + Hr / 2 - 0.4, nm, fontsize=FS, va="center", linespacing=1.2)
        for k, n in enumerate(seq):
            im = ims[n]; w, h = im.shape[1] * s, im.shape[0] * s
            x = X0 + k * (cw + GAP)
            tile(cv, x + (cw - w) / 2, y + (Hr - h), im, s)
            cv.text(x + cw / 2, y + Hr + 2.8, f"{D[key][k]:.3f} MPa", ha="center", fontsize=FS_SMALL)
    y = Y0 + 2 * RH + 6.0
    x = 0.0
    ant = [("r2_074", "Intact", MID["ant_max"][0]), ("r2_098", "Middle 40%", MID["ant_max"][4]),
           ("r2_116", "Middle 70%", MID["ant_max"][7])]
    for n, lab, v in ant:
        im = ims[n]; w, h = im.shape[1] * s, im.shape[0] * s
        tile(cv, x, y + (Hr - h), im, s)
        cv.text(x + w / 2, y + Hr + 2.8, lab, ha="center", fontsize=FS)
        cv.text(x + w / 2, y + Hr + 5.3, f"{v:.3f} MPa", ha="center", fontsize=FS_SMALL, color=INK2)
        x += w + GAP
    group(cv, 0.0, x - GAP, y - 2.6, "Anterior facet, middle-facet defects")
    x += GGAP + 2.0
    tn = [("r2_077", "Intact", POST["tn_max"][0]), ("r2_161", "Posterior 70%", POST["tn_max"][7])]
    st = Hr / max(ims[n].shape[0] for n, _, _ in tn)
    gx0 = x
    for n, lab, v in tn:
        im = ims[n]; w, h = im.shape[1] * st, im.shape[0] * st
        tile(cv, x, y + (Hr - h), im, st)
        cv.text(x + w / 2, y + Hr + 2.8, lab, ha="center", fontsize=FS)
        cv.text(x + w / 2, y + Hr + 5.3, f"{v:.3f} MPa", ha="center", fontsize=FS_SMALL, color=INK2)
        x += w + GAP
    group(cv, gx0, x - GAP, y - 2.6, "Talonavicular joint, posterior-facet defect")
    solver_key(cv, X0 + 6.0, y + Hr + 10.5, note="Solver color scale, set per map; labels give peak von Mises stress")
    cv.save(os.path.join(OUTDIR, "supp_fig4"))
    print("saved supp_fig4")


# ---------------------------------------------------------------- S5 validation
def supp_fig5():
    # the two validation crops of the report are below 300 ppi at print size; they are resampled uniformly to the
    # 600-dpi output grid (no detail is added); all other crops are embedded at their native resolution
    W = 0.8 * FIG_W_SUPP
    cv = Canvas(W, 69)
    a, b = load("r1_137"), load("r1_139")
    H = 46.0
    wa, wb = a.shape[1] * H / a.shape[0], b.shape[1] * H / b.shape[0]
    gap = (W - wa - wb) / 3
    xa, xb = gap * 0.6, gap * 0.6 + wa + gap * 1.8
    for x, im, w, t1, t2, pk in [(xa, a, wa, "Distal tibial contact stress", "Calcaneus fixed; tibia 500 N, fibula 100 N",
                                  "Peak 3.5839 MPa"),
                                 (xb, b, wb, "Plantar stress", "Ground fixed; tibia 300 N, fibula 60 N",
                                  "Peak 0.25411 MPa")]:
        cv.text(x, 3.4, t1, fontsize=FS_LABEL)
        cv.text(x, 6.4, t2, fontsize=FS_SMALL, color=INK2)
        tile(cv, x, 9.0, im, H / im.shape[0], interp="lanczos")
        cv.text(x + w / 2, 9.0 + H + 3.2, pk, ha="center", fontsize=FS)
    solver_key(cv, 10.0, 9.0 + H + 7.5, w=18.0, note="Solver color scale, set per map")
    cv.save(os.path.join(OUTDIR, "supp_fig5"))
    print("saved supp_fig5")


if __name__ == "__main__":
    for f in sys.argv[2:] or ["supp_fig1", "supp_fig2", "supp_fig3", "supp_fig4", "supp_fig5"]:
        globals()[f]()
