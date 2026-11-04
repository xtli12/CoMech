"""Figure 3: alignment re-routes load (a) and the facet-defect series (b-d) of the FE atlas.
Values are transcribed from the FE reports (fe_data.py); the maps in d are the solver's own renderings (pixels
unchanged)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figsrc"))   # figstyle, fe_data
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import *
from fe_data import *
import numpy as np
from PIL import Image
from matplotlib.cm import ScalarMappable

CROPS = os.environ.get("ATLAS_CROPS", os.path.join(HERE, "..", "crops"))   # ANSYS screenshots of the atlas (not distributed)
OUT = sys.argv[1] if len(sys.argv) > 1 else "fig3"


def crop_white(name, thr=246, pad=3):
    im = np.asarray(Image.open(os.path.join(CROPS, name + ".png")).convert("RGB"))
    m = (im < thr).any(2)
    rr, cc = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    return im[max(rr[0] - pad, 0):rr[-1] + pad, max(cc[0] - pad, 0):cc[-1] + pad]


def fig3():
    cv = Canvas(FIG_W, 134)
    # ---------------- a: alignment heatmaps ----------------
    cv.panel(0, 0.5, "a", "Hindfoot alignment re-routes load: change in peak stress versus neutral")
    metrics = [("tc_max", "Coalition"), ("mid_max", "Middle\nfacet"), ("post_max", "Posterior\nfacet"),
               ("ant_max", "Anterior\nfacet"), ("tn_max", "Talo-\nnavicular"), ("cc_max", "Calcaneo-\ncuboid")]
    # The valgus models of types III and IV contained material errors in the solver input (found 2026-10-31 when the
    # atlas was re-derived from its files; Methods): drawn hatched, without values, and not analyzed.
    BAD = {("Type III", 2), ("Type IV", 2)}
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=140)
    for j, (pos, title) in enumerate([(1, "10° varus"), (2, "10° valgus")]):
        HW = 64.0                                   # v5: wider cells for the larger column headers (2026-11-04)
        x0 = 15.0 + j * (HW + 19.0)
        ax = cv.ax(x0, 16.5, HW, 24.0)
        M = np.array([[np.nan if (t, pos) in BAD else 100 * (ALIGN[t][k][pos] - ALIGN[t][k][0]) / ALIGN[t][k][0]
                       for k, _ in metrics] for t in ALIGN_TYPES])
        heatmap(ax, M, DIV, norm, fmt="{:+.0f}", gap=1.0, fs=FS, na="")
        from matplotlib.patches import Rectangle
        for r, t in enumerate(ALIGN_TYPES):
            if (t, pos) in BAD:
                for k in range(len(metrics)):
                    ax.add_patch(Rectangle((k - 0.5, r - 0.5), 1, 1, facecolor="none", edgecolor=MUTED, hatch="//////",
                                           lw=0))
                    ax.add_patch(Rectangle((k - 0.5, r - 0.5), 1, 1, facecolor="none", edgecolor="white", lw=1.0))
        ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m for _, m in metrics], linespacing=1.1)
        ax.xaxis.tick_top()
        ax.set_yticks(range(4)); ax.set_yticklabels(ALIGN_TYPES if j == 0 else ["I", "II", "III", "IV"])
        if j == 1:
            ax.set_yticklabels(["Type I", "Type II", "Type III", "Type IV"])
        cv.text(x0 + HW / 2, 8.6, title, ha="center", fontsize=FS_LABEL)
    cbar_mm(cv, ScalarMappable(norm, DIV), 171.0, 16.5, 1.6, 24.0, "Change (%)", ticks=[-100, -50, 0, 50, 100],
            orientation="vertical")

    # ---------------- b, c: facet-defect curves ----------------
    def defect_panel(x, D, facet, tip, header, letter, ylim, yticks):
        cv.panel(x, 47.5, letter, header)
        ax = cv.ax(x + 10.0, 58.0, 47.0, 33.0)     # v5: 47 mm so the larger end labels stay on the canvas
        xs = np.array(DEFECTS)
        pct = lambda k: 100 * (np.array(D[k]) - D[k][0]) / D[k][0]
        name = "middle" if facet == "mid" else "posterior"
        series = [(f"{facet}_max", f"Residual {name} facet, peak", FACET[facet], "-", "o", True),
                  (f"{facet}_mean", f"Residual {name} facet, mean", FACET[facet], (0, (3, 1.6)), "o", False),
                  ("ant_max", "Anterior facet, peak", FACET["ant"], "-", "s", True),
                  ("tn_max", "Talonavicular, peak", FACET["tn"], "-", "^", True)]
        ax.axvspan(tip, 72, color=GRID, lw=0, zorder=0)
        ax.axvline(tip, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0, color=BASE, lw=0.5, zorder=1)
        ends = []
        for k, lab, col, ls, mk, filled in series:
            y = pct(k)
            ax.plot(xs, y, color=col, lw=0.9, ls=ls, marker=mk, ms=3.0, mfc=col if filled else "white", mec=col if not filled else "white",
                    mew=0.5 if filled else 0.7, zorder=3, clip_on=False)
            ends.append([y[-1], lab, col, ls])
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < (ylim[1] - ylim[0]) * 0.085:
                ends[i][0] = ends[i - 1][0] + (ylim[1] - ylim[0]) * 0.085
        for y, lab, col, ls in ends:
            ax.text(74.5, y, lab, fontsize=FS, color=INK, va="center", ha="left", clip_on=False)
        ax.set_xlim(-2, 72); ax.set_xticks(DEFECTS); ax.set_ylim(*ylim); ax.set_yticks(yticks)
        style(ax)
        ax.set_xlabel("Facet defect left by complete resection (% of facet area)")
        ax.set_ylabel("Change versus intact joint (%)")
        ax.text(tip - 1.2, ylim[1], f"Tipping point {tip}%", fontsize=FS_SMALL, color=INK2, va="top", ha="right")
        return ax

    ax = defect_panel(0.0, MID, "mid", 30, "Middle-facet defect", "b", (-6, 50), [0, 10, 20, 30, 40, 50])
    ax.annotate("+43.8% at 30%", xy=(30, 43.8), xytext=(42, 40.5), fontsize=FS_SMALL, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2, shrinkA=0, shrinkB=2))
    ax = defect_panel(91.0, POST, "post", 50, "Posterior-facet defect", "c", (-4, 36), [0, 10, 20, 30])
    ax.annotate("+23.7% and +33.3%\nbeyond 50%", xy=(60, 23.7), xytext=(25, 28), fontsize=FS_SMALL, color=INK2,
                va="center", ha="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2, shrinkA=0, shrinkB=2))

    # ---------------- d: residual facets (solver renderings) ----------------
    cv.letter(0, 101.5, "d")   # v5: both group titles on the letter's baseline (101.5 + LETTER_RISE)
    seqs = [("Middle facet", [("r2_075", "Intact", 0.9225), ("r2_093", "30%", 1.3265), ("r2_105", "50%", 1.0376),
                              ("r2_117", "70%", 0.9995)]),
            ("Posterior facet", [("r2_076", "Intact", 1.1949), ("r2_136", "30%", 1.2194), ("r2_148", "50%", 1.2620),
                                 ("r2_160", "70%", 1.3333)])]
    for gi, (name, items) in enumerate(seqs):
        gx = 4.8 + gi * 92.0
        cv.text(TITLE_DX if gi == 0 else gx, 101.5 + LETTER_RISE,
                f"{name}: defect and peak von Mises stress (solver output)" if gi == 0 else
                f"{name}: defect and peak von Mises stress", fontsize=FS_PTITLE5, fontweight="bold").set_gid("ptitle")
        for i, (f, lab, pk) in enumerate(items):
            x = gx - 3.0 + i * 21.6
            ax = cv.ax(x, 109.5, 19.5, 24.0)
            ax.imshow(crop_white(f), interpolation="none"); ax.set_axis_off()
            cv.text(x + 9.75, 108.6, f"{lab}  {pk:.2f} MPa", ha="center", fontsize=FS)
    cv.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    fig3()
