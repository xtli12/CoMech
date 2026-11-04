"""Figures 5-7, revision 3 (Claude, 2026-11-03): the data panels flagged by the authors are redrawn; data, numbers and
panel letters are unchanged (every value is read from results/*.json, as in fig567.py, which this script imports).

  * Fig. 5c: speed as a log-scale lollipop chart with the hardware of each timing and the 445-fold gap;
    Fig. 5d: bridge stress of the nine atlas-analogue coalitions with location icons (engine renderings,
    render_icons.py / render_icons_roz.py), CoMech as the mean and range of three seeds; defect series with the
    tolerance threshold, the overload zone and the FE and CoMech tolerance crossings.
  * Fig. 6 (all panels): a, joint plot of FE and CoMech tolerances (both facets, identity +-5 pp band, marginal
    densities); b, change of posterior tolerance with a 5-mm osteotomy per foot (FE and CoMech, direction agreement);
    c, virtual cohort against the 50% rule with the two kinds of disagreement shaded and counted, and per-type
    shares; d, precision-recall plane with iso-F1 curves; e, 2 x 2 factorial tiles of the ablation (seed 0).
  * Fig. 7b-d: rainclouds of planning regret with top-1 counts; abstention scatter with marginal densities and the
    abstention region; one- versus two-step roll-outs for three seeds with the distribution of paired differences.
Kernel densities: Silverman's rule, reflected at the bounds of the quantity (npanels.kde1d)."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from fig567 import *          # noqa: F401,F403  (load_all, render_rgba, put_fit, save, NAMES, SEEDS, ...)
import fig567 as _F
from npanels import *          # noqa: F401,F403
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon

MK = {"comech_s0": "o", "actcat": "s", "deeponet": "^", "geo": "D", "fno": "v"}
RENDERS = os.path.join(ROOT, "results", "renders")


def mcol(m):
    """Model colours of the house style (figstyle.MODEL); CoMech in blue."""
    return BLUE if m.startswith("comech") else MODEL.get(m, GREY_D)


def icon(name):
    p = os.path.join(RENDERS, name + ".npz")
    if not os.path.exists(p):
        return None
    Z = np.load(p); im = np.dstack([np.clip(Z["img"], 0, 1), np.clip(Z["alpha"], 0, 1)])
    a = im[..., 3] > 0.02
    rr, cc = np.nonzero(a.any(1))[0], np.nonzero(a.any(0))[0]
    return im[max(rr[0] - 2, 0):rr[-1] + 3, max(cc[0] - 2, 0):cc[-1] + 3]


def pct_lbl(v, nd=0):
    return f"{v:.{nd}f}".replace("-", "−")


# ---- v5 layout helpers (authors' request 2026-11-04: Times New Roman, every text 1 pt larger at save time) ----
def tw_mm(cv, s, size, **kw):
    """Printed width (mm) of a text set at base size `size` (save_fig prints it BUMP pt larger, Times New Roman)."""
    r = cv.fig.canvas.get_renderer()
    t = cv.fig.text(0, 0, s, fontsize=size + BUMP, family=TEXT_FONT, **kw)
    w = t.get_window_extent(renderer=r).width / cv.fig.dpi * 25.4
    t.remove()
    return w


def cap_mm(size):
    """Cap height (mm) of Times New Roman printed at base size `size` + BUMP (cap height 0.662 em)."""
    return 0.662 * (size + BUMP) / 72 * 25.4


class at_print_size:
    """Context: text artists and tick labels temporarily at their printed size (base + BUMP), so that layout code
    can measure what will actually be printed; restored on exit (save_fig then applies BUMP once)."""

    def __init__(self, cv, texts=(), axes_y=(), axes_x=()):
        self.cv, self.texts, self.axes_y, self.axes_x = cv, list(texts), list(axes_y), list(axes_x)

    def __enter__(self):
        self.ts = [(t, t.get_fontsize()) for t in self.texts]
        for t, f in self.ts:
            t.set_fontsize(f + BUMP)
        self.tk = []
        for ax, nm in [(a, "y") for a in self.axes_y] + [(a, "x") for a in self.axes_x]:
            lbl = [t for t in (ax.get_yticklabels() if nm == "y" else ax.get_xticklabels()) if t.get_text().strip()]
            f = lbl[0].get_fontsize() if lbl else FS
            self.tk.append((ax, nm, f))
            ax.tick_params(axis=nm, labelsize=f + BUMP)
        self.cv.fig.canvas.draw()
        return self

    def __exit__(self, *a):
        for t, f in self.ts:
            t.set_fontsize(f)
        for ax, nm, f in self.tk:
            ax.tick_params(axis=nm, labelsize=f)


def panel2(cv, x, y, letter, line1, line2):
    """Panel letter and a two-line title; the second line 1.2 x the title size below the first."""
    cv.panel(x, y, letter, line1)
    t = cv.fig.text(cv.fx(x + TITLE_DX), cv.fy(y + LETTER_RISE + 1.2 * FS_PTITLE5 / 72 * 25.4), line2,
                    fontsize=FS_PTITLE5, fontweight="bold", va="baseline", ha="left", color=INK)
    t.set_gid("ptitle")
    return t


# ============================== Figure 5 ==============================
def fig5(D):
    cv = Canvas(FIG_W, 125)
    # ---- a: rendered fields of a held-out foot (unchanged from fig567.py) ----
    _ha = D["eval_maps"]["heldout"]["act"]; _fac = "posterior" if _ha[2] >= _ha[1] else "middle"
    cv.panel(0, 0.5, "a", f"Held-out foot, {100 * max(_ha[1], _ha[2]):.0f}% {_fac}-facet defect")
    cols = [("fe", "FE (engine)"), ("cm", "CoMech"), ("err", "Absolute error")]
    xs, w = [0.8, 34.6, 68.4], 32.6
    for (k, t), x in zip(cols, xs):
        cv.text(x + w / 2, 10.0, t, ha="center", fontsize=FS_LABEL)
    y = 13.6
    cv.text(0.8, y, "Von Mises stress, calcaneal cartilage (superior view, talus removed)", fontsize=FS_SMALL,
            color=INK2)
    hs, crop = [], (world_window_crop("heldout_stress_fe") if REAL_BONE else None)
    for (k, _), x in zip(cols, xs):
        im, Z, crop = render_rgba(f"heldout_stress_{k}", cut=not REAL_BONE, crop=crop)
        hs.append(put_fit(cv, x, y + 1.6, w, 19.0, im) if REAL_BONE else put(cv, x, y + 1.6, w, im))
        if k == "fe":
            vmax = float(Z["vmax"])
    yb = y + 1.6 + max(hs) + 1.6
    cbar_mm(cv, ScalarMappable(Normalize(0, vmax), STRESS), 12.0, yb, 46.0, 1.3, "Stress (MPa)",
            ticks=mpl.ticker.MaxNLocator(4))
    cbar_mm(cv, ScalarMappable(Normalize(0, vmax / 2), ERR), 74.0, yb, 21.4, 1.3, "Error (MPa)",
            ticks=mpl.ticker.MaxNLocator(3))
    y = yb + 9.6
    cv.text(0.8, y, "Talar displacement, calcaneus-fixed frame (inferior view)", fontsize=FS_SMALL, color=INK2)
    hs, crop = [], None
    for (k, _), x in zip(cols, xs):
        im, Z, crop = render_rgba(f"heldout_disp_{k}", crop=crop)
        hs.append(put_fit(cv, x, y + 1.6, w, 19.0, im) if REAL_BONE else put(cv, x, y + 1.6, w, im))
        if k == "fe":
            umax = float(Z["umax"])
    yb = y + 1.6 + max(hs) + 1.6
    cbar_mm(cv, ScalarMappable(Normalize(0, umax), DISP), 12.0, yb, 46.0, 1.3, "Displacement (mm)",
            ticks=mpl.ticker.MaxNLocator(4))
    cbar_mm(cv, ScalarMappable(Normalize(0, umax / 2), ERR), 74.0, yb, 21.4, 1.3, "Error (mm)",
            ticks=mpl.ticker.MaxNLocator(3))

    # ---- b: field accuracy (model colours of the house style) ----
    F = D["eval_fields"]
    xb = 104.0
    cv.panel(xb, 0.5, "b", "Field accuracy")
    models = ["comech_s0"] + [m for m in BASELINES if f"{m}|test" in F]
    mets = [("stress_l2_art", "Cartilage stress", "Error (relative L₂)", (0.1, 0.7), [0.2, 0.4, 0.6]),
            ("disp_l2", "Displacement", "Error (relative L₂)", (0.05, 0.7), [0.2, 0.4, 0.6]),
            ("hot_iou", "Hotspot overlap", "IoU, top 5%", (0.1, 0.75), [0.2, 0.4, 0.6])]
    for p, (key, title, xl, lim, ticks) in enumerate(mets):
        ax = cv.ax(xb + 18.5 + p * 20.6, 13.0, 17.0, 36.0)
        for i, m in enumerate(models):
            col = mcol(m)
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=PAPER, lw=0, zorder=0)
            for sp, filled, dy in [("test", True, -0.17), ("ood", False, 0.17)]:
                rows_ = F.get(f"{m}|{sp}")
                if not rows_:
                    continue
                q1, q3 = iqr(rows_, key)
                v = (np.mean([med(F[f"{s}|{sp}"], key) for s in SEEDS if f"{s}|{sp}" in F]) if m == "comech_s0"
                     else med(rows_, key))
                ax.plot([q1, q3], [i + dy] * 2, color=col, lw=0.8, solid_capstyle="butt", zorder=2)
                ax.scatter([v], [i + dy], s=14, marker=MK.get(m, "o"), color=col if filled else "white",
                           edgecolors=col if not filled else "white", linewidths=0.8 if not filled else 0.4, zorder=3)
        ax.set_ylim(len(models) - 0.5, -0.5); ax.set_yticks(range(len(models)))
        ax.set_yticklabels([NAMES[m] for m in models] if p == 0 else [""] * len(models))
        ax.set_xlim(*lim); ax.set_xticks(ticks)
        style(ax, left=False)
        if p == 0:
            for t_, m in zip(ax.get_yticklabels(), models):
                t_.set_color(mcol(m))
        ax.set_title(title, pad=4)
        ax.set_xlabel(xl, fontsize=FS)
    # v5: the longest model name (printed 1 pt larger) starts at the panel letter b; the three axes follow it
    _axs = [a for a in cv.fig.axes[-3:]]
    with at_print_size(cv, axes_y=_axs[:1]):
        _px = 25.4 / cv.fig.dpi
        _left = min(t_.get_window_extent().x0 for t_ in _axs[0].get_yticklabels() if t_.get_text()) * _px
    _shift = max(0.0, xb - _left)
    for a in _axs:
        pos_ = a.get_position()
        a.set_position([pos_.x0 + _shift / cv.w, pos_.y0, pos_.width, pos_.height])
    lx = xb + 18.5 + _shift
    for k, (filled, lab) in enumerate([(True, "Held-out feet (40)"), (False, "Out-of-range feet (20)")]):
        ax0 = lx + k * 27.0
        cv.fig.add_artist(Line2D([cv.fx(ax0 + 1.0)], [cv.fy(60.6)], marker="o", ms=3.4, ls="none",
                                 color=INK2 if filled else "white", mec=INK2 if not filled else "white",
                                 mew=0.8 if not filled else 0.4, transform=cv.fig.transFigure))
        cv.text(ax0 + 2.6, 61.6, lab, fontsize=FS)
    # the starred baselines are explained in the legend (Nature style; footnote removed 2026-11-03)

    # ---- c: speed (log scale; hardware named; fold difference between FE solve and CoMech field) ----
    L = D["eval_latency"]
    yr = 78.0
    cv.panel(0, yr, "c", "Time per configuration")
    ax = cv.ax(17.5, yr + 12.0, 22.5, 19.0)
    rows = [("FE solve", "one GPU", L["fe_solve_s_median"], FE),
            ("CoMech field", "one CPU core", L["rollout_full_field_s"], BLUE),
            ("CoMech effects", "per candidate", L["sweep200_effects_s"] / 200, BLUE)]
    for i, (nm, hw, v, c) in enumerate(rows):
        ax.axhline(i, color=GRID, lw=0.6, zorder=0)
        ax.scatter([v], [i], s=30, color=c, edgecolors="white", linewidths=0.5, zorder=4)
        lab = f"{v:.0f} s" if v >= 1 else (f"{1000 * v:.0f} ms" if v >= 0.01 else f"{1000 * v:.1f} ms")
        ax.text(v * 1.9, i, lab, va="center", ha="left", fontsize=FS_SMALL, color=INK)
        ax.text(-0.05, i - 0.10, nm, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=FS,
                color=c)
        ax.text(-0.05, i + 0.30, hw, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=FS_SMALL, color=INK2)
    ratio = L["fe_solve_s_median"] / L["rollout_full_field_s"]
    xa, xb_ = L["rollout_full_field_s"], L["fe_solve_s_median"]
    ya = 0.52
    ax.plot([xb_, xb_], [0.15, ya], color=MUTED, lw=0.45, ls=(0, (1, 1.2)), zorder=1)
    ax.plot([xa, xa], [0.85, ya], color=MUTED, lw=0.45, ls=(0, (1, 1.2)), zorder=1)
    ax.annotate("", xy=(xa, ya), xytext=(xb_, ya),
                arrowprops=dict(arrowstyle="<->,head_length=0.25,head_width=0.14", lw=0.55, color=INK,
                                shrinkA=0, shrinkB=0))
    ax.text(np.sqrt(xa * xb_), ya - 0.08, f"{ratio:.0f}-fold", ha="center", va="bottom", fontsize=FS_SMALL,
            color=INK)
    ax.set_xscale("log"); ax.set_xlim(2e-3, 400); ax.set_ylim(2.65, -0.55)
    ax.set_yticks([])
    ax.set_xticks([0.01, 1, 100]); ax.set_xticklabels(["0.01", "1", "100"])
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    style(ax, left=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (s, log scale)")

    # ---- d: atlas analogue (zero-shot) ----
    Zs = D["eval_zeroshot"]; Z = Zs["comech_s0"]
    s = Z["scores"]; Rw = Z["rows"]
    seeds = [m for m in SEEDS if m in Zs]
    xd = 44.5
    panel2(cv, xd, yr, "d", "Atlas analogue on the reference foot", "(never seen in training)")   # v5: two lines
    ax = cv.ax(xd + 10.0, yr + 14.0, 37.0, 18.5)
    x = np.array([0, 1, 2, 3, 4, 5.6, 6.6, 7.6, 8.6])
    fe = np.array([Rw[n]["fe"]["coal_p99"] for n in COALS])
    pm = np.array([Rw[n]["pred"]["coal_p99"] for n in COALS])          # seed 0, as in the legend
    ax.axvspan(-0.55, 4.55, color=PAPER, lw=0, zorder=0)
    ax.axvspan(5.05, 9.15, color="#F4F7FB", lw=0, zorder=0)
    for i in range(len(x)):
        ax.plot([x[i]] * 2, [pm[i], fe[i]], color=BASE, lw=0.9, zorder=1, solid_capstyle="butt")
    ax.scatter(x, fe, s=17, facecolors="white", edgecolors=FE, linewidths=0.85, zorder=4)
    ax.scatter(x, pm, s=16, color=BLUE, edgecolors="white", linewidths=0.4, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(["I", "II", "III", "IV", "V", "I", "II", "III", "IV"])
    ax.set_ylim(0, 15.5); ax.set_yticks([0, 5, 10, 15]); ax.set_xlim(-0.55, 9.15)
    style(ax)
    ax.tick_params(axis="x", length=0, pad=1.2)
    ax.set_ylabel("Bridge stress (MPa)")
    ratio_med = np.mean([np.median([Zs[m]["rows"][n]["pred"]["coal_p99"] / Zs[m]["rows"][n]["fe"]["coal_p99"]
                                    for n in COALS]) for m in seeds])
    taus = [Zs[m]["scores"]["tau_all_vs_fe"] for m in seeds]
    ax.text(9.05, 15.3, f"median ratio {100 * ratio_med:.0f}%\nKendall τ {np.mean(taus):.2f}".replace("-", "−"),
            ha="right", va="top", fontsize=FS_SMALL, color=INK2, linespacing=1.2)
    # coalition-location icons (engine renderings) under the tick labels
    names_icon = ["icon_coal_RozI", "icon_coal_RozII", "icon_coal_RozIII", "icon_coal_RozIV", "icon_coal_RozV",
                  "icon_coal_I", "icon_coal_II", "icon_coal_III", "icon_coal_IV"]
    x0_mm, w_mm = xd + 10.0, 37.0
    xmm = lambda xv: x0_mm + (xv + 0.55) / 9.7 * w_mm
    iw = 3.5
    have_icons = False
    for xv, nm in zip(x, names_icon):
        im = icon(nm)
        if im is None:
            continue
        have_icons = True
        ih = iw * im.shape[0] / im.shape[1]
        cv.image(xmm(xv) - iw / 2, yr + 14.0 + 18.5 + 4.3, iw, ih, im)
    yg = yr + 14.0 + 18.5 + (4.3 + 3.2 + 2.6 if have_icons else 4.6)
    for (a_, b_, nm) in [(0, 4, "Rozansky"), (5.6, 8.6, "Anatomy-based")]:
        cv.line([xmm(a_) - 1.2, xmm(b_) + 1.2], [yg - 2.0, yg - 2.0], color=INK2, lw=0.5)
        cv.text((xmm(a_) + xmm(b_)) / 2, yg + 0.4, nm, ha="center", fontsize=FS)
    cv.fig.add_artist(Line2D([cv.fx(xd + 11.0)], [cv.fy(yr + 9.6)], marker="o", ms=3.6, ls="none", mfc="white",
                             mec=FE, mew=0.85, transform=cv.fig.transFigure))
    cv.text(xd + 12.6, yr + 10.3, "FE", fontsize=FS)
    cv.fig.add_artist(Line2D([cv.fx(xd + 18.0)], [cv.fy(yr + 9.6)], marker="o", ms=3.6, ls="none", color=BLUE,
                             mec="white", mew=0.4, transform=cv.fig.transFigure))
    cv.text(xd + 19.6, yr + 10.3, "CoMech (seed 0)", fontsize=FS)
    # defect series: overload zone above the 20% threshold, FE and CoMech tolerance crossings
    ax = cv.ax(xd + 60.5, yr + 14.0, 23.0, 18.5)
    dd = np.array([0, 10, 20, 30, 40, 50, 60, 70])
    curves = {}
    for fac in ["post", "mid"]:
        yf = np.array(s[f"curve_{fac}_fe"]); yf = 100 * (yf / yf[0] - 1)
        yp = np.mean([np.array(Zs[m]["scores"][f"curve_{fac}_effhead"]) / Zs[m]["scores"][f"curve_{fac}_effhead"][0]
                      for m in seeds], 0)
        curves[fac] = (yf, 100 * (yp - 1))
    top_ = max(60.0, 1.15 * max(max(a.max(), b.max()) for a, b in curves.values()) + 5)
    lo_ = min(-15.0, min(min(a.min(), b.min()) for a, b in curves.values()) - 8)
    ax.axhspan(20, top_, color=ZONE_OUT, lw=0, zorder=0)
    ax.axhline(20, color=RED_D, lw=0.6, ls=(0, (2.2, 1.4)), zorder=1)
    ax.text(-2.0, 20 + 2.0, "+20%", fontsize=FS_SMALL, color=RED_D, ha="left", va="bottom")
    for fac, nm in [("post", "Posterior"), ("mid", "Middle")]:
        col = FACET[fac]; yf, yp = curves[fac]
        ax.plot(dd, yf, color=col, lw=0.8, ls=(0, (3, 1.6)), marker="o", ms=2.6, mfc="white", mec=col, mew=0.6,
                zorder=3)
        ax.plot(dd, yp, color=col, lw=1.0, marker="o", ms=2.5, mec="white", mew=0.3, zorder=4)
        tf = 100 * s[f"tol_{fac}_fe"]; tp = [100 * Zs[m]["scores"][f"tol_{fac}_effhead"] for m in seeds]
        ax.plot([tf], [20], marker="D", ms=3.0, mfc="white", mec=col, mew=0.8, zorder=6, ls="none")
        ax.plot([min(tp), max(tp)], [20, 20], color=col, lw=2.2, alpha=0.55, solid_capstyle="butt", zorder=5)
        ax.plot([np.mean(tp)], [20], marker="D", ms=3.0, color=col, mec="white", mew=0.3, zorder=6, ls="none")
    # middle facet: the overestimate along the threshold
    tf = 100 * s["tol_mid_fe"]; tpm = np.mean([100 * Zs[m]["scores"]["tol_mid_effhead"] for m in seeds])
    ax.annotate("", xy=(tpm - 1.5, 20 + 0.55 * (top_ - 20)), xytext=(tf + 1.5, 20 + 0.55 * (top_ - 20)),
                arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.16", lw=0.6, color=FACET["mid"],
                                shrinkA=0, shrinkB=0))
    ax.plot([tf, tf], [20, 20 + 0.55 * (top_ - 20)], color=FACET["mid"], lw=0.4, ls=(0, (1, 1)))
    ax.plot([tpm, tpm], [20, 20 + 0.55 * (top_ - 20)], color=FACET["mid"], lw=0.4, ls=(0, (1, 1)))
    ax.text((tf + tpm) / 2, 20 + 0.58 * (top_ - 20), "middle:\noverestimated", ha="center", va="bottom",
            fontsize=FS_SMALL, color=FACET["mid"], linespacing=1.1)
    ax.set_xlim(-3, 73); ax.set_xticks([0, 35, 70]); ax.set_ylim(lo_, top_)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50 if top_ > 120 else 25))
    style(ax)
    ax.set_xlabel("Facet defect (%)"); ax.set_ylabel("Residual-facet mean\n(% change)", linespacing=1.2)
    ax.set_title("Defect series", pad=3)
    yl = yr + 14.0 + 18.5 + 9.4
    for k, (fac, nm) in enumerate([("post", "Posterior"), ("mid", "Middle")]):
        xx = xd + 52.0 + k * 15.5
        cv.line([xx, xx + 3.2], [yl - 0.7, yl - 0.7], color=FACET[fac], lw=1.0)
        cv.text(xx + 4.0, yl, nm, fontsize=FS_SMALL)
    cv.line([xd + 52.0, xd + 55.2], [yl + 1.8, yl + 1.8], color=INK2, lw=0.8, ls=(0, (3, 1.6)))
    cv.text(xd + 56.0, yl + 2.5, "FE", fontsize=FS_SMALL)
    cv.line([xd + 67.5, xd + 70.7], [yl + 1.8, yl + 1.8], color=INK2, lw=1.0)
    cv.text(xd + 71.5, yl + 2.5, "CoMech", fontsize=FS_SMALL)

    # ---- e: label-free taxonomy (unchanged content) ----
    T = D["eval_taxonomy"]
    xe = 131.0
    panel2(cv, xe, yr, "e", "Label-free taxonomy", "(240 coalitions)")   # v5: two lines (right edge)
    at = np.array(T["labels"]["atype"]); emb = np.array(T["comech_s0"]["latent"]["anatomy"]["emb"])
    ax = cv.ax(xe + 2.5, yr + 14.0, 19.0, 19.0)
    for t in ["I", "II", "III", "IV"]:
        m = at == t
        ax.scatter(emb[m, 0], emb[m, 1], s=4.0, color=TYPE[t], marker=TYPE_MARKER[t], linewidths=0, alpha=0.9)
    hide(ax)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(True); ax.spines[sp].set_color(BASE)
    ax.set_xlabel("PC 1", labelpad=1.5, fontsize=FS); ax.set_ylabel("PC 2", labelpad=1.5, fontsize=FS)
    ax.set_title("CoMech latent change", pad=4, fontsize=FS)
    _col2 = 1.4 + max(tw_mm(cv, f"Type {t_}", FS_SMALL) for t_ in ["I", "III"]) + 2.0   # v5: printed widths
    for k, t in enumerate(["I", "II", "III", "IV"]):
        xx = xe + 3.0 + (k % 2) * _col2; yy = yr + 38.6 + (k // 2) * 2.6
        cv.fig.add_artist(Line2D([cv.fx(xx)], [cv.fy(yy - 0.7)], marker=TYPE_MARKER[t], ms=2.8, ls="none",
                                 color=TYPE[t], mew=0, transform=cv.fig.transFigure))
        cv.text(xx + 1.4, yy, f"Type {t}", fontsize=FS_SMALL)
    ax = cv.ax(xe + 26.0, yr + 14.0, 24.0, 21.0)
    reps = [("geometry", "Bridge geometry"), ("fe_mechanics", "FE fingerprint"), ("mechanics", "CoMech fingerprint"),
            ("latent", "CoMech latent")]
    for i, (r, nm) in enumerate(reps):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.55, color=PAPER, lw=0, zorder=0)
        ax.text(0.0, i - 0.12, nm, fontsize=FS_SMALL, color=INK, ha="left", va="bottom")
        for task, col, dy in [("anatomy", TAXO["anatomy"], 0.12), ("rozansky", TAXO["rozansky"], 0.34)]:
            if r in ("geometry", "fe_mechanics"):
                v, e = T[r][task]["ari"], 0.0
            else:
                vv = [T[sd][r][task]["ari"] for sd in SEEDS if sd in T]
                v, e = np.mean(vv), np.std(vv)
            if e:
                ax.plot([v - e, v + e], [i + dy] * 2, color=col, lw=0.7, zorder=2)
            ax.scatter([v], [i + dy], s=12, color=col, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_yticks([]); ax.set_ylim(len(reps) - 0.45, -0.75); ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    style(ax, left=False)
    ax.set_xlabel("Adjusted Rand index", fontsize=FS)
    for k, (col, lab) in enumerate([(TAXO["anatomy"], "Anatomy-based types"), (TAXO["rozansky"], "Rozansky types")]):
        xx = xe + 26.5; yy = yr + 8.6 + k * 2.6
        cv.fig.add_artist(Line2D([cv.fx(xx)], [cv.fy(yy - 0.7)], marker="o", ms=3.0, ls="none", color=col, mew=0,
                                 transform=cv.fig.transFigure))
        cv.text(xx + 1.4, yy, lab, fontsize=FS_SMALL)
    save(cv, "fig5")


# ============================== Figure 6 ==============================
def fig6(D):
    cv = Canvas(FIG_W, 118.0)    # no footnote: the legend explains the starred baselines (2026-11-03)
    TP = D["eval_tipping"]; fe = TP["fe"]; ids = sorted(fe, key=int)

    # ---------------- a: tolerable defect, FE against CoMech (joint plot, both facets) ----------------
    cv.panel(0, 0.5, "a", "Tolerable defect, 20 held-out feet")
    # row 1: the data areas of a, b and c share their top (16 mm) and bottom (50 mm)
    AX, AY, AW = 11.0, 16.0, 34.0
    data = {}
    for fac in ["post", "mid"]:
        a = np.array([fe[i][fac]["tol"] for i in ids]) * 100
        Pp = np.array([[TP["models"][m]["pred"][i][fac]["tol"] for i in ids] for m in SEEDS]) * 100
        data[fac] = (a, Pp.mean(0), Pp.std(0))
    allv = np.r_[[np.r_[a, b - e, b + e] for a, b, e in data.values()]].ravel()
    lim, ticks = nice_lim(allv, 10.0, pad=3.0)
    ax = cv.ax(AX, AY, AW, AW)
    xx_ = np.array(lim)
    ax.fill_between(xx_, xx_ - 5, xx_ + 5, color=GRID, lw=0, zorder=0)
    ax.plot(lim, lim, color=MUTED, lw=0.6, zorder=1)
    for fac in ["post", "mid"]:
        a, b, e = data[fac]; col = FACET[fac]
        ax.errorbar(a, b, yerr=e, fmt="none", ecolor=col, elinewidth=0.5, alpha=0.6, zorder=2)
        ax.scatter(a, b, s=12, color=col, edgecolors="white", linewidths=0.4, zorder=3,
                   marker="o" if fac == "mid" else "s")
    # middle-facet bias: bracket from the identity line to the mean prediction, right of the cluster
    a, b, e = data["mid"]
    xm_, bias = float(np.mean(a)), float(np.mean(b - a))
    xa_ = float(np.max(a)) + 4.0
    ax.plot([xm_, xa_], [xm_, xm_], color=FACET["mid"], lw=0.4, ls=(0, (1, 1)), zorder=2)
    ax.plot([xm_ + 2.0, xa_], [xm_ + bias, xm_ + bias], color=FACET["mid"], lw=0.4, ls=(0, (1, 1)), zorder=2)
    ax.annotate("", xy=(xa_, xm_ + bias), xytext=(xa_, xm_),
                arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.16", lw=0.7, color=FACET["mid"],
                                shrinkA=0, shrinkB=0))
    ax.text(xa_ + 1.2, xm_ + bias / 2, f"bias\n{bias:+.1f} pp".replace("-", "−"), ha="left", va="center",
            fontsize=FS_SMALL, color=FACET["mid"], linespacing=1.15)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_xticks(ticks); ax.set_yticks(ticks)
    style(ax)
    ax.set_xlabel("FE tolerance (% of facet)"); ax.set_ylabel("CoMech tolerance (% of facet)")
    # error summaries as direct labels next to each cluster
    for fac, nm, xy, ha in [("mid", "Middle facet", (0.03, 0.97), "left"), ("post", "Posterior facet", (0.97, 0.285), "right")]:
        a, b, e = data[fac]
        loo = np.array([(a.sum() - x) / (len(a) - 1) for x in a])
        mae, cm = np.mean(np.abs(a - b)), np.mean(np.abs(a - loo))
        ax.text(*xy, nm, transform=ax.transAxes, ha=ha, va="top", fontsize=FS, color=FACET[fac])
        ax.text(xy[0], xy[1] - 0.075, f"error {mae:.1f} pp\nmean of other feet {cm:.1f}", transform=ax.transAxes,
                ha=ha, va="top", fontsize=FS_SMALL, color=INK2, linespacing=1.2)
    ax.text(lim[0] + 2.0, lim[0] + 7.6, "±5 pp", fontsize=FS_SMALL, color=INK2, ha="left", va="bottom")
    # marginal densities: FE (top), CoMech (right)
    grid = np.linspace(lim[0], lim[1], 300)
    axt = cv.ax(AX, AY - 6.0, AW, 5.4)
    marginal(axt, [data["post"][0], data["mid"][0]], [FACET["post"], FACET["mid"]], grid, "x")
    axt.set_xlim(*lim)
    axt.text(0.0, 0.95, "FE", transform=axt.transAxes, fontsize=FS_SMALL, color=INK2, va="top")
    axr = cv.ax(AX + AW + 0.8, AY, 5.4, AW)
    marginal(axr, [data["post"][1], data["mid"][1]], [FACET["post"], FACET["mid"]], grid, "y")
    axr.set_ylim(*lim)
    axr.text(0.95, 1.0, "CoMech", transform=axr.transAxes, fontsize=FS_SMALL, color=INK2, va="bottom", ha="center",
             rotation=90)

    # ---------------- b: osteotomy and posterior tolerance, per foot ----------------
    xb = 63.5
    panel2(cv, xb, 0.5, "b", "Osteotomy and", "posterior tolerance")   # v5: two lines (letter c follows)
    X = D["numbers"]["xrate_fe_check"]
    # left-aligned with the header (2026-11-03): group labels start at the header's x, the axes follow them
    BX0, BRIGHT, BY_, BH_ = xb + TITLE_DX, xb + 41.0, 16.0, 34.0
    ax = cv.ax(xb + 12.5, BY_, BRIGHT - (xb + 12.5), BH_)
    ys, labs, k_ok, n_ok, oks, centres = [], [], 0, 0, [], []
    blocks = [(0, "Lateralizing", "−5 mm"), (2, "Medializing", "+5 mm")]
    rowy = 0.0
    for bi, (j, nm, mm_) in enumerate(blocks):
        y0b = rowy
        for q, x_ in enumerate(X):
            fe3 = 100 * np.array(x_["fe"]); pr3 = 100 * np.array(x_["pred"])
            dfe, dpr = fe3[j] - fe3[1], pr3[j] - pr3[1]
            cens = (pr3[j] >= 69.95) or (pr3[1] >= 69.95)
            if q % 2 == 0:
                ax.axhspan(rowy - 0.5, rowy + 0.5, color=PAPER, lw=0, zorder=0)
            ax.plot([dfe, dpr], [rowy, rowy], color=BASE, lw=0.8, zorder=1, solid_capstyle="butt")
            ax.scatter([dfe], [rowy], s=15, facecolors="white", edgecolors=FE, linewidths=0.8, zorder=3)
            if cens:
                ax.scatter([dpr], [rowy], s=13, marker="D", facecolors="white", edgecolors=BLUE, linewidths=0.7,
                           zorder=4)
            else:
                ax.scatter([dpr], [rowy], s=15, color=BLUE, edgecolors="white", linewidths=0.4, zorder=4)
            ok = np.sign(round(dfe, 6)) == np.sign(round(dpr, 6)) and abs(dfe) > 1e-6
            k_ok += int(ok); n_ok += 1; oks.append((rowy, ok))
            ys.append(rowy); labs.append(str(q + 1))
            rowy += 1.0
        centres.append(((y0b + rowy - 1) / 2, f"{nm}\n{mm_}"))
        rowy += 0.9
    ax.axvline(0, color=INK2, lw=0.5, zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=FS_SMALL)
    ax.set_ylim(rowy - 0.5 - 0.9, -0.7)
    v_all = []
    for x_ in X:
        fe3 = 100 * np.array(x_["fe"]); pr3 = 100 * np.array(x_["pred"])
        v_all += list(fe3[[0, 2]] - fe3[1]) + list(pr3[[0, 2]] - pr3[1])
    xl_, xt_ = nice_lim(v_all, 10.0, pad=1.0)
    ax.set_xlim(*xl_); ax.set_xticks(xt_)
    style(ax, left=False)
    ax.tick_params(axis="y", pad=0.8)
    ax.set_xlabel("Change in tolerance (pp)")
    # group labels from the header's left edge; then the axes are moved so that the row numbers follow them
    ytop, ybot = -0.7, rowy - 0.5 - 0.9
    rot = [cv.text(BX0, BY_ + (yc - ytop) / (ybot - ytop) * BH_, lab, ha="left", va="center", rotation=90,
                   fontsize=FS_SMALL, color=INK, linespacing=1.15) for yc, lab in centres]
    with at_print_size(cv, texts=rot, axes_y=[ax]):          # v5: measured at the printed sizes
        rr_ = cv.fig.canvas.get_renderer(); px_mm = 25.4 / cv.fig.dpi
        rot_right = max(t_.get_window_extent(rr_).x1 for t_ in rot) * px_mm
        lab_left = min(t_.get_window_extent(rr_).x0 for t_ in ax.get_yticklabels()) * px_mm
    new_left = ax.get_position().x0 * cv.w + (rot_right + 1.0 - lab_left)
    ax.set_position([new_left / cv.w, ax.get_position().y0, (BRIGHT - new_left) / cv.w, ax.get_position().height])
    xm_ = 1.0 + 1.8 / (BRIGHT - new_left)
    for yy_, ok in oks:   # direction predicted correctly (tick) or not (cross): vector marks
        ax.plot([xm_], [yy_], transform=ax.get_yaxis_transform(), ls="none", marker=CHECK if ok else CROSS,
                ms=4.0 if ok else 3.4, mfc="none", mec=GREEN if ok else VERM, mew=0.9, clip_on=False, zorder=5)
    ax.text(1.0, -1.75, "sign", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=FS_SMALL,
            color=INK2, clip_on=False)
    ax.text(1.0, -1.0, f"{k_ok}/{n_ok}", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=FS_SMALL, color=INK2, clip_on=False)
    yl = 59.6
    for k, (h_, lab) in enumerate([(dict(marker="o", mfc="white", mec=FE, mew=0.8), "FE"),
                                   (dict(marker="o", color=BLUE, mec="white", mew=0.4), "CoMech (seed 0)"),
                                   (dict(marker="D", mfc="white", mec=BLUE, mew=0.7), "CoMech at the 70% end of its sweep")]):
        xx = BX0 + 0.85 + (k % 2) * 14.0 if k < 2 else BX0 + 0.85
        yy = yl + (0 if k < 2 else 2.5)
        cv.fig.add_artist(Line2D([cv.fx(xx)], [cv.fy(yy - 0.7)], ls="none", ms=3.2 if k < 2 else 2.8,
                                 transform=cv.fig.transFigure, **h_))
        cv.text(xx + 1.4, yy, lab, fontsize=FS_SMALL)

    # ---------------- c: virtual cohort against the 50% rule ----------------
    CF = D["eval_cohort_fe"]
    xc = 113.0
    cv.panel(xc, 0.5, "c", "The 50% rule in a virtual cohort (135 feet)")
    CX, CY, CW, CH = xc + 11.5, 16.0, 29.5, 34.0
    ax = cv.ax(CX, CY, CW, CH)
    rows_all = []
    for r in CF["rows"]:
        v = [(r[f"fe_tol_{f}"] - r[f"d_{f}"]) / r[f"fe_tol_{f}"] for f in ["mid", "post"] if f"fe_tol_{f}" in r]
        rows_all.append((r, 100 * min(v) if v else 100.0))
    _lo = min(-300.0, 100.0 * np.floor(min(v for _, v in rows_all) / 100.0))
    _xmax = max(100.0, 10.0 * np.ceil(10.0 * max(r["ratio"] for r, _ in rows_all)))
    ylo, yhi = _lo - 20, 125
    ax.add_patch(Rectangle((0, ylo), 50, 0 - ylo, color=CAT_TINT["permissive"], lw=0, zorder=0))
    ax.add_patch(Rectangle((50, 0), _xmax - 50, yhi, color=CAT_TINT["strict"], lw=0, zorder=0))
    ax.axvline(50, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color=INK2, lw=0.5, zorder=1)
    rng = np.random.default_rng(1)
    for t in ["III", "I", "II", "IV"]:
        rr = [(r, v) for r, v in rows_all if r["ctype"] == t]
        ax.scatter([100 * r["ratio"] for r, _ in rr], np.clip([v for _, v in rr], _lo, 100) + rng.uniform(-3, 3, len(rr)),
                   s=9, color=TYPE[t], marker=TYPE_MARKER[t], edgecolors="white", linewidths=0.3, zorder=3,
                   label=f"Type {t}")
    ax.set_xlim(0, _xmax); ax.set_ylim(ylo, yhi); ax.set_yticks(np.arange(_lo, 101, 100))
    ax.set_xticks([0, 50, 100] if _xmax <= 120 else [0, 50, 100, 150])
    style(ax)
    ax.text(1.02, 0.5 * (yhi) / (yhi - ylo) + (0 - ylo) / (yhi - ylo), "within", transform=ax.transAxes, rotation=90,
            fontsize=FS_SMALL, color=GREEN_D, ha="left", va="center")
    ax.text(1.02, 0.5 * (0 - ylo) / (yhi - ylo), "beyond envelope", transform=ax.transAxes, rotation=90,
            fontsize=FS_SMALL, color=RED_D, ha="left", va="center")
    ax.set_xlabel("Coalition / posterior facet (%)")
    ax.set_ylabel("FE-verified reserve (%)")
    h, l = ax.get_legend_handles_labels()
    order = [l.index(f"Type {t}") for t in ["I", "II", "III", "IV"]]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1,
              handletextpad=0.1, markerscale=1.2, borderaxespad=0.3, labelspacing=0.2, fontsize=FS_SMALL)
    xm = lambda v: CX + v / _xmax * CW
    for a_, b_, lab in [(0, 50, "resect alone"), (50, _xmax, "do not resect alone")]:
        cv.line([xm(a_) + 0.4, xm(b_) - 0.4], [CY - 1.6, CY - 1.6], color=INK2, lw=0.5)
        cv.text((xm(a_) + xm(b_)) / 2, CY - 2.4, lab, ha="center", fontsize=FS_SMALL, color=INK2)
    cv.text(CX - 1.0, CY - 5.6, "50% rule:", ha="left", fontsize=FS_SMALL, color=INK)
    # per type: share of feet in each category
    BX, BY, BW, BH = xc + 55.0, 16.0, 11.0, 19.0   # v5: 3 mm right of the 2026-11-03 layout, 1.5 mm narrower
    axb = cv.ax(BX, BY, BW, BH)
    by = {}
    for t in ["I", "II", "III", "IV"]:
        rr = [r for r in CF["rows"] if r["ctype"] == t]
        perm = sum(1 for r in rr if r["rule_ok"] and not r["fe_within"])
        strict = sum(1 for r in rr if (not r["rule_ok"]) and r["fe_within"])
        by[t] = (len(rr), len(rr) - perm - strict, perm, strict)
    for i, t in enumerate(["I", "II", "III", "IV"]):
        n, ag, pe, st = by[t]
        left = 0.0
        for v, key in [(pe, "permissive"), (ag, "agree"), (st, "strict")]:
            if v:
                axb.barh(i, 100 * v / n, left=left, height=0.64, color=CAT[key], edgecolor="white", lw=0.5)
                if key != "agree" and 100 * v / n >= 18:
                    axb.text(left + 50 * v / n, i, f"{v}", va="center", ha="center", fontsize=FS_SMALL, color="white")
            left += 100 * v / n
        axb.text(103, i, f"{n}", va="center", ha="left", fontsize=FS_SMALL, color=INK2)
    axb.set_yticks(range(4)); axb.set_yticklabels([f"Type {t}" for t in ["I", "II", "III", "IV"]], fontsize=FS_SMALL)
    for t_, t in zip(axb.get_yticklabels(), ["I", "II", "III", "IV"]):
        t_.set_color(TYPE[t])
    axb.set_ylim(3.55, -0.55); axb.set_xlim(0, 100); axb.set_xticks([0, 50, 100])
    style(axb, left=False)
    axb.tick_params(axis="y", pad=0.6)
    axb.set_xlabel("Feet (%)", fontsize=FS_SMALL, labelpad=1.0)
    axb.text(103, -0.95, "n", ha="left", va="center", fontsize=FS_SMALL, color=INK2)
    S = CF["summary"]["fe_within"]
    LX = xc + 47.0                                   # key of the bars: position of the 2026-11-03 layout
    cv.text(LX, BY + BH + 9.4, "50% rule vs envelope:", fontsize=FS_SMALL, color=INK)
    for k, (key, lab) in enumerate([("permissive", f"too permissive ({S['mis_permissive']})"),
                                    ("agree", "agree"),
                                    ("strict", f"too strict ({S['mis_strict']})")]):
        yy = BY + BH + 11.9 + k * 2.4
        cv.fig.add_artist(Rectangle((cv.fx(LX), cv.fy(yy - 0.15)), 2.4 / cv.w, 1.5 / cv.h, color=CAT[key],
                                    transform=cv.fig.transFigure, lw=0))
        cv.text(LX + 3.2, yy, lab, fontsize=FS_SMALL)


    # ---------------- d: mechanically equivalent operations (precision-recall plane) ----------------
    E = D["eval_equiv_plan"]; EQ = E["equivalence"]
    yd = 66.0
    cv.panel(0, yd, "d", "Mechanically equivalent operations")
    ax = cv.ax(11.5, yd + 9.5, 34.0, 34.0)
    XL, YL = (0.80, 0.97), (0.3, 1.0)
    r_ = np.linspace(0.5, 1.0, 400)
    for f1 in (0.6, 0.7, 0.8, 0.9):
        p_ = f1 * r_ / (2 * r_ - f1); ok = (p_ > YL[0]) & (p_ <= YL[1]) & (r_ >= XL[0]) & (r_ <= XL[1])
        ax.plot(r_[ok], p_[ok], color="#E3E3E3", lw=0.6, zorder=0)
        k_ = np.nonzero(ok)[0][-1]
        ax.text(r_[k_] - 0.002, p_[k_] - 0.014, f"F₁ {f1:.1f}", fontsize=FS_SMALL, color=MUTED, ha="right", va="top")
    models = ["comech_s0"] + [m for m in BASELINES if m in EQ]
    pts = {}
    for m in models:
        col = mcol(m)
        if m == "comech_s0":
            pr = np.array([EQ[s_]["precision"] for s_ in SEEDS]); rc = np.array([EQ[s_]["recall"] for s_ in SEEDS])
            ax.scatter(rc, pr, s=9, facecolors="none", edgecolors=col, linewidths=0.6, zorder=6)
            ax.plot([rc.mean() - rc.std(), rc.mean() + rc.std()], [pr.mean()] * 2, color=col, lw=0.8, zorder=4)
            ax.plot([rc.mean()] * 2, [pr.mean() - pr.std(), pr.mean() + pr.std()], color=col, lw=0.8, zorder=4)
            ax.scatter([rc.mean()], [pr.mean()], s=22, color=BLUE_L, edgecolors=col, linewidths=0.6, zorder=5)
            pts[m] = (rc.mean(), pr.mean())
        else:
            pts[m] = (EQ[m]["recall"], EQ[m]["precision"])
            ax.scatter([pts[m][0]], [pts[m][1]], s=20, marker=MK[m], color=col, edgecolors="white", linewidths=0.4,
                       zorder=5)
    lab_xy = {"comech_s0": (0.806, 0.975, "left"), "actcat": (0.935, 0.985, "left"), "geo": (0.868, 0.835, "center"),
              "deeponet": (0.913, 0.705, "right"), "fno": (0.888, 0.39, "left")}
    for m, (lx, ly, ha) in lab_xy.items():
        if m not in pts:
            continue
        col = mcol(m)
        ax.text(lx, ly, NAMES[m], fontsize=FS_SMALL, color=col, ha=ha, va="center", zorder=6)
        if m in ("comech_s0", "actcat", "geo"):
            ex = {"comech_s0": lx + 0.022, "actcat": lx - 0.002, "geo": lx + 0.006}[m]
            ey = {"comech_s0": ly - 0.012, "actcat": ly, "geo": ly + 0.016}[m]
            ax.plot([pts[m][0], ex], [pts[m][1], ey], color=col, lw=0.35, zorder=4)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_xticks([0.8, 0.85, 0.9, 0.95]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.80", "0.85", "0.90", "0.95"])
    style(ax)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    axd = ax
    cv.text(11.5 + 34.0 / 2, yd + 7.4, "CoMech: three seeds (open) and their mean ± s.d.", ha="center",
            fontsize=FS_SMALL, color=INK2)          # centred over the plot (2026-11-03)

    # ---------------- e: ablation, 2 x 2 factorial tiles (seed 0) ----------------
    F = D["eval_fields"]
    xe = 62.0
    cv.panel(xe, yd, "e", "Ablation of the effect-grounded constraints (seed 0)")
    cells = [["comech_s0", "comech_nosens"], ["comech_noswap", "comech_none"]]
    metrics = [("Equivalence recall", lambda m: EQ[m]["recall"], "{:.2f}", "high", 1.0),
               ("Equivalence precision", lambda m: EQ[m]["precision"], "{:.2f}", "high", 1.0),
               ("Cartilage stress error (%)", lambda m: med(F[f"{m}|test"], "stress_l2_art"), "{:.1f}", "low", 100.0),
               ("Posterior tolerance error (pp)", lambda m: float(np.mean(TP["models"][m]["tol_err_pp"]["post"])),
                "{:.1f}", "low", 1.0)]
    # aligned with d: key on d's note line, metric names starting at the top of d's plot, tile bottoms on d's x-axis
    # line (style() sets it 2 pt below the data area; the cells' white 1.6-pt edges take 0.8 pt off the visible tile)
    # and the per-seed line on d's tick-label row
    TX, TW, TH, TG = xe + 4.0, 24.0, 25.5, 5.6
    TB = yd + 9.5 + 34.0 + (2.0 + 0.8) / 72 * 25.4
    TY = TB - TH
    TT = yd + 9.5 + cap_mm(FS_SMALL)                        # baseline whose cap height starts at the top of d's plot
    with at_print_size(cv, axes_x=[axd]):                  # v5: d's tick labels at their printed size
        bb = axd.get_xticklabels()[0].get_window_extent()
        y_tick = (1 - 0.5 * (bb.y0 + bb.y1) / cv.fig.bbox.height) * cv.h
    for k, (nm, fn, fmt, better, sc) in enumerate(metrics):
        M = [[sc * fn(cells[i][j]) for j in range(2)] for i in range(2)]
        vals = np.array(M).ravel()
        span = max(np.ptp(vals), 1e-9)
        if better == "high":
            cmap, norm = BLUES, Normalize(vals.min() - 0.25 * span, vals.max() + 0.10 * span)
        else:
            cmap, norm = BLUES.reversed(), Normalize(vals.min() - 0.10 * span, vals.max() + 0.25 * span)
        x = TX + k * (TW + TG)
        ax = cv.ax(x, TY, TW, TH)
        tiles2x2(ax, M, cmap, norm, lambda v, f=fmt: f.format(v))
        cv.text(x + TW / 2, TT, nm, ha="center", fontsize=FS_SMALL, color=INK)
        cv.text(x + TW / 2, TT + 2.5, ("higher" if better == "high" else "lower") + " is better", ha="center",
                fontsize=FS_SMALL, color=INK2)
        cv.text(x + TW / 4, TY - 0.8, "+", ha="center", fontsize=FS, color=INK2)
        cv.text(x + 3 * TW / 4, TY - 0.8, "−", ha="center", fontsize=FS, color=INK2)
        cv.text(x - 0.9, TY + TH / 4 + 0.75, "+", ha="right", fontsize=FS, color=INK2)
        cv.text(x - 0.9, TY + 3 * TH / 4 + 0.75, "−", ha="right", fontsize=FS, color=INK2)
        vs = np.array([sc * fn(s_) for s_ in SEEDS])
        cv.text(x + TW / 2, y_tick, f"per seed: {fmt.format(vs.mean())} ± {fmt.format(vs.std())}",
                ha="center", va="center", fontsize=FS_SMALL, color=INK2)
    # key centred over the tiles, as the note of d over its plot (2026-11-03)
    key = cv.text(TX, yd + 7.4, "Rows: effect-swap; columns: action sensitivity; + with, − without;",
                  fontsize=FS_SMALL, color=INK2)
    fm_ = cv.text(TX, yd + 7.4, "full model", fontsize=FS_SMALL, color=INK2)
    ks = 1.8
    w1 = tw_mm(cv, key.get_text(), FS_SMALL)               # v5: printed widths
    w2 = tw_mm(cv, fm_.get_text(), FS_SMALL)
    k0 = TX + (4 * TW + 3 * TG) / 2 - (w1 + 1.3 + ks + 1.0 + w2) / 2
    key.set_x(cv.fx(k0))
    kx = k0 + w1 + 1.3
    kc = yd + 7.4 - 0.5 * cap_mm(FS_SMALL)
    cv.fig.add_artist(Rectangle((cv.fx(kx), cv.fy(kc + ks / 2)), ks / cv.w, ks / cv.h, fc="white", ec=INK, lw=0.9,
                                transform=cv.fig.transFigure))
    fm_.set_x(cv.fx(kx + ks + 1.0))
    save(cv, "fig6")


# ============================== Figure 7 ==============================
def fig7(D):
    from maps import child_map
    cv = Canvas(FIG_W, 131)
    Mp = D["eval_maps"]["child"]
    fac = Mp.get("facet", "post"); fname = {"post": "posterior", "mid": "middle"}[fac]
    t, ip, fu = Mp["choice"]; side = "medializing" if t > 0 else "lateralizing"
    align = Mp["foot"]["align"]
    _tol = f"tolerance {100 * Mp['tol_' + fac]:.0f}%"
    if REAL_BONE:
        _fr = [r for r in (D.get("eval_cohort_fe") or {}).get("rows", []) if r.get("i") == Mp.get("i")]
        if _fr and f"fe_tol_{fac}" in _fr[0]:
            _tol = f"tolerance {100 * Mp['tol_' + fac]:.0f}% predicted, {100 * _fr[0]['fe_tol_' + fac]:.0f}% by FE"
    cv.panel(0, 0.5, "a", f"Virtual child: type {Mp['foot']['ctype']} coalition, {abs(align):.0f}° "
             f"{'valgus' if align > 0 else 'varus'}; complete resection leaves a {100 * Mp['d_' + fac]:.0f}% "
             f"{fname}-facet defect ({_tol})")
    mk = np.load(os.path.join(ROOT, "results", "map_child_masks.npz"))
    fm = mk["post"] | mk["mid"] | mk["ant"] | mk["coal"]
    ix, iy = np.nonzero(fm.any(1))[0], np.nonzero(fm.any(0))[0]
    xs_, ys_ = mk["xs"], mk["ys"]
    x0_, x1_, y0_, y1_ = xs_[ix[0]] - 4.0, xs_[ix[-1]] + 4.0, ys_[iy[0]] - 3.5, ys_[iy[-1]] + 3.5
    W, H = 33.0, 24.0
    cxm, cym = (x0_ + x1_) / 2, (y0_ + y1_) / 2
    half = max(x1_ - x0_, (y1_ - y0_) * W / H) / 2
    cols = [("comech", "CoMech ensemble", "neural field, continuous query"),
            ("fe", "FE verification", "engine grid"),
            ("fefine", "FE, refined grid", "1.0-mm voxels (engine: 1.5 mm)" if REAL_BONE else "twofold in every direction")]
    rows = [("resect", ["Resection", "only"]),
            ("plan", ["Rehearsed plan:"] + ([f"{abs(t):.0f}-mm {side}", "calcaneal", "osteotomy"] if t else []) +
             ([("+ " if t else "") + "interposition"] if ip else []) + ([("+ " if (t or ip) else "") + "fusion"] if fu else []) +
             ([] if (t or ip or fu) else ["resection alone"]))]
    X0, Y0, GX, GY = 19.0, 14.5, 1.6, 1.8
    uref = None
    for r_, (tag, rlab) in enumerate(rows):
        y = Y0 + r_ * (H + GY)
        for k, s_ in enumerate(rlab):
            cv.text(0.8, y + H / 2 - (len(rlab) - 1) * 1.25 + k * 2.5 + 0.9, s_, fontsize=FS if k == 0 else FS_SMALL,
                    color=INK if k == 0 else INK2)
        for c_, (src, ct, cs) in enumerate(cols):
            x = X0 + c_ * (W + GX)
            ax = cv.ax(x, y, W, H)
            uref = child_map(ax, tag, src, uref=uref, lw=0.75, arrow_width=0.0075, arrow_scale=11)
            ax.set_xlim(cxm - half, cxm + half); ax.set_ylim(cym - half * H / W, cym + half * H / W)
            if r_ == 0:
                cv.text(x + W / 2, Y0 - 3.4, ct, ha="center", fontsize=FS_LABEL)
                cv.text(x + W / 2, Y0 - 0.9, cs, ha="center", fontsize=FS_SMALL, color=INK2)
            if r_ == 0 and c_ == 0:
                ox, oy = x + 2.2, y + 7.4
                cv.arrow(ox, oy, ox + 4.2, oy, lw=0.5, color=INK2, arrowstyle="-|>,head_length=1.8,head_width=0.9")
                cv.arrow(ox, oy, ox, oy - 4.2, lw=0.5, color=INK2, arrowstyle="-|>,head_length=1.8,head_width=0.9")
                cv.text(ox + 4.8, oy + 0.7, "A", fontsize=FS_SMALL, color=INK2)
                cv.text(ox - 0.75, oy - 4.9, "M", fontsize=FS_SMALL, color=INK2)
    yk = Y0 + 2 * H + GY + 3.4
    cbar_mm(cv, ScalarMappable(Normalize(0, 1.6), STRESS), X0 + 1.0, yk, 31.0, 1.3, "Von Mises stress (MPa)",
            ticks=[0, 0.4, 0.8, 1.2, 1.6])
    kx = X0 + W + GX + 2.0
    cv.line([kx, kx + 5.0], [yk + 0.6, yk + 0.6], color=INK, lw=0.9)
    cv.text(kx + 6.4, yk + 1.5, "Defect at the predicted tolerance" if REAL_BONE else f"Defect at the {fname}-facet tolerance",
            fontsize=FS_SMALL)
    cv.line([kx, kx + 5.0], [yk + 3.6, yk + 3.6], color=BLUE, lw=0.8, ls=(0, (2.4, 1.4)))
    cv.text(kx + 6.4, yk + 4.5, "Coalition footprint (resected)", fontsize=FS_SMALL)
    cv.arrow(kx, yk + 6.6, kx + 5.0, yk + 6.6, lw=0.7, arrowstyle="-|>,head_length=2.2,head_width=1.1")
    cv.text(kx + 6.4, yk + 7.5, "Talar displacement relative to the calcaneus", fontsize=FS_SMALL)
    cv.text(kx + 6.4 + 39.5, yk + 1.5, "Superior view, talus removed;", fontsize=FS_SMALL, color=INK2)
    cv.text(kx + 6.4 + 39.5, yk + 4.5, "A, anterior; M, medial", fontsize=FS_SMALL, color=INK2)
    # effect of the rehearsed adjunct (descriptor changes)
    ax = cv.ax(146.0, Y0 + 2.0, 34.0, 2 * H + GY - 4.0)
    keys = [("post_mean", "Posterior mean"), ("mid_mean", "Middle mean"), ("tn_p99", "Talonavicular peak"),
            ("talus_evert", "Talar rotation"), ("nav_drop", "Navicular drop")]
    ax.axvline(0, color=INK, lw=0.5, zorder=1)
    _vals = [0.0]
    for j, (k, nm) in enumerate(keys):
        if j % 2 == 0:
            ax.axhspan(j - 0.5, j + 0.5, color=PAPER, lw=0, zorder=0)
        dp = (abs(Mp["eff_pred_plan"][k]) - abs(Mp["eff_pred_resect"][k])) / (abs(Mp["eff_pred_resect"][k]) + 1e-9)
        df = (abs(Mp["eff_fe_plan"][k]) - abs(Mp["eff_fe_resect"][k])) / (abs(Mp["eff_fe_resect"][k]) + 1e-9)
        _vals += [100 * dp, 100 * df]
        for v, dy, col, filled in [(100 * dp, -0.15, BLUE, True), (100 * df, 0.15, FE, False)]:
            ax.plot([0, v], [j + dy] * 2, color=col if filled else MUTED, lw=0.7, zorder=2, solid_capstyle="butt")
            ax.scatter([v], [j + dy], s=15, color=col if filled else "white", edgecolors="white" if filled else col,
                       linewidths=0.4 if filled else 0.8, zorder=3)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([k[1] for k in keys])
    ax.set_ylim(len(keys) - 0.5, -0.5)
    _lim, _tk = nice_lim(_vals, 25, pad=4)
    ax.set_xlim(*_lim); ax.set_xticks(_tk)
    style(ax, left=False)
    ax.set_xlabel("Change versus resection only (%)")
    ax.set_title("Effect of the rehearsed plan", pad=11)
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, color=BLUE, mec="white", mew=0.4,
                              label="CoMech ensemble"),
                       Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=FE, mew=0.8,
                              label="FE verification")],
              loc="lower center", bbox_to_anchor=(0.42, 0.995), ncol=2, handletextpad=0.05, columnspacing=0.8,
              borderaxespad=0.0)

    # ---- b: planning regret as rainclouds ----
    E = D["eval_equiv_plan"]; PM = E["planning"]["models"]
    yr = 81.0
    cv.panel(0, yr, "b", "Rehearsal on 16 held-out feet")
    models = ["comech_s0"] + [m for m in BASELINES if m in PM]
    Jf = [np.array(v["J_fe"]) for v in E["planning"]["feet"].values()]
    rand = [np.mean((J - J.min()) / (J.max() - J.min())) for J in Jf]
    fixed = [[(J[c] - J.min()) / (J.max() - J.min()) for J in Jf] for c in range(len(Jf[0]))]
    cb_ = int(np.argmin([np.median(f) for f in fixed]))
    entries = [(NAMES[m] + (" (seed 0)" if m == "comech_s0" else ""), np.array([r["regret"] for r in PM[m]]), mcol(m),
                f"{sum(r['top1'] for r in PM[m])}/{len(PM[m])}") for m in models]
    entries += [("Random adjunct", np.array(rand), GREY, ""), ("Best fixed adjunct", np.array(fixed[cb_]), GREY_D, "")]
    ax = cv.ax(25.5, yr + 10.0, 24.0, 31.5)
    pos = list(range(len(models))) + [len(models) + 0.3, len(models) + 1.3]
    grid = np.linspace(0, 1, 400)
    for (nm, v, col, top), p in zip(entries, pos):
        raincloud_h(ax, p, v, col, lo=0.0, hi=1.0, grid=grid, height=0.40, seed=int(10 * p) + 1, ms=5.5)
    ax.axhline(len(models) - 0.35, color=BASE, lw=0.5, zorder=0)
    ax.set_yticks(pos); ax.set_yticklabels([e[0] for e in entries])
    for t_, e in zip(ax.get_yticklabels(), entries):
        t_.set_color(e[2] if e[2] not in (GREY,) else INK2)
    ax.set_ylim(pos[-1] + 0.55, -0.62); ax.set_xlim(-0.03, 1.03); ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    style(ax, left=False)
    ax.set_xlabel("Regret (fraction of FE range)")
    # left-aligned with the header (2026-11-03): the longest model name starts at the header's x, right edge kept
    with at_print_size(cv, axes_y=[ax]):                   # v5: measured at the printed size
        rr_ = cv.fig.canvas.get_renderer(); px_mm = 25.4 / cv.fig.dpi
        lab_left = min(t_.get_window_extent(rr_).x0 for t_ in ax.get_yticklabels()) * px_mm
    new_left = 25.5 + (TITLE_DX - lab_left)                 # the title now starts at TITLE_DX
    ax.set_position([new_left / cv.w, ax.get_position().y0, (25.5 + 24.0 - new_left) / cv.w, ax.get_position().height])
    for (nm, v, col, top), p in zip(entries, pos):
        if top:
            y_mm = yr + 10.0 + (p + 0.62) / (pos[-1] + 0.55 + 0.62) * 31.5
            pill(cv, 25.5 + 24.0 + 6.2, y_mm, top, fc=col, w=6.4, h=2.4)
    cv.text(25.5 + 24.0 + 6.2, yr + 9.0, "Top-1", ha="center", fontsize=FS_SMALL, color=INK2)

    # ---- c: abstention gate with marginal densities ----
    A = E["abstention"]
    xc = 62.5
    cv.panel(xc, yr, "c", "Abstention gate")
    MX, MY, MW, MH = xc + 13.0, yr + 18.5, 30.0, 22.5
    ax = cv.ax(MX, MY, MW, MH)
    thr = A["threshold"]
    xlim_, ylim_ = (0.02, 0.36), (0.0, 1.5)
    ax.add_patch(Rectangle((thr, ylim_[0]), xlim_[1] - thr, ylim_[1] - ylim_[0], color=PAPER, lw=0, zorder=0))
    ax.axvline(thr, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(xlim_[1] - 0.006, ylim_[1] - 0.04, "abstain", fontsize=FS_SMALL, color=INK2, va="top", ha="right")
    grp = [("test", BLUE, True, "Held-out"), ("ood", VERM, False, "Out-of-range")]
    for sp, col, filled, nm in grp:
        d = np.array(A[sp]["dis"]); e = np.clip(np.array(A[sp]["err"]), None, ylim_[1] - 0.01)
        if filled:
            ax.scatter(d, e, s=3.5, color=col, edgecolors="none", alpha=0.45, zorder=2)
        else:
            ax.scatter(d, e, s=4.5, facecolors="none", edgecolors=col, linewidths=0.4, alpha=0.75, zorder=3)
    ax.set_xlim(*xlim_); ax.set_ylim(*ylim_); ax.set_xticks([0.1, 0.2, 0.3]); ax.set_yticks([0, 0.5, 1.0, 1.5])
    style(ax)
    ax.set_xlabel("Ensemble disagreement"); ax.set_ylabel("Effect error\n(RMS, standardized)", linespacing=1.2)
    axt = cv.ax(MX, MY - 6.3, MW, 5.6)
    gx = np.linspace(xlim_[0], xlim_[1], 300)
    marginal(axt, [A["test"]["dis"], A["ood"]["dis"]], [BLUE, VERM], gx, "x", lo=0.0)
    axt.set_xlim(*xlim_); axt.axvline(thr, color=INK2, lw=0.6, ls=(0, (3, 2)))
    axr = cv.ax(MX + MW + 1.0, MY, 5.6, MH)
    gy = np.linspace(ylim_[0], ylim_[1], 300)
    marginal(axr, [A["test"]["err"], A["ood"]["err"]], [BLUE, VERM], gy, "y", lo=0.0)
    axr.set_ylim(*ylim_)
    for k, (sp, col, filled, nm) in enumerate(grp):
        yy = yr + 7.6 + k * 2.4
        xx = xc + 6.0
        cv.fig.add_artist(Line2D([cv.fx(xx)], [cv.fy(yy - 0.7)], marker="o", ms=2.8, ls="none",
                                 color=col if filled else "white", mec=col, mew=0.5 if filled else 0.7,
                                 transform=cv.fig.transFigure))
        cv.text(xx + 1.4, yy, f"{nm}: abstains on {100 * A[sp]['abstain_rate']:.0f}%; "
                              f"ρ = {A[sp]['spearman_dis_err']:.2f}", fontsize=FS_SMALL)

    # ---- d: one- versus two-step roll-outs (three seeds) ----
    Rr = D["eval_rollout"]
    xd = 123.0
    cv.panel(xd, yr, "d", "Composed roll-outs: cartilage stress error")
    ax = cv.ax(xd + 12.5, yr + 11.0, 28.5, 28.5)
    sds = [s_ for s_ in SEEDS if s_ in Rr]
    shades = {"comech_s0": "#0B4F80", "comech_s1": BLUE, "comech_s2": SKY}
    mks = {"comech_s0": "o", "comech_s1": "s", "comech_s2": "^"}
    one_all, two_all = [], []
    for sd in sds:
        one_all.append(100 * np.array([r["one_step"]["stress_l2_art"] for r in Rr[sd]]))
        two_all.append(100 * np.array([r["two_step"]["stress_l2_art"] for r in Rr[sd]]))
    vmin = np.floor(min(np.min(o) for o in one_all + two_all) / 5) * 5 - 2
    vmax = np.ceil(max(np.max(o) for o in one_all + two_all) / 5) * 5 + 2
    lim = (vmin, vmax)
    ax.fill_between(lim, [lim[0] - 2, lim[1] - 2], [lim[0] + 2, lim[1] + 2], color=GRID, lw=0, zorder=0)
    ax.plot(lim, lim, color=MUTED, lw=0.6, zorder=1)
    for sd, one, two in zip(sds, one_all, two_all):
        ax.scatter(one, two, s=8, marker=mks[sd], color=shades[sd], edgecolors="white", linewidths=0.3, alpha=0.9,
                   zorder=3, label=f"Seed {sd[-1]}")
    m1 = np.mean([np.median(o) for o in one_all]); m2 = np.mean([np.median(o) for o in two_all])
    ax.scatter([m1], [m2], s=46, marker="+", color=INK, linewidths=1.0, zorder=5)
    ax.text(lim[1] - 0.03 * (lim[1] - lim[0]), lim[0] + 0.03 * (lim[1] - lim[0]),
            f"+ medians\none step {m1:.1f}%\ntwo steps {m2:.1f}%", fontsize=FS_SMALL, color=INK,
            va="bottom", ha="right", linespacing=1.2)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    tk = [t_ for t_ in np.arange(np.ceil(lim[0] / 10) * 10, lim[1] + 0.1, 10)]
    ax.set_xticks(tk); ax.set_yticks(tk)
    style(ax)
    ax.set_xlabel("One transition, combined action (%)")
    ax.set_ylabel("Two transitions:\nresection, then osteotomy (%)", linespacing=1.2)
    ax.text(0.5, 0.5, "±2 pp", transform=ax.transAxes, fontsize=FS_SMALL, color=MUTED, rotation=45, ha="center",
            va="center", alpha=0.0)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.01), handletextpad=0.1, markerscale=1.3, borderaxespad=0.2,
              fontsize=FS_SMALL, labelspacing=0.25)
    # paired differences (two - one), all seeds
    diff = np.concatenate([t_ - o_ for o_, t_ in zip(one_all, two_all)])
    axh = cv.ax(xd + 42.0, yr + 11.0, 8.0, 28.5)
    gmax = np.ceil(max(abs(diff)) + 1)
    gy = np.linspace(-gmax, gmax, 300)
    d_ = kde1d(diff, gy)
    axh.fill_betweenx(gy, 0, d_ / d_.max(), color=BLUE, alpha=0.25, lw=0)
    axh.plot(d_ / d_.max(), gy, color=BLUE, lw=0.7)
    axh.axhline(0, color=INK2, lw=0.5)
    axh.axhline(np.median(diff), color=BLUE, lw=0.8, ls=(0, (2, 1)))
    axh.set_xlim(0, 1.1); axh.set_ylim(-gmax, gmax)
    axh.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2.5))   # v5: the ticks of the 2026-11-03 version
    axh.set_xticks([])
    for sp_ in ("top", "left", "bottom"):
        axh.spines[sp_].set_visible(False)
    axh.spines["right"].set_visible(True); axh.spines["right"].set_linewidth(0.5)
    axh.yaxis.tick_right(); axh.yaxis.set_label_position("right")
    axh.tick_params(axis="y", direction="out", length=2.0, width=0.5, pad=1.4)
    axh.set_ylabel("Two − one (pp)", labelpad=2.0)
    axh.text(0.5, 1.01, "paired\ndifference", transform=axh.transAxes, ha="center", va="bottom", fontsize=FS_SMALL,
             color=INK2, linespacing=1.1)
    save(cv, "fig7")


if __name__ == "__main__":
    D = load_all()
    for f in sys.argv[2:] if len(sys.argv) > 2 else ["fig5", "fig6", "fig7"]:
        globals()[f](D)
