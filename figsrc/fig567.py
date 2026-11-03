"""Figures 5-7 from the CoMech experiment results (results/*.json, results/*.npz, results/renders/*.npz).
House style v2: figstyle.py (millimetre grid, plain headers, detached hairline axes, Okabe-Ito colours, CoMech in blue
and comparators in grey wherever the axis names them)."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from figstyle import *
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "figures")
NAMES = {"comech_s0": "CoMech", "actcat": "Action-concat.", "deeponet": "DeepONet", "geo": "Geometry→field*",
         "fno": "FNO*"}
BASELINES = ["actcat", "deeponet", "geo", "fno"]
SEEDS = ["comech_s0", "comech_s1", "comech_s2"]
STAR = "*Given the post-operative geometry instead of the action."
COALS = ["Roz I", "Roz II", "Roz III", "Roz IV", "Roz V", "Type I", "Type II", "Type III", "Type IV"]


def mcol(m):
    return BLUE if m.startswith("comech") else GREY_D


def med(rows, k):
    v = [r[k] for r in rows if k in r and r[k] == r[k]]
    return float(np.median(v)) if v else np.nan


def iqr(rows, k):
    v = [r[k] for r in rows if k in r and r[k] == r[k]]
    return np.percentile(v, 25), np.percentile(v, 75)


def load_all():
    D = {}
    for n in ["eval_fields", "eval_rollout", "eval_latency", "eval_zeroshot", "eval_taxonomy", "eval_tipping",
              "eval_cohort", "eval_cohort_fe", "eval_equiv_plan", "eval_maps", "numbers"]:
        p = os.path.join(ROOT, "results", n + ".json")
        D[n] = json.load(open(p)) if os.path.exists(p) else None
    return D


def render_rgba(name, cut=False, crop=None):
    """Rendering as RGBA cropped to the object; `cut` drops the shaded side wall of the bone below the articular
    surfaces (no field data there). Pass the returned crop to the other renderings of the same row so that they stay
    pixel-aligned."""
    Z = np.load(os.path.join(ROOT, "results", "renders", name + ".npz"))
    img = np.clip(Z["img"], 0, 1); al = np.clip(Z["alpha"], 0, 1)
    if crop is None:
        m = al > 0.02
        rr, cc = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        r0, r1, c0, c1 = max(rr[0] - 3, 0), rr[-1] + 3, max(cc[0] - 3, 0), cc[-1] + 3
        if cut:
            im = img[r0:r1, c0:c1]; a = al[r0:r1, c0:c1]
            sat = (im.max(2) - im.min(2)) > 0.3 * im.max(2)
            last = np.nonzero(sat.any(1))[0][-1]
            L = im.mean(2)
            for r in range(last, len(L)):
                if (a[r] > 0.02).any() and L[r][a[r] > 0.02].mean() < 0.4:
                    r1 = r0 + max(r - 2, last + 1)
                    break
        crop = (r0, r1, c0, c1)
    r0, r1, c0, c1 = crop
    return np.dstack([img, al])[r0:r1, c0:c1], Z, crop


REAL_BONE = getattr(__import__("fe.geometry", fromlist=["x"]), "ENGINE", "") == "real-bone"


def put_fit(cv, x, y, w, hmax, im):
    """As put, but no taller than hmax mm (narrower and centred in its column if needed; real-bone renderings are
    less elongated than the parametric ones; Claude, 2026-11-01)."""
    h = w * im.shape[0] / im.shape[1]
    if h > hmax:
        w2 = hmax * im.shape[1] / im.shape[0]; x += (w - w2) / 2; w, h = w2, hmax
    ax = cv.ax(x, y, w, h)
    ax.imshow(im, interpolation="none"); ax.set_axis_off()
    return h


def world_window_crop(name, key="ext_top", pad=4.0):
    """Pixel crop (r0, r1, c0, c1) of a top-view rendering to the subtalar facets of the held-out foot (+- pad mm)."""
    import fe.geometry as FG
    M = np.load(os.path.join(ROOT, "results", "map_heldout.npz"), allow_pickle=True)
    xs, ys = FG.map_axes(FG.Foot(**json.loads(str(M["foot"]))))
    Z = np.load(os.path.join(ROOT, "results", "renders", name + ".npz")); ext = Z[key]; H, W = Z["alpha"].shape
    x0, x1, y0, y1 = xs[0] + 8 - pad, xs[-1] - 8 + pad, ys[0] + 8 - pad, ys[-1] - 8 + pad
    c0 = int(np.floor((x0 - ext[0]) / (ext[1] - ext[0]) * W)); c1 = int(np.ceil((x1 - ext[0]) / (ext[1] - ext[0]) * W))
    r0 = int(np.floor((ext[3] - y1) / (ext[3] - ext[2]) * H)); r1 = int(np.ceil((ext[3] - y0) / (ext[3] - ext[2]) * H))
    return (max(r0, 0), min(r1, H), max(c0, 0), min(c1, W))


def nice_lim(vals, step, pad=0.0):
    """Axis limits on multiples of step around the data (real-bone figures; Claude, 2026-11-01)."""
    v = np.asarray(vals, float).ravel(); v = v[np.isfinite(v)]
    a = np.floor((v.min() - pad) / step) * step; b = np.ceil((v.max() + pad) / step) * step
    return (float(a), float(b)), [float(t) for t in np.arange(a, b + 1e-9, step)]


def put(cv, x, y, w, im):
    """Image of width w mm at (x, y); returns its height in mm."""
    h = w * im.shape[0] / im.shape[1]
    ax = cv.ax(x, y, w, h)
    ax.imshow(im, interpolation="none"); ax.set_axis_off()
    return h


def save(cv, name):
    os.makedirs(OUT, exist_ok=True)
    cv.save(os.path.join(OUT, name))
    print("saved", name)


# ============================== Figure 5 ==============================
def fig5(D):
    cv = Canvas(FIG_W, 122)
    # ---- a: rendered fields of a held-out foot ----
    _ha = D["eval_maps"]["heldout"]["act"]; _fac = "posterior" if _ha[2] >= _ha[1] else "middle"   # (Claude, 2026-11-01)
    cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, f"Held-out foot, {100 * max(_ha[1], _ha[2]):.0f}% {_fac}-facet defect")
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

    # ---- b: field accuracy, held-out and out-of-range ----
    F = D["eval_fields"]
    xb = 104.0
    cv.letter(xb, 0.5, "b"); cv.header(xb + 4.8, 3.4, "Field accuracy")
    models = ["comech_s0"] + [m for m in BASELINES if f"{m}|test" in F]
    mets = [("stress_l2_art", "Cartilage stress", "Error (relative L₂)", (0.1, 0.7), [0.2, 0.4, 0.6]),
            ("disp_l2", "Displacement", "Error (relative L₂)", (0.05, 0.7), [0.2, 0.4, 0.6]),
            ("hot_iou", "Hotspot overlap", "IoU, top 5%", (0.1, 0.75), [0.2, 0.4, 0.6])]
    for p, (key, title, xl, lim, ticks) in enumerate(mets):
        ax = cv.ax(xb + 18.5 + p * 20.6, 13.0, 17.0, 36.0)
        for i, m in enumerate(models):
            col = mcol(m)
            for sp, filled, dy in [("test", True, -0.17), ("ood", False, 0.17)]:
                rows_ = F.get(f"{m}|{sp}")
                if not rows_:
                    continue
                q1, q3 = iqr(rows_, key)
                if m == "comech_s0":
                    v = np.mean([med(F[f"{s}|{sp}"], key) for s in SEEDS if f"{s}|{sp}" in F])
                else:
                    v = med(rows_, key)
                ax.plot([q1, q3], [i + dy] * 2, color=col, lw=0.7, solid_capstyle="butt", zorder=2)
                ax.scatter([v], [i + dy], s=13, color=col if filled else "white", edgecolors=col if not filled else "white",
                           linewidths=0.8 if not filled else 0.4, zorder=3)
        ax.set_ylim(len(models) - 0.45, -0.55); ax.set_yticks(range(len(models)))
        ax.set_yticklabels([NAMES[m] for m in models] if p == 0 else [""] * len(models))
        ax.set_xlim(*lim); ax.set_xticks(ticks)
        style(ax, left=False)
        if p == 0:
            ax.get_yticklabels()[0].set_color(BLUE)
        ax.set_title(title, pad=4)
        ax.set_xlabel(xl, fontsize=FS)
    lx = xb + 18.5
    for k, (filled, lab) in enumerate([(True, "Held-out feet (40)"), (False, "Out-of-range feet (20)")]):
        ax0 = lx + k * 27.0
        cv.fig.add_artist(Line2D([cv.fx(ax0 + 1.0)], [cv.fy(60.6)], marker="o", ms=3.4, ls="none",
                                 color=INK2 if filled else "white", mec=INK2 if not filled else "white",
                                 mew=0.8 if not filled else 0.4, transform=cv.fig.transFigure))
        cv.text(ax0 + 2.6, 61.6, lab, fontsize=FS)
    cv.text(lx, 65.6, STAR, fontsize=FS_SMALL, color=INK2)

    # ---- c: speed ----
    L = D["eval_latency"]
    yr = 78.0
    cv.letter(0, yr, "c"); cv.header(4.8, yr + 2.9, "Speed" if REAL_BONE else "Speed (one CPU core)")
    if REAL_BONE:      # the real-bone FE iterates on a GPU; hardware on a line of its own (Claude, 2026-11-01)
        cv.text(4.8, yr + 7.2, "CoMech: one CPU core; FE: one GPU", fontsize=FS_SMALL, color=INK2)
    ax = cv.ax(17.5, yr + 12.0, 22.0, 14.0)
    vals = [L["fe_solve_s_median"], L["rollout_full_field_s"], L["sweep200_effects_s"] / 200]
    labs = ["FE solve", "CoMech field", "CoMech effects"]
    for i, (v, c) in enumerate(zip(vals, [FE, BLUE, BLUE])):
        ax.scatter([v], [i], s=16, color=c, edgecolors="white", linewidths=0.4, zorder=3)
        ax.text(v * 1.6, i, f"{v:.2g} s", va="center", fontsize=FS)
    ax.set_xscale("log"); ax.set_xlim(0.01, 100); ax.set_ylim(2.5, -0.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(labs)
    ax.get_yticklabels()[1].set_color(BLUE); ax.get_yticklabels()[2].set_color(BLUE)
    ax.set_xticks([0.01, 0.1, 1, 10, 100]); ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    style(ax, left=False)
    ax.set_xlabel("Time per configuration (s)")

    # ---- d: atlas analogue (zero-shot) ----
    Z = D["eval_zeroshot"]["comech_s0"]
    s = Z["scores"]; Rw = Z["rows"]
    xd = 44.0
    cv.letter(xd, yr, "d"); cv.header(xd + 4.8, yr + 2.9, "Atlas analogue (reference foot, never seen in training)")
    ax = cv.ax(xd + 10.5, yr + 14.0, 33.0, 22.0)
    x = np.array([0, 1, 2, 3, 4, 5.7, 6.7, 7.7, 8.7])
    fe = [Rw[n]["fe"]["coal_p99"] for n in COALS]; pr = [Rw[n]["pred"]["coal_p99"] for n in COALS]
    for i in range(len(x)):
        ax.plot([x[i]] * 2, [pr[i], fe[i]], color=BASE, lw=0.6, zorder=1)
    ax.scatter(x, fe, s=15, facecolors="white", edgecolors=FE, linewidths=0.8, zorder=3)
    ax.scatter(x, pr, s=14, color=BLUE, edgecolors="white", linewidths=0.4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(["I", "II", "III", "IV", "V", "I", "II", "III", "IV"])
    ax.set_ylim(0, 16); ax.set_yticks([0, 5, 10, 15]); ax.set_xlim(-0.6, 9.3)
    style(ax)
    ax.set_ylabel("Bridge stress (MPa)")
    for (a_, b_, nm) in [(0, 4, "Rozansky"), (5.7, 8.7, "Anatomy-based")]:
        ax.annotate("", xy=(a_ - 0.25, -0.20), xytext=(b_ + 0.25, -0.20), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2, shrinkA=0, shrinkB=0))
        ax.text((a_ + b_) / 2, -0.25, nm, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS)
    taus = [D["eval_zeroshot"][m]["scores"]["tau_all_vs_fe"] for m in SEEDS if m in D["eval_zeroshot"]]
    ax.set_title(f"Nine coalitions: Kendall τ = {np.mean(taus):.2f}".replace("-", "−"), pad=9)
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=FE, mew=0.8, label="FE"),
                       Line2D([], [], ls="none", marker="o", ms=3.6, color=BLUE, mec="white", mew=0.4,
                              label="CoMech")],
              loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=2, handletextpad=0.05, columnspacing=1.0,
              borderaxespad=0.0)
    ax = cv.ax(xd + 58.0, yr + 14.0, 24.0, 22.0)
    dd = np.array([0, 10, 20, 30, 40, 50, 60, 70])
    ax.axhline(20, color=MUTED, lw=0.6, ls=(0, (1, 1.4)), zorder=0)
    _ymax, _labs = 0.0, []
    for fac, nm, lx_, ly_ in [("post", "Posterior", 57, 128), ("mid", "Middle", 70, 47)]:
        col = FACET[fac]
        yf = np.array(s[f"curve_{fac}_fe"])
        ax.plot(dd, 100 * (yf / yf[0] - 1), color=col, lw=0.8, ls=(0, (3, 1.6)), marker="o", ms=2.6, mfc="white",
                mec=col, mew=0.6, zorder=3)
        yp = np.mean([np.array(D["eval_zeroshot"][m]["scores"][f"curve_{fac}_effhead"]) /
                      D["eval_zeroshot"][m]["scores"][f"curve_{fac}_effhead"][0] for m in SEEDS], 0)
        ax.plot(dd, 100 * (yp - 1), color=col, lw=0.9, marker="o", ms=2.4, mec="white", mew=0.3, zorder=4)
        if REAL_BONE:      # labels placed after both facets are drawn; axis range from the data (Claude, 2026-11-01)
            _ymax = max(_ymax, 100 * (yf / yf[0] - 1).max(), 100 * (yp - 1).max())
            _labs.append((nm, 100 * (yf[4] / yf[0] - 1), 100 * (yp[4] - 1)))
        else:
            ax.text(lx_, ly_, nm, color=INK, fontsize=FS, ha="right", va="center")
    if REAL_BONE:
        _labs = sorted(_labs, key=lambda t: max(t[1], t[2]))       # lower pair first
        for k_, (nm_, a_, b_) in enumerate(_labs):
            if k_ == len(_labs) - 1:
                ax.text(40, max(a_, b_) + 7, nm_, color=INK, fontsize=FS, ha="center", va="bottom")
            else:
                ax.text(40, min(a_, b_) - 5, nm_, color=INK, fontsize=FS, ha="center", va="top")
        top_ = max(60.0, 1.25 * _ymax + 10)
        ax.set_xlim(-3, 73); ax.set_xticks([0, 35, 70]); ax.set_ylim(min(-15.0, min(min(t[1], t[2]) for t in _labs) - 22), top_)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50 if top_ > 120 else 25))
    else:
        ax.set_xlim(-3, 73); ax.set_xticks([0, 35, 70]); ax.set_ylim(-8, 195); ax.set_yticks([0, 50, 100, 150])
    style(ax)
    ax.set_xlabel("Facet defect (%)"); ax.set_ylabel("Residual-facet mean\n(% change)", linespacing=1.2)
    ax.set_title("Defect series", pad=9)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 1.6)), marker="o", ms=2.6, mfc="white",
                              label="FE"),
                       Line2D([], [], color=INK2, lw=0.9, marker="o", ms=2.4, mec="white", mew=0.3, label="CoMech")],
              loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=2, handlelength=2.2, columnspacing=0.8,
              borderaxespad=0.0)

    # ---- e: label-free taxonomy ----
    T = D["eval_taxonomy"]
    xe = 130.5
    cv.letter(xe, yr, "e"); cv.header(xe + 4.8, yr + 2.9, "Label-free taxonomy (240 coalitions)")
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
    for k, t in enumerate(["I", "II", "III", "IV"]):
        xx = xe + 3.0 + (k % 2) * 9.5; yy = yr + 38.6 + (k // 2) * 2.6
        cv.fig.add_artist(Line2D([cv.fx(xx)], [cv.fy(yy - 0.7)], marker=TYPE_MARKER[t], ms=2.8, ls="none",
                                 color=TYPE[t], mew=0, transform=cv.fig.transFigure))
        cv.text(xx + 1.4, yy, f"Type {t}", fontsize=FS_SMALL)
    ax = cv.ax(xe + 26.0, yr + 14.0, 24.0, 21.0)
    reps = [("geometry", "Bridge geometry"), ("fe_mechanics", "FE fingerprint"), ("mechanics", "CoMech fingerprint"),
            ("latent", "CoMech latent")]
    for i, (r, nm) in enumerate(reps):
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
    cv = Canvas(FIG_W, 109)
    TP = D["eval_tipping"]; fe = TP["fe"]; ids = sorted(fe, key=int)

    # ---------------- a: tolerable defect, FE against CoMech ----------------
    cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, "Tolerable defect, 20 held-out feet")
    for k, (fac, nm, lim, ticks) in enumerate([("post", "Posterior facet", (10, 40), [10, 20, 30, 40]),
                                               ("mid", "Middle facet", (20, 80), [20, 40, 60, 80])]):
        col = FACET[fac]
        ax = cv.ax(11 + k * 38, 15.5, 29, 29)
        a = np.array([fe[i][fac]["tol"] for i in ids]) * 100
        P = np.array([[TP["models"][m]["pred"][i][fac]["tol"] for i in ids] for m in SEEDS]) * 100
        b = P.mean(0)
        if REAL_BONE:      # ranges from the data (the parametric ranges do not hold for the real-bone engine)
            vv = np.r_[a, b - P.std(0), b + P.std(0)]
            st = 10.0 if np.ptp(vv) < 45 else 20.0
            lim, ticks = nice_lim(vv, st, pad=2.0)
        ax.plot(lim, lim, color=BASE, lw=0.6, zorder=1)
        loo = np.array([(a.sum() - x) / (len(a) - 1) for x in a])
        ax.axhline(a.mean(), color=MUTED, lw=0.6, ls=(0, (1, 1.4)), zorder=1)
        ax.errorbar(a, b, yerr=P.std(0), fmt="none", ecolor=col, elinewidth=0.5, alpha=0.7, zorder=2)
        ax.scatter(a, b, s=9, color=col, edgecolors="white", linewidths=0.4, zorder=3)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_aspect("equal")
        style(ax)
        mae = np.mean(np.abs(a - b)); cm = np.mean(np.abs(a - loo))
        ax.set_title(nm, pad=19)
        ax.text(0.5, 1.03, f"CoMech error {mae:.1f} pp\nMean of others {cm:.1f} pp", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=FS_SMALL, color=INK2, linespacing=1.2)
        ax.set_xlabel("FE (% of facet)")
        if k == 0:
            ax.set_ylabel("CoMech (% of facet)")

    # ---------------- b: osteotomy and tolerance ----------------
    cv.letter(84, 0.5, "b"); cv.header(88.8, 3.4, "Osteotomy")
    ax = cv.ax(95, 15.5, 22, 29)
    for x in D["numbers"]["xrate_fe_check"]:
        ax.plot([-5, 0, 5], 100 * np.array(x["fe"]), color=FE, lw=0.6, marker="o", ms=2.8, mfc="white", mew=0.6,
                zorder=3)
        ax.plot([-5, 0, 5], 100 * np.array(x["pred"]), color=BLUE, lw=0.6, marker="o", ms=2.6, mec="white", mew=0.3,
                zorder=4, alpha=0.95)
    ax.set_xticks([-5, 0, 5]); ax.set_xticklabels(["−5", "0", "+5"]); ax.set_xlim(-6.5, 6.5)
    if REAL_BONE:
        vv = np.r_[[100 * np.array(x["fe"]) for x in D["numbers"]["xrate_fe_check"]],
                   [100 * np.array(x["pred"]) for x in D["numbers"]["xrate_fe_check"]]]
        lim_, ticks_ = nice_lim(vv, 10.0 if np.ptp(vv) < 45 else 20.0, pad=2.0)
        ax.set_ylim(*lim_); ax.set_yticks(ticks_)
    else:
        ax.set_ylim(-2, 44); ax.set_yticks([0, 10, 20, 30, 40])
    style(ax)
    ax.set_xlabel("Osteotomy (mm)\n(− lateralizing, + medializing)", linespacing=1.3)
    ax.set_ylabel("Posterior tolerance (% of facet)")
    ax.legend(handles=[Line2D([], [], color=FE, lw=0.6, marker="o", ms=2.8, mfc="white", mew=0.6, label="FE"),
                       Line2D([], [], color=BLUE, lw=0.6, marker="o", ms=2.6, mec="white", mew=0.3,
                              label="CoMech")],
              loc="lower right", bbox_to_anchor=(1.04, -0.02), handlelength=1.6)

    # ---------------- c: virtual cohort against the 50% rule ----------------
    CF = D["eval_cohort_fe"]
    cv.letter(123, 0.5, "c"); cv.header(127.8, 3.4, "Virtual cohort (135 feet) and the 50% rule")
    ax = cv.ax(134, 15.5, 45.5, 29)
    rng = np.random.default_rng(1)
    ax.axvspan(50, 100, color=GRID, lw=0, zorder=0)
    ax.axvline(50, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color=INK2, lw=0.5, zorder=1)
    _lo, _xmax = -300.0, 100.0
    if REAL_BONE:      # axes from the data: real-bone feet reach reserves below -300% and ratios above 100% (Claude, 2026-11-01)
        _allres = []
        for r in CF["rows"]:
            v = [(r[f"fe_tol_{f}"] - r[f"d_{f}"]) / r[f"fe_tol_{f}"] for f in ["mid", "post"] if f"fe_tol_{f}" in r]
            _allres.append(100 * min(v) if v else 100.0)
        _lo = min(-300.0, 100.0 * np.floor(min(_allres) / 100.0))
        _xmax = max(100.0, 10.0 * np.ceil(10.0 * max(r["ratio"] for r in CF["rows"])))
        ax.axvspan(100, _xmax, color=GRID, lw=0, zorder=0)
    for t in ["III", "I", "II", "IV"]:
        rr = [r for r in CF["rows"] if r["ctype"] == t]
        res = []
        for r in rr:
            v = [(r[f"fe_tol_{f}"] - r[f"d_{f}"]) / r[f"fe_tol_{f}"] for f in ["mid", "post"] if f"fe_tol_{f}" in r]
            res.append(100 * min(v) if v else 100.0)
        ax.scatter([100 * r["ratio"] for r in rr], np.clip(res, _lo, 100) + rng.uniform(-2, 2, len(rr)), s=8,
                   color=TYPE[t], marker=TYPE_MARKER[t], edgecolors="white", linewidths=0.3, zorder=3,
                   label=f"Type {t}")
    _ylab = _lo - 40 if REAL_BONE else _lo          # real bone: rule labels below the lowest foot
    ax.set_xlim(0, _xmax); ax.set_ylim(_lo - (45 if REAL_BONE else 10), 150); ax.set_yticks(np.arange(_lo, 101, 100))
    style(ax)
    ax.set_xlabel("Coalition / posterior facet (%)")
    ax.set_ylabel("Mechanical reserve, FE (%)")
    ax.text(48.5, _ylab, "Rule: resect alone", ha="right", va="bottom", fontsize=FS_SMALL, color=INK2)
    ax.text(64, _ylab, "Rule: do not\nresect alone", ha="left", va="bottom", fontsize=FS_SMALL, color=INK2,
            linespacing=1.15)
    S = CF["summary"]["fe_within"]
    ax.text(0.585 * _xmax if REAL_BONE else 51.5, 140, f"Within envelope: {S['within']}/{CF['summary']['n']}\nRule too permissive: {S['mis_permissive']}"
            f"\nRule too strict: {S['mis_strict']}", ha="left", va="top", fontsize=FS_SMALL, color=INK, linespacing=1.25)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(f"Type {t}") for t in ["I", "II", "III", "IV"]]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 0.995), ncol=4,
              handletextpad=0.05, columnspacing=0.7, markerscale=1.2, borderaxespad=0.0)

    # ---------------- d: mechanically equivalent operations ----------------
    E = D["eval_equiv_plan"]; EQ = E["equivalence"]
    cv.letter(0, 58.5, "d"); cv.header(4.8, 61.4, "Mechanically equivalent operations")
    models = ["comech_s0"] + [m for m in BASELINES if m in EQ]
    ax = cv.ax(23, 74, 34, 22)
    for i, m in enumerate(models):
        col = BLUE if m == "comech_s0" else GREY_D
        if m == "comech_s0":
            p = [EQ[s][ "precision"] for s in SEEDS]; r = [EQ[s]["recall"] for s in SEEDS]
            pv, rv, ps, rs = np.mean(p), np.mean(r), np.std(p), np.std(r)
        else:
            pv, rv, ps, rs = EQ[m]["precision"], EQ[m]["recall"], 0, 0
        ax.plot([pv, rv], [i, i], color=col, lw=0.6, zorder=2)
        if ps:
            ax.plot([pv - ps, pv + ps], [i, i], color=col, lw=1.6, alpha=0.35, solid_capstyle="butt", zorder=2)
            ax.plot([rv - rs, rv + rs], [i, i], color=col, lw=1.6, alpha=0.35, solid_capstyle="butt", zorder=2)
        ax.scatter([pv], [i], s=16, color=col, edgecolors="white", linewidths=0.4, zorder=4)
        ax.scatter([rv], [i], s=16, facecolors="white", edgecolors=col, linewidths=0.8, zorder=4)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([NAMES[m] for m in models])
    ax.set_ylim(len(models) - 0.5, -0.5); ax.set_xlim(0.35, 1.0); ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    style(ax, left=False)
    ax.get_yticklabels()[0].set_color(BLUE)
    ax.set_xlabel("Precision, recall")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, color=INK2, mec="white", mew=0.4, label="Precision"),
                       Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=INK2, mew=0.8, label="Recall")],
              loc="lower center", bbox_to_anchor=(0.5, 0.995), ncol=2, handletextpad=0.05, borderaxespad=0.0)

    # ---------------- e: ablation of the effect-grounded constraints ----------------
    cv.letter(64, 58.5, "e"); cv.header(68.8, 61.4, "Ablation of the effect-grounded constraints")
    F = D["eval_fields"]
    abl = ["full", "comech_noswap", "comech_nosens", "comech_none"]
    labs = ["Full model", "− Effect-swap", "− Sensitivity", "− Both"]
    metrics = [("Equivalence\nrecall", lambda m: EQ[m]["recall"], (0.6, 1.0), [0.6, 0.8, 1.0]),
               ("Equivalence\nprecision", lambda m: EQ[m]["precision"], (0.6, 1.0), [0.6, 0.8, 1.0]),
               ("Cartilage\nstress error", lambda m: med(F[f"{m}|test"], "stress_l2_art"), (0.2, 0.32),
                [0.2, 0.25, 0.3]),
               ("Posterior tolerance\nerror (pp)", lambda m: float(np.mean(TP["models"][m]["tol_err_pp"]["post"])),
                (3, 5.5), [3, 4, 5])]
    for p, (nm, fn, xl, xt) in enumerate(metrics):
        ax = cv.ax(88 + p * 24.5, 74, 19, 22)
        for i, m in enumerate(abl):
            if m == "full":
                vs = [fn(s) for s in SEEDS]
                ax.plot([np.mean(vs) - np.std(vs), np.mean(vs) + np.std(vs)], [i, i], color=BLUE, lw=0.7, zorder=2)
                ax.scatter(vs, [i] * 3, s=7, facecolors="white", edgecolors=BLUE, linewidths=0.5, zorder=3)
                ax.scatter([np.mean(vs)], [i], s=16, color=BLUE, edgecolors="white", linewidths=0.4, zorder=4)
                ax.axvline(np.mean(vs), color=BLUE_L, lw=0.6, zorder=0)
            else:
                ax.scatter([fn(m)], [i], s=16, color=GREY_D, edgecolors="white", linewidths=0.4, zorder=4)
        if REAL_BONE:
            vv = [fn(s_) for s_ in SEEDS] + [fn(m) for m in abl[1:]]
            span = max(np.ptp(vv), 1e-6); pad_ = 0.15 * span
            xl = (min(vv) - pad_, max(vv) + pad_)
            loc = mpl.ticker.MaxNLocator(3); xt = [t for t in loc.tick_values(*xl) if xl[0] <= t <= xl[1]]
        ax.set_ylim(len(abl) - 0.5, -0.5); ax.set_xlim(*xl); ax.set_xticks(xt)
        ax.set_yticks(range(len(abl)))
        ax.set_yticklabels(labs if p == 0 else [""] * len(abl))
        style(ax, left=False)
        if p == 0:
            ax.get_yticklabels()[0].set_color(BLUE)
        ax.set_title(nm, pad=4, linespacing=1.15)
    cv.text(1, 107.6, STAR, fontsize=FS_SMALL, color=INK2)
    save(cv, "fig6")


# ============================== Figure 7 ==============================
def fig7(D):
    from maps import child_map
    cv = Canvas(FIG_W, 126)
    Mp = D["eval_maps"]["child"]
    fac = Mp.get("facet", "post"); fname = {"post": "posterior", "mid": "middle"}[fac]   # facet of the child's defect
    t, ip, fu = Mp["choice"]; side = "medializing" if t > 0 else "lateralizing"
    align = Mp["foot"]["align"]
    cv.letter(0, 0.5, "a")
    _tol = f"tolerance {100 * Mp['tol_' + fac]:.0f}%"
    if REAL_BONE:      # the child's tolerance in eval_maps is CoMech's prediction; FE value from the cohort sweeps (Claude, 2026-11-01)
        _fr = [r for r in (D.get("eval_cohort_fe") or {}).get("rows", []) if r.get("i") == Mp.get("i")]
        if _fr and f"fe_tol_{fac}" in _fr[0]:
            _tol = f"tolerance {100 * Mp['tol_' + fac]:.0f}% predicted, {100 * _fr[0]['fe_tol_' + fac]:.0f}% by FE"
    cv.header(4.8, 3.4, f"Virtual child: type {Mp['foot']['ctype']} coalition, {abs(align):.0f}° "
              f"{'valgus' if align > 0 else 'varus'}; complete resection leaves a {100 * Mp['d_' + fac]:.0f}% "
              f"{fname}-facet defect ({_tol})")
    # ---- maps: zoomed to the articular facets ----
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
            ("fefine", "FE, refined grid", "1.0-mm voxels (engine: 1.5 mm)" if getattr(__import__("fe.geometry", fromlist=["x"]),
                                                     "ENGINE", "") == "real-bone" else "twofold in every direction")]
    rows = [("resect", ["Resection", "only"]),
            ("plan", ["Rehearsed plan:"] + ([f"{abs(t):.0f}-mm {side}", "calcaneal osteotomy"] if t else []) +
             ([("+ " if t else "") + "interposition"] if ip else []) + ([("+ " if (t or ip) else "") + "fusion"] if fu else []) +
             ([] if (t or ip or fu) else ["resection alone"]))]
    X0, Y0, GX, GY = 19.0, 14.5, 1.6, 1.8
    uref = None
    for r_, (tag, rlab) in enumerate(rows):
        y = Y0 + r_ * (H + GY)
        for k, s in enumerate(rlab):
            cv.text(0.8, y + H / 2 - (len(rlab) - 1) * 1.25 + k * 2.5 + 0.9, s, fontsize=FS if k == 0 else FS_SMALL,
                    color=INK if k == 0 else INK2)
        for c_, (src, ct, cs) in enumerate(cols):
            x = X0 + c_ * (W + GX)
            ax = cv.ax(x, y, W, H)
            uref = child_map(ax, tag, src, uref=uref, lw=0.75, arrow_width=0.0075, arrow_scale=11)
            ax.set_xlim(cxm - half, cxm + half); ax.set_ylim(cym - half * H / W, cym + half * H / W)
            if r_ == 0:
                cv.text(x + W / 2, Y0 - 3.4, ct, ha="center", fontsize=FS_LABEL)
                cv.text(x + W / 2, Y0 - 0.9, cs, ha="center", fontsize=FS_SMALL, color=INK2)
            if r_ == 0 and c_ == 0:      # orientation
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

    # ---- effect of the rehearsed adjunct (descriptor changes) ----
    ax = cv.ax(146.0, Y0 + 2.0, 34.0, 2 * H + GY - 4.0)
    keys = [("post_mean", "Posterior mean"), ("mid_mean", "Middle mean"), ("tn_p99", "Talonavicular peak"),
            ("talus_evert", "Talar rotation"), ("nav_drop", "Navicular drop")]
    ax.axvline(0, color=INK, lw=0.5, zorder=1)
    _vals = [0.0]
    for j, (k, nm) in enumerate(keys):
        # change in magnitude relative to resection alone (rotation and drop are signed quantities)
        dp = (abs(Mp["eff_pred_plan"][k]) - abs(Mp["eff_pred_resect"][k])) / (abs(Mp["eff_pred_resect"][k]) + 1e-9)
        df = (abs(Mp["eff_fe_plan"][k]) - abs(Mp["eff_fe_resect"][k])) / (abs(Mp["eff_fe_resect"][k]) + 1e-9)
        _vals += [100 * dp, 100 * df]
        for v, dy, col, filled in [(100 * dp, -0.15, BLUE, True), (100 * df, 0.15, FE, False)]:
            ax.plot([0, v], [j + dy] * 2, color=col if filled else MUTED, lw=0.7, zorder=2, solid_capstyle="butt")
            ax.scatter([v], [j + dy], s=15, color=col if filled else "white", edgecolors="white" if filled else col,
                       linewidths=0.4 if filled else 0.8, zorder=3)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([k[1] for k in keys])
    ax.set_ylim(len(keys) - 0.45, -0.55); ax.set_xlim(-100, 50); ax.set_xticks([-100, -50, 0, 50])
    if REAL_BONE:      # range from the data (the navicular drop of the real-bone child rises by more than 50%)
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

    # ---- b: planning regret ----
    E = D["eval_equiv_plan"]; PM = E["planning"]["models"]
    yr = 81.0
    cv.letter(0, yr, "b"); cv.header(4.8, yr + 2.9, "Rehearsal on 16 held-out feet")
    models = ["comech_s0"] + [m for m in BASELINES if m in PM]
    Jf = [np.array(v["J_fe"]) for v in E["planning"]["feet"].values()]
    rand = [np.mean((J - J.min()) / (J.max() - J.min())) for J in Jf]
    fixed = [[(J[c] - J.min()) / (J.max() - J.min()) for J in Jf] for c in range(len(Jf[0]))]
    cb_ = int(np.argmin([np.median(f) for f in fixed]))
    entries = [(NAMES[m], np.array([r["regret"] for r in PM[m]]), mcol(m),
                f"{sum(r['top1'] for r in PM[m])}/{len(PM[m])}") for m in models]
    entries += [("Random adjunct", np.array(rand), GREY, ""), ("Best fixed adjunct", np.array(fixed[cb_]), GREY, "")]
    ax = cv.ax(24.0, yr + 9.5, 26.0, 28.0)
    pos = list(range(len(models))) + [len(models) + 0.35, len(models) + 1.35]
    for (nm, v, col, top), p in zip(entries, pos):
        rng = np.random.default_rng(int(10 * p))
        ax.scatter(v, p + rng.uniform(-0.2, 0.2, len(v)), s=6, color=col, edgecolors="white", linewidths=0.25,
                   zorder=3, alpha=0.9)
        ax.plot([np.median(v)] * 2, [p - 0.33, p + 0.33], color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
        if top:
            ax.text(1.36, p, top, ha="right", va="center", fontsize=FS, transform=ax.get_yaxis_transform())
    ax.text(1.36, -1.05, "Top-1", ha="right", va="center", fontsize=FS_SMALL, color=INK2,
            transform=ax.get_yaxis_transform())
    ax.set_yticks(pos); ax.set_yticklabels([e[0] for e in entries])
    ax.get_yticklabels()[0].set_color(BLUE)
    ax.set_ylim(pos[-1] + 0.55, -0.6); ax.set_xlim(-0.03, 1.03); ax.set_xticks([0, 0.5, 1])
    style(ax, left=False)
    ax.set_xlabel("Regret (fraction of FE range)")

    # ---- c: abstention ----
    A = E["abstention"]
    xc = 63.0
    cv.letter(xc, yr, "c"); cv.header(xc + 4.8, yr + 2.9, "Abstention gate")
    ax = cv.ax(xc + 12.0, yr + 12.5, 37.0, 25.0)
    for sp, col, filled, nm in [("test", BLUE, True, "Held-out"), ("ood", VERM, False, "Out-of-range")]:
        d = np.array(A[sp]["dis"]); e = np.array(A[sp]["err"])
        if filled:
            ax.scatter(d, e, s=4.0, color=col, edgecolors="none", alpha=0.55, zorder=2,
                       label=f"{nm} (abstains on {100 * A[sp]['abstain_rate']:.0f}%)")
        else:
            ax.scatter(d, e, s=5.0, facecolors="none", edgecolors=col, linewidths=0.45, alpha=0.8, zorder=3,
                       label=f"{nm} (abstains on {100 * A[sp]['abstain_rate']:.0f}%)")
    ax.axvline(A["threshold"], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(A["threshold"] + 0.006, 1.62, "Threshold", fontsize=FS_SMALL, color=INK2, va="top")
    ax.set_xlim(0.03, 0.38); ax.set_ylim(0, 1.65); ax.set_xticks([0.1, 0.2, 0.3]); ax.set_yticks([0, 0.5, 1.0, 1.5])
    style(ax)
    ax.set_xlabel("Ensemble disagreement"); ax.set_ylabel("Effect error\n(RMS, standardized)", linespacing=1.2)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=1, markerscale=1.8, handletextpad=0.05,
              borderaxespad=0.0)

    # ---- d: roll-out composition ----
    Rr = D["eval_rollout"]
    xd = 124.0
    cv.letter(xd, yr, "d"); cv.header(xd + 4.8, yr + 2.9, "Composed roll-outs: cartilage stress error")
    ax = cv.ax(xd + 22.0, yr + 9.5, 28.0, 28.0)
    one, two = [], []
    for sd in [s_ for s_ in SEEDS if s_ in Rr]:
        one += [r["one_step"]["stress_l2_art"] for r in Rr[sd]]; two += [r["two_step"]["stress_l2_art"] for r in Rr[sd]]
    lim = (0, 0.8)
    ax.plot(lim, lim, color=BASE, lw=0.6, zorder=0)
    ax.scatter(one, two, s=6, color=BLUE, edgecolors="white", linewidths=0.25, alpha=0.8, zorder=2)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    style(ax)
    ax.set_xlabel("One transition (combined action)")
    ax.set_ylabel("Two transitions\n(resection, then osteotomy)", linespacing=1.2)
    save(cv, "fig7")


if __name__ == "__main__":
    D = load_all()
    for f in sys.argv[2:] if len(sys.argv) > 2 else ["fig5", "fig6", "fig7"]:
        globals()[f](D)
