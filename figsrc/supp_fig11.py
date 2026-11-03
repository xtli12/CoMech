"""Supplementary Fig. 11: the GPU scaling experiment -- field error against training data and model capacity, defect
tolerance, and the discretization error of the engine itself (twofold-refined grid).

    python figsrc/supp_fig11.py [results_gpu_dir] [out_dir]

results_gpu_dir is the unpacked results_gpu_<date>.zip (default: results_gpu/); the paper's CPU models are read from
results/.  Every number in the figure is read from these files.
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from figstyle import *
from scipy.stats import spearmanr

G = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_gpu")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "out", "supp_figures")
RG = lambda n: json.load(open(os.path.join(G, "results", n + ".json")))
RP = lambda n: json.load(open(os.path.join(ROOT, "results", n + ".json")))
NG, NP = RG("numbers_gpu"), RP("numbers")
FG = RG("eval_fields")
FEET = [180, 450, 900, 1800]
SERIES = {  # run name per number of training feet
    "comech": {180: "comechL_s0_n180", 450: "comechL_s0_n450", 900: "comechL_s0_n900", 1800: "comechL_s0"},
    "actcat": {180: "actcatL_n180", 450: "actcatL_n450", 900: "actcatL_n900", 1800: "actcatL"},
    "gino": {180: "gino_n180", 1800: "gino"},
    "geofno": {180: "geofno_n180", 1800: "geofno"},
}
COL = {"comech": MODEL["comech"], "actcat": MODEL["actcat"], "gino": MODEL["gino"], "geofno": MODEL["geofno"]}
LAB = {"comech": "CoMech-L", "actcat": "Action-concat.-L", "gino": "GINO*", "geofno": "Geo-FNO*"}
ENS = ["comechL_s0", "comechL_s1", "comechL_s2"]


def art(name, split="test"):
    """median and IQR of the cartilage stress error of a GPU run"""
    v = np.array([r["stress_l2_art"] for r in FG.get(f"{name}|{split}", [])])
    return (np.median(v), np.percentile(v, 25), np.percentile(v, 75)) if len(v) else None


from matplotlib.lines import Line2D
cv = Canvas(FIG_W_SUPP, 154)

# ---------------- a: field error against training feet ----------------
cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, "Field error against training data")
ax = cv.ax(14.0, 11.0, 52.0, 44.0)
for kind, ser in SERIES.items():
    xs, ms, lo, hi = [], [], [], []
    for n, run in ser.items():
        a = art(run)
        if a:
            xs.append(n); ms.append(a[0]); lo.append(a[1]); hi.append(a[2])
    if not xs:
        continue
    geo = kind in ("gino", "geofno")
    ax.vlines(xs, lo, hi, color=COL[kind], lw=0.5, alpha=0.7, zorder=2)
    ax.plot(xs, ms, color=COL[kind], lw=0.9, ls=(0, (3, 1.6)) if geo else "-", marker="o", ms=3.0, mec="white", mew=0.3,
            label=LAB[kind], zorder=3)
seeds = [art(n)[0] for n in ENS if art(n)]
if len(seeds) == 3:
    ax.errorbar([1800 * 1.12], [np.mean(seeds)], yerr=[np.std(seeds)], fmt="s", ms=3.0, color=COL["comech"],
                mfc="white", mew=0.7, lw=0.6, capsize=0, zorder=4)
    ax.text(1800 * 1.2, np.mean(seeds), "3 seeds", fontsize=FS_SMALL, color=INK2, va="center")
# the paper's CPU models (180 feet, 4,000 steps on one CPU core)
cpu = [("comech_seeds", MODEL["comech"]), ("actcat", MODEL["actcat"]), ("geo", MODEL["geo"]), ("fno", MODEL["fno"])]
for j, (m, c) in enumerate(cpu):
    d = NP["fidelity"].get(m, {}).get("test", {}).get("stress_l2_art")
    if d is None:
        continue
    y = d.get("median", d.get("mean"))
    ax.plot([180 * (0.74 + 0.035 * j)], [y], marker="D", ms=3.0, mfc="white", mec=c, mew=0.7, ls="none", zorder=4)
ax.plot([], [], "D", ms=3.0, mfc="white", mec=INK2, mew=0.7, label="Main-study models (CPU)")
ax.set_xscale("log"); ax.set_xticks(FEET); ax.set_xticklabels([f"{n:,}" for n in FEET]); ax.minorticks_off()
_RB = os.path.isdir(os.path.join(ROOT, "rb"))   # real-bone code tree (Claude, 2026-11-01)
_ytop = max([0.32] + [l.get_ydata().max() * 1.08 for l in ax.lines if len(l.get_ydata())]) if _RB else 0.32
ax.set_xlim(115, 2700); ax.set_ylim(0, _ytop); ax.set_yticks([t for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5] if t <= _ytop])
style(ax)
ax.set_xlabel("Training feet"); ax.set_ylabel("Cartilage stress error")
ax.legend(loc="lower left", handlelength=2.0)

# ---------------- b: defect tolerance of both facets ----------------
xb = 74.0
cv.letter(xb, 0.5, "b"); cv.header(xb + 4.8, 3.4, "Defect tolerance, 20 held-out feet")
TP = NG.get("tipping", {})
T0 = RP("eval_tipping"); ids = sorted(T0["fe"], key=int)
AXB = {}
for r_, key in enumerate(["mae", "rho"]):
    for c_, fac in enumerate(["post", "mid"]):
        AXB[(key, fac)] = cv.ax(xb + 13.0 + c_ * 40.0, 12.0 + r_ * 26.0, 32.0, 19.0)
for (key, fac), axx in AXB.items():
    for kind, ser in SERIES.items():
        xs = [n for n, run in ser.items() if run in TP]
        if xs:
            axx.plot(xs, [TP[ser[n]][fac][key] for n in xs], color=COL[kind], lw=0.9,
                     ls=(0, (3, 1.6)) if kind in ("gino", "geofno") else "-", marker="o", ms=2.8, mec="white", mew=0.3,
                     zorder=3)
    if "comechL_ens" in TP:
        axx.plot([1800 * 1.12], [TP["comechL_ens"][fac][key]], "s", ms=3.0, color=COL["comech"], mfc="white", mew=0.7,
                 zorder=4)
    # the paper's CoMech ensemble on the same 20 feet, recomputed with the same definitions
    a = np.array([T0["fe"][i][fac]["tol"] for i in ids if fac in T0["fe"][i]]) * 100
    p = np.mean([[T0["models"][m]["pred"][i][fac]["tol"] for i in ids if fac in T0["fe"][i]]
                 for m in ["comech_s0", "comech_s1", "comech_s2"]], 0) * 100
    axx.plot([180 * 0.76], [np.abs(p - a).mean() if key == "mae" else spearmanr(p, a)[0]], "D", ms=3.0, mfc="white",
             mec=COL["comech"], mew=0.7, zorder=4)
    if key == "mae":
        loo = TP.get(f"loo_mae_{fac}")
        if loo is not None:
            axx.axhline(loo, color=INK2, lw=0.6, ls=(0, (1, 1.4)), zorder=1)
        axx.set_ylim(0, 6 if fac == "post" else 20)
        axx.set_yticks([0, 2, 4, 6] if fac == "post" else [0, 5, 10, 15, 20])
        axx.set_title({"post": "Posterior facet", "mid": "Middle facet"}[fac], fontsize=FS, pad=4)
    else:
        axx.axhline(0, color=INK2, lw=0.5, zorder=1); axx.set_ylim(-1, 1); axx.set_yticks([-1, 0, 1])
    axx.set_xscale("log"); axx.set_xticks(FEET); axx.minorticks_off(); axx.set_xlim(115, 2700)
    axx.set_xticklabels(["180", "450", "900", "1,800"] if key == "rho" else [])
    style(axx)
    if key == "rho":
        axx.set_xlabel("Training feet")
    if fac == "post":
        axx.set_ylabel("Tolerance error (pp)" if key == "mae" else "Spearman ρ with FE")
AXB[("mae", "post")].text(250, TP.get("loo_mae_post", 2.2) - 0.3, "Mean of others", fontsize=FS_SMALL, color=INK2,
                          ha="left", va="top")
hs = [Line2D([], [], color=COL[k], lw=0.9, ls=(0, (3, 1.6)) if k in ("gino", "geofno") else "-", marker="o", ms=2.8,
             mec="white", mew=0.3, label=LAB[k]) for k in ["comech", "actcat", "gino", "geofno"]]
hs += [Line2D([], [], ls="none", marker="s", ms=3.0, color=COL["comech"], mfc="white", mew=0.7, label="CoMech-L ensemble"),
       Line2D([], [], ls="none", marker="D", ms=3.0, mfc="white", mec=COL["comech"], mew=0.7,
              label="CoMech ensemble (CPU)")]
cv.fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(cv.fx(xb + 12.0), cv.fy(65.5)), ncol=3, handlelength=2.0,
              columnspacing=1.0)

# ---------------- c: against the twofold-refined FE ----------------
yc = 79.0
cv.letter(0, yc, "c"); cv.header(4.8, yc + 2.9, "Error against the refined FE (1.0-mm voxels)" if _RB else
                                 "Error against the twofold-refined FE")
RF = NG.get("refined", {})
order = [("comech_s0", "CoMech", MODEL["comech"], True), ("actcat", "Action-concat.", MODEL["actcat"], True),
         ("geo", "Geometry→field*", MODEL["geo"], True), ("fno", "FNO*", MODEL["fno"], True),
         ("comechL_s0", "CoMech-L", COL["comech"], False), ("actcatL", "Action-concat.-L", COL["actcat"], False),
         ("gino", "GINO*", COL["gino"], False), ("geofno", "Geo-FNO*", COL["geofno"], False)]
order = [o for o in order if o[0] in RF]
ncpu = sum(o[3] for o in order)
pos = [i if i < ncpu else i + 1.0 for i in range(len(order))]
axc = cv.ax(24.0, yc + 12.0, 44.0, 44.0)
if "fe_base" in RF:
    axc.axvline(RF["fe_base"]["stress_l2_art"], color=INK2, lw=0.6, ls=(0, (1, 1.4)), zorder=1)
for y, (m, lab, c, is_cpu) in zip(pos, order):
    s_, r_ = RF[m]["vs_standard"]["stress_l2_art"], RF[m]["vs_refined"]["stress_l2_art"]
    axc.plot([s_, r_], [y, y], color=c, lw=0.7, zorder=2)
    axc.plot(s_, y, "o", ms=3.2, color=c, mfc="white" if is_cpu else c, mew=0.8, zorder=3)
    axc.plot(r_, y, "^", ms=3.6, color=c, mfc="white" if is_cpu else c, mew=0.8, zorder=3)
axc.set_yticks(pos); axc.set_yticklabels([o[1] for o in order])
axc.set_ylim(pos[-1] + 0.6, -1.4); axc.set_xlim(0, 0.5); axc.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
style(axc, left=False)
axc.text(-0.02, -1.0, "CPU, 180 feet", transform=axc.get_yaxis_transform(), ha="right", va="center", fontsize=FS_SMALL,
         color=INK2)
axc.text(-0.02, ncpu, "GPU, 1,800 feet", transform=axc.get_yaxis_transform(), ha="right", va="center", fontsize=FS_SMALL,
         color=INK2)
axc.set_xlabel("Cartilage stress error (median)")
axc.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=INK2, mew=0.8,
                           label="Against standard-grid FE (training target)"),
                    Line2D([], [], ls="none", marker="^", ms=3.6, mfc="white", mec=INK2, mew=0.8,
                           label="Against refined FE (1.0 mm)" if _RB else "Against twofold-refined FE"),
                    Line2D([], [], ls=(0, (1, 1.4)), lw=0.6, color=INK2, label="Standard-grid FE against refined FE")],
           loc="upper left", bbox_to_anchor=(-0.45, -0.17), handlelength=1.6)

# ---------------- d: effect descriptors, refined vs standard grid ----------------
xd = 73.0
cv.letter(xd, yc, "d"); cv.header(xd + 4.8, yc + 2.9, "Effect descriptors, refined versus standard grid")
RE = RG("eval_refined_effects")
rows = [r for f in RE["feet"] for r in f["actions"]]
DESC = [("post_mean", "Posterior facet, mean"), ("mid_mean", "Middle facet, mean"), ("ant_mean", "Anterior facet, mean"),
        ("post_p99", "Posterior facet, peak"), ("mid_p99", "Middle facet, peak"), ("ant_p99", "Anterior facet, peak"),
        ("tn_p99", "Talonavicular, peak"), ("cc_p99", "Calcaneocuboid, peak"), ("tc_p99", "Talocalcaneal, peak"),
        ("share_post", "Posterior load share"), ("share_mid", "Middle load share"), ("share_ant", "Anterior load share"),
        ("heel_share", "Heel share"), ("talus_dz", "Talar vertical translation"), ("talus_dy", "Talar medial translation"),
        ("talus_evert", "Talar eversion"), ("nav_drop", "Navicular drop")]
axd = cv.ax(xd + 30.0, yc + 10.0, 28.0, 54.0)
axd.axvline(0, color=INK2, lw=0.5, zorder=1)
for j, (e, lab) in enumerate(DESC):
    s_ = np.array([r["std"][e] for r in rows]); rf = np.array([r["ref"][e] for r in rows])
    ok = np.abs(s_) > 1e-6 * np.abs(s_).max()
    d = (rf - s_)[ok] / np.abs(s_)[ok] * 100
    q1, md, q3 = np.percentile(d, [25, 50, 75])
    gc = BLUE if j < 9 else GREEN if j < 13 else PINK
    axd.plot([q1, q3], [j, j], color=gc, lw=0.7, solid_capstyle="butt", zorder=2)
    axd.plot(md, j, "o", ms=3.0, color=gc, mec="white", mew=0.3, zorder=3)
axd.set_yticks(range(len(DESC))); axd.set_yticklabels([l for _, l in DESC], fontsize=FS_SMALL)
axd.set_ylim(len(DESC) - 0.4, -0.6); axd.set_xlim(-100, 100); axd.set_xticks([-100, -50, 0, 50, 100])
style(axd, left=False)
axd.set_xlabel("Refined minus standard (%)")
S = RE["summary"]; W = S.get("within_foot", {}); EQ = S.get("equivalence_standard_vs_refined", {})
lines = [("320 held-out configurations", INK),
         (f"Standardized effect RMSE {S['eff_rmse']['median']:.2f}", INK2),
         ("Within-foot rank agreement:", INK),
         (f"posterior mean, ρ {W['post_mean']['median_rho']:.2f}", INK2),
         (f"middle mean, ρ {W['mid_mean']['median_rho']:.2f}", INK2),
         ("Equivalent pairs, standard", INK),
         ("against refined grid:", INK),
         (f"precision {EQ['precision']:.2f}, recall {EQ['recall']:.2f}", INK2)]
for k, (t_, c_) in enumerate(lines):
    cv.text(xd + 61.5, yc + 13.0 + 2.6 * k + (1.0 if k >= 2 else 0) + (1.0 if k >= 5 else 0), t_, fontsize=FS_SMALL,
            color=c_)
for k, (c_, t_) in enumerate([(BLUE, "Stresses"), (GREEN, "Load shares"), (PINK, "Kinematics")]):
    yy = yc + 45.0 + 2.6 * k
    cv.fig.add_artist(Line2D([cv.fx(xd + 62.5)], [cv.fy(yy - 0.7)], marker="o", ms=3.0, ls="none", color=c_, mew=0,
                             transform=cv.fig.transFigure))
    cv.text(xd + 64.0, yy, t_, fontsize=FS_SMALL)

os.makedirs(OUT, exist_ok=True)
cv.save(os.path.join(OUT, "supp_fig11"))
print("saved supp_fig11")
