"""Supplementary Fig. 8: training curves of all models and held-out error after 1,600 and 4,000 steps (house style v2)."""
import os, sys, json, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from figstyle import *
from matplotlib.lines import Line2D
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "supp_figures")
MODELS = os.path.join(ROOT, "results", "models_long")
NAME = {"comech_s0": "CoMech, seed 0", "comech_s1": "CoMech, seed 1", "comech_s2": "CoMech, seed 2",
        "comech_noswap": "− effect-swap", "comech_nosens": "− sensitivity", "comech_none": "− both",
        "actcat": "Action-concat.", "deeponet": "DeepONet", "geo": "Geometry→field", "fno": "FNO"}
COL = {"comech_s0": BLUE, "comech_s1": BLUE, "comech_s2": BLUE, "comech_noswap": GREY_D, "comech_nosens": GREY,
       "comech_none": "#C4C4C4", "actcat": MODEL["actcat"], "deeponet": MODEL["deeponet"], "geo": MODEL["geo"],
       "fno": MODEL["fno"]}
DASH = (0, (3, 1.6))


def curve(m, k=50):
    p = os.path.join(MODELS, m + ".pkl")
    if os.path.exists(p):
        f = np.array([x["field"] for x in pickle.load(open(p, "rb"))["hist"]])
    else:      # real-bone packs carry only the CoMech seeds' weights; the other training histories come in
        hj = os.path.join(ROOT, "results", "train_hist.json")      # results/train_hist.json (Claude, 2026-11-01)
        H = json.load(open(hj)) if os.path.exists(hj) else {}
        if m not in H:
            return None
        f = np.array(H[m], float)
    sm = np.convolve(f, np.ones(k) / k, mode="valid")
    return np.arange(len(sm)) + k, sm


cv = Canvas(FIG_W_SUPP, 70)
cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, "Training curves (field loss on training batches, moving average)")
for p, (ttl, ms) in enumerate([("CoMech and its ablations", ["comech_none", "comech_nosens", "comech_noswap", "comech_s1",
                                                            "comech_s2", "comech_s0"]),
                               ("CoMech and the comparators", ["fno", "deeponet", "geo", "actcat", "comech_s0"])]):
    ax = cv.ax(12.0 + p * 52.0, 13.0, 44.0, 43.0)
    for m in ms:
        c = curve(m)
        if c is None:
            continue
        ax.plot(*c, color=COL[m], lw=0.7, ls=DASH if m in ("comech_s1", "comech_s2") else "-",
                zorder=3 if m == "comech_s0" else 2)
    ax.plot([1600, 1600], [0.035, 1.3], color=MUTED, lw=0.5, ls=(0, (1, 1.4)), zorder=1)
    ax.text(1680, 0.55, "Warm restart", fontsize=FS_SMALL, color=INK2, va="bottom")   # above the curves (Claude, 2026-11-01)
    ax.set_yscale("log"); ax.set_ylim(0.035, 4.0); ax.set_xlim(0, 4100); ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xticklabels(["0", "1,000", "2,000", "3,000", "4,000"])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, q: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    style(ax)
    ax.set_title(ttl, fontsize=FS, pad=4)
    ax.set_xlabel("Training step")
    if p == 0:
        ax.set_ylabel("Field loss")
        hs = [Line2D([], [], color=BLUE, lw=1.0, label="CoMech, seed 0"),
              Line2D([], [], color=BLUE, lw=1.0, ls=DASH, label="CoMech, seeds 1, 2"),
              Line2D([], [], color=COL["comech_noswap"], lw=1.0, label="− effect-swap"),
              Line2D([], [], color=COL["comech_nosens"], lw=1.0, label="− sensitivity"),
              Line2D([], [], color=COL["comech_none"], lw=1.0, label="− both")]
    else:
        hs = [Line2D([], [], color=COL[m], lw=1.0, label=NAME[m]) for m in ["comech_s0", "actcat", "deeponet", "geo", "fno"]]
    ax.legend(handles=hs, loc="upper right", ncol=2, handlelength=1.6, columnspacing=0.8)

cv.letter(112.0, 0.5, "b"); cv.header(116.8, 3.4, "Held-out error after the two phases")
F1 = json.load(open(os.path.join(ROOT, "results", "preview", "eval_fields_m1600.json")))
F2 = json.load(open(os.path.join(ROOT, "results", "eval_fields.json")))
ms = [m for m in ["comech_s0", "comech_s2", "deeponet", "geo"] if f"{m}|test" in F1 and f"{m}|test" in F2]
ax = cv.ax(133.5, 17.0, 28.5, 39.0)
for i, m in enumerate(ms):
    for j, (FF, mk, filled) in enumerate([(F1, "o", False), (F2, "s", True)]):
        v = [r["stress_l2_art"] for r in FF[f"{m}|test"]]
        y = i + (j - 0.5) * 0.36
        q1, q3 = np.percentile(v, [25, 75])
        ax.plot([q1, q3], [y, y], color=COL[m], lw=0.7, solid_capstyle="butt", zorder=2)
        ax.plot(np.median(v), y, marker=mk, ms=3.6, color=COL[m], mfc=COL[m] if filled else "white", mec=COL[m],
                mew=0.8, ls="none", zorder=3)
ax.set_yticks(range(len(ms))); ax.set_yticklabels([NAME[m] for m in ms])
ax.set_ylim(len(ms) - 0.45, -0.55); ax.set_xlim(0.15, 0.5); ax.set_xticks([0.2, 0.3, 0.4, 0.5])
style(ax, left=False)
ax.set_xlabel("Cartilage stress error")
ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=INK2, mew=0.8, label="1,600 steps"),
                   Line2D([], [], ls="none", marker="s", ms=3.6, color=INK2, label="4,000 steps")],
          loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.2)
os.makedirs(OUT, exist_ok=True)
cv.save(os.path.join(OUT, "supp_fig8"))
print("saved supp_fig8")
