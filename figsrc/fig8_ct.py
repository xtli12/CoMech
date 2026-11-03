"""Supplementary Fig. 10: CT-informed twins of 39 real feet (preliminary readings), per foot.
Also provides the shared loaders and type colours used by fig8_v2.py (main Fig. 8)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE_COL = {"I": TYPE["I"], "III": TYPE["III"], "indeterminate": TYPE["?"]}
TYPE_MK = {"I": TYPE_MARKER["I"], "III": TYPE_MARKER["III"], "indeterminate": "o"}
OPER = lambda t: t.startswith("resection")
ENV_COL = BLUE         # share of twins within the envelope
RULE_COL = VERM        # share of twins for which the 50% rule recommends resection alone


def load():
    T = json.load(open(f"{ROOT}/results/eval_ct_twins.json"))
    F = json.load(open(f"{ROOT}/results/eval_ct_twins_fe.json"))
    return T, F


def fig8(out):
    T, F = load()
    rows = T["rows"]
    from matplotlib.lines import Line2D
    cv = Canvas(FIG_W_SUPP, 139)
    # ---------- a: per-foot strip ----------
    cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, "Preliminary CT readings and twin envelopes, 39 feet")
    order = sorted(rows, key=lambda r: ({"I": 0, "III": 1, "indeterminate": 2}[r["type"]], -r["p_within"], r["foot"]))
    Y0, RH = 15.0, 3.0
    XT, XC, XTR, XD = 10.0, 17.9, 25.0, 31.8
    XB = [36.5, 50.0, 63.5]; BW = 12.0; XA = 81.0
    heads = [(XT + 2.2, "Type"), (XC, "Conf."), (XTR, "Treated"), (XD, "Defor-\nmity"), (XB[0] + BW / 2, "Within\nenvelope"),
             (XB[1] + BW / 2, "Within,\ncalibrated"), (XB[2] + BW / 2, "50% rule:\nresect alone"), (XA, "Abstain")]
    for x, h in heads:
        cv.text(x, Y0 - 1.6, h, ha="center", va="bottom", fontsize=FS_SMALL, linespacing=1.1)
    cv.line([0.5, 86.0], [Y0 - 0.9, Y0 - 0.9], color=INK2, lw=0.4)
    R = lambda x, y, w, h, **kw: cv.fig.add_artist(Rectangle((cv.fx(x), cv.fy(y + h)), w / cv.w, h / cv.h,
                                                             transform=cv.fig.transFigure, **kw))
    for i, r in enumerate(order):
        y = Y0 + i * RH
        yc = y + RH / 2 + 0.75                     # text baseline, centred in the row
        cv.text(8.5, yc, r["foot"], ha="right", fontsize=FS_SMALL, color=INK2)
        R(XT, y + 0.25, 4.4, RH - 0.5, color=TYPE_COL[r["type"]], lw=0)
        cv.text(XT + 2.2, yc, {"I": "I", "III": "III", "indeterminate": "?"}[r["type"]], ha="center", fontsize=FS_SMALL,
                color="white" if r["type"] != "III" else INK)
        cv.text(XC, yc, r["conf"], ha="center", fontsize=FS_SMALL, color=INK2)
        if OPER(r["treatment"]):
            R(XTR - 1.8, y + 0.45, 3.6, RH - 0.9, color=INK, lw=0)
            if "osteotomy" in r["treatment"]:
                cv.text(XTR, yc, "O", ha="center", fontsize=FS_SMALL, color="white")
        else:
            R(XTR - 1.8, y + 0.45, 3.6, RH - 0.9, fill=False, ec=MUTED, lw=0.4)
        if r["deformity"]:
            cv.text(XD, yc, "valgus" if r["deformity"].startswith("valgus") else "varus", ha="center", fontsize=FS_SMALL,
                    color=INK2)
        for j, key in enumerate(["p_within", "p_within_cal", "p_rule_ok"]):
            v = r[key]
            R(XB[j], y + 0.55, BW, RH - 1.1, color=GRID, lw=0)
            if v > 0:
                R(XB[j], y + 0.55, BW * v, RH - 1.1, color=RULE_COL if key == "p_rule_ok" else ENV_COL, lw=0)
        if r["abstain_frac"] > 0:
            cv.text(XA, yc, f"{100 * r['abstain_frac']:.0f}%", ha="center", fontsize=FS_SMALL)
    yb = Y0 + len(order) * RH + 3.0
    K = T.get("K", 24)
    cv.text(XB[0], yb, f"Bars: share of the {K} twins of each foot (full width, all {K})", fontsize=FS_SMALL, color=INK2)
    # ---------- b: coalition ratio vs share within the envelope ----------
    xb = 91.0
    cv.letter(xb, 0.5, "b"); cv.header(xb + 4.8, 3.4, "Envelope versus the 50% rule")
    ax = cv.ax(xb + 11.0, 11.0, 34.0, 30.0)
    rng = np.random.default_rng(3)
    ax.axvline(50, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for r in rows:
        x = 100 * r["ratio"][1]; y = r["p_within"] + rng.uniform(-0.015, 0.015)
        col = TYPE_COL[r["type"]]
        ax.plot([100 * r["ratio"][0], 100 * r["ratio"][2]], [y, y], color=col, lw=0.5, alpha=0.7, zorder=2)
        ax.plot(x, y, TYPE_MK[r["type"]], ms=3.0, mfc=col if OPER(r["treatment"]) else "white", mec=col, mew=0.7,
                zorder=3)
    ax.set_ylim(-0.06, 1.06); ax.set_xlim(0, 60); ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "0.5", "1"])
    style(ax)
    ax.set_xlabel("Coalition / posterior facet (%)")
    ax.set_ylabel("Share within the envelope")
    hs = [Line2D([], [], ls="none", marker=TYPE_MK[t], ms=3.0, color=TYPE_COL[t], mew=0.7, label=lab)
          for t, lab in [("I", "Type I"), ("III", "Type III"), ("indeterminate", "Indeterminate")]]
    hs += [Line2D([], [], ls="none", marker="o", ms=3.0, color=INK2, mew=0.7, label="Operated"),
           Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=INK2, mew=0.7, label="Not operated")]
    ax.legend(handles=hs, loc="center left", bbox_to_anchor=(1.03, 0.5), handletextpad=0.1)
    # ---------- c: FE-verified point twins (type I) ----------
    yc_ = 53.0
    cv.letter(xb, yc_, "c"); cv.header(xb + 4.8, yc_ + 2.9, "FE-verified point twins, type I")
    ax = cv.ax(xb + 11.0, yc_ + 9.0, 34.0, 38.0)
    Rr = sorted(F["rows"], key=lambda r: r["d_mid"])
    for i, r in enumerate(Rr):
        ax.plot([100 * r["fe_tol"], 100 * r["d_mid"]], [i, i], color=BASE, lw=0.6, zorder=1)
        ax.plot(100 * r["fe_tol"], i, "|", ms=5.0, color=FE, mew=1.0, zorder=3)
        ax.plot(100 * r["pred_tol"], i, "x", ms=3.0, color=BLUE, mew=0.8, zorder=4)
        ax.plot(100 * r["d_mid"], i, "o", ms=3.0, mfc=VERM if r["d_mid"] > r["fe_tol"] else "white", mec=VERM, mew=0.7,
                zorder=5)
    ax.set_yticks(range(len(Rr))); ax.set_yticklabels([r["foot"] for r in Rr], fontsize=FS_SMALL)
    ax.set_ylim(len(Rr) - 0.4, -0.6); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    style(ax, left=False)
    ax.set_xlabel("Middle facet (%)")
    hs = [Line2D([], [], ls="none", marker="|", ms=5.0, color=FE, mew=1.0, label="FE tolerance"),
          Line2D([], [], ls="none", marker="x", ms=3.0, color=BLUE, mew=0.8, label="CoMech tolerance"),
          Line2D([], [], ls="none", marker="o", ms=3.0, color=VERM, mew=0.7, label="Defect, beyond"),
          Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=VERM, mew=0.7, label="Defect, within")]
    ax.legend(handles=hs, loc="center left", bbox_to_anchor=(1.03, 0.5), handletextpad=0.1)
    # ---------- d: counts ----------
    yd = yc_ + 9.0 + 38.0 + 9.5
    cv.letter(xb, yd, "d"); cv.header(xb + 4.8, yd + 2.9, "Counts")
    L = T["summary"]["table_lines"]
    cols = [L[:6], L[6:]]
    for c, lines in enumerate(cols):
        for k, s in enumerate(lines):
            sub = s.startswith("  ")
            cv.text(xb + 1.0 + c * 36.0 + (2.0 if sub else 0.0), yd + 7.6 + 2.5 * k, s.strip(), fontsize=FS_SMALL,
                    color=INK2 if sub else INK)
    cv.save(out[:-4] if out.endswith(".pdf") else out)
    print("saved", out)


if __name__ == "__main__":
    fig8(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/paper/figures/fig8.pdf")
