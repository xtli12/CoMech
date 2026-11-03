"""Supplementary figures: the open-source data engine (geometry, actions, fields) and its comparison with the atlas."""
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from figstyle import *
from fe_data import CLS_MODELS, CLS, MID, POST, ALIGN, ALIGN_TYPES, DEFECTS
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from fe.geometry import *
from fe.simulate import get_grid
from exp.scenarios import atlas_scenarios

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "supp_figures")
grid = get_grid()
SC = atlas_scenarios()
REC = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results/atlas_analog/rec_*.json"))}
IDX = {s[1]: i for i, s in enumerate(SC)}

# label palette: the house anatomical palette (figstyle.ANAT), identical to the voxel section of Fig. 4a
LCOL = {VOID: "#ffffff", TALUS: ANAT["talus"], CALC: ANAT["calc"], NAV: ANAT["nav"], CUB: ANAT["cub"],
        C_ANT: ANAT["ant"], C_MID: ANAT["mid"], C_POST: ANAT["post"], C_TN: ANAT["tn"], C_CC: ANAT["cc"],
        COAL: ANAT["coal"], FIB: ANAT["fib"], INTERP: ANAT["interp"], FUSE: ANAT["fuse"], LIG: ANAT["lig"]}
REAL_BONE = getattr(__import__("fe.geometry", fromlist=["x"]), "ENGINE", "") == "real-bone"
NL = 15
if REAL_BONE:      # tissues of the full atlas foot (Claude, 2026-11-01)
    LCOL.update({TIB: "#D9CFBC", FIBU: "#E3DBCB", C_ANKLE: "#C9DFEC", OBONE: "#CFC7B2", C_OTHER: "#CFE6DD",
                 SOFT: "#F4EFEA", FASC: "#DDD0EA"})
    NL = 22
CM = ListedColormap([LCOL.get(i, "#ffffff") for i in range(NL)]); NORM = BoundaryNorm(np.arange(-0.5, NL + 0.5), NL)
TOP_Z = (46.0, 72.0) if REAL_BONE else (33.0, 40.0)                 # joint layer of the top views
TOP_LIM = ((-72.0, -14.0), (-18.0, 22.0)) if REAL_BONE else ((17, 72), (-14, 24))


def labels_of(foot, act):
    fr = voxelise(FootGeom(foot, act), grid)
    occ = fr[:, 1:].sum(1); lab = fr[:, 1:].argmax(1) + 1
    cart = fr[:, [C_ANT, C_MID, C_POST, C_TN, C_CC]].sum(1)
    lab[~np.where(cart > 0.05, occ > 0.15, occ >= 0.5)] = VOID
    return lab.reshape(grid.shape)


def topview(lab3, zlo=TOP_Z[0], zhi=TOP_Z[1]):
    ks = [k for k in range(grid.nz) if zlo < grid.zc[k] < zhi]
    L = np.zeros(lab3.shape[:2], int)
    for pri in [TALUS, C_ANT, C_MID, C_POST, INTERP, FUSE, FIB, COAL]:
        for k in ks:
            L[lab3[:, :, k] == pri] = pri
    return L


def show_top(ax, L):
    ax.pcolormesh(grid.xe, grid.ye, L.T, cmap=CM, norm=NORM, rasterized=True)
    ax.set_xlim(*TOP_LIM[0]); ax.set_ylim(*TOP_LIM[1]); ax.set_aspect("equal"); hide(ax)


def group(cv, x0, x1, y, s):
    cv.text(x0, y, s, fontsize=FS)
    cv.line([x0, x1], [y + 1.2, y + 1.2], color=INK2, lw=0.4)


def fig_engine():
    cv = Canvas(FIG_W_SUPP, 171)
    ref = labels_of(Foot(), Action())
    # ---- a: sections of the reference foot ----
    cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, "Reference foot of the open-source engine (voxel model, partial-volume materials)")
    W = 49.0; H = W * 64 / 96
    for i, (yy, ttl) in enumerate([(-2.0, "Sagittal, through the posterior facet"), (11.0, "Sagittal, through the middle facet")]):
        ax = cv.ax(10.0 + i * 57.0, 15.0, W, H)
        j = np.searchsorted(grid.ye, yy) - 1
        ax.pcolormesh(grid.xe, grid.ze, ref[:, j, :].T, cmap=CM, norm=NORM, rasterized=True)
        ax.set_aspect("equal"); ax.set_xlim(0, 96); ax.set_ylim(0, 64)
        ax.set_xticks([0, 20, 40, 60, 80]); ax.set_yticks([0, 20, 40, 60])
        style(ax)
        ax.set_title(ttl, fontsize=FS, pad=13)
        ax.set_xlabel("x (mm), posterior to anterior")
        if i == 0:
            ax.set_ylabel("z (mm)")
            ax.annotate("", xy=(42, 60.5), xytext=(42, 70), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK),
                        annotation_clip=False)
            ax.text(44.0, 67.5, "360 N", fontsize=FS_SMALL, va="center")
            ax.plot([2, 34], [-0.9, -0.9], color=INK2, lw=1.4, clip_on=False, solid_capstyle="butt")
            ax.text(3, 3.0, "Heel pad", fontsize=FS_SMALL)
            ax.plot([76, 92], [10.8, 10.8], color=INK2, lw=1.4, clip_on=False, solid_capstyle="butt")
            ax.text(92, 5.0, "Lateral column", fontsize=FS_SMALL, ha="right")
        else:
            ax.plot([74, 92], [29.6, 29.6], color=INK2, lw=1.4, solid_capstyle="butt")
            ax.text(74, 25.0, "Medial column", fontsize=FS_SMALL, ha="left")
    ax = cv.ax(130.0, 15.0, H * 48 / 64, H)
    i = np.searchsorted(grid.xe, 35.0) - 1
    ax.pcolormesh(grid.ye, grid.ze, ref[i, :, :].T, cmap=CM, norm=NORM, rasterized=True)
    ax.set_aspect("equal"); ax.set_xlim(-18, 30); ax.set_ylim(0, 64); ax.set_xticks([-10, 10, 30]); ax.set_yticks([0, 20, 40, 60])
    style(ax)
    ax.set_title("Coronal, x = 35 mm", fontsize=FS, pad=13)
    ax.set_xlabel("y (mm), lateral to medial")
    items = [(TALUS, "Talus"), (CALC, "Calcaneus"), (NAV, "Navicular"), (CUB, "Cuboid"), (LIG, "Ligament"),
             (C_ANT, "Anterior facet"), (C_MID, "Middle facet"), (C_POST, "Posterior facet"), (C_TN, "TN cartilage"),
             (C_CC, "CC cartilage"), (COAL, "Coalition (bone)"), (FIB, "Fibrous interval"), (INTERP, "Interposition"),
             (FUSE, "Fusion bone")]
    yl = 15.0 + H + 12.5
    for n, (l, t) in enumerate(items):
        x0 = 1.0 + (n % 7) * 23.2; y0 = yl + (n // 7) * 3.2
        cv.fig.add_artist(mpl.patches.Rectangle((cv.fx(x0), cv.fy(y0 + 0.2)), 2.4 / cv.w, 2.0 / cv.h,
                                                transform=cv.fig.transFigure, facecolor=LCOL[l], edgecolor=MUTED, lw=0.3))
        cv.text(x0 + 3.3, y0, t, fontsize=FS_SMALL)
    # ---- b: coalition morphologies ----
    yb = yl + 9.5
    cv.letter(0, yb, "b"); cv.header(4.8, yb + 2.9, "Coalition morphologies of the atlas analogue (superior view of the subtalar joint layer)")
    tw, th = 25.6, 25.6 * 38 / 55
    def tiles(y, entries):
        x = 0.0
        for grp, items_ in entries:
            gx = x
            for name, lab in items_:
                g, nm, foot, act = SC[IDX[name]]
                ax = cv.ax(x, y, tw, th)
                show_top(ax, topview(labels_of(foot, act)))
                cv.text(x + tw / 2, y + th + 3.0, lab, ha="center")
                x += tw + 1.2
            if grp:
                group(cv, gx, x - 1.2, y - 2.4, grp)
            x += 1.6
    y1 = yb + 10.5
    tiles(y1, [(None, [("Normal", "Intact")]),
               ("Rozansky types", [("Roz I", "I"), ("Roz II", "II"), ("Roz III", "III"), ("Roz IV", "IV"), ("Roz V", "V")])])
    y2 = y1 + th + 10.0
    tiles(y2, [("Anatomy-based types", [("Type I", "I"), ("Type II", "II"), ("Type III", "III"), ("Type IV", "IV")]),
               (None, [("Fusion", "Complete fusion")])])
    # ---- c: actions ----
    yc = y2 + th + 7.5
    cv.letter(0, yc, "c"); cv.header(4.8, yc + 2.9, "Surgical actions and posture")
    y3 = yc + 10.5
    x = 0.0
    for name, lab in [("post 30", "30%"), ("post 50", "50%"), ("post 70", "70%")]:
        g, nm, foot, act = SC[IDX[name]]
        ax = cv.ax(x, y3, tw, th); show_top(ax, topview(labels_of(foot, act)))
        cv.text(x + tw / 2, y3 + th + 3.0, lab, ha="center")
        x += tw + 1.2
    group(cv, 0.0, x - 1.2, y3 - 2.4, "Posterior-facet defect")
    x += 1.6
    g, nm, foot, act = SC[IDX["mid 30"]]
    ax = cv.ax(x, y3, tw, th); show_top(ax, topview(labels_of(foot, act)))
    cv.text(x + tw / 2, y3 + th + 3.0, "30%", ha="center")
    group(cv, x, x + tw, y3 - 2.4, "Middle-facet defect")
    x += tw + 4.0
    ch = th + 1.5; cw_ = ch * 32 / 40
    for n, (foot, act, t1, t2) in enumerate([(Foot(align=10.0), Action(), "10° valgus", ""),
                                             (Foot(align=10.0), Action(resect=1, osteo=6.0), "10° valgus +", "6-mm medializing osteotomy")]):
        xx = x + n * (cw_ + 10.0)
        ax = cv.ax(xx, y3 - 0.6, cw_, ch)
        L = labels_of(foot, act); i = np.searchsorted(grid.xe, 10.0) - 1
        ax.pcolormesh(grid.ye, grid.ze, L[i].T, cmap=CM, norm=NORM, rasterized=True)
        ax.axvline(-3.0, color=MUTED, lw=0.4, ls=(0, (3, 2)))
        ax.set_aspect("equal"); ax.set_xlim(-18, 14); ax.set_ylim(0, 40); hide(ax)
        cv.text(xx + cw_ / 2, y3 + th + 3.0, t1, ha="center")
        if t2:
            cv.text(xx + cw_ / 2, y3 + th + 5.5, t2, ha="center", fontsize=FS_SMALL, color=INK2)
    group(cv, x, x + 2 * cw_ + 10.0, y3 - 2.4, "Coronal section, x = 10 mm")
    cv.save(os.path.join(OUT, "supp_fig6"))
    print("saved supp_fig6")


def topview_rb(foot, act, zlo=TOP_Z[0], zhi=TOP_Z[1]):
    """Top view of the joint layer from the partial-volume fractions: a column shows a tissue when one of its cells in
    the layer holds at least 25% of it (the facet-cell definition of the effect descriptors; 50% for the talus), so
    that thin inclined facets are not lost to the argmax of bone-dominated cells (real-bone engine)."""
    fr = voxelise(FootGeom(foot, act), grid).reshape(grid.shape + (-1,))
    ks = [k for k in range(grid.nz) if zlo < grid.zc[k] < zhi]
    L = np.zeros(grid.shape[:2], int)
    for pri in [TALUS, C_ANT, C_MID, C_POST, INTERP, FUSE, FIB, COAL]:
        thr = 0.5 if pri == TALUS else 0.25
        L[(fr[:, :, ks, pri] >= thr).any(2)] = pri
    return L


def fig_engine_rb():
    """Supplementary Fig. 6 for the real-bone engine: the atlas foot as a voxel model on the analysis grid,
    coalition morphologies and actions (Claude, 2026-11-01; same layout as fig_engine)."""
    cv = Canvas(FIG_W_SUPP, 176)
    ref = labels_of(Foot(), Action())
    (x0b, x1b), (y0b, y1b), (z0b, z1b) = ML_BOX
    cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, "Reference foot of the data engine (atlas foot on the 2-mm analysis grid, partial-volume materials)")
    W = 49.0; H = W * (z1b - z0b) / (x1b - x0b)
    for i, (yy, ttl) in enumerate([(-1.1, "Sagittal, through the posterior facet"), (16.4, "Sagittal, through the middle facet")]):
        ax = cv.ax(10.0 + i * 57.0, 15.0, W, H)
        j = np.searchsorted(grid.ye, yy) - 1
        ax.pcolormesh(grid.xe, grid.ze, ref[:, j, :].T, cmap=CM, norm=NORM, rasterized=True)
        ax.set_aspect("equal"); ax.set_xlim(x0b, x1b); ax.set_ylim(z0b, z1b)
        ax.set_xticks([-100, -75, -50, -25, 0]); ax.set_yticks([20, 40, 60, 80, 100])
        style(ax)
        ax.set_title(ttl, fontsize=FS, pad=13)
        ax.set_xlabel("x (mm), posterior to anterior")
        if i == 0:
            ax.set_ylabel("z (mm)")
            ax.annotate("", xy=(-42, z1b + 0.5), xytext=(-42, z1b + 10), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK),
                        annotation_clip=False)
            ax.text(-39.0, z1b + 7.0, "360 N", fontsize=FS_SMALL, va="center")
    xc_ = -52.0
    ax = cv.ax(130.0, 15.0, H * (y1b - y0b) / (z1b - z0b), H)
    i = np.searchsorted(grid.xe, xc_) - 1
    ax.pcolormesh(grid.ye, grid.ze, ref[i, :, :].T, cmap=CM, norm=NORM, rasterized=True)
    ax.set_aspect("equal"); ax.set_xlim(y0b, y1b); ax.set_ylim(z0b, z1b); ax.set_xticks([-20, 0, 20, 40]); ax.set_yticks([20, 40, 60, 80, 100])
    style(ax)
    ax.set_title(f"Coronal, x = {xc_:.0f} mm".replace("-", "−"), fontsize=FS, pad=13)
    ax.set_xlabel("y (mm), lateral to medial")
    items = [(TALUS, "Talus"), (CALC, "Calcaneus"), (NAV, "Navicular"), (CUB, "Cuboid"), (TIB, "Tibia"), (FIBU, "Fibula"),
             (OBONE, "Other bones"), (C_ANT, "Anterior facet"), (C_MID, "Middle facet"), (C_POST, "Posterior facet"),
             (C_TN, "TN cartilage"), (C_CC, "CC cartilage"), (C_ANKLE, "Ankle cartilage"), (C_OTHER, "Other cartilage"),
             (COAL, "Coalition (bone)"), (FIB, "Fibrous interval"), (INTERP, "Interposition"), (FUSE, "Fusion bone"),
             (SOFT, "Soft tissue"), (FASC, "Plantar fascia")]
    yl = 15.0 + H + 12.5
    for n, (l, t) in enumerate(items):
        x0 = 1.0 + (n % 7) * 23.2; y0 = yl + (n // 7) * 3.2
        cv.fig.add_artist(mpl.patches.Rectangle((cv.fx(x0), cv.fy(y0 + 0.2)), 2.4 / cv.w, 2.0 / cv.h,
                                                transform=cv.fig.transFigure, facecolor=LCOL[l], edgecolor=MUTED, lw=0.3))
        cv.text(x0 + 3.3, y0, t, fontsize=FS_SMALL)
    # ---- b: coalition morphologies ----
    yb = yl + 12.7
    cv.letter(0, yb, "b"); cv.header(4.8, yb + 2.9, "Coalition morphologies of the atlas analogue (superior view of the subtalar joint layer)")
    (tx0, tx1), (ty0, ty1) = TOP_LIM
    tw, th = 25.6, 25.6 * (ty1 - ty0) / (tx1 - tx0)
    def tiles(y, entries):
        x = 0.0
        for grp, items_ in entries:
            gx = x
            for name, lab in items_:
                g, nm, foot, act = SC[IDX[name]]
                ax = cv.ax(x, y, tw, th)
                show_top(ax, topview_rb(foot, act))
                cv.text(x + tw / 2, y + th + 3.0, lab, ha="center")
                x += tw + 1.2
            if grp:
                group(cv, gx, x - 1.2, y - 2.4, grp)
            x += 1.6
    y1 = yb + 10.5
    tiles(y1, [(None, [("Normal", "Intact")]),
               ("Rozansky types", [("Roz I", "I"), ("Roz II", "II"), ("Roz III", "III"), ("Roz IV", "IV"), ("Roz V", "V")])])
    y2 = y1 + th + 10.0
    tiles(y2, [("Anatomy-based types", [("Type I", "I"), ("Type II", "II"), ("Type III", "III"), ("Type IV", "IV")]),
               (None, [("Fusion", "Complete fusion")])])
    # ---- c: actions ----
    yc = y2 + th + 7.5
    cv.letter(0, yc, "c"); cv.header(4.8, yc + 2.9, "Surgical actions and posture")
    y3 = yc + 10.5
    x = 0.0
    for name, lab in [("post 30", "30%"), ("post 50", "50%"), ("post 70", "70%")]:
        g, nm, foot, act = SC[IDX[name]]
        ax = cv.ax(x, y3, tw, th); show_top(ax, topview_rb(foot, act))
        cv.text(x + tw / 2, y3 + th + 3.0, lab, ha="center")
        x += tw + 1.2
    group(cv, 0.0, x - 1.2, y3 - 2.4, "Posterior-facet defect")
    x += 1.6
    g, nm, foot, act = SC[IDX["mid 30"]]
    ax = cv.ax(x, y3, tw, th); show_top(ax, topview_rb(foot, act))
    cv.text(x + tw / 2, y3 + th + 3.0, "30%", ha="center")
    group(cv, x, x + tw, y3 - 2.4, "Middle-facet defect")
    x += tw + 4.0
    # coronal sections through the calcaneal tuberosity, neutral midline of the tuberosity dashed
    xt_ = -84.0; it = np.searchsorted(grid.xe, xt_) - 1
    cy_ = float(grid.yc[np.nonzero((ref[it] == CALC).any(1))[0]].mean())
    cyl, czl = (cy_ - 26.0, cy_ + 22.0), (12.0, 72.0)
    ch = th + 1.5; cw_ = ch * (cyl[1] - cyl[0]) / (czl[1] - czl[0])
    for n, (foot, act, t1, t2) in enumerate([(Foot(align=10.0), Action(), "10° valgus", ""),
                                             (Foot(align=10.0), Action(resect=1, osteo=6.0), "10° valgus +", "6-mm medializing osteotomy")]):
        xx = x + n * (cw_ + 10.0)
        ax = cv.ax(xx, y3 - 0.6, cw_, ch)
        L = labels_of(foot, act)
        ax.pcolormesh(grid.ye, grid.ze, L[it].T, cmap=CM, norm=NORM, rasterized=True)
        ax.axvline(cy_, color=MUTED, lw=0.4, ls=(0, (3, 2)))
        ax.set_aspect("equal"); ax.set_xlim(*cyl); ax.set_ylim(*czl); hide(ax)
        cv.text(xx + cw_ / 2, y3 + th + 3.0, t1, ha="center")
        if t2:
            cv.text(xx + cw_ / 2, y3 + th + 5.5, t2, ha="center", fontsize=FS_SMALL, color=INK2)
    group(cv, x, x + 2 * cw_ + 10.0, y3 - 2.4, f"Coronal section, x = {xt_:.0f} mm".replace("-", "−"))
    cv.save(os.path.join(OUT, "supp_fig6"))
    print("saved supp_fig6 (real-bone engine)")


def fig_compare():
    E = {k: v["eff"] for k, v in REC.items()}
    cv = Canvas(FIG_W_SUPP, 141)
    # ---- a: bridge stress concentration ----
    cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, "Stress concentration at the bridge")
    ax = cv.ax(11.0, 13.0, 66.0, 38.0)
    coals = ["Roz I", "Roz II", "Roz III", "Roz IV", "Roz V", "Type I", "Type II", "Type III", "Type IV"]
    ea = [E[n]["coal_p99"] / E["Normal"]["tc_p99"] for n in coals]
    aa = [CLS["tc_max"][CLS_MODELS.index(n)] / CLS["tc_max"][0] for n in coals]
    x = np.array([0, 1, 2, 3, 4, 5.7, 6.7, 7.7, 8.7])
    ax.bar(x - 0.19, aa, 0.36, color=GREY, label="Atlas (peak / intact)")
    ax.bar(x + 0.19, ea, 0.36, color=BLUE, label="Engine (99th percentile / intact)")
    ax.set_xticks(x); ax.set_xticklabels(["I", "II", "III", "IV", "V", "I", "II", "III", "IV"])
    ax.set_xlim(-0.6, 9.3); ax.set_ylim(0, 8); ax.set_yticks([0, 2, 4, 6, 8])
    style(ax)
    ax.set_ylabel("Bridge stress relative to the intact joint")
    for (a_, b_, nm) in [(0, 4, "Rozansky"), (5.7, 8.7, "Anatomy-based")]:
        ax.annotate("", xy=(a_ - 0.3, -0.12), xytext=(b_ + 0.3, -0.12), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2, shrinkA=0, shrinkB=0))
        ax.text((a_ + b_) / 2, -0.15, nm, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.04))
    # ---- b: defect curves ----
    cv.letter(87.0, 0.5, "b"); cv.header(91.8, 3.4, "Facet-defect series")
    for n, (fac, key_e, key_a, ttl, D, yl_, yt_) in enumerate(
            [("post", "post_mean", "post_mean", "Posterior defect:\nresidual-facet mean", POST, (-5, 150), [0, 50, 100, 150]),
             ("mid", "ant_p99", "ant_max", "Middle defect:\nanterior-facet peak", MID, (-3, 75), [0, 25, 50, 75])]):
        ax = cv.ax(101.0 + n * 33.0, 17.0, 24.0, 34.0)
        ye = [0] + [100 * (E[f"{fac} {d}"][key_e] / E["Normal"][key_e] - 1) for d in DEFECTS[1:]]
        ya = [100 * (v / D[key_a][0] - 1) for v in D[key_a]]
        ax.axhline(20, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.plot(DEFECTS, ya, color=GREY_D, marker="o", ms=2.6, lw=0.8, mec="white", mew=0.3, label="Atlas")
        ax.plot(DEFECTS, ye, color=BLUE, marker="s", ms=2.6, lw=0.8, mec="white", mew=0.3, label="Engine")
        if n == 0:
            ax.legend(loc="upper left", handlelength=1.8)
        ax.set_xlim(-3, 73); ax.set_xticks([0, 35, 70]); ax.set_ylim(*yl_); ax.set_yticks(yt_)
        style(ax)
        ax.set_xlabel("Defect (% of facet)")
        if n == 0:
            ax.set_ylabel("Change versus intact (%)")
        ax.set_title(ttl, fontsize=FS, pad=4, linespacing=1.15)
    # ---- c: alignment heatmaps ----
    yc = 66.0
    cv.letter(0, yc, "c"); cv.header(4.8, yc + 2.9, "Alignment effects, atlas and engine: change in peak stress versus neutral alignment (%)")
    regs = [("tc_max", "coal_p99", "Bridge"), ("mid_max", "mid_p99", "Middle"), ("post_max", "post_p99", "Posterior"),
            ("ant_max", "ant_p99", "Anterior"), ("tn_max", "tn_p99", "TN"), ("cc_max", "cc_p99", "CC")]
    # The atlas valgus models of types III and IV contained material errors in the solver input (found 2026-10-31; Methods):
    # hatched in the atlas panel and not compared (no sign marks for these rows in the engine panel).
    BAD = {("Type III", "valgus10"), ("Type IV", "valgus10")}
    from matplotlib.patches import Rectangle
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=140)
    for c, (src, pos, j, ttl) in enumerate([("atlas", "varus10", 1, "Atlas, 10° varus"), ("atlas", "valgus10", 2, "Atlas, 10° valgus"),
                                            ("engine", "varus10", 1, "Engine, 10° varus"), ("engine", "valgus10", 2, "Engine, 10° valgus")]):
        ax = cv.ax(14.0 + (c % 2) * 74.0, yc + 15.0 + (c // 2) * 31.0, 58.0, 20.0)
        M = np.zeros((4, 6)); A = np.zeros((4, 6))
        for r, t in enumerate(ALIGN_TYPES):
            for k, (ka, ke, _) in enumerate(regs):
                A[r, k] = 100 * (ALIGN[t][ka][j] - ALIGN[t][ka][0]) / ALIGN[t][ka][0]
                M[r, k] = 100 * (E[f"{t} {pos}"][ke] - E[t][ke]) / max(E[t][ke], 1e-9)
        V = A.copy() if src == "atlas" else M
        bad_rows = [r for r, t in enumerate(ALIGN_TYPES) if (t, pos) in BAD]
        if src == "atlas":
            V[bad_rows, :] = np.nan
        heatmap(ax, V, DIV, norm, fmt="{:+.0f}", gap=1.0, fs=FS_SMALL, na="")
        if src == "atlas":
            for r in bad_rows:
                for k in range(6):
                    ax.add_patch(Rectangle((k - 0.5, r - 0.5), 1, 1, facecolor="none", edgecolor=MUTED, hatch="//////", lw=0))
                    ax.add_patch(Rectangle((k - 0.5, r - 0.5), 1, 1, facecolor="none", edgecolor="white", lw=1.0))
        if src == "engine":       # mark sign disagreements with the atlas
            for r in range(4):
                if r in bad_rows:
                    continue
                for k in range(6):
                    if np.sign(A[r, k]) != np.sign(M[r, k]):
                        ax.text(k + 0.42, r - 0.18, "*", ha="right", va="center", fontsize=FS, color=INK)
        ax.set_xticks(range(6)); ax.set_xticklabels([r[2] for r in regs]); ax.xaxis.tick_top()
        ax.set_yticks(range(4)); ax.set_yticklabels(ALIGN_TYPES)
        ax.set_title(ttl, fontsize=FS_LABEL, pad=11)
    cbar_mm(cv, ScalarMappable(norm, DIV), 150.5, yc + 15.0, 1.6, 51.0, "Change (%)", ticks=[-100, -50, 0, 50, 100],
            orientation="vertical")
    cv.text(14.0, yc + 15.0 + 31.0 + 20.0 + 6.0, "*Sign differs from the atlas. Hatched: atlas models with input errors, "
            "not compared (Methods).", fontsize=FS_SMALL, color=INK2)
    cv.save(os.path.join(OUT, "supp_fig7"))
    print("saved supp_fig7")


if __name__ == "__main__":
    fig_engine_rb() if REAL_BONE else fig_engine()
    fig_compare()
