"""Figure 4: the open-source data engine, the co-constructed world model and the planner, drawn in the visual language
of the authors' pipeline figures (tinted bands with side tabs, pale chevrons, gradient encoders, a glossy transition
sphere, swap/sensitivity badges).

Pictures are engine outputs (renderings of virtual feet, coalition types and the virtual child of Fig. 7a before and
after complete resection, a voxel section of the material labels, an FE stress section, FE solutions of TCC-MechBank
and the FE-verified map of the virtual child), CoMech predictions for a held-out foot (Fig. 5a) and CT crops of
Supplementary Fig. 9 (clinical stream).  v4 (2026-10-28): the surgical actions of a are engine renderings of the
virtual child (render_actions.py) and the latent rehearsal of c shows the real latent states of the adjuncts rehearsed
for the virtual child (child_latent.py), as in Fig. 1.  Icons, badges, latent tokens and the transition sphere are
schematic and drawn by this code."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from figstyle import *
from schem import *
from figparts import Placed, common_crop, world_crop, annotate_action, latent_fan, j_bar, REAL_BONE
from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm
from matplotlib.path import Path as MPath
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "figures")
R = lambda n: np.load(os.path.join(ROOT, "results", "renders", n + ".npz"))
CT = lambda f: os.path.join(ROOT, "results", "ct_tiles_s9", f + ".png")
DJ = {"fontfamily": "DejaVu Sans"}          # script L is not in Arial/Liberation Sans
NAVY = PAL["navy"]
VIOLET = LinearSegmentedColormap.from_list("act", ["#EEE8F8", "#C6B5E6", "#9277C6", PAL["act"], "#4B347E"])
BLUE_TOK = LinearSegmentedColormap.from_list("tok", ["#DCEBF6", BLUE_L, SKY, BLUE, "#004A75"])


def lines(bd, x, y, items, ha="left", step=2.25, size=FS_SMALL, color=INK2):
    for k, it in enumerate(items):
        if isinstance(it, str):
            it = (it, {})
        s, kw = it
        kw = dict(kw); kw.setdefault("size", size); kw.setdefault("color", color)
        bd.t(x, y + k * step, s, ha=ha, **kw)


def title(bd, x, y, s, ha="center"):
    bd.t(x, y, s, size=FS_LABEL, ha=ha, color=INK)   # headers as in the other figures (2026-11-03)


def ell_L(bd, x, y, sub, size=FS + 0.5, color=PAL["loss"]):
    """Loss symbol (italic L, as the script L of Methods but in the text font) with a subscript; returns the end x."""
    return bd.tsub(x, y, "L", sub, size=size, style="italic", sub_style="normal", color=color)


def ct_tile(f, crop=(0.08, 0.86, 0.10, 0.90)):
    im = np.asarray(Image.open(CT(f)).convert("RGB")).astype(float) / 255
    h, w = im.shape[:2]
    r0, r1, c0, c1 = crop
    return im[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)]


def bezier(bd, p0, p3, color, lw=0.8, ls="-", z=3, head=True, k=0.55):
    """Smooth S-shaped stream line from p0 to p3 (horizontal tangents), arrowhead at p3."""
    (x0, y0), (x3, y3) = p0, p3
    dx = (x3 - x0) * k
    verts = [(x0, y0), (x0 + dx, y0), (x3 - dx, y3), (x3, y3)]
    pp = PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]), fc="none", ec=color,
                   lw=lw, ls=ls, zorder=z, capstyle="butt")
    bd.b.add_patch(pp)
    if head:
        bd.arr(x3 - 1.6, y3, x3 + 0.05, y3, color=color, lw=lw, head=(1.6, 0.95), z=z + 0.1)


def labels_section():
    """Sagittal voxel section (material labels) of the reference foot with a type I coalition, through the middle
    facet, in the house palette."""
    from fe.geometry import Foot, Action, FootGeom, voxelise, VOID, TALUS, CALC, NAV, CUB, C_ANT, C_MID, C_POST, \
        C_TN, C_CC, COAL as LCOAL, FIB, INTERP, FUSE, LIG
    from fe.simulate import get_grid
    from exp.scenarios import COAL_MORPH
    g = get_grid()
    fr = voxelise(FootGeom(Foot(**COAL_MORPH["Type I"]), Action()), g)
    occ = fr[:, 1:].sum(1); lab = fr[:, 1:].argmax(1) + 1
    cart = fr[:, [C_ANT, C_MID, C_POST, C_TN, C_CC]].sum(1)
    lab[~np.where(cart > 0.05, occ > 0.15, occ >= 0.5)] = VOID
    lab = lab.reshape(g.shape)
    col = {VOID: "#FFFFFF", TALUS: ANAT["talus"], CALC: ANAT["calc"], NAV: ANAT["nav"], CUB: ANAT["cub"],
           C_ANT: ANAT["ant"], C_MID: ANAT["mid"], C_POST: ANAT["post"], C_TN: ANAT["tn"], C_CC: ANAT["cc"],
           LCOAL: ANAT["coal"], FIB: ANAT["fib"], INTERP: ANAT["interp"], FUSE: ANAT["fuse"], LIG: ANAT["lig"]}
    yy = 11.0
    if REAL_BONE:      # tissues of the full atlas foot; section through the middle facet (Claude, 2026-11-01)
        import fe.geometry as FG
        col.update({FG.TIB: "#D9CFBC", FG.FIBU: "#D9CFBC", FG.C_ANKLE: "#C9DFEC", FG.OBONE: "#CFC7B2",
                    FG.C_OTHER: "#CFE6DD", FG.SOFT: "#FFFFFF", FG.FASC: "#FFFFFF"})   # soft tissue left white
        yy = 16.4
    n = fr.shape[1]
    cm = ListedColormap([col.get(i, "#FFFFFF") for i in range(n)]); nm = BoundaryNorm(np.arange(-0.5, n + 0.5), n)
    j = np.searchsorted(g.ye, yy) - 1
    return g, lab[:, j, :], cm, nm


def stress_section():
    from fe.simulate import get_grid
    g = get_grid()
    z = np.load(os.path.join(ROOT, "data", "hires", "child_resect_f1.00.npz"), allow_pickle=True)
    vm = z["vm"].reshape(g.shape); lab = z["lab"].reshape(g.shape)
    j = np.searchsorted(g.ye, -1.1 if REAL_BONE else -2.0) - 1
    if REAL_BONE:      # hard tissues only (the soft-tissue envelope of the atlas foot is not shown)
        import fe.geometry as FG
        hard = [FG.TALUS, FG.CALC, FG.NAV, FG.CUB, FG.C_ANT, FG.C_MID, FG.C_POST, FG.C_TN, FG.C_CC, FG.COAL, FG.FIB,
                FG.INTERP, FG.FUSE, FG.TIB, FG.FIBU, FG.C_ANKLE, FG.OBONE, FG.C_OTHER]
        v = np.where(np.isin(lab[:, j, :], hard), vm[:, j, :], np.nan)
    else:
        v = np.where(lab[:, j, :] > 0, vm[:, j, :], np.nan)
    return g, v, lab[:, j, :]


def map_axes(bd, x, y, w, h, tag, src, pad=3.0):
    """The deformation-guided map of the virtual child (Fig. 7a), zoomed to the articular facets."""
    from maps import child_map
    ax = bd.ax(x, y, w, h)
    child_map(ax, tag, src, lw=0.45, arrow_width=0.012, arrow_scale=8)
    mk = np.load(os.path.join(ROOT, "results", "map_child_masks.npz"))
    fm = mk["post"] | mk["mid"] | mk["ant"]
    ix, iy = np.nonzero(fm.any(1))[0], np.nonzero(fm.any(0))[0]
    xs_, ys_ = mk["xs"], mk["ys"]
    cx, cy = (xs_[ix[0]] + xs_[ix[-1]]) / 2, (ys_[iy[0]] + ys_[iy[-1]]) / 2
    half = max(xs_[ix[-1]] - xs_[ix[0]] + 2 * pad, (ys_[iy[-1]] - ys_[iy[0]] + 2 * pad) * w / h) / 2
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half * h / w, cy + half * h / w)
    return ax


def fig4():
    H = 104.5
    bd = Board(FIG_W, H)
    SX = [6.6 + k * 30.2 for k in range(6)]; SW = 24.0          # stage columns of bands a and c

    # ============================================================ a: data engine (simulation stream)
    bd.panel(0.5, 0.5, 182.0, 30.5, "teal", tab="Data engine", letter="a")
    for k in range(5):
        bd.chevron(SX[k] + SW + 1.0, 14.2, w=4.2, h=5.0, fc=PAL["teal_chev"])
    heads = ["Virtual feet", "Coalition types", "Voxelization", "Surgical actions", "Voxel FE", "TCC-MechBank"]
    for k, s in enumerate(heads):
        title(bd, SX[k] + SW / 2, 5.2, s)
    # 1 virtual feet: a fan of renderings of bank feet
    for k, n in enumerate(["icon_foot3", "icon_foot2", "icon_foot1"]):
        bd.card(SX[0] + 0.6 + 4.0 * k, 7.2 + 2.3 * k, 15.4, 9.4, cut_dark_base(rgba_render(R(n))), z=3 + 3 * k)
    # 2 coalition types (reference foot of the engine)
    for k, t in enumerate(["I", "II", "III", "IV"]):
        xx, yy = SX[1] + (k % 2) * 12.3, 7.0 + (k // 2) * 7.4
        bd.img(xx, yy, 11.7, 7.0, cut_dark_base(rgba_render(R("icon_coal_" + t))), z=3)
        bd.box(xx + 0.2, yy + 0.3, 3.6 if len(t) > 2 else 2.9, 2.9, fc=NAVY, ec="none", r=0.6, z=5)
        bd.t(xx + 0.2 + (3.6 if len(t) > 2 else 2.9) / 2, yy + 2.45, t, size=FS_SMALL, color="white",
             weight="bold", ha="center", z=6)
    # 3 voxelization: sagittal section of the material labels
    bd.card(SX[2], 7.0, SW, 14.6, under=True)
    g, sec, cm, nm = labels_section()
    ax = bd.ax(SX[2] + 0.8, 7.7, SW - 1.6, 13.2)
    ax.pcolormesh(g.xe, g.ze, sec.T, cmap=cm, norm=nm, edgecolors="white", linewidth=0.04 if REAL_BONE else 0.1,
                  rasterized=True)
    if REAL_BONE:      # equal axes, centred in the card (real anatomy, not stretched)
        ax.set_xlim(-70, -8); ax.set_ylim(38, 74.5); ax.set_aspect("equal", adjustable="box", anchor="C"); hide(ax)
    else:
        ax.set_xlim(20, 92); ax.set_ylim(21, 60); hide(ax)
    # 4 surgical actions: engine renderings of the virtual child of Fig. 7a (resection with the resected coalition
    #   dashed, medializing osteotomy with its cut, interposition, subtalar fusion, load)
    from fe.geometry import Foot
    child = json.load(open(os.path.join(ROOT, "results", "eval_maps.json")))["child"]
    foot = Foot(**child["foot"])
    names = ["resect", "osteo", "interp", "fusion", "load"]
    s_ = foot.scale
    if REAL_BONE:      # box around the top of the calcaneus and the facets (template mm, scaled about the foot centre)
        import rbgeom as _G
        _c0 = _G.Template.get().c0
        wbox = tuple((_c0[i] + (a - _c0[i]) * s_, _c0[i] + (b - _c0[i]) * s_)
                     for i, (a, b) in enumerate(((-92.0, -16.0), (-20.0, 26.0), (42.0, 70.0))))
    else:
        wbox = ((35 + (6 - 35) * s_, 35 + (64 - 35) * s_), (-14.0 * s_, 22.0 * s_), (35 + (24 - 35) * s_, 35 + (37 - 35) * s_))
    crop = world_crop("resect", wbox)
    aw_, ah_ = 7.6, 6.6
    for k, nm in enumerate(names):
        r_, c_ = (0, k) if k < 3 else (1, k - 3)
        ix = SX[3] + 0.2 + c_ * (aw_ + 0.4) + (r_ * (aw_ + 0.4) / 2)
        iy = 7.0 + r_ * (ah_ + 0.9)
        bd.card(ix, iy, aw_, ah_, r=0.5, lw=0.35, pad=0.0)
        pl = Placed(bd, nm, ix + 0.25, iy + 0.25, aw_ - 0.5, ah_ - 0.5, z=4, crop_common=crop)
        annotate_action(pl, nm, foot, lw=0.4, clip=(pl.x0, pl.y0, pl.w, pl.h))
    # 5 voxel FE: von Mises stress, sagittal section through the posterior facet (engine solution, virtual child)
    bd.card(SX[4], 7.0, SW, 14.6, under=True)
    g, v, lb = stress_section()
    ax = bd.ax(SX[4] + 0.8, 7.7, SW - 1.6, 13.2)
    ax.pcolormesh(g.xe, g.ze, np.ma.masked_invalid(v.T), cmap=STRESS, norm=PowerNorm(0.6, 0, 3.0), rasterized=True)
    from fe.geometry import TALUS, CALC, NAV, CUB
    xc_, zc_ = (g.xe[:-1] + g.xe[1:]) / 2, (g.ze[:-1] + g.ze[1:]) / 2
    for L_ in (TALUS, CALC, NAV, CUB):
        m = (lb == L_).astype(float)
        if m.any():
            ax.contour(xc_, zc_, m.T, levels=[0.5], colors=[INK], linewidths=0.35)
    if REAL_BONE:
        ax.set_xlim(-100, 6); ax.set_ylim(14, 92); ax.set_aspect("equal", adjustable="box", anchor="C"); hide(ax)
    else:
        ax.set_xlim(4, 90); ax.set_ylim(8, 54.5); hide(ax)
    # 6 TCC-MechBank: FE solutions of a held-out TCC-MechBank foot (displacement, von Mises stress)
    for k, n in enumerate(["heldout_disp_fe", "heldout_stress_fe"]):
        bd.card(SX[5] + 1.6 + 5.4 * k, 7.6 + 3.6 * k, 15.6, 9.6, cut_dark_base(rgba_render(R(n))), z=3 + 3 * k)
    desc = [(["atlas hindfoot, warped by", "12 parameters; coalition", "types as in the 3-center cohort"] if REAL_BONE else
             ["parametric hindfoot (12", "parameters); coalition types", "as in the 3-center cohort"]),
            ["anatomy-based I–IV;", "Rozansky I–V morphologies"],
            (["1.5-mm voxels; 60×40×44", "analysis grid (2 mm)"] if REAL_BONE else ["fixed 57×24×32 grid,", "partial-volume materials"]),
            ["complete resection (defect),", "osteotomy t, interposition,", "fusion, load"],
            ["multigrid-CG solver,", "refinement-checked;", "calcaneus-fixed u"],
            ["2,016 solutions:", "σ(x), u(x), effects δ"]]
    for k, d in enumerate(desc):
        lines(bd, SX[k] + SW / 2, 24.6, d, ha="center")

    # ============================================================ b: co-constructed world model (training)
    yb = 33.0
    bd.panel(0.5, yb, 182.0, 45.0, "blue", tab="Co-constructed world model", letter="b")
    # ---- the two streams (soft-edged collages with label boxes, after the pipeline style)
    sims = [cut_dark_base(rgba_render(R(n))) for n in ["heldout_stress_fe", "child_resect_fe", "child_plan_fe",
                                                        "heldout_disp_fe"]]
    for k, im in enumerate(sims):
        bd.box(6.9 + (k % 2) * 13.5, 35.2 + (k // 2) * 7.6, 13.2, 7.3, fc="white", ec="none", r=0.6, z=2)
        bd.img(6.9 + (k % 2) * 13.5 + 0.3, 35.2 + (k // 2) * 7.6 + 0.3, 12.6, 6.7, im, z=3)
    bd.label_box(20.2, 42.85, "Simulation stream", fs=FS, ec=PAL["sim"], color=PAL["sim"])
    lines(bd, 20.2, 53.6, ["exact actions, full fields",
                         "(FE engine, atlas foot)" if REAL_BONE else "(open-source FE engine)"],   # (Claude, 2026-11-03)
          ha="center", color=INK2)
    cts = [ct_tile(f) for f in ["b2", "c3", "b4", "d2"]]
    for k, im in enumerate(cts):
        bd.img(6.9 + (k % 2) * 13.5, 56.6 + (k // 2) * 6.6, 13.2, 6.4, feather(rounded_mask(im, 0.06), 0.10),
               z=3, fit=False)
    bd.label_box(20.2, 63.2, "Clinical stream", fs=FS, ec=PAL["clin"], color=PAL["clin"], lw=0.55, ls=(0, (2.0, 1.0)))
    lines(bd, 20.2, 72.0, ["next step, not trained here:", "pre-/post-op CT, operation", "and outcome"], ha="center",
          color=INK2)
    # ---- crossing stream lines into the two encoders
    ys_, ya_ = 46.0, 66.5
    bezier(bd, (34.0, 42.8), (39.6, ys_ - 2.5), PAL["sim"], lw=0.8)
    bezier(bd, (34.0, 42.8), (39.6, ya_ - 1.8), PAL["sim"], lw=0.8)
    bezier(bd, (34.0, 60.0), (39.6, ys_ + 2.5), PAL["clin"], lw=0.8, ls=(0, (2.2, 1.3)))
    bezier(bd, (34.0, 60.0), (39.6, ya_ + 1.8), PAL["clin"], lw=0.8, ls=(0, (2.2, 1.3)))
    # ---- encoders
    bd.gtrap(39.8, ys_, 11.4, 16.0, 9.4, label="State\nencoder", fs=FS)
    bd.tsub(45.5, ys_ + 4.8, "E", "s", size=FS_LABEL, ha="center", color=NAVY)
    bd.gtrap(39.8, ya_ - 0.8, 11.4, 11.6, 6.4, label="Action\nencoder", fs=FS)
    bd.t(45.5, 75.4, "u = g(a)", size=FS, style="italic", ha="center", color=NAVY)
    # ---- latent state s and latent action u
    xs1, _ = tokens_round(bd, 53.4, 41.2, 3, 5, size=1.55, gap=0.36, seed=3, cmap=BLUE_TOK)
    bd.t((53.4 + xs1) / 2, 40.2, "s", size=FS_LABEL + 0.5, style="italic", ha="center", color=NAVY)
    xu1, _ = tokens_round(bd, 55.4, 63.4, 1, 4, size=1.55, gap=0.36, seed=5, cmap=VIOLET, lo=0.35, hi=0.95)
    bd.t(56.2, 62.4, "u", size=FS_LABEL + 0.5, style="italic", ha="center", color=PAL["act"])
    bd.arr(51.5, ys_, 53.0, ys_, color=NAVY, lw=0.6, head=(1.4, 0.85))
    bd.arr(51.5, ya_, 55.0, ya_, color=NAVY, lw=0.6, head=(1.4, 0.85))
    # ---- the composable latent transition D
    cx, cy, r = 70.6, 55.6, 6.3
    bd.sphere(cx, cy, r, ring=True, label="D")
    bd.arr(xs1 + 0.4, 46.0, cx - r * 0.82, cy - r * 0.45, color=NAVY, lw=0.6, head=(1.5, 0.9), rad=-0.15)
    bd.arr(xu1 + 0.4, 66.5, cx - r * 0.80, cy + r * 0.52, color=PAL["act"], lw=0.6, head=(1.5, 0.9), rad=0.15)
    bd.arr(cx + 4.6, cy - r - 1.1, cx - 4.6, cy - r - 1.1, color=INK2, lw=0.55, head=(1.4, 0.85), rad=0.55)
    bd.t(cx, cy - r - 5.3, "composable roll-out", size=FS_SMALL, ha="center", color=INK2)
    bd.t(cx, cy + r + 3.6, "latent transition", size=FS, ha="center", weight="bold", color=NAVY)
    bd.t(cx, cy + r + 6.0, "s′ = D(s, u)", size=FS, style="italic", ha="center", color=NAVY)
    # ---- predicted state and read-out heads
    xp1, _ = tokens_round(bd, 81.0, 51.0, 3, 5, size=1.55, gap=0.36, seed=8, cmap=BLUE_TOK, lo=0.35, hi=0.95)
    bd.t((81.0 + xp1) / 2, 50.0, "s′", size=FS_LABEL + 0.5, style="italic", ha="center", color=NAVY)
    bd.arr(cx + r + 0.3, cy, 80.6, cy, color=NAVY, lw=0.6, head=(1.4, 0.85))
    hx, hw, hh = 90.4, 31.6, 8.9
    HY = [35.6, 45.8, 56.0, 66.2]
    bus = 88.6
    bd.seg([xp1 + 0.3, bus], [cy, cy], color=NAVY, lw=0.6)
    bd.seg([bus, bus], [HY[0] + hh / 2, HY[2] + hh / 2], color=NAVY, lw=0.6)
    bd.seg([bus, bus], [HY[2] + hh / 2, HY[3] + hh / 2], color=MUTED, lw=0.6, ls=(0, (1.8, 1.2)))
    for k, hy in enumerate(HY):
        dashed = k == 3
        bd.card(hx, hy, hw, hh, shadow=not dashed, ec=MUTED if dashed else "#B8C2CF",
                ls=(0, (1.8, 1.2)) if dashed else "-", lw=0.5)
        bd.arr(bus, hy + hh / 2, hx - 0.1, hy + hh / 2, color=MUTED if dashed else NAVY, lw=0.6, head=(1.4, 0.85),
               ls=(0, (1.8, 1.2)) if dashed else "-")
    for k, (sym, nm) in enumerate([("σ", "heldout_stress_cm"), ("u", "heldout_disp_cm")]):
        hy = HY[k]
        bd.t(hx + 1.2, hy + 3.5, "Neural field", size=FS, weight="bold", color=NAVY)
        e = bd.tsub(hx + 1.2, hy + 6.6, "Φ", sym, size=FS, style="normal", sub_style="italic", color=NAVY)
        bd.t(e + 0.3, hy + 6.6, "(x | s′)", size=FS, color=NAVY)
        bd.img(hx + hw - 13.0, hy + 0.5, 12.4, hh - 1.0, cut_dark_base(rgba_render(R(nm))), z=5)
    hy = HY[2]
    bd.t(hx + 1.2, hy + 3.5, "Effect head Q(s′)", size=FS, weight="bold", color=NAVY)
    bd.t(hx + 1.2, hy + 6.6, "17 effects", size=FS, color=NAVY)
    bd.seal(hx + 14.8, hy + 6.25, 1.8)
    vals = [0.55, 0.85, 0.35, 0.7, 0.5, 0.95, 0.4]
    for j, vv in enumerate(vals):
        bd.rect(hx + hw - 10.4 + j * 1.35, hy + 7.6 - 5.6 * vv, 0.95, 5.6 * vv, fc=BLUE_L, ec="none", z=5)
    bd.seg([hx + hw - 10.8, hx + hw - 1.0], [hy + 7.65, hy + 7.65], lw=0.4, color=INK2, z=5)
    hy = HY[3]
    bd.t(hx + 1.2, hy + 3.5, "Outcome head R(s′)", size=FS, color=INK2)
    bd.t(hx + 1.2, hy + 6.6, "clinical stream, not trained here", size=FS_SMALL, color=INK2)
    # ---- effect-grounded objectives (losses in red)
    lx, ly, lw_, lh = 125.2, 35.6, 56.0, 40.6
    bd.card(lx, ly, lw_, lh, ec=NAVY, lw=0.6, r=1.2)
    bd.t(lx + 1.6, ly + 3.6, "Objectives", size=FS_LABEL, weight="bold", color=NAVY)
    e = ell_L(bd, lx + 1.6, ly + 7.6, "field"); bd.t(e + 0.15, ly + 7.6, ",", size=FS, color=PAL["loss"])
    e = ell_L(bd, e + 1.2, ly + 7.6, "eff")
    bd.t(e + 1.2, ly + 7.6, "reconstruct fields and effects", size=FS_SMALL, color=INK2)
    e = ell_L(bd, lx + 1.6, ly + 11.6, "roll")
    bd.t(e + 1.2, ly + 11.6, "two-step roll-out: resection → osteotomy", size=FS_SMALL, color=INK2)

    def pair(yr, c2, fill2, lab2, seed):
        bd.cluster(lx + 10.6, yr - 1.8, 1.9, 1.25, "#D9534F", n=4, seed=seed)
        bd.cluster(lx + 10.6, yr + 1.8, 1.9, 1.25, c2, n=4, seed=seed + 3, fill=fill2)
        bd.t(lx + 8.0, yr - 3.3, "a", size=FS_SMALL, style="italic", color=INK2, ha="center")
        if lab2.endswith("⁻"):   # superscript minus set in the text font (2026-11-03)
            bd.tsup(lx + 8.0, yr + 4.7, lab2[:-1], "−", size=FS_SMALL, color=INK2, ha="center")
        else:
            bd.t(lx + 8.0, yr + 4.7, lab2, size=FS_SMALL, style="italic", color=INK2, ha="center")
        for dy in (-1.8, 1.8):
            bd.seg([lx + 12.3, lx + 14.5], [yr + dy, yr], color=NAVY, lw=0.5)
        bd.circ(lx + 15.8, yr, 1.45, fc=NAVY, ec="none", z=6)
        bd.t(lx + 15.8, yr + 0.75, "D", size=FS_SMALL, style="italic", weight="bold", color="white", ha="center", z=7)

    # effect-swap: actions whose FE effects agree must predict compatible futures
    yr = ly + 18.0
    bd.icon_swap(lx + 4.2, yr, 2.5)
    pair(yr, "#2F6FB5", None, "a′", 1)
    bd.t(lx + 18.6, yr + 0.9, "»", size=FS_LABEL, color=NAVY, ha="center")
    for dy, c in [(-1.8, "#F2B79F"), (1.8, "#A9D6E5")]:
        for q, hgt in enumerate([1.5, 2.6, 1.1]):
            bd.rect(lx + 20.6 + q * 1.15, yr + dy + 1.0 - hgt, 0.85, hgt, fc=c, ec="none", z=6)
    bd.badge(lx + 25.6, yr, 1.15, "check")
    e = ell_L(bd, lx + 28.4, yr - 0.9, "swap")
    bd.t(e + 1.0, yr - 0.9, "Eq. (2)", size=FS_SMALL, color=INK2)
    bd.t(lx + 28.4, yr + 2.4, "equal FE effects →", size=FS_SMALL, color=INK2)
    bd.t(lx + 28.4, yr + 4.6, "compatible futures", size=FS_SMALL, color=INK2)
    # action sensitivity: neighbouring actions whose effects differ must stay apart
    yr = ly + 28.4
    bd.icon_sens(lx + 4.2, yr, 2.5)
    pair(yr, "#D9534F", "#E99A8F", "a⁻", 6)
    bd.arr(lx + 17.4, yr, lx + 19.6, yr, color=NAVY, lw=0.5, head=(1.1, 0.7))
    bd.rect(lx + 20.6, yr - 2.4, 0.95, 4.0, fc="#F2B79F", ec="none", z=6)
    bd.rect(lx + 22.0, yr - 0.2, 0.95, 1.8, fc=GREY, ec="none", z=6)
    for s_ in (1, -1):
        bd.seg([lx + 24.4, lx + 26.0], [yr - 0.8 * s_, yr + 0.8 * s_], color=GREY_D, lw=0.7, z=6)
    e = ell_L(bd, lx + 28.4, yr - 0.9, "sens")
    bd.t(e + 1.0, yr - 0.9, "Eq. (3)", size=FS_SMALL, color=INK2)
    bd.t(lx + 28.4, yr + 2.4, "near actions, different", size=FS_SMALL, color=INK2)
    bd.t(lx + 28.4, yr + 4.6, "effects: kept apart", size=FS_SMALL, color=INK2)
    # clinical-stream objectives (dashed: not trained)
    yd = ly + 36.9
    bd.box(lx + 1.2, yd - 2.5, lw_ - 2.4, 4.3, fc="none", ec=MUTED, lw=0.5, ls=(0, (1.8, 1.2)), r=0.7, z=5)
    e = ell_L(bd, lx + 2.4, yd + 0.6, "bind", color=MUTED); bd.t(e + 0.15, yd + 0.6, ",", size=FS, color=MUTED)
    e = ell_L(bd, e + 1.2, yd + 0.6, "out", color=MUTED)
    bd.t(e + 1.2, yd + 0.6, "clinical stream; not trained here", size=FS_SMALL, color=INK2)

    # ============================================================ c: rehearsal and planning (use)
    yc = 80.0
    bd.panel(0.5, yc, 182.0, 24.0, "green", tab="Rehearsal", letter="c")
    for k in range(5):
        bd.chevron(SX[k] + SW + 1.0, yc + 10.6, w=4.2, h=5.0, fc=PAL["green_chev"])
    heads = ["Coalition foot", "Complete resection", "Candidate adjuncts", "Latent rehearsal", "Ensemble gate",
             "FE verification"]
    for k, s in enumerate(heads):
        title(bd, SX[k] + SW / 2, yc + 4.2, s)
    iy, ih = yc + 5.4, 11.6
    bd.img(SX[0], iy, SW, ih, cut_dark_base(rgba_render(R("icon_child_pre"))), z=3)
    bd.img(SX[1], iy, SW, ih, cut_dark_base(rgba_render(R("icon_child_resect"))), z=3)
    # candidate adjuncts: osteotomy translations x interposition
    for row in range(2):
        for q, tt in enumerate(["−10", "…", "0", "…", "+10"]):
            xx, yy = SX[2] + 0.2 + q * 4.75, iy + 1.2 + row * 5.0
            bd.box(xx, yy, 4.25, 3.9, fc=PAL["act_l"], ec=PAL["act"], lw=0.4, r=0.6, z=4)
            bd.t(xx + 2.12, yy + 2.65, tt, size=FS_SMALL, ha="center", color=NAVY, z=5)
            if row == 1:
                bd.circ(xx + 3.55, yy + 0.7, 0.42, fc="#F5CE55", ec="#B8922B", lw=0.3, z=6)
    # latent rehearsal of the virtual child: real latent states (CoMech seed 0), ensemble objective J (Fig. 1c)
    latent_fan(bd, SX[3] - 0.6, iy - 0.6, 15.4, ih + 0.4, ms=0.62, star=0.62, lw_ray=0.16, sphere=True)
    lines(bd, SX[3] + 15.4, iy + 3.4, ["rank by J", "(Eq. 4)"], step=2.3, color=NAVY, size=FS)
    j_bar(bd, SX[3] + 15.6, iy + 7.4, 7.6, h=0.8, label=False)
    bd.t(SX[3] + 15.4, iy + 10.6, "low", size=FS_SMALL, color=INK2)
    bd.t(SX[3] + 23.4, iy + 10.6, "high", size=FS_SMALL, color=INK2, ha="right")
    # ensemble gate: three models agree (proceed) or disagree (abstain)
    for k in range(3):
        bd.sphere(SX[4] + 3.4, iy + 2.0 + k * 3.8, 1.55, c0="#20306E", c1="#7D93E6")
    for k in range(3):
        bd.seg([SX[4] + 5.1, SX[4] + 9.4], [iy + 2.0 + k * 3.8, iy + 5.8], color=NAVY, lw=0.45)
    bd.badge(SX[4] + 11.2, iy + 3.1, 1.6, "check"); bd.badge(SX[4] + 11.2, iy + 8.5, 1.6, "cross")
    bd.t(SX[4] + 13.6, iy + 3.9, "agree", size=FS_SMALL, color=INK2)
    bd.t(SX[4] + 13.6, iy + 9.3, "abstain", size=FS_SMALL, color=INK2)
    # FE verification: the FE-verified map of the virtual child (Fig. 7a)
    bd.card(SX[5], iy - 0.2, 16.6, ih + 0.4, under=True)
    map_axes(bd, SX[5] + 0.5, iy + 0.3, 15.6, ih - 0.6, "plan", "fe")
    bd.circ(SX[5] + 19.6, iy + 5.6, 2.3, fc=FE, ec="white", lw=0.5, z=8)
    bd.t(SX[5] + 19.6, iy + 6.45, "FE", size=FS_SMALL, weight="bold", color="white", ha="center", z=9)
    _ct = json.load(open(os.path.join(ROOT, "results", "eval_maps.json")))["child"]["foot"]["ctype"]   # (Claude, 2026-11-01)
    desc = [["virtual child of Fig. 7a", f"(type {_ct} coalition)"], ["first action fixed", "(hard constraint)"],
            ["osteotomy t (mm),", "± interposition"], ["overload, transfer,", "deformation, invasiveness"],
            ["three models", "(different seeds)"], ["surgical map, tolerance", "envelope, equivalent plans"]]
    for k, d in enumerate(desc):
        lines(bd, SX[k] + SW / 2, yc + 19.9, d, ha="center", step=2.15)
    os.makedirs(OUT, exist_ok=True)
    bd.save(os.path.join(OUT, "fig4"))
    print("saved fig4")


if __name__ == "__main__":
    fig4()
