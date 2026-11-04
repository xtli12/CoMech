"""Figure 1 (concept, v4, 2026-10-28; panel d redrawn as pictures on 2026-11-03): planning today, the co-constructed mechanical world model, rehearsal before
the incision for one virtual child, and the in silico evidence of this study.  Drawn in the visual language of the
authors' pipeline figures (tinted panels, pale chevrons, gradient encoders, a glossy transition sphere,
swap/sensitivity badges), with every picture taken from this study:

  * a: CT crop (Supplementary Fig. 9), atlas mesh (Fig. 2a), engine rendering of the virtual child of Fig. 7a;
  * b: engine renderings of the surgical actions applied to that child (render_actions.py; annotations projected
    with the rendering camera), the CT reconstruction of P05-R (Fig. 8a), CoMech predictions for a held-out foot
    (Fig. 5a), engine FE solutions (simulation stream) and CT crops (clinical stream);
  * c: the latent states of the 42 adjuncts that CoMech rehearsed for the virtual child (child_latent.py: real
    seed-0 states, ensemble objective J; Fig. 7a), its FE-verified maps (Fig. 7a) and its FE tolerance curve
    (virtual cohort, Fig. 6);
  * d: counts and headline results, each read from the result files named below.
Icons, badges, latent tokens and the transition sphere are schematic.  All numbers are read from results/*.json
(and results_gpu/numbers_gpu_text.json for the scaling extension); nothing is typed in by hand except counts of the
study design that are fixed (33 atlas models, 8 actions per foot)."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from figstyle import *
from schem import *
from figparts import *
from matplotlib.path import Path as MPath
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "figures")
CROPS = os.path.join(ROOT, "..", "crops")
RES = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
R = lambda n: np.load(os.path.join(ROOT, "results", "renders", n + ".npz"))
CT = lambda f: os.path.join(ROOT, "results", "ct_tiles_s9", f + ".png")
NAVY = PAL["navy"]
BLUE_TOK = LinearSegmentedColormap.from_list("tok", ["#DCEBF6", BLUE_L, SKY, BLUE, "#004A75"])
VIOLET = LinearSegmentedColormap.from_list("act", ["#EEE8F8", "#C6B5E6", "#9277C6", PAL["act"], "#4B347E"])
GAP = 1.6                                   # gap between panels (mm)


def lines(bd, x, y, items, ha="left", step=2.15, size=FS_SMALL, color=INK2):
    """Stacked lines; the step grows with the printed size (base + BUMP, v5) so the leading stays as designed."""
    step = step * (size + BUMP) / size
    for k, it in enumerate(items):
        if isinstance(it, str):
            it = (it, {})
        s, kw = it
        kw = dict(kw); kw.setdefault("size", size); kw.setdefault("color", color)
        bd.t(x, y + k * step, s, ha=ha, **kw)


def head(bd, x, y, letter, s, color=None):
    """Panel letter (Times New Roman Bold) and title (Microsoft YaHei Bold) on one baseline (v5, 2026-11-04)."""
    bd.t(x, y + 3.0, letter, size=FS_PANEL, weight="bold", color=INK, gid="pletter")
    bd.t(x + TITLE_DX, y + 3.0, s, size=FS_PTITLE5, weight="bold", color=INK, gid="ptitle")


def tile(path, crop=None):
    im = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255
    if crop:
        r0, r1, c0, c1 = crop
        h, w = im.shape[:2]
        im = im[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)]
    return im


def white_to_alpha(im, thr=0.94):
    im = np.asarray(im, float)
    a = 1 - np.clip((im.min(2) - thr) / (1 - thr), 0, 1)
    return np.dstack([im, a])


def vbezier(bd, p0, p3, color, lw=0.8, ls="-", z=3, k=0.55):
    """Smooth stream line with vertical end tangents, arrowhead at p3 (pointing up)."""
    (x0, y0), (x3, y3) = p0, p3
    dy = (y3 - y0) * k
    verts = [(x0, y0), (x0, y0 + dy), (x3, y3 - dy), (x3, y3)]
    bd.b.add_patch(PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]), fc="none",
                             ec=color, lw=lw, ls=ls, zorder=z, capstyle="butt"))
    bd.arr(x3, y3 + 1.4, x3, y3 - 0.05, color=color, lw=lw, head=(1.4, 0.85), z=z + 0.1)


def pixel_frame(seed, h=9, w=12):
    """A blurred 'video frame' of pixel blocks (schematic of image/video world models)."""
    from scipy.ndimage import zoom, gaussian_filter
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.random((h, w)), 1.2)
    base = (base - base.min()) / (base.max() - base.min())
    im = zoom(base, 24, order=0)
    return np.dstack([0.25 + 0.65 * im] * 3)


def pct(a, b):
    return 100.0 * (b / a - 1.0)


def fig1():
    H = 102.5                                                     # v5: +6 mm (larger text, formulas in c)
    bd = Board(FIG_W, H)
    child = RES("eval_maps.json")["child"]
    foot = __import__("fe.geometry", fromlist=["Foot"]).Foot(**child["foot"])
    TOP, HT = 0.5, 72.0                                           # panels a-c
    PA, PB, PC = (0.5, 34.4), (0.5 + 34.4 + GAP, 95.4), (0.5 + 34.4 + GAP + 95.4 + GAP, 0)
    PC = (PC[0], FIG_W - 0.5 - PC[0])
    bd.panel(PA[0], TOP, PA[1], HT, "warm")
    bd.panel(PB[0], TOP, PB[1], HT, "blue")
    bd.panel(PC[0], TOP, PC[1], HT, "green")
    bd.chevron(PA[0] + PA[1] - 1.2, 37.0, w=3.8, h=5.6, fc=PAL["warm_chev"], z=8)
    bd.chevron(PB[0] + PB[1] - 1.2, 37.0, w=3.8, h=5.6, fc=PAL["green_chev"], z=8)

    # ================================================================== a: planning today
    x0 = PA[0] + 1.5
    head(bd, x0, 1.0, "a", "Planning today")
    rows = [("ct", 7.0, ["Imaging"], ["shape, not force"]),
            ("fe", 19.2, ["Finite element", "analysis"], ["hours to weeks per", "patient and scenario"]),
            ("wm", 31.6, ["Image/video", "world models"], ["imagine pixels,", "not mechanics"])]
    for kind, y, hd, body in rows:
        if kind == "ct":
            bd.card(x0, y, 10.4, 10.4, rounded_mask(tile(CT("a3"), (0.1, 0.9, 0.1, 0.9)), 0.06), r=0.7, pad=0.35)
        elif kind == "fe":
            bd.img(x0 + 0.1, y - 0.4, 10.4, 11.6, white_to_alpha(tile(os.path.join(CROPS, "r1_061.png"))), z=3)
        else:
            bd.film(x0, y + 0.8, 10.8, 8.4, [pixel_frame(3), pixel_frame(8)], gap=0.4)
        for j, h_ in enumerate(hd):
            bd.t(x0 + 12.3, y + 2.6 + 2.6 * j, h_, size=FS, weight="bold", color=NAVY)
        lines(bd, x0 + 12.3, y + 5.2 + 2.6 * (len(hd) - 1), body, step=2.05)
    # the population rule, drawn on the virtual child of Fig. 7a (engine rendering, as in b)
    yr = 45.4
    bd.arr(x0 + 5.4, 42.6, x0 + 5.4, yr - 0.3, color=PAL["warm_edge"], lw=0.7, head=(1.4, 0.85))
    bd.t(x0 + 7.0, 44.3, "so decisions rest on", size=FS_SMALL, color=INK2)
    bd.card(x0, yr, PA[1] - 3.0, HT - yr - 0.9, ec=PAL["warm_edge"], lw=0.55, r=1.0)
    bd.t(x0 + 1.3, yr + 3.3, "Population rule", size=FS, weight="bold", color=PAL["warm_edge"])
    pl = Placed(bd, "pre", x0 + 0.6, yr + 4.0, PA[1] - 4.2, 11.6, z=4)
    if REAL_BONE:      # real-bone engine: centroids measured on the label raster (rbannot.py; Claude, 2026-11-01)
        from rbannot import mask_centroid
        cC = pl.mm(mask_centroid(foot, "coal")[None])[0]; cP = pl.mm(mask_centroid(foot, "post", exclude_coal=True)[None])[0]
    else:
        from fe.geometry import FootGeom, Action as _A
        gpre = FootGeom(foot, _A())
        xs_ = np.arange(10, 80, 0.25); ys_ = np.arange(-16, 26, 0.25)
        Xg, Yg = np.meshgrid(xs_, ys_, indexing="ij")
        br, fib = gpre.coalition_xy(Xg, Yg); ant_, mid_, post_ = gpre.facet_masks(Xg, Yg)
        def centroid(m):
            xm, ym = Xg[m].mean(), Yg[m].mean()
            return pl.mm(world_mm(foot, [xm], [ym], [gpre.z_joint(xm, ym) + foot.cart_t]))[0]
        cC = centroid(br | fib); cP = centroid(post_ & ~(br | fib))
    bd.t(cC[0], cC[1] + 0.9, "C", size=FS, weight="bold", color="white", ha="center", z=9)
    bd.t(cP[0], cP[1] + 0.9, "P", size=FS, weight="bold", color=NAVY, ha="center", z=9)
    # C/P bar: the child's ratio against the 50% threshold of the rule
    bx, by, bw = x0 + 1.4, yr + 18.0, PA[1] - 6.0
    r_ = child["ratio"]
    bd.rect(bx, by, bw, 1.5, fc="#ECE7E4", z=5)
    bd.rect(bx, by, bw * r_, 1.5, fc=COAL, z=6)
    bd.seg([bx + 0.5 * bw, bx + 0.5 * bw], [by - 0.8, by + 2.3], color=INK, lw=0.6, z=7)
    bd.t(bx + 0.5 * bw + 0.5, by - 0.4, "50%", size=FS_SMALL, color=INK, z=7)
    bd.t(bx, by + 4.4, f"C/P = {100 * r_:.0f}%: " + ("resect alone" if r_ < 0.5 else "do not resect alone"),
         size=FS_SMALL, color=INK)          # what the 50% rule says for this child (Claude, 2026-11-01)
    bd.t(bx, by + 6.8, "(one number for every child)", size=FS_SMALL, color=INK2)

    # ================================================================== b: CoMech
    xb = PB[0] + 1.5
    head(bd, xb, 1.0, "b", "CoMech: a co-constructed biomechanical world model")   # same term as the title (2026-11-03)
    # ---- surgical actions u: engine renderings of the virtual child
    ax0, ay0, aw, ah = xb, 5.7, PB[1] - 3.0, 14.6
    bd.box(ax0, ay0, aw, ah, fc="#F5F1FB", ec=PAL["act"], lw=0.5, r=1.0, z=1)
    # "Surgical / action u" in bold, u on the second line (authors, 2026-11-04)
    bd.t(ax0 + 1.2, ay0 + 3.4, "Surgical", size=FS, weight="bold", color=PAL["act"], gid="bold")
    bd.t(ax0 + 1.2, ay0 + 6.1, "action", size=FS, weight="bold", color=PAL["act"], gid="bold")
    bd.t(ax0 + 1.2 + bd.tw("action ", FS, fontweight="bold"), ay0 + 6.1, "u", size=FS, style="italic",
         weight="bold", color=PAL["act"], gid="bold")
    lines(bd, ax0 + 1.2, ay0 + 9.6, ["on the virtual", "child (engine)"], step=1.95, color=INK2)
    acts = [("resect", "resection", "defect d"), ("osteo", "osteotomy", "translation t"),
            ("interp", "interposition", "fat graft"), ("fusion", "fusion", "subtalar"), ("load", "load", "× body weight")]
    crop = common_crop([a[0] for a in acts])
    aw1, gap1 = 14.0, 1.15
    axs = ax0 + 14.4
    for k, (nm, lab, sub) in enumerate(acts):
        ix = axs + k * (aw1 + gap1)
        pl = Placed(bd, nm, ix, ay0 + 0.7, aw1, 9.0, z=4, crop_common=crop)
        annotate_action(pl, nm, foot)
        bd.t(ix + aw1 / 2, ay0 + 11.4, lab, size=FS_SMALL, ha="center", color=INK)
        bd.t(ix + aw1 / 2, ay0 + 13.45, sub, size=FS_SMALL, ha="center", color=INK2,
             style="italic" if len(sub) == 1 else "normal")
    # ---- main row: child's hindfoot -> state encoder -> s -> D -> s' -> decoders -> fields and effects
    yc = 35.6
    ct = rgba_render(R("ct_P05R_medial"))
    bd.img(xb + 0.2, 24.0, 15.4, 19.0, ct, z=3)
    lines(bd, xb + 8.2, 45.0, [("Child's hindfoot", {"color": INK}), "(pre-operative state)"],
          ha="center", step=1.95)
    bd.gtrap(xb + 16.6, yc, 9.0, 12.0, 6.6, label="State\nencoder", fs=FS_SMALL)
    bd.tsub(xb + 21.1, yc + 8.3, "E", "s", size=FS, ha="center", color=NAVY)
    xs1, _ = tokens_round(bd, xb + 27.0, yc - 4.0, 3, 5, size=1.3, gap=0.32, seed=3, cmap=BLUE_TOK)
    bd.t((xb + 27.0 + xs1) / 2, yc - 4.9, "s", size=FS_LABEL, style="italic", ha="center", color=NAVY)
    bd.arr(xb + 25.7, yc, xb + 26.8, yc, color=NAVY, lw=0.55, head=(1.1, 0.7))
    sx, sr = xb + 42.2, 5.4
    bd.sphere(sx, yc, sr, ring=True, label="D")
    bd.arr(xs1 + 0.4, yc, sx - sr - 0.3, yc, color=NAVY, lw=0.55, head=(1.2, 0.75))
    bd.arr(sx, ay0 + ah + 0.1, sx, yc - sr - 0.3, color=PAL["act"], lw=0.65, head=(1.4, 0.85))
    xu, _ = tokens_round(bd, sx + 1.5, ay0 + ah + 1.5, 4, 1, size=1.15, gap=0.28, seed=5, cmap=VIOLET, lo=0.35, hi=0.95)
    bd.t(xu + 0.8, ay0 + ah + 2.65, "u", size=FS, style="italic", color=PAL["act"])
    lines(bd, sx - 1.3, 25.2, [("latent transition", {"weight": "bold", "color": NAVY}),
                               ("s′ = D(s, u)", {"color": NAVY, "style": "italic"})], ha="right", step=2.05)
    xp0 = sx + sr + 2.6
    xp1, _ = tokens_round(bd, xp0, yc - 4.0, 3, 5, size=1.3, gap=0.32, seed=8, cmap=BLUE_TOK, lo=0.35, hi=0.95)
    bd.t((xp0 + xp1) / 2, yc - 4.9, "s′", size=FS_LABEL, style="italic", ha="center", color=NAVY)
    bd.arr(sx + sr + 0.3, yc, xp0 - 0.3, yc, color=NAVY, lw=0.55, head=(1.2, 0.75))
    # read-out heads: neural fields (stress, displacement) and the effect head (17 descriptors)
    hx = xp1 + 2.0
    bd.gtrap(hx, yc, 7.6, 6.6, 12.6, c0=PAL["enc1"], c1=PAL["enc0"], label="Read-\nout", fs=FS_SMALL)
    bd.arr(xp1 + 0.3, yc, hx - 0.2, yc, color=NAVY, lw=0.55, head=(1.1, 0.7))
    ox = hx + 9.2
    st = cut_dark_base(rgba_render(R("heldout_stress_cm"))); dp = rgba_render(R("heldout_disp_cm"))
    cw, chh = 12.2, 7.6
    bd.card(ox, 23.6, cw, chh, st, pad=0.35)
    bd.card(ox, 32.2, cw, chh, dp, pad=0.35)
    for yy in (23.6 + chh / 2, 32.2 + chh / 2, 43.6):
        bd.elbow([(hx + 7.6, yc), (hx + 8.6, yc), (hx + 8.6, yy), (ox - 0.2, yy)], color=NAVY, lw=0.5,
                 head=(1.1, 0.7))
    tx = ox + cw + 0.9
    lines(bd, tx, 26.4, [("stress", {"color": INK}), ("σ(x)", {"color": INK, "style": "italic"})], step=2.05)
    lines(bd, tx, 35.0, [("displacement", {"color": INK}), ("u(x)", {"color": INK, "style": "italic"})], step=2.05)
    # effect head: 17 effect descriptors (schematic bars) with the delta seal
    ex, ey = ox, 40.6
    bd.box(ex, ey, cw, 6.0, fc="white", ec="#B8C2CF", lw=0.45, r=0.7, z=3)
    vals = [0.55, 0.85, 0.35, 0.7, 0.5, 0.95, 0.4]
    for j, vv in enumerate(vals):
        bd.rect(ex + 1.0 + j * 1.0, ey + 5.0 - 3.9 * vv, 0.72, 3.9 * vv, fc=BLUE_L, ec="none", z=4)
    bd.seal(ex + cw - 1.9, ey + 3.0, 1.4, fs=FS)
    lines(bd, tx, ey + 2.6, [("17 effects", {"color": INK}), ("δ", {"color": INK, "style": "italic"})], step=2.05)
    # ---- the two data streams and the co-construction constraints
    ys0 = 51.5
    sims = [cut_dark_base(rgba_render(R(n))) for n in ["heldout_stress_fe", "child_resect_fe", "child_plan_fe",
                                                        "heldout_disp_fe"]]
    for k, im in enumerate(sims):
        bx, by = xb + 0.2 + (k % 2) * 12.6, ys0 + (k // 2) * 6.4
        bd.box(bx, by, 12.2, 6.1, fc="white", ec="none", r=0.5, z=2)
        bd.img(bx + 0.25, by + 0.25, 11.7, 5.6, im, z=3)
    bd.label_box(xb + 12.6, ys0 + 6.25, "Simulation stream", fs=FS_SMALL, ec=PAL["sim"], color=PAL["sim"])
    lines(bd, xb + 12.6, ys0 + 14.6, ["exact actions, full fields",
                                      "(FE engine, atlas foot)" if REAL_BONE else "(open-source FE engine)"],   # (Claude, 2026-11-03)
          ha="center", step=1.95)
    cts = [tile(CT(f), (0.08, 0.86, 0.10, 0.90)) for f in ["b2", "c3", "b4", "d2"]]
    cx0 = PB[0] + PB[1] - 1.5 - 25.0
    for k, im in enumerate(cts):
        bd.img(cx0 + (k % 2) * 12.6, ys0 + (k // 2) * 6.4, 12.2, 6.1, feather(rounded_mask(im, 0.06), 0.10), z=3,
               fit=False)
    bd.label_box(cx0 + 12.4, ys0 + 6.25, "Clinical stream", fs=FS_SMALL, ec=PAL["clin"], color=PAL["clin"],
                 ls=(0, (2.0, 1.0)), lw=0.55)
    lines(bd, cx0 + 12.4, ys0 + 14.6, ["next step: pre-/post-op", "CT, operation, outcome", "(not used)"],
          ha="center", step=1.95)
    vbezier(bd, (xb + 23.0, ys0 - 0.2), (sx - 3.0, yc + sr - 0.6), PAL["sim"], lw=0.8)
    vbezier(bd, (cx0 + 2.4, ys0 - 0.2), (sx + 3.0, yc + sr - 0.6), PAL["clin"], lw=0.8, ls=(0, (2.2, 1.3)))
    # co-construction: effect-swap and action sensitivity (pairs of actions -> D -> effects), as in Fig. 4b
    cx = xb + 27.6
    bd.t(cx, ys0 + 0.9, "Co-construction of the action space", size=FS_SMALL, weight="bold", color=NAVY)

    def pair(yr, c2, fill2, lab2, seed, same):   # graphics 0.5 mm right of the 2026-11-02 layout: room for a⁻
        bd.cluster(cx + 7.1, yr - 1.45, 1.45, 0.95, "#D9534F", n=4, seed=seed)
        bd.cluster(cx + 7.1, yr + 1.45, 1.45, 0.95, c2, n=4, seed=seed + 3, fill=fill2)
        bd.t(cx + 4.4, yr - 1.0, "a", size=FS_SMALL, style="italic", color=INK2, ha="center")
        if lab2.endswith("⁻"):   # superscript minus set in the text font (2026-11-03)
            bd.tsup(cx + 4.4, yr + 2.5, lab2[:-1], "−", size=FS_SMALL, color=INK2, ha="center")
        else:
            bd.t(cx + 4.4, yr + 2.5, lab2, size=FS_SMALL, style="italic", color=INK2, ha="center")
        for dy in (-1.45, 1.45):
            bd.seg([cx + 8.4, cx + 9.9], [yr + dy, yr], color=NAVY, lw=0.45)
        bd.circ(cx + 11.0, yr, 1.15, fc=NAVY, ec="none", z=6)
        bd.t(cx + 11.0, yr + 0.62, "D", size=FS_SMALL, style="italic", weight="bold", color="white", ha="center", z=7)
        bd.arr(cx + 12.3, yr, cx + 13.7, yr, color=NAVY, lw=0.45, head=(0.9, 0.55))
        if same:
            for dy, c in [(-1.45, "#F2B79F"), (1.45, "#A9D6E5")]:
                for q, hgt in enumerate([1.1, 1.9, 0.8]):
                    bd.rect(cx + 14.3 + q * 0.95, yr + dy + 0.75 - hgt, 0.7, hgt, fc=c, ec="none", z=6)
            bd.badge(cx + 18.3, yr, 0.85, "check")
        else:
            bd.rect(cx + 14.3, yr - 1.9, 0.75, 3.3, fc="#F2B79F", ec="none", z=6)
            bd.rect(cx + 15.4, yr - 0.1, 0.75, 1.5, fc=GREY, ec="none", z=6)
            bd.badge(cx + 18.3, yr, 0.85, "cross")

    yr1, yr2 = ys0 + 6.3, ys0 + 13.8
    bd.icon_swap(cx + 1.6, yr1, 1.6)
    pair(yr1, "#2F6FB5", None, "a′", 1, True)
    lines(bd, cx + 20.0, yr1 - 1.3, [("effect-swap:", {"color": INK}), "equal FE effects →", "compatible futures"],
          step=1.9)
    bd.icon_sens(cx + 1.6, yr2, 1.6)
    pair(yr2, "#D9534F", "#E99A8F", "a⁻", 6, False)
    lines(bd, cx + 20.0, yr2 - 1.3, [("action sensitivity:", {"color": INK}), "near actions, other",
                                     "effects → kept apart"], step=1.9)

    # ================================================================== c: rehearsal before the incision
    xc = PC[0] + 1.5
    head(bd, xc, 1.0, "c", "Rehearsal before surgery")
    # latent rehearsal of the virtual child: real latent states (seed 0), ensemble objective J
    bd.card(xc, 5.7, 27.4, 26.5, under=True, r=0.8, shadow=True)
    info = latent_fan(bd, xc + 0.6, 6.3, 26.2, 21.6, ms=1.0, star=1.05, lw_ray=0.25)
    sp = info["s"]
    bd.t(sp[0] + 1.6, sp[1] - 0.6, "s", size=FS, style="italic", color=NAVY, z=9)
    ch_pt = info["pts"][info["choice"]]
    # marker legend (two lines) and the J scale
    def mark(x_, y_, kind):
        if kind == "D":
            bd.b.scatter([x_], [y_], s=(0.85 * 2.835) ** 2, marker="D", color="#9BD25A", edgecolors="white",
                         linewidths=0.25, zorder=8)
        elif kind == "o":
            bd.b.scatter([x_], [y_], s=(1.0 * 2.835) ** 2, marker="o", color="#36A88A", edgecolors="white",
                         linewidths=0.25, zorder=8)
        elif kind == "s":
            bd.b.scatter([x_], [y_], s=(1.25 * 2.835) ** 2, marker="s", facecolors="none", edgecolors=INK,
                         linewidths=0.45, zorder=8)
        else:
            bd.b.scatter([x_], [y_], s=(1.05 * 2.835 * 2.2) ** 2, marker="*", color="#F5A623", edgecolors=INK,
                         linewidths=0.35, zorder=8)
    for row, items in enumerate([[("D", "with"), ("o", "without interposition")],
                                 [("s", "resection alone"), ("*", "plan")]]):
        lx, ly = xc + 0.75, 24.6 + row * 2.5         # v5: tighter spacing so that the larger text stays in the card
        for kind, txt in items:
            mark(lx + (0.35 if kind == "*" else 0.0), ly - 0.75, kind)
            off = {"*": 1.55, "s": 1.1}.get(kind, 0.85)
            bd.t(lx + off, ly, txt, size=FS_SMALL, color=INK2, z=8)
            lx += off + 0.95 + bd.tw(txt, FS_SMALL)
    j_bar(bd, xc + 9.6, 29.8, 9.0, h=0.9)
    bd.t(xc + 1.3, 30.65, "J", size=FS, style="italic", color=INK)
    tx = xc + 28.6
    t_ch, ip_ch, _ = child["choice"]
    side = "lateralizing" if t_ch < 0 else "medializing"
    L = latent_data()
    al = child["foot"]["align"]
    lines(bd, tx, 8.4, [("Latent", {"weight": "bold", "color": NAVY, "size": FS}),
                        ("rehearsal", {"weight": "bold", "color": NAVY, "size": FS}),
                        ("virtual child of a", {"color": INK}),
                        (f"(type {child['foot']['ctype']}, {abs(al):.0f}° {'valgus' if al > 0 else 'varus'})", {"color": INK}),
                        f"{len(L['cands'])} adjuncts after", "complete resection,", "ranked by J"],
          step=1.95)
    # the ranking objective J (Methods, equation (4)), bottom right of the card (authors, 2026-11-04)
    for k, eq in enumerate([r"$J = z_{\mathrm{P}} + z_{\mathrm{M}}$",
                            r"$\;+\,½\,(z_{\mathrm{TN}} + z_{\mathrm{A}})$",
                            r"$\;+\,½\,(|z_{\mathrm{ev}}| + |z_{\mathrm{nd}}|)$",
                            r"$\;+\,0.3\,(|t|/10 + 0.1\,i)$"]):
        bd.t(tx, 24.9 + 2.55 * k, eq, size=FS_SMALL, color=INK, z=8)
    # FE verification of the chosen plan (Fig. 7a) and its effects
    from maps import child_map
    my, mw, mh = 35.9, 12.0, 9.0
    bd.t(xc, my + 0.2, "Plan:", size=FS, weight="bold", color=NAVY)
    _parts = ([f"{abs(t_ch):.0f}-mm {side} osteotomy"] if t_ch else []) + (["interposition"] if ip_ch else []) + \
             (["fusion"] if child["choice"][2] else [])          # the chosen plan, whatever it is (Claude, 2026-11-01)
    bd.t(xc + 6.4, my + 0.2, " + ".join(_parts) if _parts else "resection alone", size=FS_SMALL, color=INK)
    mk = np.load(os.path.join(ROOT, "results", "map_child_masks.npz"))
    fm = mk["post"] | mk["mid"] | mk["ant"]
    ix, iy = np.nonzero(fm.any(1))[0], np.nonzero(fm.any(0))[0]
    xs_, ys_ = mk["xs"], mk["ys"]
    x0_, x1_, y0_, y1_ = xs_[ix[0]] - 2.5, xs_[ix[-1]] + 2.5, ys_[iy[0]] - 2.5, ys_[iy[-1]] + 2.5
    cxm, cym = (x0_ + x1_) / 2, (y0_ + y1_) / 2
    half = max(x1_ - x0_, (y1_ - y0_) * mw / mh) / 2
    uref = None
    for k, (tag, lab) in enumerate([("resect", "resection alone"), ("plan", "plan")]):
        mx = xc + k * (mw + 4.0)
        bd.card(mx, my + 1.6, mw, mh, under=True, r=0.6, pad=0.0)
        ax = bd.ax(mx + 0.3, my + 1.9, mw - 0.6, mh - 0.6)
        uref = child_map(ax, tag, "fe", uref=uref, lw=0.45, arrow_width=0.012, arrow_scale=8)
        ax.set_xlim(cxm - half, cxm + half); ax.set_ylim(cym - half * mh / mw, cym + half * mh / mw)
        bd.t(mx + mw / 2, my + mh + 3.6, lab, size=FS_SMALL, ha="center", color=INK2)
    bd.arr(xc + mw + 0.6, my + 1.6 + mh / 2, xc + mw + 3.4, my + 1.6 + mh / 2, color=NAVY, lw=0.6, head=(1.1, 0.7))
    # FE-verified changes of the plan relative to resection alone (FE, CoMech prediction)
    rows_ = [("post", "post_mean"), ("mid", "mid_mean"), ("TN", "tn_p99")]
    tx2 = xc + 2 * mw + 4.0 + 1.0
    c1, c2, c3 = tx2 + 9.0, tx2 + 14.7, tx2 + 16.4
    bd.t(tx2, my + 3.3, "Δ", size=FS_SMALL, color=INK2)
    bd.t(c1, my + 3.3, "FE", size=FS_SMALL, weight="bold", color=INK, ha="right")
    bd.t(c2, my + 3.3, "pred.", size=FS_SMALL, color=INK2, ha="right")
    for k, (nm, key) in enumerate(rows_):
        fe_ = pct(child["eff_fe_resect"][key], child["eff_fe_plan"][key])
        pr_ = pct(child["eff_pred_resect"][key], child["eff_pred_plan"][key])
        yy = my + 6.1 + k * 2.75
        bd.t(tx2, yy, nm, size=FS_SMALL, color=INK2)
        bd.t(c1, yy, f"{fe_:+.0f}%".replace("-", "−"), size=FS_SMALL, color=INK, ha="right")
        bd.t(c2, yy, f"{pr_:+.0f}%".replace("-", "−"), size=FS_SMALL, color=INK2, ha="right")
        ok = np.sign(fe_) == np.sign(pr_)
        bd.badge(c3, yy - 0.68, 0.78, "check" if ok else "cross")

    # tolerance envelope of the child: FE defect sweep on its coalition-free twin (virtual cohort, Fig. 6)
    fe_row = [r for r in RES("eval_cohort_fe.json")["rows"] if r["i"] == child["i"]][0]
    fac = child.get("facet", "post")      # facet of the child's defect (v3e: middle facet, type I child)
    cur = np.array(fe_row[f"fe_curve_{fac}"]); dd = np.arange(len(cur)) * 10.0
    rise = 100 * (cur / cur[0] - 1)
    ty = 51.2
    ax = bd.ax(xc + 6.4, ty + 1.4, 16.4, 10.6)
    tol = 100 * fe_row[f"fe_tol_{fac}"]; dch = 100 * fe_row[f"d_{fac}"]
    ax.axvspan(tol, 75, color="#F3DCD7", lw=0, zorder=0)
    ax.plot(dd, rise, color=FE, lw=0.75, marker="o", ms=1.6, mfc="white", mew=0.45, zorder=3)
    ax.axhline(20, color=MUTED, lw=0.5, ls=(0, (2, 1.2)), zorder=1)
    ax.text(-1.5, 27, "+20%", fontsize=FS_SMALL, color=INK2, ha="left", va="bottom")
    ax.plot([tol, tol], [-8, 20], color=PAL["green_edge"], lw=0.7, zorder=2)
    ax.scatter([tol], [20], s=9, color=PAL["green_edge"], zorder=4, edgecolors="white", linewidths=0.4)
    yd = float(np.interp(dch, dd, rise))
    ax.scatter([dch], [yd], s=14, marker="X", color=PAL["cross"], zorder=5, edgecolors="white", linewidths=0.3)
    ytop = max(170.0, 1.08 * float(np.nanmax(rise)))
    ax.set_xlim(-3, 73); ax.set_ylim(-10, ytop)
    ax.set_xticks([0, 35, 70]); ax.set_yticks([0, 80, 160] if ytop < 200 else list(np.arange(0, ytop, 100.0)))
    ax.tick_params(labelsize=FS_SMALL, length=1.4, pad=0.8)
    style(ax, offset=1.0)
    ax.set_xlabel(f"{'Posterior' if fac == 'post' else 'Middle'}-facet defect (%)", fontsize=FS_SMALL, labelpad=0.4,
                  x=0.36, ha="center")   # v5: shifted left, clear of the formulas at right
    ax.set_ylabel("Stress rise (%)", fontsize=FS_SMALL, labelpad=0.4)
    lines(bd, xc + 25.4, ty + 2.9, [("Tolerance envelope", {"weight": "bold", "color": NAVY, "size": FS}),
                                    (f"FE tolerance {tol:.0f}%", {"color": PAL["green_edge"]}),
                                    (f"defect left {dch:.0f}%", {"color": PAL["cross"]}),
                                    f"though C/P = {100 * child['ratio']:.0f}%",
                                    "is below 50%" if child["ratio"] < 0.5 else "is above 50%"], step=1.95)
    # tolerance and reserve as defined in Methods, bottom right (authors, 2026-11-04): d* is where the residual-facet
    # mean stress first rises by 20% over the defect-free joint; R < 0: the defect left exceeds the tolerance
    for k, eq in enumerate([r"$d^{\star}{:}\ \bar{\sigma}(d^{\star}) = 1.2\,\bar{\sigma}(0)$",
                            r"$R = (d^{\star} - d)\,/\,d^{\star} " + ("< 0$" if tol < dch else "\\geq 0$")]):
        bd.t(xc + 26.4, ty + 15.0 + 2.6 * k, eq, size=FS_SMALL, color=INK, z=8)

    # ================================================================== d: in silico evidence of this study
    # (revision 3, Claude 2026-11-03): one picture per result instead of text cards -- the atlas solver output
    # (cropped, pixels unchanged), engine renderings of virtual feet, FE against CoMech for a held-out foot, the virtual
    # cohort against the 50% rule (drawn from eval_cohort_fe.json), the FE-verified plan of the virtual child and the
    # CT reconstruction and point twin of P05-R; each tile names its figure and its size.
    yd0 = TOP + HT + 1.4
    hd = H - 0.5 - yd0
    bd.panel(0.5, yd0, FIG_W - 1.0, hd, "teal")
    head(bd, 2.0, yd0 + 0.5, "d", "In silico evidence in this study", color=PAL["teal_edge"])
    B = RES("bank_stats.json")         # TCC-MechBank summary written by exp/rb_bank_stats.py
    T = {"BANK_SPLIT": "/".join(str(B[k]["feet"]) for k in ["train", "val", "test", "ood"]),
         "BANK_SOL": f"{B['total_solutions_bank']:,}",
         "BANK_FEET": f"{sum(B[k]['feet'] for k in ['train', 'val', 'test', 'ood']):,}"}
    SC = RES("eval_cohort_fe.json")
    ctw = RES("eval_ct_twins.json")["summary"]
    nsplit = [int(v) for v in T["BANK_SPLIT"].split("/")]                    # train/val/test/out-of-range feet
    n = 6
    cx0, cgap = 2.2, 3.3
    cwid = (FIG_W - 2 * cx0 - (n - 1) * cgap) / n
    ty = yd0 + 8.3                     # tile titles (baseline)
    cy, ch = yd0 + 9.4, hd - 17.4      # picture cards
    ny = cy + ch + 4.1                 # numbers (baseline)

    def crop_white(path, thr=246, pad=6):
        im = np.asarray(Image.open(path).convert("RGB"))
        m = (im < thr).any(2)
        rr, cc = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        return im[max(rr[0] - pad, 0):rr[-1] + pad, max(cc[0] - pad, 0):cc[-1] + pad].astype(float) / 255

    def etile(k, title, big, unit, sub=None):
        x = cx0 + k * (cwid + cgap)
        if sub:
            bd.t(x + 0.3, ny + 2.45, sub, size=FS_SMALL, color=INK2)
        bd.t(x + 0.3, ty, title, size=FS_SMALL, weight="bold", color=PAL["teal_edge"])
        bd.card(x, cy, cwid, ch, r=0.8, lw=0.4)
        bd.t(x + 0.3, ny, big, size=FS_MAX, color=NAVY)          # Nature: text at most 7 pt (2026-11-03)
        bd.t(x + 0.3 + bd.tw(big, FS_MAX) + 0.8, ny, unit, size=FS_SMALL, color=NAVY)
        if k < n - 1:
            bd.chevron(x + cwid + 0.45, cy + ch / 2, w=2.4, h=3.4, fc=PAL["teal_chev"], z=6)
        return x

    def sub_imgs(x, ims, labels=None, pad=0.5, gap=0.5, z=5):
        w_ = (cwid - 2 * pad - (len(ims) - 1) * gap) / len(ims)
        h_ = ch - 2 * pad - (2.0 if labels else 0.0)
        for j, im in enumerate(ims):
            xx = x + pad + j * (w_ + gap)
            bd.img(xx, cy + pad, w_, h_, im, z=z)
            if labels:
                bd.t(xx + w_ / 2, cy + ch - 0.7, labels[j], size=FS_SMALL, color=INK2, ha="center", z=z + 1)

    # 1. FE atlas: the solver's mesh and a coalition stress map (cropped, pixels unchanged)
    x = etile(0, "FE atlas · Figs. 2, 3", "33", "FE models", "one adult CT template")
    sub_imgs(x, [crop_white(os.path.join(CROPS, "r1_061.png")), crop_white(os.path.join(CROPS, "r1_115.png"))])
    # 2. TCC-MechBank: engine renderings of virtual feet (intact and with coalitions)
    x = etile(1, "TCC-MechBank · Fig. 4", T["BANK_SOL"], "FE solutions", f"{T['BANK_FEET']} virtual feet × 8 actions")
    sub_imgs(x, [rgba_render(R(nm)) for nm in ["icon_foot1", "icon_coal_I", "icon_coal_IV"]],
             labels=["intact", "type I", "type IV"], gap=0.2, pad=0.3)
    # 3. Fidelity: FE and CoMech stress of a held-out foot
    x = etile(2, "Fidelity · Fig. 5", str(nsplit[2]), "held-out feet", f"+ {nsplit[3]} out of the training range")
    sub_imgs(x, [cut_dark_base(rgba_render(R("heldout_stress_fe"))), cut_dark_base(rgba_render(R("heldout_stress_cm")))],
             labels=["FE", "CoMech"])
    # 4. 50% rule: the virtual cohort against the rule (data of Fig. 6c)
    x = etile(3, "50% rule · Fig. 6", str(SC["summary"]["n"]), "virtual feet", "FE-verified envelopes")
    axm = bd.ax(x + 1.2, cy + 0.9, cwid - 2.2, ch - 1.8)
    axm.set_zorder(7); axm.patch.set_alpha(0.0)
    res_ = []
    for r in SC["rows"]:
        v = [(r[f"fe_tol_{f}"] - r[f"d_{f}"]) / r[f"fe_tol_{f}"] for f in ["mid", "post"] if f"fe_tol_{f}" in r]
        res_.append((100 * r["ratio"], 100 * min(v) if v else 100.0, r["ctype"]))
    lo_ = min(-300.0, 100.0 * np.floor(min(v for _, v, _ in res_) / 100.0))
    xmax_ = max(100.0, 10.0 * np.ceil(max(v for v, _, _ in res_) / 10.0))
    from matplotlib.patches import Rectangle as _Rect
    axm.add_patch(_Rect((0, lo_ - 20), 50, 20 - lo_, color="#FBE7E3", lw=0, zorder=0))
    axm.add_patch(_Rect((50, 0), xmax_ - 50, 130, color="#E7F0F8", lw=0, zorder=0))
    axm.axvline(50, color=INK2, lw=0.45, ls=(0, (2.5, 1.5)), zorder=1); axm.axhline(0, color=INK2, lw=0.4, zorder=1)
    rng_ = np.random.default_rng(1)
    for t_ in ["III", "I", "II", "IV"]:
        pts_ = [(a_, b_) for a_, b_, c_ in res_ if c_ == t_]
        axm.scatter([p[0] for p in pts_], np.clip([p[1] for p in pts_], lo_, 100) + rng_.uniform(-3, 3, len(pts_)),
                    s=2.2, color=TYPE[t_], marker=TYPE_MARKER[t_], linewidths=0, zorder=3)
    axm.set_xlim(0, xmax_); axm.set_ylim(lo_ - 20, 130); axm.set_xticks([]); axm.set_yticks([])
    for sp_ in axm.spines.values():
        sp_.set_visible(False)
    axm.text(52.0, lo_ - 14, "50%", fontsize=FS_SMALL, color=INK2, va="bottom")
    # 5. Rehearsal: the FE-verified rehearsed plan of the virtual child (Fig. 7a)
    x = etile(4, "Rehearsal · Fig. 7", "16", "feet × 10 adjuncts", "plans verified by FE")
    sub_imgs(x, [cut_dark_base(rgba_render(R("child_resect_fe"))), cut_dark_base(rgba_render(R("child_plan_fe")))],
             labels=["resection", "plan"])
    # 6. CT-informed twins: reconstruction of P05-R and its point twin after complete resection (Fig. 8a,d)
    x = etile(5, "CT-informed twins · Fig. 8", str(ctw["n_feet"]), f"feet, {ctw['n_patients']} patients",
              "preliminary; no outcomes")
    sub_imgs(x, [rgba_render(R("ct_P05R_medial")), cut_dark_base(rgba_render(R("twin_P05-R_resect")))],
             labels=["CT", "twin"])
    os.makedirs(OUT, exist_ok=True)
    bd.save(os.path.join(OUT, "fig1"))
    print("saved fig1")


if __name__ == "__main__":
    fig1()
