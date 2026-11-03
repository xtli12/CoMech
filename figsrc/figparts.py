"""Shared picture elements of Figs. 1 and 4 (v4, 2026-10-28): engine renderings of the surgical actions with
projected annotations, and the latent rehearsal of the virtual child (real latent states, results/child_latent.json).
Everything is placed in millimetres on a schem.Board."""
import os, json, numpy as np
from figstyle import *
from schem import *
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RENDERS = os.environ.get("ACT_RENDERS", os.path.join(ROOT, "results", "renders"))
import fe.geometry as _FGm
REAL_BONE = getattr(_FGm, "ENGINE", "parametric") == "real-bone"   # annotations measured on the label raster (rbannot.py)
J_CMAP = LinearSegmentedColormap.from_list("J", ["#F6E04B", "#9BD25A", "#36A88A", "#2B7C8E", "#35508F", "#3B2D6B"])


# ------------------------------------------------------------------ action renderings
def _camera_proj(Z):
    import render_actions as ra
    eye, tg, ortho = Z["eye"], Z["target"], float(Z["ortho"])
    H, W = Z["alpha"].shape
    return lambda P: ra.project(P, eye, tg, ra.UP, W, H, ortho)


def act_rgba(name, pad=6):
    """RGBA rendering of act_<name>.npz cropped to its silhouette; returns (rgba, (row0, col0), projector)."""
    Z = np.load(os.path.join(RENDERS, f"act_{name}.npz"))
    img = np.clip(Z["img"], 0, 1); al = np.clip(Z["alpha"], 0, 1)
    m = al > 0.02
    rr, cc = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    r0, c0 = max(rr[0] - pad, 0), max(cc[0] - pad, 0)
    r1, c1 = rr[-1] + pad, cc[-1] + pad
    return np.dstack([img, al])[r0:r1, c0:c1], (r0, c0), _camera_proj(Z)


class Placed:
    """An image placed on the board; maps 3D points (mm, engine frame) to board millimetres."""

    def __init__(self, bd, name, x, y, w, h, z=4, pad=6, crop_common=None):
        im, (r0, c0), proj = act_rgba(name, pad)
        if crop_common is not None:                       # same crop for a family of renderings
            R0, R1, C0, C1 = crop_common
            Z = np.load(os.path.join(RENDERS, f"act_{name}.npz"))
            im = np.dstack([np.clip(Z["img"], 0, 1), np.clip(Z["alpha"], 0, 1)])[R0:R1, C0:C1]
            r0, c0 = R0, C0
        self.x0, self.y0, self.w, self.h = bd.img(x, y, w, h, im, z=z)
        self.ih, self.iw = im.shape[:2]
        self.r0, self.c0, self.proj, self.bd = r0, c0, proj, bd

    def mm(self, P):
        q = self.proj(np.atleast_2d(P))
        return np.c_[self.x0 + (q[:, 0] - self.c0) / self.iw * self.w, self.y0 + (q[:, 1] - self.r0) / self.ih * self.h]


def common_crop(names, pad=6):
    boxes = []
    for n in names:
        Z = np.load(os.path.join(RENDERS, f"act_{n}.npz")); m = Z["alpha"] > 0.02
        rr, cc = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        boxes.append((rr[0], rr[-1], cc[0], cc[-1]))
    b = np.array(boxes)
    H, W = np.load(os.path.join(RENDERS, f"act_{names[0]}.npz"))["alpha"].shape
    return (max(b[:, 0].min() - pad, 0), min(b[:, 1].max() + pad, H), max(b[:, 2].min() - pad, 0), min(b[:, 3].max() + pad, W))


def world_crop(name, box, pad=4):
    """Pixel crop (r0, r1, c0, c1) of act_<name>.npz that contains the projection of a world box
    ((x0, x1), (y0, y1), (z0, z1)) in mm, clipped to the image."""
    import itertools
    Z = np.load(os.path.join(RENDERS, f"act_{name}.npz"))
    H, W = Z["alpha"].shape
    q = _camera_proj(Z)(np.array(list(itertools.product(*box)), float))
    c0, r0 = np.floor(q.min(0)).astype(int) - pad
    c1, r1 = np.ceil(q.max(0)).astype(int) + pad
    return (max(r0, 0), min(r1, H), max(c0, 0), min(c1, W))


def world_mm(foot, x, y, z):
    """Canonical (unscaled) engine coordinates -> world millimetres (FootGeom.classify undoes this scaling)."""
    s = foot.scale
    return np.c_[35.0 + (np.asarray(x) - 35.0) * s, np.asarray(y) * s, 35.0 + (np.asarray(z) - 35.0) * s]


def coalition_outline(foot):
    """Closed outline(s) of the pre-operative coalition footprint on the cartilage surface (world mm)."""
    if REAL_BONE:
        from rbannot import coalition_outline as _rb_outline
        return _rb_outline(foot)
    from fe.geometry import FootGeom, Action
    import matplotlib.pyplot as _plt
    g = FootGeom(foot, Action())
    xs = np.arange(10, 80, 0.2); ys = np.arange(-16, 26, 0.2)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    br, fib = g.coalition_xy(X, Y)
    fig = _plt.figure(); cs = _plt.contour(xs, ys, (br | fib).T.astype(float), levels=[0.5]); _plt.close(fig)
    out = []
    for seg in cs.allsegs[0]:
        zt = g.z_joint(seg[:, 0], seg[:, 1]) + foot.cart_t
        out.append(world_mm(foot, seg[:, 0], seg[:, 1], zt))
    return out


def annotate_action(pl, name, foot, color=INK, lw=0.5, clip=None):
    """Vector annotations projected onto an action rendering; clip = (x, y, w, h) mm keeps them inside a frame."""
    bd = pl.bd
    before = {id(a) for a in list(bd.b.lines) + list(bd.b.patches)}
    _annotate(pl, name, foot, color, lw)
    if clip is not None:
        r = Rectangle((clip[0], clip[1]), clip[2], clip[3], transform=bd.b.transData, fc="none", ec="none")
        for a in list(bd.b.lines) + list(bd.b.patches):
            if id(a) not in before:
                a.set_clip_path(r)


def _annotate(pl, name, foot, color, lw):
    bd = pl.bd
    if REAL_BONE and name in ("osteo", "load"):
        import rbannot, render_actions as _ra
        act = _ra.ACTIONS[name][0]
        if name == "osteo":
            cut, a = rbannot.osteotomy_lines(foot, act)
            for P in cut:
                q = pl.mm(P)
                bd.b.plot(q[:, 0], q[:, 1], color=COAL, lw=0.6, ls=(0, (1.6, 0.9)), zorder=8)
            a = pl.mm(a)
            bd.arr(a[0, 0], a[0, 1], a[1, 0], a[1, 1], color=color, lw=lw, head=(1.1, 0.7), z=9)
        else:
            for P in rbannot.load_arrows(foot, act):
                a = pl.mm(P)
                bd.arr(a[0, 0], a[0, 1], a[1, 0], a[1, 1], color=color, lw=lw, head=(1.1, 0.7), z=9)
        return
    if name == "osteo":
        from fe.geometry import x_cut
    if name == "resect":
        for P in coalition_outline(foot):
            q = pl.mm(P)
            bd.b.plot(q[:, 0], q[:, 1], color=COAL, lw=0.6, ls=(0, (1.6, 0.9)), zorder=8, solid_capstyle="butt")
    elif name == "osteo":
        # cut line on the top and lateral surfaces of the calcaneus, and the medial translation t of the tuberosity
        zt = np.linspace(30.0, 31.0, 2)
        ys = np.linspace(-14.5, 8.5, 20)
        top = world_mm(foot, np.full_like(ys, x_cut(30.0)), ys, np.full_like(ys, 30.0))
        zs = np.linspace(30.0, 6.0, 20)
        lat = world_mm(foot, x_cut(zs), np.full_like(zs, -14.5), zs)
        q = pl.mm(np.r_[top[::-1], lat])
        bd.b.plot(q[:, 0], q[:, 1], color=COAL, lw=0.6, ls=(0, (1.6, 0.9)), zorder=8)
        a = pl.mm(world_mm(foot, [8.0, 8.0], [-6.0, 4.0], [31.5, 31.5]))
        bd.arr(a[0, 0], a[0, 1], a[1, 0], a[1, 1], color=color, lw=lw, head=(1.1, 0.7), z=9)
    elif name == "load":
        for xx in (26.0, 36.0, 46.0):
            a = pl.mm(world_mm(foot, [xx, xx], [1.0, 1.0], [78.0, 63.0]))
            bd.arr(a[0, 0], a[0, 1], a[1, 0], a[1, 1], color=color, lw=lw, head=(1.1, 0.7), z=9)


# ------------------------------------------------------------------ latent rehearsal of the virtual child
def latent_data():
    return json.load(open(os.path.join(ROOT, "results", "child_latent.json")))


def _oblique(Q, az, el):
    """Orthographic view of 3D principal-component coordinates (azimuth, elevation in degrees); returns (x, up)."""
    az, el = np.deg2rad(az), np.deg2rad(el)
    x = -np.sin(az) * Q[:, 0] + np.cos(az) * Q[:, 1]
    y = -np.cos(az) * np.sin(el) * Q[:, 0] - np.sin(az) * np.sin(el) * Q[:, 1] + np.cos(el) * Q[:, 2]
    return np.c_[x, y]


def latent_fan(bd, x, y, w, h, az=-112.0, el=28.0, ms=1.15, lw_ray=0.28, star=1.25, floor=True, z=4, sphere=True,
               margin=0.04, triad=False):
    """Latent rehearsal of the virtual child (real latent states of CoMech seed 0): the pre-operative state s and the
    42 states s' = D(s, u) predicted for the rehearsed adjuncts, in the first three principal components of these 43
    states (orthographic view).  Lines join s to every s'; colour = ensemble objective J (lower is better); circles,
    no interposition; diamonds, with interposition; star, chosen plan; open square, resection alone; grey, the
    states projected onto the floor plane.  Drawn into the box (x, y, w, h) mm; returns positions (mm) of key points."""
    L = latent_data()
    S = np.array(L["s"])[None]; P = np.array(L["pc"]); J = np.array(L["J"]); C = np.array(L["cands"])
    norm = mpl.colors.Normalize(J.min(), J.max())
    z0 = P[:, 2].min() - 0.35
    x0, x1 = min(P[:, 0].min(), S[0, 0]) - 0.5, max(P[:, 0].max(), S[0, 0]) + 0.5
    y0, y1 = min(P[:, 1].min(), S[0, 1]) - 0.5, max(P[:, 1].max(), S[0, 1]) + 0.5
    F = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])
    allq = _oblique(np.r_[S, P, F], az, el)
    lo, hi = allq.min(0), allq.max(0)
    sc = min(w * (1 - 2 * margin) / (hi[0] - lo[0]), h * (1 - 2 * margin) / (hi[1] - lo[1]))
    cx, cy = x + w / 2, y + h / 2
    mid = (lo + hi) / 2

    def mm(Q):
        q = _oblique(np.atleast_2d(Q), az, el)
        return np.c_[cx + (q[:, 0] - mid[0]) * sc, cy - (q[:, 1] - mid[1]) * sc]

    fq = mm(F)
    if floor:
        bd.poly(fq, fc="#FFFFFF", ec="#C9D3C1", lw=0.4, z=z - 0.6, alpha=0.95)
        for k in range(1, 4):                                       # faint grid on the floor plane
            a = k / 4.0
            p0 = fq[0] + a * (fq[1] - fq[0]); p1 = fq[3] + a * (fq[2] - fq[3])
            bd.seg([p0[0], p1[0]], [p0[1], p1[1]], color="#E3E9DE", lw=0.3, z=z - 0.5)
            p0 = fq[0] + a * (fq[3] - fq[0]); p1 = fq[1] + a * (fq[2] - fq[1])
            bd.seg([p0[0], p1[0]], [p0[1], p1[1]], color="#E3E9DE", lw=0.3, z=z - 0.5)
        sh = mm(np.c_[P[:, :2], np.full(len(P), z0)])
        bd.b.scatter(sh[:, 0], sh[:, 1], s=(0.55 * ms * 2.835) ** 2, color="#C3CBBE", linewidths=0, zorder=z - 0.4)
        ssh = mm(np.c_[S[:, :2], [z0]])
        sp = mm(S)
        bd.seg([sp[0, 0], ssh[0, 0]], [sp[0, 1], ssh[0, 1]], color=MUTED, lw=0.35, ls=(0, (1.0, 0.8)), z=z - 0.3)
        bd.b.scatter(ssh[:, 0], ssh[:, 1], s=(0.8 * ms * 2.835) ** 2, color="#AEB8A8", linewidths=0, zorder=z - 0.4)
    sp = mm(S)[0]; pq = mm(P)
    order = np.argsort(-J)                                          # draw the best rays last (on top)
    segs = [[sp, pq[i]] for i in order]
    bd.b.add_collection(LineCollection(segs, colors=[J_CMAP(norm(J[i])) for i in order], linewidths=lw_ray,
                                       alpha=0.75, zorder=z))
    pt = ms * 2.835                                                 # marker size in points (ms in mm)
    for ip, mk, sz in [(0, "o", pt), (1, "D", pt * 0.85)]:
        sel = np.nonzero(C[:, 1] == ip)[0]
        bd.b.scatter(pq[sel, 0], pq[sel, 1], c=J[sel], cmap=J_CMAP, norm=norm, s=sz ** 2, marker=mk,
                     edgecolors="white", linewidths=0.25, zorder=z + 0.5)
    i0 = int(np.nonzero((C[:, 0] == 0) & (C[:, 1] == 0))[0][0]); ch = int(L["choice"])
    bd.b.scatter([pq[i0, 0]], [pq[i0, 1]], s=(2.0 * pt) ** 2, marker="s", facecolors="none", edgecolors=INK,
                 linewidths=0.5, zorder=z + 0.8)
    bd.b.scatter([pq[ch, 0]], [pq[ch, 1]], s=(star * 2.835 * 2.2) ** 2, marker="*", color="#F5A623", edgecolors=INK,
                 linewidths=0.4, zorder=z + 1.0)
    if sphere:
        bd.sphere(sp[0], sp[1], 1.15, c0="#1B2A62", c1="#6D86DE", z=z + 1.2)
    else:
        bd.b.scatter([sp[0]], [sp[1]], s=(1.6 * pt) ** 2, color=PAL["navy"], zorder=z + 1.2)
    if triad:                                                       # principal-component axes at a floor corner
        o3 = F[3].copy(); L3 = 0.16 * (x1 - x0)
        o = mm(o3)[0]
        for k, nm in enumerate(["PC1", "PC2", "PC3"]):
            e = o3.copy(); e[k] += L3 if k != 1 else -L3
            q = mm(e)[0]
            bd.arr(o[0], o[1], q[0], q[1], color=INK2, lw=0.4, head=(0.8, 0.5), z=z + 1.5)
            dv = (q - o) / (np.linalg.norm(q - o) + 1e-9)
            bd.t(q[0] + 0.9 * dv[0], q[1] + 0.9 * dv[1] + 0.7, nm, size=FS_SMALL - 0.0, color=INK2,
                 ha="center", z=z + 1.5)
    ends = {}
    for ip in (0, 1):
        sel = np.nonzero(C[:, 1] == ip)[0]
        ends[ip] = (pq[sel[np.argmin(C[sel, 0])]], pq[sel[np.argmax(C[sel, 0])]])
    return dict(s=sp, pts=pq, i0=i0, choice=ch, J=J, C=C, norm=norm, ends=ends, floor=fq)


def j_bar(bd, x, y, w, h=1.1, label=True, fs=FS_SMALL, z=6):
    """Small horizontal J scale (low = better)."""
    n = 64
    for k in range(n):
        bd.rect(x + k * w / n, y, w / n + 0.02, h, fc=J_CMAP(k / (n - 1)), z=z)
    bd.box(x, y, w, h, fc="none", ec=INK2, lw=0.3, r=0.0, z=z + 0.1)
    if label:
        bd.t(x - 0.6, y + h - 0.05, "low", size=fs, ha="right", color=INK2)
        bd.t(x + w + 0.6, y + h - 0.05, "high", size=fs, ha="left", color=INK2)
