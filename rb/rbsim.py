"""One simulation of the real-bone CoMech engine (Claude, 2026-10-29): Foot + Action -> effect descriptors and fields.

The FE solve is the atlas replica (rbfe.py) on the real foot; descriptors follow fe/simulate.effects; fields (von Mises
stress and displacement relative to the calcaneus) are resampled on a fixed 2-mm analysis grid around the hindfoot,
together with the tissue fractions of the analysis cells, for the learning experiments."""
import os, sys, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.environ.get("COMECH", os.path.dirname(HERE)))
import rbgeom as G
import rbfe as RF
from fe.solver import Grid

# ---- analysis (ML) grid: 2 mm, fixed in the engine frame, contains the hindfoot of every virtual foot
ML_BOX = ((-106.0, 14.0), (-36.0, 44.0), (12.0, 100.0))
ML_H = 2.0
ML_PATCH = (5, 4, 4)
# analysis labels: the 15 labels of fe/geometry, then the tissues of the full foot
(VOID, TALUS, CALC, NAV, CUB, C_ANT, C_MID, C_POST, C_TN, C_CC, COAL, FIB, INTERP, FUSE, LIG,
 TIB, FIBU, C_ANKLE, OBONE, C_OTHER, SOFT, FASC) = range(22)
LABEL_NAMES = ["void", "talus", "calcaneus", "navicular", "cuboid", "cart_ant", "cart_mid", "cart_post", "cart_TN",
               "cart_CC", "coalition", "fibrous", "interposition", "fusion_bone", "ligament", "tibia", "fibula",
               "cart_ankle", "other_bone", "cart_other", "soft_tissue", "fascia"]
CART_LABELS = [C_ANT, C_MID, C_POST, C_TN, C_CC, C_ANKLE, C_OTHER]
HARD_LABELS = [TALUS, CALC, NAV, CUB, COAL, FIB, INTERP, FUSE, TIB, FIBU, OBONE] + CART_LABELS
EFFECT_NAMES = ["ant_p99", "ant_mean", "mid_p99", "mid_mean", "post_p99", "post_mean", "tn_p99", "cc_p99",
                "share_ant", "share_mid", "share_post", "talus_dz", "talus_dy", "talus_evert", "nav_drop",
                "heel_share", "tc_p99"]
H_FE, HC_FE, NS_FE = 1.5, 4.0, 3


def body_to_ml(T=None):
    T = T or G.Template.get()
    nb = max(max(T.cls), G.FUSE, G.GAP) + 1
    m = np.zeros(nb, np.int8)
    for b, c in T.cls.items():
        if c in (1, 2):
            m[b] = OBONE
        elif c == 3:
            m[b] = C_OTHER
        elif c == 4:
            m[b] = FASC
        elif c == 5:
            m[b] = SOFT
    for b in G.BONES["talus"]: m[b] = TALUS
    for b in G.BONES["calcaneus"]: m[b] = CALC
    for b in G.BONES["navicular"]: m[b] = NAV
    for b in G.BONES["cuboid"]: m[b] = CUB
    for b in G.BONES["tibia"]: m[b] = TIB
    for b in G.BONES["fibula"]: m[b] = FIBU
    m[G.CART["ant"]] = C_ANT; m[G.CART["mid"]] = C_MID; m[G.CART["post"]] = C_POST; m[G.CART["tn"]] = C_TN
    m[G.CART["cc"]] = C_CC
    for k in ("ankle", "syndesmosis", "talofibular"):
        m[G.CART[k]] = C_ANKLE
    m[G.COAL] = COAL; m[G.FIB] = FIB; m[G.INTERP] = INTERP; m[G.FUSE] = FUSE
    return m


def ml_edges(fine=1.0):
    h = ML_H * fine
    return tuple(np.round(np.arange(a, b + 1e-9, h), 6) for a, b in ML_BOX)


_GRIDS = {}


def get_grid(fine=1.0):
    if fine not in _GRIDS:
        _GRIDS[fine] = Grid(*ml_edges(fine))
    return _GRIDS[fine]


def ml_fractions(foot, act, grid, ss=3, T=None, defect_fill="soft"):
    """(ncell, 22) label fractions of the analysis cells (ss^3 sub-samples)."""
    T = T or G.Template.get()
    m = body_to_ml(T)
    o = (np.arange(ss) + 0.5) / ss
    X = (grid.xe[:-1, None] + grid.dx[:, None] * o).ravel(); Y = (grid.ye[:-1, None] + grid.dy[:, None] * o).ravel()
    Z = (grid.ze[:-1, None] + grid.dz[:, None] * o).ravel()
    nx, ny, nz = grid.shape
    lab = m[G.label_lattice(X, Y, Z, foot, act, T, defect_fill)]
    L = lab.reshape(nx, ss, ny, ss, nz, ss).transpose(0, 2, 4, 1, 3, 5).reshape(nx * ny * nz, ss ** 3)
    frac = np.zeros((L.shape[0], len(LABEL_NAMES)), np.float32)
    for l in range(len(LABEL_NAMES)):
        frac[:, l] = (L == l).mean(1)
    return frac


def ml_labels(frac):
    """Dominant hard-tissue label of each analysis cell (soft tissue and fascia count as void)."""
    hard = frac[:, HARD_LABELS]
    occ = hard.sum(1); cart = frac[:, CART_LABELS].sum(1)
    lab = np.array(HARD_LABELS)[hard.argmax(1)].astype(np.int8)
    act = np.where(cart > 0.05, occ > 0.15, occ >= 0.5)
    lab[~act] = VOID
    return lab


def rigid_fit(X, U):
    c = X.mean(0); r = X - c
    A = np.zeros((3 * len(X), 6))
    A[0::3, 0] = 1; A[1::3, 1] = 1; A[2::3, 2] = 1
    A[0::3, 4] = r[:, 2]; A[0::3, 5] = -r[:, 1]
    A[1::3, 3] = -r[:, 2]; A[1::3, 5] = r[:, 0]
    A[2::3, 3] = r[:, 1]; A[2::3, 4] = -r[:, 0]
    sol, *_ = np.linalg.lstsq(A, U.ravel(), rcond=None)
    return sol[:3], sol[3:], c


def sample_fe(o, grid_ml, Urel, sub=2):
    """Average von Mises stress and relative displacement of every analysis cell from sub^3 points in the FE cells."""
    g = o["g"]; cn = o["cn"]; act = o["active"]
    oo = (np.arange(sub) + 0.5) / sub
    X = (grid_ml.xe[:-1, None] + grid_ml.dx[:, None] * oo).ravel(); Y = (grid_ml.ye[:-1, None] + grid_ml.dy[:, None] * oo).ravel()
    Z = (grid_ml.ze[:-1, None] + grid_ml.dz[:, None] * oo).ravel()
    i = np.clip(np.searchsorted(g.xe, X) - 1, 0, g.nx - 1); j = np.clip(np.searchsorted(g.ye, Y) - 1, 0, g.ny - 1)
    k = np.clip(np.searchsorted(g.ze, Z) - 1, 0, g.nz - 1)
    tx = np.clip((X - g.xe[i]) / g.dx[i], 0, 1); ty = np.clip((Y - g.ye[j]) / g.dy[j], 0, 1); tz = np.clip((Z - g.ze[k]) / g.dz[k], 0, 1)
    inside_x = (X >= g.xe[0]) & (X <= g.xe[-1]); inside_y = (Y >= g.ye[0]) & (Y <= g.ye[-1]); inside_z = (Z >= g.ze[0]) & (Z <= g.ze[-1])
    I, J, K = np.meshgrid(i, j, k, indexing="ij")
    c = ((I * g.ny + J) * g.nz + K).ravel()
    ok = (np.broadcast_to(inside_x[:, None, None], I.shape) & inside_y[None, :, None] & inside_z[None, None, :]).ravel() & act[c]
    TX, TY, TZ = np.meshgrid(tx, ty, tz, indexing="ij")
    TX, TY, TZ = TX.ravel(), TY.ravel(), TZ.ravel()
    from fe.solver import _LOC
    u = np.zeros((len(c), 3))
    for l in range(8):
        w = np.where(_LOC[l, 0], TX, 1 - TX) * np.where(_LOC[l, 1], TY, 1 - TY) * np.where(_LOC[l, 2], TZ, 1 - TZ)
        u += w[:, None] * np.nan_to_num(Urel[cn[c, l]])
    vm = np.where(ok, o["VM"][c], 0.0)
    sg = np.where(ok[:, None], o["SIG"][c], 0.0)
    nx, ny, nz = grid_ml.shape
    def avg(a):
        a = a.reshape(nx, sub, ny, sub, nz, sub, *a.shape[1:]).transpose(0, 2, 4, 1, 3, 5, *range(6, a.ndim + 5))
        return a.reshape(nx * ny * nz, sub ** 3, *a.shape[6:])
    okc = avg(ok.astype(np.float32)).sum(1)
    vm_c = avg(vm).sum(1) / np.maximum(okc, 1)
    u_c = avg(u * ok[:, None]).sum(1) / np.maximum(okc, 1)[:, None]
    sample_fe.sig = (avg(sg).sum(1) / np.maximum(okc, 1)[:, None]).astype(np.float32)
    return vm_c.astype(np.float32), u_c.astype(np.float32), okc / sub ** 3


def sample_nodes(o, grid_ml, Urel):
    """Relative displacement at the nodes of the analysis grid (trilinear in the FE cell that contains the node; nodes
    in empty FE cells take the value of the nearest active FE cell, clamped to it)."""
    from scipy import ndimage
    from fe.solver import _LOC
    g = o["g"]; cn = o["cn"]
    P = grid_ml.node_xyz
    i = np.clip(np.searchsorted(g.xe, P[:, 0]) - 1, 0, g.nx - 1); j = np.clip(np.searchsorted(g.ye, P[:, 1]) - 1, 0, g.ny - 1)
    k = np.clip(np.searchsorted(g.ze, P[:, 2]) - 1, 0, g.nz - 1)
    _, ind = ndimage.distance_transform_edt(~o["active"].reshape(g.shape), return_indices=True)
    ci, cj, ck = ind[0, i, j, k], ind[1, i, j, k], ind[2, i, j, k]
    c = (ci * g.ny + cj) * g.nz + ck
    tx = np.clip((P[:, 0] - g.xe[ci]) / g.dx[ci], 0, 1); ty = np.clip((P[:, 1] - g.ye[cj]) / g.dy[cj], 0, 1)
    tz = np.clip((P[:, 2] - g.ze[ck]) / g.dz[ck], 0, 1)
    u = np.zeros((len(P), 3))
    for l in range(8):
        w = np.where(_LOC[l, 0], tx, 1 - tx) * np.where(_LOC[l, 1], ty, 1 - ty) * np.where(_LOC[l, 2], tz, 1 - tz)
        u += w[:, None] * np.nan_to_num(Urel[cn[c, l]])
    return u.astype(np.float32)


def effects(o, foot, act, T=None):
    T = T or G.Template.get()
    g = o["g"]; F = o["F"]; VM = o["VM"]; act_c = o["active"]; part = o["part"]; bnames = o["bnames"]; U = o["U"]
    L = o["L"]; vol = g.cell_vol
    bf = lambda bodies: np.isin(L, list(bodies)).mean(1)
    d = {}
    for key, b in (("ant", G.CART["ant"]), ("mid", G.CART["mid"]), ("post", G.CART["post"]), ("tn", G.CART["tn"]), ("cc", G.CART["cc"])):
        fb = bf((b,))
        m = (fb >= 0.25) & act_c
        mm = (fb > 0) & act_c
        d[key + "_p99"] = float(np.percentile(VM[m], 99)) if m.sum() >= 3 else 0.0
        w = fb[mm] * vol[mm]
        d[key + "_mean"] = float((VM[mm] * w).sum() / w.sum()) if mm.any() else 0.0
    Ft = o["res"]["F_on_talus"]
    fz = {k: max(Ft.get(k, [0, 0, 0])[2], 0.0) for k in ("ant", "mid", "post")}
    tot = sum(fz.values()) + 1e-9
    for k in fz:
        d["share_" + k] = float(fz[k] / tot)
    xyz = g.node_xyz[o["orig"]]
    nodes_c = o["bone_nodes"]["calcaneus"]
    t, w, c = rigid_fit(xyz[nodes_c], np.nan_to_num(U[nodes_c]))
    Urel = np.nan_to_num(U) - (t[None, :] + np.cross(w[None, :], xyz - c[None, :]))
    nodes_t = o["bone_nodes"]["talus"]
    tt, wt, ct = rigid_fit(xyz[nodes_t], Urel[nodes_t])
    d["talus_dz"] = float(tt[2]); d["talus_dy"] = float(tt[1]); d["talus_evert"] = float(np.rad2deg(wt[0]))
    d["nav_drop"] = float(-np.mean(Urel[o["bone_nodes"]["navicular"], 2]))
    gz = o["res"]["ground_Fz"]
    d["heel_share"] = float(gz["heel"] / max(o["res"]["ground_total_Fz"], 1e-9))
    # talocalcaneal complex: talus, calcaneus, subtalar cartilage and bars within 6 mm of both bones (template frame)
    P = g.cell_xyz
    ps, pm, frag = G.inverse(T, foot, act, P)
    dt = T.sub_lookup(T.dt, ps); dc = T.sub_lookup(T.dc, ps)
    kt = [i for i, n in bnames.items() if n == "talus"][0]; kc = [i for i, n in bnames.items() if n == "calcaneus"][0]
    sub_ = bf((G.CART["post"], G.CART["mid"], G.CART["ant"], G.COAL, G.FUSE, G.FIB))
    band = act_c & (dt <= 6) & (dc <= 6) & ((part == kt) | (part == kc) | (sub_ > 0))
    d["tc_p99"] = float(np.percentile(VM[band], 99)) if band.sum() > 3 else 0.0
    cm = (bf((G.COAL,)) >= 0.25) & act_c
    d["coal_p99"] = float(np.percentile(VM[cm], 99)) if cm.sum() > 3 else 0.0
    d["coal_max"] = float(VM[cm].max()) if cm.sum() > 0 else 0.0
    return d, Urel


def simulate(foot, act, fine=1.0, h=None, hc=None, ns=NS_FE, return_fields=True, tol=1e-6, defect_fill="soft", T=None):
    T = T or G.Template.get()
    t0 = time.time()
    h = h or max(1.0, H_FE * fine); hc = hc or HC_FE
    V = RF.voxels(foot, act, h, hc, ns, T=T, defect_fill=defect_fill)
    t1 = time.time()
    o = RF.solve(V, load=act.load, cart_E=10.0 * 2.0 / foot.cart_t, tol=tol)
    o["L"] = V["L"]
    t2 = time.time()
    eff, Urel = effects(o, foot, act, T)
    out = dict(eff=eff, res=o["res"], timing=(round(t1 - t0, 1), round(t2 - t1, 1)), iters=o["res"]["iters"], ndof=o["res"]["ndof"])
    if return_fields:
        gml = get_grid(fine)
        frac = ml_fractions(foot, act, gml, T=T, defect_fill=defect_fill)
        vm, u, cov = sample_fe(o, gml, Urel)
        lab = ml_labels(frac)
        out.update(frac=frac, lab=lab, vm=vm, ucell=u, cover=cov, Urel=sample_nodes(o, gml, Urel), active=lab != VOID,
                   sig=sample_fe.sig)
    out["timing"] = out["timing"] + (round(time.time() - t2, 1),)
    return out
