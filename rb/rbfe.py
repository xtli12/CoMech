"""Finite-element solve of a real-bone foot (Claude, 2026-10-29): the voxel replica of the atlas (replica3: split nodes
between bones, constrained-modulus mixing across joints, selective reduced integration, ground contact footprint,
leg loads on their line of action, 15 ligament springs) applied to body labels produced by rbgeom.label_points.

grid_template(h, hc, ns)          FE grid of the template foot (engine frame), as in replica2.voxelize
voxels(foot, act, h, hc, ns)      labels of every sub-sample of the (scaled) FE grid + contact, springs, load lines
solve(V, ...)                      solution and post-processing (forces on talus/calcaneus by joint, ground reaction ...)"""
import os, sys, time, json, numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.environ.get("COMECH", os.path.dirname(HERE)))
import rbgeom as G
import splitfe as SF
import replica2 as R2
import replica3 as R3
import voxel_replica as VR
from fe.solver import Grid, lame
import fe.gmg as _gmg
try:                                   # iterations on a GPU when one is available (same algorithm, gpusolve.py)
    import gpusolve as _gs
    if _gs.pick_device() is not None:
        _gmg._pcg_gmg_cpu = getattr(_gmg, "_pcg_gmg_cpu", _gmg.pcg_gmg)     # kept for the GPU solver's CPU fallback
        _gmg.pcg_gmg = _gs.pcg_gmg_gpu
except Exception:                      # pragma: no cover
    pass

ROLES = {"bones": dict(G.BONES), "cart": dict(G.CART), "bonded_extra": [G.COAL, G.FUSE]}
FORCES = {"tibia": 300.0, "fibula": 60.0}
BOX = ((-105, 125), (-50, 45), (0, 100))          # fine region of the replica (whole foot)
Z_CUT = 150.0


def grid_template(h, hc, box=BOX, T=None):
    T = T or G.Template.get()
    lo, hi = T.bounds
    (bx0, bx1), (by0, by1), (bz0, bz1) = box
    xe = VR.axis_edges(np.floor(lo[0]) - 1, np.ceil(hi[0]) + 1, bx0, bx1, h, hc)
    ye = VR.axis_edges(np.floor(lo[1]) - 1, np.ceil(hi[1]) + 1, by0, by1, h, hc)
    ze = VR.axis_edges(0.0, Z_CUT, bz0, bz1, h, hc)
    return xe, ye, ze


def voxels(foot, act, h=1.5, hc=4.0, ns=3, box=BOX, T=None, defect_fill="soft", chunk=4_000_000):
    """Body label of the ns^3 sub-samples of every cell of the foot's FE grid (the template grid scaled with the foot)."""
    T = T or G.Template.get()
    xe, ye, ze = grid_template(h, hc, box, T)
    s = foot.scale; c0 = T.c0
    xe, ye, ze = c0[0] + s * (xe - c0[0]), c0[1] + s * (ye - c0[1]), s * ze
    sub = lambda e: ((e[:-1, None] + (np.arange(ns) + 0.5)[None] / ns * np.diff(e)[:, None])).ravel()
    sx, sy, sz = sub(xe), sub(ye), sub(ze)
    nx, ny, nz = len(xe) - 1, len(ye) - 1, len(ze) - 1
    lab = G.label_lattice(sx, sy, sz, foot, act, T, defect_fill)
    L = lab.reshape(nx, ns, ny, ns, nz, ns).transpose(0, 2, 4, 1, 3, 5).reshape(nx * ny * nz, ns ** 3)
    # ground contact footprint per bottom cell (0.25 mm raster in the template, scaled)
    rr = 0.25 * s
    gx = np.arange(xe[0] + rr / 2, xe[-1], rr); gy = np.arange(ye[0] + rr / 2, ye[-1], rr)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    c = G.contact_points(np.column_stack([GX.ravel(), GY.ravel()]), foot, act, T).reshape(GX.shape)
    ii = np.clip(np.searchsorted(xe, gx) - 1, 0, nx - 1); jj = np.clip(np.searchsorted(ye, gy) - 1, 0, ny - 1)
    cnt = np.zeros((nx, ny)); hit = np.zeros((nx, ny))
    np.add.at(cnt, (ii[:, None].repeat(len(gy), 1), jj[None, :].repeat(len(gx), 0)), 1)
    np.add.at(hit, (ii[:, None].repeat(len(gy), 1), jj[None, :].repeat(len(gx), 0)), c)
    contact = (hit / np.maximum(cnt, 1)).astype(np.float32)
    springs = []
    for sp_ in T.springs:
        a, b = G.forward(T, foot, act, np.array([sp_["a"], sp_["b"]]))
        springs.append({"name": "", "k": sp_["k"], "a": a.tolist(), "b": b.tolist()})
    lines = {k: (c0[:2] + s * (np.array(v) - c0[:2])).tolist() for k, v in T.load_line.items()}
    return dict(xe=xe, ye=ye, ze=ze, L=L, ns=ns, cls=dict(T.cls), springs_json=json.dumps(springs), contact=contact,
                contact_area=float(c.sum() * rr * rr), load_line=lines, z_cut=float(s * Z_CUT), scale=s)


def materials(V, roles, cart_E=10.0, nu_soft=0.49):
    """Class fractions and the constrained-modulus mixing of replica2.materials (rule iface3), extended with the fibrous
    interval of shingled coalitions (60 MPa, a joint layer like cartilage) and interposed fat (1.15 MPa, nu 0.45, with
    the soft tissue).  cart_E: cartilage modulus (equivalent modulus for the cartilage thickness of the foot)."""
    from scipy import ndimage
    cls = {int(k): int(v) for k, v in V["cls"].items()}
    L = V["L"]; nb = max(max(cls), int(L.max()))
    lut = np.zeros(nb + 1, np.int16)
    for b, c in cls.items():
        lut[b] = c
    C = lut[L]
    ncl = len(G.CLASSES)
    F = np.stack([(C == k).mean(1) for k in range(ncl)], 1).astype(np.float64)      # void cortical cancellous cartilage fascia soft plate fibrous fat
    mat = dict(G.MAT); mat["cartilage"] = (cart_E, 0.4); mat["soft"] = (1.15, nu_soft)
    E_ = np.array([1.0] + [mat[c][0] for c in G.CLASSES[1:]]); nu_ = np.array([0.3] + [mat[c][1] for c in G.CLASSES[1:]])
    lam_i = E_ * nu_ / ((1 + nu_) * (1 - 2 * nu_)); mu_i = E_ / (2 * (1 + nu_))
    H = [1, 2, 4]                  # hard: cortical, cancellous, fascia
    LAY = [3, 7]                   # joint layers: cartilage, fibrous
    SOFTC = [5, 8]                 # soft: soft tissue, fat
    PH = H + LAY + SOFTC
    GAPC = [G.CLASSES.index("gap")]   # removed material (resection, facet defect): a compliant layer in joint cells
    f = F[:, PH]; occ = f.sum(1)
    hard = F[:, H].sum(1); lay = F[:, LAY].sum(1); soft = F[:, SOFTC].sum(1); fgap = F[:, GAPC].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lamV = (F[:, PH] * lam_i[PH]).sum(1) / occ; muV = (F[:, PH] * mu_i[PH]).sum(1) / occ
        lam_h = (F[:, H] * lam_i[H]).sum(1) / hard; mu_h = (F[:, H] * mu_i[H]).sum(1) / hard
        M_h = lam_h + 2 * mu_h; G_h = mu_h
        hM = np.where(hard > 0, hard / M_h, 0.0); hG = np.where(hard > 0, hard / G_h, 0.0)
        Mi = lam_i + 2 * mu_i
        # a gap (removed material) lies in series with the layer and the bone of a joint cell; without a gap these
        # are exactly the formulas of replica2 (rule iface3)
        gapM = (F[:, GAPC] / Mi[GAPC]).sum(1); gapG = (F[:, GAPC] / mu_i[GAPC]).sum(1)
        layM = (F[:, LAY] / Mi[LAY]).sum(1) + gapM; layG = (F[:, LAY] / mu_i[LAY]).sum(1) + gapG
        fch = lay + hard + fgap; occ_g = occ + fgap
        M_ch = fch / (layM + hM); G_ch = fch / (layG + hG)
        lam_soft = (F[:, SOFTC] * lam_i[SOFTC]).sum(1); mu_soft = (F[:, SOFTC] * mu_i[SOFTC]).sum(1)
        lam_c = (fch * (M_ch - 2 * G_ch) + lam_soft) / occ_g; mu_c = (fch * G_ch + mu_soft) / occ_g
        softM = (F[:, SOFTC] / Mi[SOFTC]).sum(1); softG = (F[:, SOFTC] / mu_i[SOFTC]).sum(1)
        M_s = occ_g / (softM + hM + gapM); G_s = occ_g / (softG + hG + gapG)
        lam_s = M_s - 2 * G_s; mu_s = G_s
    # cells within one cell of two different bones (joint cells) as in replica2
    grp, _ = R2.bone_groups(cls, roles)
    lutg = np.zeros(nb + 1, np.int16)
    for b, k in grp.items():
        lutg[b] = k
    Gb = lutg[L]
    nx, ny, nz = len(V["xe"]) - 1, len(V["ye"]) - 1, len(V["ze"]) - 1
    count = np.zeros(nx * ny * nz, np.int16)
    st = np.ones((3, 3, 3), bool)
    for k in np.unique(Gb[Gb > 0]):
        pres = (Gb == k).any(1).reshape(nx, ny, nz)
        count += ndimage.binary_dilation(pres, st).ravel()
    cart = lay > 0
    ser = cart | (count >= 2)
    lamX = np.where(cart, lam_c, np.where(ser, lam_s, lamV)); muX = np.where(cart, mu_c, np.where(ser, mu_s, muV))
    with np.errstate(divide="ignore", invalid="ignore"):
        E = muX * (3 * lamX + 2 * muX) / (lamX + muX); nu = lamX / (2 * (lamX + muX))
    active = np.where(ser, occ > 0.15, occ >= 0.5)
    E = np.where(active, E, 1.0); nu = np.where(active, np.clip(nu, 0.0, 0.4995), 0.3)
    return np.nan_to_num(E, nan=1.0), np.nan_to_num(nu, nan=0.3), active, F, ser


def solve(V, roles=ROLES, forces=FORCES, load=1.0, cart_E=10.0, tol=1e-6, rigid=True, nu_soft=0.49, keep=True):
    t0 = time.time()
    g = Grid(V["xe"], V["ye"], V["ze"])
    E, nu, active, F, ser = materials(V, roles, cart_E, nu_soft)
    lam, mu = lame(E, nu)
    L = V["L"]
    bf = lambda bodies: np.isin(L, list(bodies)).mean(1)
    S, bnames = R3.bone_share(V, roles)
    cart = (F[:, 3] + F[:, 7]) > 0
    part = np.where((S.max(1) >= 0.5) & ~cart, S.argmax(1), 0)
    cn, orig, nsplit = SF.split_connectivity_fast(g, active, part, S) if hasattr(SF, "split_connectivity_fast") else SF.split_connectivity(g, active, part, S)
    cells = np.arange(g.ncell)
    cf = V["contact"].ravel()[(g.cI * g.ny + g.cJ)]
    fixed_cells = cells[(g.cK == 0) & active & (cf > 0)]
    cfw = cf[fixed_cells]
    fixed = SF.face_nodes_cn(g, cn, fixed_cells, "z-")
    fixed = (fixed[0], fixed[1] * np.repeat(cfw, 4))
    top = cells[(g.cK == g.nz - 1) & active]
    loads = []
    for name, Fz in forces.items():
        Fz = Fz * load
        c = top[bf(roles["bones"][name])[top] > 0.5]
        n_, a_ = SF.face_nodes_cn(g, cn, c, "z+")
        X = g.node_xyz[orig[n_]][:, :2]; w = a_
        xb = (w[:, None] * X).sum(0) / w.sum(); dX = X - xb
        I = np.einsum("n,ni,nj->ij", w, dX, dX)
        e = np.array(V["load_line"][name]) - xb
        ab = np.linalg.solve(I, w.sum() * e)
        tz = -Fz / w.sum() * (1 + dX @ ab)
        loads.append((n_, a_ * tz, np.array([0.0, 0.0, 1.0])))
    xyz = g.node_xyz[orig]
    bone_nodes = {}
    for name in roles["bones"]:
        k = [i for i, nm in bnames.items() if nm == name][0]
        bone_nodes[name] = np.unique(cn[cells[(part == k) & active]].ravel())
    springs = json.loads(V["springs_json"])
    hc = float(np.diff(V["ze"]).max())
    springs = [s_ for s_ in springs if max(s_["a"][2], s_["b"][2]) < V["z_cut"] - 2 * hc]
    trees = {nm: cKDTree(xyz[nd]) for nm, nd in bone_nodes.items() if len(nd)}
    for s_ in springs:
        for key, p in (("A", s_["a"]), ("B", s_["b"])):
            s_[key] = min(trees, key=lambda nm: trees[nm].query(p)[0])
    extra = R3.spring_matrix_cn(xyz, springs, bone_nodes) if springs else None
    groups = [np.unique(cn[cells[(part == k) & active]].ravel()) for k in bnames if ((part == k) & active).sum() > 3] if rigid else None
    t_prep = time.time() - t0
    U, info, R = SF.solve_cn(g, lam, mu, active, cn, orig, fixed, loads, extra, groups, tol)
    t1 = time.time()
    act = cells[active]
    sig, vm, eps = SF.cell_stress_cn(g, lam, mu, U, act, cn)
    VM = np.zeros(g.ncell, np.float32); VM[act] = vm
    SIG = np.zeros((g.ncell, 6), np.float32); SIG[act] = sig
    res = {"ndof": int(info["ndof"]), "iters": info.get("iters"), "flag": info.get("flag"), "t_prep_s": round(t_prep, 1),
           "t_solve_s": round(t1 - t0 - t_prep, 1), "solver": {k: v for k, v in info.items() if k in ("t_hier", "t_gpu", "device", "relres", "coarse", "gpu_attempt",
                                                                       "fallback", "gpu_error")},
           "n_split_nodes": int(nsplit), "contact_area_mm2": V["contact_area"], "grid": list(g.shape)}
    # forces on the talus / calcaneus from each neighbouring region (as replica3)
    names = list(roles["cart"]) + ["coalition", "fibrous", "fat", "soft", "fascia", "bone", "other"]
    lab = np.full(g.ncell, len(names) - 1, np.int16)
    dom = np.argmax(F[:, 1:], 1) + 1
    lab[dom == G.CLASSES.index("soft")] = names.index("soft")
    lab[dom == G.CLASSES.index("fascia")] = names.index("fascia")
    lab[np.isin(dom, [1, 2])] = names.index("bone")
    lab[bf((G.COAL, G.FUSE)) > 0] = names.index("coalition")
    lab[bf((G.FIB,)) > 0] = names.index("fibrous")
    lab[bf((G.INTERP,)) > 0] = names.index("fat")
    for key, b in roles["cart"].items():
        lab[bf((b,)) > 0] = names.index(key)
    for bone in ("talus", "calcaneus"):
        k = [i for i, nm in bnames.items() if nm == bone][0]
        tgt = (part == k) & active
        tn = np.zeros(len(orig), bool); tn[cn[cells[tgt]].ravel()] = True
        oc = cells[active & ~tgt]
        oc = oc[tn[cn[oc]].any(1)]
        fe = SF.element_forces_cn(g, lam, mu, U, oc, cn)
        fsum = -(fe * tn[cn[oc]][:, :, None]).sum(1)
        res[f"F_on_{bone}"] = {nm: [round(float(x), 2) for x in fsum[lab[oc] == i].sum(0)] for i, nm in enumerate(names)
                               if (lab[oc] == i).any()}
    xc = xyz[:, 0]
    s = V["scale"]; c0x = G.Template.get().c0[0]
    xh = c0x + s * (-40.0 - c0x); xf = c0x + s * (20.0 - c0x)
    res["ground_Fz"] = {"heel": round(float(R[xc < xh, 2].sum()), 2), "mid": round(float(R[(xc >= xh) & (xc < xf), 2].sum()), 2),
                        "fore": round(float(R[xc >= xf, 2].sum()), 2)}
    res["ground_total_Fz"] = float(R[:, 2].sum())
    res["time_s"] = round(time.time() - t0, 1)
    out = dict(res=res, g=g, U=U, VM=VM, SIG=SIG, part=part, bnames=bnames, cn=cn, orig=orig, active=active, F=F, lab=lab,
               names=names, R=R, bone_nodes=bone_nodes, E=E, nu=nu)
    return out
