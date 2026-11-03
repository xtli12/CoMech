"""Voxel replica of the ANSYS atlas model on the CoMech engine's solver (Claude, 2026-10-28).

The atlas's own tetrahedral mesh (ds.dat) is sampled on a rectilinear grid in the engine frame (frame.py): every cell
gets the fraction of each body from ns^3 sub-samples (point-in-tetrahedron, corner nodes of the 10-node tetrahedra).
Materials follow the atlas (cortical bone 7300 MPa, cancellous 100, cartilage 10, plantar fascia 350, soft tissue 1.15,
Poisson's ratios as in the atlas) and are mixed per cell as in the engine: in series (Reuss) where a cell contains
cartilage, by volume (Voigt) elsewhere.  All atlas contacts are bonded, as are the engine's interfaces.  Supports and
loads follow the atlas: the ground plate (17 GPa, fixed) is replaced by a fixed boundary at its top face (z = 0) and
the leg forces (tibia 297.6 N, fibula 59.0 N, along the plate normal) are applied on the cross-sections of the two
bones where the leg is cut (z = z_cut).  Not modelled: the 15 ligament springs (80 N/mm) of the atlas."""
import os, sys, time, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from frame import to_engine

MAT = {"cortical": (7300.0, 0.3), "cancellous": (100.0, 0.3), "cartilage": (10.0, 0.4), "fascia": (350.0, 0.4),
       "soft": (1.15, 0.49), "plate": (17000.0, 0.1)}
CLASSES = ["void", "cortical", "cancellous", "cartilage", "fascia", "soft", "plate"]
E_TO_CLASS = {7300.0: "cortical", 100.0: "cancellous", 10.0: "cartilage", 350.0: "fascia", 1.15: "soft", 17000.0: "plate"}
# body indices of the normal model (ds.dat "Elements for Body k")
BODY = dict(calcaneus=(1, 53), talus=(2, 52), fibula=(3, 54), tibia=(4, 13), navicular=(40, 45), cuboid=(43, 49),
            ankle=(5,), syndesmosis=(6,), talofibular=(7,), mid=(8,), tn=(9,), ant=(10,), cc=(11,), post=(12,))
FORCES = {"tibia": 300.0, "fibula": 60.0}           # N (ds.dat: surface-effect pressure on the upper 70 mm of each bone = 300 N and 60 N)


def load_model(pkl):
    D = pickle.load(open(pkl, "rb"))
    ids = np.array(sorted(D["nodes"])); X = np.array([D["nodes"][k] for k in ids])
    pos = np.full(ids.max() + 1, -1, np.int64); pos[ids] = np.arange(len(ids))
    XE = to_engine(X)
    T, body, cls = [], [], {}
    for b in D["bodies"]:
        E = np.array(b["elems"])[:, :4]
        T.append(XE[pos[E]]); body.append(np.full(len(E), b["index"], np.int16))
        ex = float(D["mats"][b["mat"][0]]["EX"])
        cls[b["index"]] = CLASSES.index(E_TO_CLASS[ex])
    return np.concatenate(T), np.concatenate(body), cls, D


def axis_edges(lo, hi, fine_lo, fine_hi, h_fine, h_coarse):
    """Cell edges: spacing h_fine inside [fine_lo, fine_hi], about h_coarse outside, ending exactly at lo and hi."""
    e = list(np.arange(fine_lo, fine_hi + 1e-9, h_fine))
    if e[-1] < fine_hi - 1e-9:
        e.append(fine_hi)
    left = fine_lo - lo; nl = max(1, int(np.ceil(left / h_coarse))) if left > 1e-9 else 0
    right = hi - fine_hi; nr = max(1, int(np.ceil(right / h_coarse))) if right > 1e-9 else 0
    L = list(np.linspace(lo, fine_lo, nl + 1)[:-1]) if nl else []
    Rr = list(np.linspace(fine_hi, hi, nr + 1)[1:]) if nr else []
    return np.array(L + e + Rr)


def locate(T, body, sx, sy, sz, chunk=40000):
    """Body index (0 = none) at every point of the tensor-product lattice sx x sy x sz."""
    lab = np.zeros((len(sx), len(sy), len(sz)), np.int16)
    v0 = T[:, 0]; M = np.stack([T[:, 1] - v0, T[:, 2] - v0, T[:, 3] - v0], -1)        # (n,3,3), columns edges
    det = np.linalg.det(M); ok = np.abs(det) > 1e-12
    T, body, v0, M = T[ok], body[ok], v0[ok], M[ok]
    Mi = np.linalg.inv(M)
    lo = T.min(1); hi = T.max(1)
    i0 = np.searchsorted(sx, lo[:, 0]); i1 = np.searchsorted(sx, hi[:, 0], "right")
    j0 = np.searchsorted(sy, lo[:, 1]); j1 = np.searchsorted(sy, hi[:, 1], "right")
    k0 = np.searchsorted(sz, lo[:, 2]); k1 = np.searchsorted(sz, hi[:, 2], "right")
    nx, ny, nz = i1 - i0, j1 - j0, k1 - k0
    has = (nx > 0) & (ny > 0) & (nz > 0)
    key = np.stack([nx, ny, nz], 1)
    order = np.lexsort((nz, ny, nx))
    order = order[has[order]]
    groups = {}
    for idx in order:
        groups.setdefault(tuple(key[idx]), []).append(idx)
    for (a, b, c), members in groups.items():
        members = np.array(members)
        oi, oj, ok_ = np.meshgrid(np.arange(a), np.arange(b), np.arange(c), indexing="ij")
        oi, oj, ok_ = oi.ravel(), oj.ravel(), ok_.ravel()
        for s in range(0, len(members), max(1, chunk // max(1, a * b * c) + 1)):
            m = members[s:s + max(1, chunk // max(1, a * b * c) + 1)]
            I = i0[m][:, None] + oi[None]; J = j0[m][:, None] + oj[None]; K = k0[m][:, None] + ok_[None]
            P = np.stack([sx[I], sy[J], sz[K]], -1) - v0[m][:, None]                      # (m,k,3)
            lam = np.einsum("mij,mkj->mki", Mi[m], P)
            inside = (lam >= -1e-9).all(-1) & (lam.sum(-1) <= 1 + 1e-9)
            mm, kk = np.nonzero(inside)
            lab[I[mm, kk], J[mm, kk], K[mm, kk]] = body[m][mm]
    return lab


def cell_fractions(lab, ns, cls_of_body, nbody):
    nx, ny, nz = (np.array(lab.shape) // ns)
    L = lab[:nx * ns, :ny * ns, :nz * ns].reshape(nx, ns, ny, ns, nz, ns).transpose(0, 2, 4, 1, 3, 5).reshape(nx, ny, nz, -1)
    nsub = ns ** 3
    lut = np.zeros(nbody + 1, np.int16)
    for b, c in cls_of_body.items():
        lut[b] = c
    C = lut[L]
    fcls = np.stack([(C == k).sum(-1) for k in range(len(CLASSES))], -1).astype(np.float32) / nsub
    return L, fcls


def body_fraction(L, bodies, ns):
    return np.isin(L, list(bodies)).sum(-1).astype(np.float32) / ns ** 3


def materials(fcls, cart_thr=0.0):
    E_ = np.array([np.nan] + [MAT[c][0] for c in CLASSES[1:]]); nu_ = np.array([np.nan] + [MAT[c][1] for c in CLASSES[1:]])
    f = fcls[..., 1:6]                                        # cortical .. soft (no plate inside the domain)
    occ = f.sum(-1)
    Ev, nv = E_[1:6], nu_[1:6]
    with np.errstate(divide="ignore", invalid="ignore"):
        voigt = (f * Ev).sum(-1) / occ
        reuss = occ / (f / Ev).sum(-1)
        nu = (f * nv).sum(-1) / occ
    cart = fcls[..., 3] > cart_thr
    E = np.where(cart, reuss, voigt)
    active = np.where(cart, occ > 0.15, occ >= 0.5)
    E = np.where(active, E, 1.0); nu = np.where(active, np.clip(nu, 0.0, 0.49), 0.3)
    return E.astype(float), nu.astype(float), active, occ
