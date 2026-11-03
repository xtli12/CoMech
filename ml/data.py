"""Load TCC-MechBank splits into compact arrays for training/evaluation."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fe.geometry import *
from fe.simulate import get_grid, EFFECT_NAMES

import fe.geometry as _FG
GRID = get_grid()
GSHAPE = GRID.shape                    # (57, 24, 32) parametric engine; (60, 40, 44) real-bone engine
PATCH = getattr(_FG, "ML_PATCH", (3, 4, 4))
TOK = tuple(g // p for g, p in zip(GSHAPE, PATCH))      # (19, 6, 8)
NTOK = int(np.prod(TOK))
SIG0 = 0.1                              # MPa, stress scale for log transform

# geometry channels from label fractions
CH = getattr(_FG, "ML_CH", [[TALUS, CALC, NAV, CUB, FUSE], [C_ANT, C_MID, C_POST, C_TN, C_CC], [COAL, FIB], [LIG], [INTERP]])
NCH = len(CH)
EFF_STRESS = [i for i, n in enumerate(EFFECT_NAMES) if n.endswith("_p99") or n.endswith("_mean")]
JOINT_LABELS = [C_ANT, C_MID, C_POST, C_TN, C_CC, COAL, FIB, INTERP]


def channels(frac_u8):
    """(..., ncell, 15) uint8 -> (..., NCH, 57, 24, 32) float32 in [0,1]."""
    f = frac_u8.astype(np.float32) / 255.0
    out = np.stack([f[..., ch].sum(-1) for ch in CH], -2)          # (..., NCH, ncell)
    return out.reshape(out.shape[:-1] + GSHAPE)


def patchify(g):
    """(B, C, 57, 24, 32) -> (B, NTOK, C*48)."""
    B, C = g.shape[:2]
    px, py, pz = PATCH
    tx, ty, tz = TOK
    g = g.reshape(B, C, tx, px, ty, py, tz, pz).transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return g.reshape(B, tx * ty * tz, C * px * py * pz)


def cell_norm_coords():
    xyz = GRID.cell_xyz.copy()
    box = getattr(_FG, "ML_BOX", ((0, 96), (-18, 30), (0, 63)))
    lo = np.array([b[0] for b in box], float); hi = np.array([b[1] for b in box], float)
    return ((xyz - lo) / (hi - lo)).astype(np.float32)


CELL_UVW = cell_norm_coords()


def fourier(uvw, nf=5):
    f = (2.0 ** np.arange(nf)) * np.pi
    a = uvw[..., None] * f                                  # (..., 3, nf)
    return np.concatenate([uvw, np.sin(a).reshape(uvw.shape[:-1] + (-1,)),
                           np.cos(a).reshape(uvw.shape[:-1] + (-1,))], -1).astype(np.float32)


CELL_FOURIER = fourier(CELL_UVW)                           # (ncell, 33)


def _token_interp_table():
    """For every cell: 8 token indices and trilinear weights on the token grid."""
    I, J, K = GRID.cI, GRID.cJ, GRID.cK
    idx = np.stack([I, J, K], 1).astype(np.float64) + 0.5
    t = idx / np.array(PATCH) - 0.5
    t0 = np.floor(t).astype(int)
    w = t - t0
    tids, wts = [], []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                ti = np.clip(t0[:, 0] + a, 0, TOK[0] - 1)
                tj = np.clip(t0[:, 1] + b, 0, TOK[1] - 1)
                tk = np.clip(t0[:, 2] + c, 0, TOK[2] - 1)
                tids.append((ti * TOK[1] + tj) * TOK[2] + tk)
                wts.append((w[:, 0] if a else 1 - w[:, 0]) * (w[:, 1] if b else 1 - w[:, 1]) * (w[:, 2] if c else 1 - w[:, 2]))
    return np.stack(tids, 1), np.stack(wts, 1).astype(np.float32)


TOK_IDX, TOK_W = _token_interp_table()


def effect_transform(E):
    """Raw effect matrix (n,17) -> transformed (stress-like via log1p(x/SIG0))."""
    E = np.array(E, dtype=np.float64)
    E[:, EFF_STRESS] = np.log1p(np.maximum(E[:, EFF_STRESS], 0) / SIG0)
    return E


class Split:
    """All feet of a split, held in memory."""

    def __init__(self, split, root="data/bank", max_feet=None, load_post=True):
        files = sorted(glob.glob(f"{root}/{split}/foot_*.json"))
        if max_feet:
            files = files[:max_feet]
        self.meta = [json.load(open(f)) for f in files]
        cache_fn = f"{root}/{split}/intact_volumes.json"
        cache = json.load(open(cache_fn)) if os.path.exists(cache_fn) else {}
        for m, f in zip(self.meta, files):
            key = os.path.basename(f)
            if key in cache:
                m["v_intact_mid"], m["v_intact_post"] = cache[key]
        self.nf = len(self.meta)
        self.K = len(self.meta[0]["actions"]) if self.nf else 0
        geo_pre, geo_post, vm, uc, lab, eff, acts, acts_raw, metas = [], [], [], [], [], [], [], [], []
        for m, f in zip(self.meta, files):
            d = np.load(f.replace(".json", ".npz"))
            fr = d["frac"]                                      # (K, ncell, 15)
            geo_pre.append(channels(fr[0]).astype(np.float16))
            if load_post:
                geo_post.append(channels(fr).astype(np.float16))
            vm.append(d["vm"]); uc.append(d["ucell"]); lab.append(d["lab"])
            eff.append([[a["eff"][k] for k in EFFECT_NAMES] for a in m["actions"]])
            acts_raw.append([a["act"] for a in m["actions"]])
            metas.append([a["meta"] for a in m["actions"]])
            # effective defect extents measured from the voxelised facets
            fr32 = fr.astype(np.float32) / 255.0
            vol = GRID.cell_vol[None, :]
            vmid = (fr32[:, :, C_MID] * vol).sum(1); vpost = (fr32[:, :, C_POST] * vol).sum(1)
            # reference: intact facet volume of the same foot (stored at bank time if present, else action-0 + coalition)
            ref_mid, ref_post = m.get("v_intact_mid"), m.get("v_intact_post")
            if ref_mid is None:
                ref_mid, ref_post = intact_volumes(m["foot"])
                m["v_intact_mid"], m["v_intact_post"] = ref_mid, ref_post
            dm = np.clip(1 - vmid / ref_mid, 0, 1); dp = np.clip(1 - vpost / ref_post, 0, 1)
            av = []
            for k, a in enumerate(m["actions"]):
                A = a["act"]
                av.append([A["resect"], dm[k] if A["resect"] else 0.0, dp[k] if A["resect"] else 0.0,
                           A["osteo"] / 10.0, A["interp"], A["fusion"], A["load"] - 1.0])
            acts.append(av)
        new = {os.path.basename(f): [m["v_intact_mid"], m["v_intact_post"]] for m, f in zip(self.meta, files)}
        if any(k not in cache for k in new):
            cache.update(new)
            try:
                json.dump(cache, open(cache_fn, "w"))
            except OSError:
                pass
        self.geo_pre = np.stack(geo_pre)                         # (nf, C, X, Y, Z) f16
        self.geo_post = np.stack(geo_post) if load_post else None   # (nf, K, C, X, Y, Z) f16
        self.vm = np.stack(vm)                                   # (nf, K, ncell) f16
        self.uc = np.stack(uc)                                   # (nf, K, ncell, 3) f16
        self.lab = np.stack(lab)                                 # (nf, K, ncell) i8
        self.eff_raw = np.array(eff)                             # (nf, K, 17)
        self.act = np.array(acts, dtype=np.float32)              # (nf, K, 7)
        self.acts_raw = acts_raw
        self.metas = metas
        self.foot_vec = np.array([[m["foot"][k] for k in Foot.vec_names()] for m in self.meta], np.float32)
        self.ctype = [m["foot"]["ctype"] for m in self.meta]
        self.rtype = [m["foot"]["rtype"] for m in self.meta]


_INTACT_CACHE = {}


def intact_volumes(foot_dict):
    """Facet cartilage volumes of the same foot without coalition or defect."""
    key = json.dumps(foot_dict, sort_keys=True)
    if key in _INTACT_CACHE:
        return _INTACT_CACHE[key]
    fd = dict(foot_dict); fd.update(ctype="none", rtype="none", align=0.0)
    fr = voxelise(FootGeom(Foot(**fd), Action()), GRID)
    v = (float((fr[:, C_MID] * GRID.cell_vol).sum()), float((fr[:, C_POST] * GRID.cell_vol).sum()))
    _INTACT_CACHE[key] = v
    return v


class Norm:
    """Normalisation statistics computed on the training split (cached in results/norm.json)."""

    @staticmethod
    def cached(path="results/norm.json"):
        if os.path.exists(path):
            n = Norm.__new__(Norm)
            d = json.load(open(path))
            n.s_mu, n.s_sd = d["s_mu"], d["s_sd"]
            n.u_mu, n.u_sd = np.array(d["u_mu"], np.float32), np.array(d["u_sd"], np.float32)
            n.e_mu, n.e_sd = np.array(d["e_mu"]), np.array(d["e_sd"])
            return n
        n = Norm(Split("train"))
        json.dump(dict(s_mu=n.s_mu, s_sd=n.s_sd, u_mu=n.u_mu.tolist(), u_sd=n.u_sd.tolist(), e_mu=n.e_mu.tolist(),
                       e_sd=n.e_sd.tolist()), open(path, "w"))
        return n

    def __init__(self, S: Split):
        occ = S.lab != VOID
        ys = np.log1p(S.vm.astype(np.float32)[occ] / SIG0)
        self.s_mu, self.s_sd = float(ys.mean()), float(ys.std())
        u = S.uc.astype(np.float32)[occ]
        self.u_mu = u.mean(0); self.u_sd = u.std(0) + 1e-6
        E = effect_transform(S.eff_raw.reshape(-1, S.eff_raw.shape[-1]))
        self.e_mu = E.mean(0); self.e_sd = E.std(0) + 1e-6

    def eff(self, raw):
        sh = raw.shape
        return ((effect_transform(raw.reshape(-1, sh[-1])) - self.e_mu) / self.e_sd).reshape(sh).astype(np.float32)

    def eff_inv(self, z):
        sh = z.shape
        E = z.reshape(-1, sh[-1]) * self.e_sd + self.e_mu
        E[:, EFF_STRESS] = (np.expm1(E[:, EFF_STRESS])) * SIG0
        return E.reshape(sh)

    def targets(self, vm, uc):
        ys = (np.log1p(vm.astype(np.float32) / SIG0) - self.s_mu) / self.s_sd
        yu = (uc.astype(np.float32) - self.u_mu) / self.u_sd
        return np.concatenate([ys[..., None], yu], -1)

    def fields_inv(self, y):
        vm = np.expm1(y[..., 0] * self.s_sd + self.s_mu) * SIG0
        u = y[..., 1:] * self.u_sd + self.u_mu
        return vm, u
