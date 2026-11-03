"""Real-bone geometry for the CoMech data engine (Claude, 2026-10-29).

The parametric hindfoot of fe/geometry.py is replaced by the full foot of the FE atlas (normal model: 63 bodies from
the CT segmentation, engine frame of frame.py: x anterior, y medial, z height above the ground plate).  Virtual feet and
surgical actions keep the parameters of fe/geometry.Foot and fe/geometry.Action, implemented on the real bones:

  scale            isotropic scale of the whole foot and leg about a ground point below the subtalar joint
  post_a/b, mid_a/b, ant_a/b
                   in-plane size of each subtalar facet (factor = value / nominal value of fe/geometry, applied to the
                   atlas facet about its centre, along the facet's anterior (a) and medial (b) axes); cartilage is removed
                   outside the scaled facet and added in the thin joint gap inside it
  ant_mode         0 separate anterior facet (atlas), 1 confluent with the middle facet, 2 absent
  sust_y           medial position of the sustentaculum / middle facet: smooth local shift of both bones (nominal 11 mm)
  post_dome        convexity of the posterior facet: smooth local displacement of both bones along the facet normal
  cart_t           combined cartilage thickness; enters as the equivalent compressive modulus of the layer
                   (E = 10 MPa x 2.0 / cart_t), the atlas geometry being kept
  align            hindfoot alignment (deg, + valgus): the hindfoot below the subtalar joint is sheared laterally
                   (valgus) or medially (varus) about the joint level, fading out across the midfoot
  ctype/rtype/cov_mid/cov_post/seed_ang/fib_frac
                   coalitions as in fe/geometry: bars on the middle facet (Rozansky R1-R4: medial band, posterior hook,
                   fibrous interval, complete), posterior-facet coalitions grown from the medial edge (type II), the
                   extra-articular bar in the tarsal canal between the posterior and middle facets (type III), both (IV)
  Action: resect (complete resection: bar removed, facet defect of d_mid / d_post of the facet area; v3e: the facet
          part of the resection -- facet defect and the bar's footprint on the facet -- is soft tissue as in the FE
          atlas, the extra-articular part of the bar an empty gap; all of it fat if interp),
          osteo (calcaneal osteotomy: the tuberosity fragment behind an oblique cut is
          translated medially (+) or laterally (-)), fusion (subtalar facets and the thin joint gap become bone), load.

Labels are body indices of the atlas (0 = empty) plus COAL, FIB, INTERP, FUSE, GAP (bodies 101-105)."""
import os, sys, json, pickle, numpy as np
from dataclasses import dataclass, asdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COAL, FIB, INTERP, FUSE = 101, 102, 103, 104
GAP = 105            # removed material that is not refilled (v3e: extra-articular part of a resected bar): an empty gap
SOFT, FASCIA = 63, 62
# atlas bodies of the normal model (cortical, cancellous) and the cartilage of each joint (replica2.NORMAL_ROLES)
BONES = {"calcaneus": (1, 53), "talus": (2, 52), "fibula": (3, 54), "tibia": (4, 13), "navicular": (40, 45),
         "cuboid": (43, 49), "cun_med": (39, 51), "cun_int": (41, 46), "cun_lat": (42, 48)}
CART = {"ankle": 5, "syndesmosis": 6, "talofibular": 7, "mid": 8, "tn": 9, "ant": 10, "cc": 11, "post": 12}
FACETS = {"post": 12, "mid": 8, "ant": 10}
NOMINAL = {"post": (12.0, 9.5), "mid": (6.5, 4.5), "ant": (4.5, 4.0)}      # fe/geometry nominal semi-axes (mm)
# classes of voxel_replica.CLASSES + two new ones
CLASSES = ["void", "cortical", "cancellous", "cartilage", "fascia", "soft", "plate", "fibrous", "fat", "gap"]
NEW_CLASS = {COAL: 1, FIB: 7, INTERP: 8, FUSE: 1, GAP: 9}
MAT = {"cortical": (7300.0, 0.3), "cancellous": (100.0, 0.3), "cartilage": (10.0, 0.4), "fascia": (350.0, 0.4),
       "soft": (1.15, 0.49), "plate": (17000.0, 0.1), "fibrous": (60.0, 0.4), "fat": (1.15, 0.45),
       "gap": (0.01, 0.3)}

TEMPLATE = os.environ.get("RB_TEMPLATE", os.path.join(os.environ.get("ATLAS_W", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "atlas")), "rb_template.npz"))


@dataclass
class Foot:
    scale: float = 1.0
    post_a: float = 12.0
    post_b: float = 9.5
    mid_a: float = 6.5
    mid_b: float = 4.5
    ant_a: float = 4.5
    ant_b: float = 4.0
    ant_mode: int = 0
    sust_y: float = 11.0
    post_dome: float = 1.5
    cart_t: float = 2.0
    align: float = 0.0
    ctype: str = "none"
    rtype: str = "none"
    cov_mid: float = 0.0
    cov_post: float = 0.0
    seed_ang: float = 0.0
    fib_frac: float = 0.0

    def as_vec(self):
        return np.array([self.scale, self.post_a, self.post_b, self.mid_a, self.mid_b, self.ant_a, self.ant_b,
                         float(self.ant_mode), self.sust_y, self.post_dome, self.cart_t, self.align])

    @staticmethod
    def vec_names():
        return ["scale", "post_a", "post_b", "mid_a", "mid_b", "ant_a", "ant_b", "ant_mode", "sust_y",
                "post_dome", "cart_t", "align"]


@dataclass
class Action:
    resect: int = 0
    d_mid: float = 0.0
    d_post: float = 0.0
    osteo: float = 0.0
    interp: int = 0
    fusion: int = 0
    load: float = 1.0

    def as_vec(self):
        return np.array([self.resect, self.d_mid, self.d_post, self.osteo / 10.0, self.interp, self.fusion,
                         self.load - 1.0])

    @staticmethod
    def vec_names():
        return ["resect", "d_mid", "d_post", "osteo/10", "interp", "fusion", "load-1"]


# ---------------------------------------------------------------- template
def build_template(pkl, out, h0=0.5, sub_box=((-100, 10), (-30, 45), (10, 100))):
    """Label raster of the normal atlas foot (h0 mm), distance maps of the subtalar region, facet frames and footprints,
    ground contact footprint, ligament springs and leg load lines (all in the engine frame)."""
    from scipy import ndimage
    import voxel_replica as VR, run_atlas as RA
    T, body, cls, D = VR.load_model(pkl)
    keep = (T[:, :, 2].max(1) > -1.0) & (T[:, :, 2].min(1) < 151.0) & (np.vectorize(cls.get)(body) != VR.CLASSES.index("plate"))
    T, body = T[keep], body[keep]
    lo = T.reshape(-1, 3).min(0); hi = T.reshape(-1, 3).max(0)
    x0, y0, z0 = np.floor(lo[0]) - 1, np.floor(lo[1]) - 1, 0.0
    nx = int(np.ceil((np.ceil(hi[0]) + 1 - x0) / h0)); ny = int(np.ceil((np.ceil(hi[1]) + 1 - y0) / h0)); nz = int(np.ceil(150.0 / h0))
    sx = x0 + (np.arange(nx) + 0.5) * h0; sy = y0 + (np.arange(ny) + 0.5) * h0; sz = z0 + (np.arange(nz) + 0.5) * h0
    lab = VR.locate(T, body, sx, sy, sz).astype(np.int8)
    # ground contact: soft tissue below z = 0.4 mm on a 0.25 mm raster (replica2.voxelize criterion)
    rr = 0.25
    gx = np.arange(x0 + rr / 2, x0 + nx * h0, rr); gy = np.arange(y0 + rr / 2, y0 + ny * h0, rr)
    ms = body == SOFT
    contact = (VR.locate(T[ms], body[ms], gx, gy, np.array([0.4]))[:, :, 0] > 0)
    # subtalar region: distances to talus and calcaneus
    (bx0, bx1), (by0, by1), (bz0, bz1) = sub_box
    i0, i1 = int(round((bx0 - x0) / h0)), int(round((bx1 - x0) / h0))
    j0, j1 = int(round((by0 - y0) / h0)), int(round((by1 - y0) / h0))
    k0, k1 = int(round((bz0 - z0) / h0)), int(round((bz1 - z0) / h0))
    S = lab[i0:i1, j0:j1, k0:k1].astype(np.int16)
    tal = np.isin(S, BONES["talus"]); cal = np.isin(S, BONES["calcaneus"])
    dt = (ndimage.distance_transform_edt(~tal) * h0).astype(np.float16)
    dc = (ndimage.distance_transform_edt(~cal) * h0).astype(np.float16)
    sub_origin = np.array([x0 + i0 * h0, y0 + j0 * h0, z0 + k0 * h0])
    # facet frames: centre, normal (least variance), e1 = anterior axis in the facet plane, e2 = n x e1 (medial)
    fr = {}
    grad = np.stack(np.gradient(dc.astype(np.float32) - dt.astype(np.float32)), -1)      # points from calcaneus to talus
    for nm, b in FACETS.items():
        P = (np.argwhere(S == b) + 0.5) * h0 + sub_origin
        c = P.mean(0)
        gv = grad[S == b]; gv = gv / (np.linalg.norm(gv, axis=1, keepdims=True) + 1e-9)
        n = gv.mean(0); n /= np.linalg.norm(n)
        e1 = np.array([1.0, 0, 0]) - n[0] * n; e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        if e2[1] < 0:
            e2 = -e2
        u = (P - c) @ e1; v = (P - c) @ e2; w = (P - c) @ n
        # footprint raster (0.25 mm) of the cartilage projected on the facet plane, holes closed
        res = 0.25
        ua, va = np.floor(u.min()) - 2, np.floor(v.min()) - 2
        nu_, nv_ = int((np.ceil(u.max()) + 2 - ua) / res), int((np.ceil(v.max()) + 2 - va) / res)
        F0 = np.zeros((nu_, nv_), bool)
        # splat each voxel as a h0 x h0 square
        for du in np.arange(-h0 / 2 + res / 2, h0 / 2, res):
            for dv in np.arange(-h0 / 2 + res / 2, h0 / 2, res):
                iu = np.clip(((u + du - ua) / res).astype(int), 0, nu_ - 1); iv = np.clip(((v + dv - va) / res).astype(int), 0, nv_ - 1)
                F0[iu, iv] = True
        F0 = ndimage.binary_closing(F0, np.ones((5, 5)), iterations=2)
        F0 = ndimage.binary_fill_holes(F0)
        gw = (dt.astype(float) + dc.astype(float))[S == b]
        fr[nm] = dict(c=c, n=n, e1=e1, e2=e2, ua=ua, va=va, res=res, F0=F0, a0=float(u.max() - u.min()) / 2,
                      xyz_lo=P.min(0) - h0 / 2, xyz_hi=P.max(0) + h0 / 2,
                      b0=float(v.max() - v.min()) / 2, uc=float((u.max() + u.min()) / 2), vc=float((v.max() + v.min()) / 2),
                      wmax=float(np.abs(w).max()), gap=float(np.median(gw)), area=float(F0.sum() * res * res))
    # ligament springs (engine frame) and leg load lines
    RA.set_model_frame(D)
    roles = RA.model_roles_from_D(D)
    from frame import to_engine
    springs = []
    for s in D["springs"]:
        a, b = s["nodes"]
        springs.append({"k": float(s["k"]), "a": to_engine(np.array(D["nodes"][a])).tolist(), "b": to_engine(np.array(D["nodes"][b])).tolist()})
    out_d = dict(lab=lab, origin=np.array([x0, y0, z0]), h=h0, cls_json=json.dumps({int(k): int(v) for k, v in cls.items()}),
                 contact=contact, contact_origin=np.array([x0, y0]), contact_res=rr, dt=dt, dc=dc, sub_origin=sub_origin,
                 sub_shape=np.array(S.shape), facets=json.dumps({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) and vv.dtype != bool else vv)
                                                                  for kk, vv in v.items() if kk != "F0"} for k, v in fr.items()}),
                 springs_json=json.dumps(springs), load_line=json.dumps(roles["load_line"]),
                 bounds=np.array([lo, hi]))
    for k, v in fr.items():
        out_d[f"F0_{k}"] = v["F0"]
    np.savez_compressed(out, **out_d)
    return out


class Template:
    _inst = None

    @classmethod
    def get(cls, path=None):
        if cls._inst is None:
            cls._inst = Template(path or TEMPLATE)
        return cls._inst

    def __init__(self, path):
        Z = np.load(path, allow_pickle=True)
        self.lab = Z["lab"]; self.o = Z["origin"].astype(float); self.h = float(Z["h"])
        self.cls = {int(k): int(v) for k, v in json.loads(str(Z["cls_json"])).items()}
        self.cls.update(NEW_CLASS)
        self.contact = Z["contact"]; self.co = Z["contact_origin"].astype(float); self.cres = float(Z["contact_res"])
        self.dt = Z["dt"].astype(np.float32); self.dc = Z["dc"].astype(np.float32)
        self.so = Z["sub_origin"].astype(float)
        self.facets = {k: {kk: (np.array(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in json.loads(str(Z["facets"])).items()}
        for k in self.facets:
            self.facets[k]["F0"] = Z[f"F0_{k}"]
        self.springs = json.loads(str(Z["springs_json"]))
        self.load_line = json.loads(str(Z["load_line"]))
        self.bounds = Z["bounds"]
        fp, fm = self.facets["post"], self.facets["mid"]
        # scale centre: ground point below the middle of the posterior and middle facets
        cxy = 0.5 * (fp["c"][:2] + fm["c"][:2])
        self.c0 = np.array([cxy[0], cxy[1], 0.0])
        # alignment shear: below the subtalar joint (z_ref, a little under the facets), fading out across the midfoot
        self.z_ref = float(min(fp["c"][2], fm["c"][2]) - 8.0)
        self.x_w0, self.x_w1 = -30.0, 10.0
        # osteotomy plane: 15 mm behind the posterior facet centre at its height, inclined like fe/geometry.x_cut
        # (x decreases by 0.5 mm per mm of height); fragment = everything behind it below z_frag_top except the talus
        self.x_cut0, self.z_cut0, self.cut_slope = float(fp["c"][0] - 15.0), float(fp["c"][2]), -0.5
        self.z_frag_top = 76.0
        self.static_bodies = np.array(sorted(set(BONES["talus"]) | set(BONES["tibia"]) | set(BONES["fibula"]) |
                                             {CART["ankle"], CART["syndesmosis"], CART["talofibular"]}))

    # ---- lookups
    def lookup(self, P):
        idx = np.floor((P - self.o) / self.h).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(self.lab.shape)), 1)
        out = np.zeros(len(P), np.int16)
        i = idx[ok]
        out[ok] = self.lab[i[:, 0], i[:, 1], i[:, 2]]
        return out

    def sub_lookup(self, arr, P, fill=99.0):
        idx = np.floor((P - self.so) / self.h).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(arr.shape)), 1)
        out = np.full(len(P), fill, np.float32)
        i = idx[ok]
        out[ok] = arr[i[:, 0], i[:, 1], i[:, 2]]
        return out

    def contact_lookup(self, XY):
        idx = np.floor((XY - self.co) / self.cres).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(self.contact.shape)), 1)
        out = np.zeros(len(XY), bool)
        out[ok] = self.contact[idx[ok, 0], idx[ok, 1]]
        return out

    def x_cut(self, z):
        return self.x_cut0 + self.cut_slope * (z - self.z_cut0)


# ---------------------------------------------------------------- warps
def _local_disp(T, foot, P):
    """Smooth local displacement (template -> foot) of the sustentaculum (sust_y) and of the posterior facet dome."""
    d = np.zeros_like(P)
    fm, fp = T.facets["mid"], T.facets["post"]
    if foot.sust_y != 11.0:
        r2 = ((P - fm["c"]) ** 2).sum(1)
        d[:, 1] += (foot.sust_y - 11.0) * np.exp(-r2 / (2 * 8.0 ** 2))
    if foot.post_dome != 1.5:
        q = P - fp["c"]
        u = (q @ fp["e1"] - fp["uc"]) / (fp["a0"] + 2.0); v = (q @ fp["e2"] - fp["vc"]) / (fp["b0"] + 2.0); w = q @ fp["n"]
        amp = (foot.post_dome - 1.5) * np.clip(1 - u ** 2 - v ** 2, 0, None) * np.exp(-w ** 2 / (2 * 6.0 ** 2))
        d += amp[:, None] * fp["n"][None, :]
    return d


def _shear(T, foot, P):
    """Lateral (valgus, align > 0) shift of the hindfoot below the joint level: -tan(align) (z_ref - z)+ w(x)."""
    if foot.align == 0.0:
        return np.zeros(len(P))
    w = np.clip((T.x_w1 - P[:, 0]) / (T.x_w1 - T.x_w0), 0, 1)
    return -np.tan(np.deg2rad(foot.align)) * np.clip(T.z_ref - P[:, 2], 0, None) * w


def forward(T, foot, act, P):
    """Template -> foot (world) coordinates (for spring ends and load lines)."""
    P = np.asarray(P, float).reshape(-1, 3)
    p1 = P + _local_disp(T, foot, P)
    p2 = p1.copy(); p2[:, 1] += _shear(T, foot, p1)
    p3 = p2.copy()
    if act is not None and act.osteo != 0.0:
        frag = (p2[:, 0] < T.x_cut(p2[:, 2])) & (p2[:, 2] < T.z_frag_top)
        p3[frag, 1] += act.osteo
    return T.c0 + foot.scale * (p3 - T.c0)


def inverse(T, foot, act, P):
    """Foot (world) -> template coordinates.  Returns (p_static, p_moved, frag): p_moved is the template point of the
    osteotomy fragment content at P (used where P lies behind the cut)."""
    p3 = T.c0 + (np.asarray(P, float) - T.c0) / foot.scale
    frag = np.zeros(len(p3), bool)
    outs = []
    variants = [p3]
    if act is not None and act.osteo != 0.0:
        frag = (p3[:, 0] < T.x_cut(p3[:, 2])) & (p3[:, 2] < T.z_frag_top)
        pm = p3.copy(); pm[:, 1] -= act.osteo
        variants.append(pm)
    for p2 in variants:
        p1 = p2.copy(); p1[:, 1] -= _shear(T, foot, p2)
        if foot.sust_y != 11.0 or foot.post_dome != 1.5:
            p = p1 - _local_disp(T, foot, p1)
            for _ in range(3):
                p = p1 - _local_disp(T, foot, p)
        else:
            p = p1
        outs.append(p)
    return outs[0], (outs[1] if len(outs) > 1 else None), frag


# ---------------------------------------------------------------- facets, coalitions, defects
def _facet_uv(F, P):
    q = P - F["c"]
    return q @ F["e1"], q @ F["e2"], q @ F["n"]


def _in_footprint(F, u, v):
    iu = np.floor((u - F["ua"]) / F["res"]).astype(np.int64); iv = np.floor((v - F["va"]) / F["res"]).astype(np.int64)
    ok = (iu >= 0) & (iu < F["F0"].shape[0]) & (iv >= 0) & (iv < F["F0"].shape[1])
    out = np.zeros(len(u), bool)
    out[ok] = F["F0"][iu[ok], iv[ok]]
    return out


def _footprint_samples(F, fa, fb):
    """(u, v) of the pixel centres of the scaled facet footprint."""
    iu, iv = np.nonzero(F["F0"])
    u = F["ua"] + (iu + 0.5) * F["res"]; v = F["va"] + (iv + 0.5) * F["res"]
    return F["uc"] + fa * (u - F["uc"]), F["vc"] + fb * (v - F["vc"])


def _scaled_in(F, u, v, fa, fb):
    return _in_footprint(F, F["uc"] + (u - F["uc"]) / fa, F["vc"] + (v - F["vc"]) / fb)


def _grown(F, fa, fb, d, frac, u, v):
    """Part of the scaled facet reached by a front moving along d (in facet (u, v) axes) from the facet edge, covering
    frac of its area (fe/geometry.FootGeom._grow)."""
    if frac <= 0:
        return np.zeros(len(u), bool)
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    us, vs = _footprint_samples(F, fa, fb)
    proj = (us - F["uc"]) * d[0] + (vs - F["vc"]) * d[1]
    thr = np.quantile(proj, min(frac, 1.0)) if frac < 1 else np.inf
    pq = (u - F["uc"]) * d[0] + (v - F["vc"]) * d[1]
    return pq <= thr


def coalition_dirs(foot):
    rot = np.deg2rad(foot.seed_ang)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    return R @ np.array([1.0, -0.35]), R @ np.array([0.25, -1.0])        # middle (anterior), posterior (lateral)


FACET_PAR = {"post": ("post_a", "post_b"), "mid": ("mid_a", "mid_b"), "ant": ("ant_a", "ant_b")}


def facet_factors(foot):
    return {k: (getattr(foot, a) / NOMINAL[k][0], getattr(foot, b) / NOMINAL[k][1]) for k, (a, b) in FACET_PAR.items()}


def label_points(P, foot, act=None, T=None, defect_fill="soft"):
    """Body label of every point P (N, 3) of the foot (world frame)."""
    T = T or Template.get()
    act = act or Action()
    ps, pm, frag = inverse(T, foot, act, P)
    L = T.lookup(ps)
    if pm is not None and frag.any():
        Lm = T.lookup(pm[frag])
        Ls = L[frag]
        moving_s = ~np.isin(Ls, T.static_bodies)
        moving_m = ~np.isin(Lm, T.static_bodies)
        L[frag] = np.where(moving_s, np.where(moving_m, Lm, SOFT), Ls)
    # ---- subtalar edits (points near the joint, outside the osteotomy fragment)
    so, sh = T.so, np.array(T.dt.shape) * T.h
    near = np.all((ps >= so) & (ps < so + sh), 1) & ~frag
    idx = np.nonzero(near)[0]
    if not len(idx):
        return L
    q = ps[idx]; Lq = L[idx].copy()
    dt = T.sub_lookup(T.dt, q); dc = T.sub_lookup(T.dc, q)
    gapw = dt + dc
    fac = facet_factors(foot)
    uvw = {k: _facet_uv(T.facets[k], q) for k in FACETS}
    infacet = {}
    # facet size: remove cartilage outside the scaled footprint, add it in the thin gap inside it
    for k, b in FACETS.items():
        F = T.facets[k]; fa, fb = fac[k]
        u, v, w = uvw[k]
        cand = (np.abs(w) <= F["wmax"] + 1.0) & (np.abs(u - F["uc"]) <= 1.7 * F["a0"] * max(fa, 1)) & \
               (np.abs(v - F["vc"]) <= 1.7 * F["b0"] * max(fb, 1))
        ins = cand & _scaled_in(F, u, v, fa, fb)
        if fa != 1.0 or fb != 1.0:
            grow = ins & (gapw <= 1.6 * F["gap"]) & np.isin(Lq, [SOFT, 0])
            Lq[grow] = b
            Lq[(Lq == b) & ~ins] = SOFT
        infacet[k] = ins
    # anterior facet mode
    fm_, fa_ = T.facets["mid"], T.facets["ant"]
    if foot.ant_mode == 1:
        seg = fa_["c"] - fm_["c"]; t = ((q - fm_["c"]) @ seg) / (seg @ seg)
        dist = np.linalg.norm(q - fm_["c"] - t[:, None] * seg[None, :], axis=1)
        conf = (t > 0) & (t < 1) & (dist < 0.75 * min(fm_["b0"] * fac["mid"][1], fa_["b0"] * fac["ant"][1])) & \
               (gapw <= 1.6 * max(fm_["gap"], fa_["gap"])) & np.isin(Lq, [SOFT, 0])
        Lq[conf] = FACETS["ant"]
    elif foot.ant_mode == 2:
        Lq[Lq == FACETS["ant"]] = SOFT
    # coalition and defects
    d_mid, d_post = coalition_dirs(foot)
    mid_c, post_c = Lq == FACETS["mid"], Lq == FACETS["post"]
    um, vm_, wm = uvw["mid"]; up, vp, wp = uvw["post"]
    Fm, Fp = T.facets["mid"], T.facets["post"]
    br = np.zeros(len(q), bool); fib = np.zeros(len(q), bool)
    if foot.ctype != "none":
        gap_bar = (dt <= 4.0) & (dc <= 4.0) & np.isin(Lq, [SOFT, 0])            # soft tissue between the two bones
        if foot.ctype in ("I", "IV"):
            if foot.rtype in ("R1", "R2", "R3"):
                b_ = Fm["b0"] * fac["mid"][1]
                band = mid_c & (vm_ >= Fm["vc"] + b_ * (1 - 2 * foot.cov_mid))
                br |= band
                if foot.rtype == "R2":
                    # posterior hook into the canal (fe/geometry: 6 mm behind the facet's posterior edge, 1.5 mm medial)
                    # (rounded: elliptical footprint of the same area as the 7 x 4 mm box of fe/geometry)
                    xe_ = Fm["c"][0] - (Fm["c"][0] - Fm["xyz_lo"][0]) * fac["mid"][0]
                    k_ = 2.0 / np.sqrt(np.pi)
                    hook = gap_bar & (((q[:, 0] - (xe_ - 2.5)) / (3.5 * k_)) ** 2 + ((q[:, 1] - (Fm["c"][1] + 1.5)) / (2.0 * k_)) ** 2 <= 1) & \
                           (np.abs(wm) < 6.0)
                    br |= hook
                if foot.rtype == "R3":
                    a_ = Fm["a0"] * fac["mid"][0]
                    fib = band & (um < Fm["uc"] - a_ + 2 * a_ * foot.fib_frac)
                    br &= ~fib
            else:
                br |= mid_c & _grown(Fm, *fac["mid"], d_mid, foot.cov_mid, um, vm_)
        if foot.ctype in ("II", "IV"):
            br |= post_c & _grown(Fp, *fac["post"], d_post, foot.cov_post, up, vp)
        if foot.ctype in ("III", "IV"):
            # extra-articular bar in the tarsal canal: medial gap between the posterior and the middle facet
            x_lo = Fp["c"][0] + 2.0; x_hi = Fm["c"][0] - (Fm["c"][0] - Fm["xyz_lo"][0]) * fac["mid"][0] + 1.0
            y_lo = Fp["c"][1] + (Fp["xyz_hi"][1] - Fp["c"][1]) * fac["post"][1] - 3.0; y_hi = Fm["c"][1] + 3.0
            # elliptical footprint with the area of the x-y window of fe/geometry (a bar, not a box)
            k_ = 2.0 / np.sqrt(np.pi)
            ex, ey = 0.5 * (x_hi - x_lo) * k_, 0.5 * (y_hi - y_lo) * k_
            corr = gap_bar & (((q[:, 0] - 0.5 * (x_lo + x_hi)) / ex) ** 2 + ((q[:, 1] - 0.5 * (y_lo + y_hi)) / ey) ** 2 <= 1)
            br |= corr
    dm = mid_c & _grown(Fm, *fac["mid"], d_mid, act.d_mid, um, vm_)
    dp = post_c & _grown(Fp, *fac["post"], d_post, act.d_post, up, vp)
    if act.fusion:
        fz = np.isin(Lq, list(FACETS.values())) | br | fib | ((gapw <= 4.0) & (dt <= 4.0) & (dc <= 4.0) & np.isin(Lq, [SOFT, 0]))
        Lq[fz] = FUSE
    elif act.resect:
        hole = br | fib | dm | dp
        if act.interp:
            Lq[hole] = INTERP
        elif defect_fill == "void":
            Lq[hole] = GAP
        else:
            # v3e (2026-11-01): the facet part of the resection (facet defect and the bar's footprint on the facet)
            # is soft tissue bonded to both bones, as in the FE atlas; the extra-articular part of the bar is empty
            on_facet = hole & (mid_c | post_c)
            Lq[hole & ~on_facet] = GAP
            Lq[on_facet] = SOFT
    else:
        Lq[br] = COAL
        Lq[fib] = FIB
        Lq[(dm | dp) & ~br & ~fib] = GAP if defect_fill == "void" else SOFT
    L[idx] = Lq
    return L


def contact_points(XY, foot, act=None, T=None):
    """Ground contact (soft tissue bonded to the plate) at world points (x, y, 0)."""
    T = T or Template.get()
    P = np.column_stack([XY, np.zeros(len(XY))])
    ps, pm, frag = inverse(T, foot, act or Action(), P)
    c = T.contact_lookup(ps[:, :2])
    if pm is not None and frag.any():
        c[frag] = T.contact_lookup(pm[frag, :2])
    return c


if __name__ == "__main__":
    out = build_template(sys.argv[1], sys.argv[2])
    print("template written", out)


# ---------------------------------------------------------------- fast labelling of tensor-product lattices
JOINT_BOX = ((-82.0, -12.0), (-32.0, 32.0), (36.0, 92.0))      # template frame: every subtalar edit happens inside


def _raster_index(T, t, ax):
    i = np.floor((t - T.o[ax]) / T.h).astype(np.int64)
    ok = (i >= 0) & (i < T.lab.shape[ax])
    return np.where(ok, i, 0), ok


def label_lattice(sx, sy, sz, foot, act=None, T=None, defect_fill="soft", chunk=2_000_000):
    """Labels of the lattice sx x sy x sz (world frame), identical to label_points on its points: the scale is undone
    separably and the raster read with separable indices; only points in regions where the alignment shear, the local
    warps, the osteotomy or the subtalar edits can act are labelled point by point."""
    T = T or Template.get()
    act = act or Action()
    s = foot.scale; c0 = T.c0
    tx = c0[0] + (np.asarray(sx, float) - c0[0]) / s
    ty = c0[1] + (np.asarray(sy, float) - c0[1]) / s
    tz = c0[2] + (np.asarray(sz, float) - c0[2]) / s
    (ix, okx), (iy, oky), (iz, okz) = _raster_index(T, tx, 0), _raster_index(T, ty, 1), _raster_index(T, tz, 2)
    L = T.lab[np.ix_(ix, iy, iz)].astype(np.int16)
    L[~okx] = 0; L[:, ~oky] = 0; L[:, :, ~okz] = 0
    need = np.zeros(L.shape, bool)
    (jx0, jx1), (jy0, jy1), (jz0, jz1) = JOINT_BOX
    need |= ((tx >= jx0) & (tx <= jx1))[:, None, None] & ((ty >= jy0) & (ty <= jy1))[None, :, None] & ((tz >= jz0) & (tz <= jz1))[None, None, :]
    if foot.align != 0.0:
        need |= (tx < T.x_w1 + 1.0)[:, None, None] & (tz < T.z_ref + 1.0)[None, None, :]
    if act.osteo != 0.0:
        need |= ((tx[:, None] < T.x_cut(tz)[None, :] + 1.0) & (tz < T.z_frag_top + 1.0)[None, :])[:, None, :]
    if foot.sust_y != 11.0:
        c = T.facets["mid"]["c"]; r = 4 * 8.0 + abs(foot.sust_y - 11.0) + 1.0
        need |= (np.abs(tx - c[0]) < r)[:, None, None] & (np.abs(ty - c[1]) < r)[None, :, None] & (np.abs(tz - c[2]) < r)[None, None, :]
    if foot.post_dome != 1.5:
        c = T.facets["post"]["c"]; r = 40.0
        need |= (np.abs(tx - c[0]) < r)[:, None, None] & (np.abs(ty - c[1]) < r)[None, :, None] & (np.abs(tz - c[2]) < r)[None, None, :]
    I, J, K = np.nonzero(need)
    for a in range(0, len(I), chunk):
        i, j, k = I[a:a + chunk], J[a:a + chunk], K[a:a + chunk]
        P = np.column_stack([np.asarray(sx, float)[i], np.asarray(sy, float)[j], np.asarray(sz, float)[k]])
        L[i, j, k] = label_points(P, foot, act, T, defect_fill)
    return L
