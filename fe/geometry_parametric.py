"""Parametric hindfoot (talus, calcaneus, navicular, cuboid) with three subtalar facets,
coalitions, facet defects left by complete resection, hindfoot alignment, calcaneal osteotomy,
interposition and subtalar fusion.  Geometry is defined by point-membership functions and
voxelised with partial-volume fractions on a rectilinear grid.

Frame (mm): x posterior->anterior, y lateral->medial (right foot), z inferior->superior.
"""
import numpy as np
from dataclasses import dataclass, field, asdict

# ---------------- labels ----------------
VOID, TALUS, CALC, NAV, CUB, C_ANT, C_MID, C_POST, C_TN, C_CC, COAL, FIB, INTERP, FUSE, LIG = range(15)
LABEL_NAMES = ["void", "talus", "calcaneus", "navicular", "cuboid", "cart_ant", "cart_mid", "cart_post",
               "cart_TN", "cart_CC", "coalition", "fibrous", "interposition", "fusion_bone", "ligament"]
# materials (E MPa, nu) -- bone/cartilage/fascia-like values follow the FE atlas template
MAT = {
    "bone": (7300.0, 0.30),
    "cart": (10.0, 0.40),
    "fib": (60.0, 0.40),        # fibrous/cartilaginous interval of a shingled coalition
    "interp": (1.15, 0.45),     # interposed fat graft (soft-tissue modulus of the atlas)
    "lig": (260.0, 0.30),       # ligament solids (atlas value)
}
LABEL_MAT = {TALUS: "bone", CALC: "bone", NAV: "bone", CUB: "bone", C_ANT: "cart", C_MID: "cart",
             C_POST: "cart", C_TN: "cart", C_CC: "cart", COAL: "bone", FIB: "fib", INTERP: "interp",
             FUSE: "bone", LIG: "lig"}

COAL_TYPES = ["none", "I", "II", "III", "IV"]                 # anatomy-based (Li 2026)
ROZ_TYPES = ["none", "R1", "R2", "R3", "R4", "R5", "unclass"]  # Rozansky


@dataclass
class Foot:
    """Pre-operative state parameters of one (virtual) foot."""
    scale: float = 1.0          # isotropic size factor (about the facet centre)
    post_a: float = 12.0        # posterior facet semi-axes (mm)
    post_b: float = 9.5
    mid_a: float = 6.5          # middle facet
    mid_b: float = 4.5
    ant_a: float = 4.5          # anterior facet
    ant_b: float = 4.0
    ant_mode: int = 0           # 0 separate anterior facet, 1 confluent with middle, 2 absent
    sust_y: float = 11.0        # middle facet (sustentaculum) medial position
    post_dome: float = 1.5      # convexity of posterior facet (mm)
    cart_t: float = 2.0         # combined cartilage thickness (mm)
    align: float = 0.0          # hindfoot alignment (deg, + valgus, - varus)
    # coalition
    ctype: str = "none"         # anatomy-based type none/I/II/III/IV
    rtype: str = "none"         # Rozansky R1..R5 / unclass
    cov_mid: float = 0.0        # fraction of the middle facet occupied by coalition
    cov_post: float = 0.0       # fraction of the posterior facet occupied by coalition
    seed_ang: float = 0.0       # growth direction perturbation (deg)
    fib_frac: float = 0.0       # shingled: fraction of bridge that is fibrous

    def as_vec(self):
        return np.array([self.scale, self.post_a, self.post_b, self.mid_a, self.mid_b, self.ant_a, self.ant_b,
                         float(self.ant_mode), self.sust_y, self.post_dome, self.cart_t, self.align])

    @staticmethod
    def vec_names():
        return ["scale", "post_a", "post_b", "mid_a", "mid_b", "ant_a", "ant_b", "ant_mode", "sust_y",
                "post_dome", "cart_t", "align"]


@dataclass
class Action:
    """Surgical action. Complete resection of the coalition is implied whenever resect=1; the facet
    defect it leaves is described by d_mid/d_post (fraction of facet area), grown from the coalition."""
    resect: int = 0
    d_mid: float = 0.0
    d_post: float = 0.0
    osteo: float = 0.0          # calcaneal osteotomy translation (mm, + medialising)
    interp: int = 0             # interposition of fat graft in the resection gap
    fusion: int = 0             # subtalar fusion branch
    load: float = 1.0           # load multiplier (1 = single-leg stance, 360 N)

    def as_vec(self):
        return np.array([self.resect, self.d_mid, self.d_post, self.osteo / 10.0, self.interp, self.fusion,
                         self.load - 1.0])

    @staticmethod
    def vec_names():
        return ["resect", "d_mid", "d_post", "osteo/10", "interp", "fusion", "load-1"]


# ---------------- reference anatomy (scale = 1) ----------------
POST_C = np.array([33.0, -2.0])
MID_C0 = np.array([50.5, 11.0])
ANT_C0 = np.array([61.0, 9.0])
TN_SLOPE = 0.7        # inclination of the talonavicular interface (dx/dz)
Z_REF = 29.0          # alignment shear acts below the subtalar band (z < Z_REF)


def x_cut(z):
    """Oblique calcaneal osteotomy plane: behind the posterior facet superiorly, anterior to the
    weight-bearing tuberosity inferiorly."""
    return 18.0 + 0.5 * (30.0 - z)


def _ell(x, y, c, a, b):
    return ((x - c[0]) / a) ** 2 + ((y - c[1]) / b) ** 2


def _sig(t):
    return 1 / (1 + np.exp(-t))


class FootGeom:
    """Point-membership model for one foot + action."""

    def __init__(self, foot: Foot, act: Action = None):
        self.f = foot
        self.a = act or Action()
        f = foot
        self.mid_c = np.array([MID_C0[0], f.sust_y])
        self.ant_c = np.array([ANT_C0[0], f.sust_y - 2.0])
        self.t_align = np.tan(np.deg2rad(f.align))

    # ---- joint surface and facets (in calcaneal frame x,y) ----
    def z_joint(self, x, y):
        f = self.f
        z = 34.0 + 0.0 * x
        r = _ell(x, y, POST_C, f.post_a + 2, f.post_b + 2)
        z = z + f.post_dome * np.clip(1 - r, 0, None)
        return z

    def facet_masks(self, x, y):
        f = self.f
        post = _ell(x, y, POST_C, f.post_a, f.post_b) <= 1
        mid = _ell(x, y, self.mid_c, f.mid_a, f.mid_b) <= 1
        if f.ant_mode == 0:
            ant = _ell(x, y, self.ant_c, f.ant_a, f.ant_b) <= 1
        elif f.ant_mode == 1:   # confluent: elongated middle facet bridging to the anterior one
            ant = (_ell(x, y, self.ant_c, f.ant_a, f.ant_b) <= 1) | (
                (x > self.mid_c[0]) & (x < self.ant_c[0]) & (np.abs(y - (self.mid_c[1] - 1)) < 0.75 * min(f.mid_b, f.ant_b)))
            ant = ant & ~mid
        else:
            ant = np.zeros_like(post)
        return ant, mid, post

    # ---- coalition footprint (in x,y) and defect footprints ----
    def _grow(self, x, y, facet, centre, a, b, direction, frac):
        """Region of the facet ellipse reached by a front moving along `direction` from the facet edge,
        covering `frac` of the facet area (closed form via a supersampled CDF)."""
        if frac <= 0:
            return np.zeros_like(x, dtype=bool), None
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        # sample the facet to find the threshold on projection
        uu, vv = np.meshgrid(np.linspace(-1, 1, 161), np.linspace(-1, 1, 161))
        inside = uu ** 2 + vv ** 2 <= 1
        px = centre[0] + a * uu[inside]; py = centre[1] + b * vv[inside]
        proj = (px - centre[0]) * d[0] + (py - centre[1]) * d[1]
        thr = np.quantile(proj, min(frac, 1.0)) if frac < 1 else np.inf
        projq = (x - centre[0]) * d[0] + (y - centre[1]) * d[1]
        return facet & (projq <= thr), thr

    def coalition_dirs(self):
        rot = np.deg2rad(self.f.seed_ang)
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        d_post = R @ np.array([0.25, -1.0])      # posterior facet: from medial edge laterally
        d_mid = R @ np.array([1.0, -0.35])       # middle facet: from posterior-medial edge anteriorly
        return d_mid, d_post

    def coalition_xy(self, x, y):
        """Return (bridge, fibrous) boolean masks in x,y for the pre-operative coalition."""
        f = self.f
        ant, mid, post = self.facet_masks(x, y)
        d_mid, d_post = self.coalition_dirs()
        br = np.zeros_like(x, dtype=bool)
        fib = np.zeros_like(x, dtype=bool)
        if f.ctype == "none":
            return br, fib
        midc = self.mid_c
        # extra-articular corridor: medial gap posterior to the middle facet, medial to the posterior facet
        corridor = (x > POST_C[0] + 2) & (x < midc[0] - f.mid_a + 1.0) & (y > POST_C[1] + f.post_b - 3.0) & \
                   (y < midc[1] + 3.0) & ~(ant | mid | post)
        if f.ctype in ("I", "IV"):
            if f.rtype in ("R1", "R2", "R3"):
                # linear bar: band along the facet long axis on its medial side
                band = mid & (y >= midc[1] + f.mid_b * (1 - 2 * f.cov_mid))
                br |= band
                if f.rtype == "R2":      # posterior hook extending into the canal toward the posterior facet
                    hook = (x > midc[0] - f.mid_a - 6) & (x < midc[0] - f.mid_a + 1) & (np.abs(y - (midc[1] + 1.5)) < 2.0)
                    br |= hook
                if f.rtype == "R3":      # shingled: part of the bar is a fibrous interval
                    fib = band & (x < midc[0] - f.mid_a + 2 * f.mid_a * f.fib_frac)
                    br &= ~fib
            else:
                m, _ = self._grow(x, y, mid, midc, f.mid_a, f.mid_b, d_mid, f.cov_mid)
                br |= m
        if f.ctype in ("II", "IV"):
            p, _ = self._grow(x, y, post, POST_C, f.post_a, f.post_b, d_post, f.cov_post)
            br |= p
        if f.ctype in ("III", "IV"):
            br |= corridor
        return br, fib

    def defect_xy(self, x, y):
        """Facet defect left by complete resection (fractions d_mid/d_post of facet area)."""
        f, a = self.f, self.a
        ant, mid, post = self.facet_masks(x, y)
        d_mid, d_post = self.coalition_dirs()
        dm, _ = self._grow(x, y, mid, self.mid_c, f.mid_a, f.mid_b, d_mid, a.d_mid)
        dp, _ = self._grow(x, y, post, POST_C, f.post_a, f.post_b, d_post, a.d_post)
        return dm, dp

    # ---- bone footprints ----
    def talus_xy(self, x, y):
        body = _ell(x, y, np.array([34.0, 1.0]), 18.0, 15.0) <= 1
        t = np.clip((x - 42.0) / 40.0, 0, 1)
        lo = -9.0 + 7.0 * t
        hi = 13.0 + 6.0 * t + 4.0 * np.exp(-((x - self.mid_c[0]) / 7.0) ** 2) * (self.f.sust_y - 6) / 5
        neck = (x >= 42.0) & (x <= 84.0) & (y >= lo) & (y <= hi)
        return body | neck

    def talus_top(self, x, y):
        dome = 61.0 - 0.022 * (x - 34.0) ** 2 - 0.012 * (y - 1.0) ** 2
        head = 53.0 - 0.004 * (x - 70.0) ** 2
        w = _sig((x - 50.0) / 2.0)
        return (1 - w) * dome + w * head

    def calc_xy(self, x, y, z, zj):
        body = (x >= 2.0) & (x <= 79.0) & (y >= -14.5) & (y <= 8.5) & ~((y > 0.0) & (x > 70.0))
        # anterior process carries the anterior facet medially
        antp = (x >= 54.0) & (x <= 70.0) & (y > 0.0) & (y <= self.ant_c[1] + 5.0) & (z >= zj - 14.0)
        # sustentaculum tali: robust medial shelf carrying the middle facet, tapering medially
        thick = 13.0 - 0.5 * np.clip(y - 8.5, 0, None)
        sus = (x >= self.mid_c[0] - self.f.mid_a - 4) & (x <= self.ant_c[0] + 6) & (y > 6.0) & \
              (y <= self.f.sust_y + 8.0) & (z >= zj - thick)
        return body | sus | antp

    def calc_bottom(self, x):
        return np.clip((x - 28.0) * 12.0 / 50.0, 0, None)

    @staticmethod
    def x_tn(y, z):
        """Talonavicular interface: inclined (normal antero-inferior) so the talar head sits in the socket."""
        return 80.0 + TN_SLOPE * (z - 44.0) - 0.02 * (y - 11.0) ** 2

    @staticmethod
    def x_cc(z):
        """Calcaneocuboid interface: the anterior process overhangs the cuboid."""
        return 78.0 + 0.6 * (z - 21.0)

    def to_calc_y(self, x, y, z):
        """Map a point to the calcaneal frame: valgus (+align) moves the heel laterally (-y) below the
        subtalar band; a medialising osteotomy translates the posterior fragment by +osteo."""
        yc = y + np.clip(Z_REF - z, 0, None) * self.t_align
        return np.where(x < x_cut(z), yc - self.a.osteo, yc)

    # ---- point classification ----
    def classify(self, X, Y, Z):
        """Label every point (arrays of equal shape)."""
        f, a = self.f, self.a
        s = f.scale
        # undo isotropic scaling about the facet centre (35, 0, 35)
        x = 35.0 + (X - 35.0) / s
        y = (Y) / s
        z = 35.0 + (Z - 35.0) / s
        lab = np.zeros(X.shape, dtype=np.int8)

        # ---- calcaneal frame: alignment shear about the joint level + osteotomy translation ----
        yc = self.to_calc_y(x, y, z)
        zj_c = self.z_joint(x, y)
        cal_in = self.calc_xy(x, yc, z, zj_c) & (z >= self.calc_bottom(x))
        # anterior process ends at the CC joint
        x_cc = self.x_cc(z)
        cal_in &= ~((x > x_cc) & (yc < 0.0)) & (x <= 79.0)
        # calcaneal top: joint surface under the talus, lower elsewhere
        tal_xy = self.talus_xy(x, y)
        cal_top = np.where(tal_xy, zj_c, np.minimum(zj_c, 31.0))
        cal_top = np.where(x < 18, 30.0 - 0.3 * (18 - x), cal_top)
        cal_in &= z <= cal_top

        # ---- joint gap ----
        ant, mid, post = self.facet_masks(x, y)
        facet = ant | mid | post
        gap = np.where(facet, f.cart_t, 5.0)
        zj = self.z_joint(x, y)
        in_gap = tal_xy & (z > zj) & (z <= zj + gap) & (x > 18)
        # calcaneus support under the gap (only articulate where calcaneus exists below)
        cal_below = self.calc_xy(x, self.to_calc_y(x, y, zj - 0.5), zj - 0.5, zj)
        in_gap &= cal_below

        # ---- talus ----
        x_tn = self.x_tn(y, z)
        tal_in = tal_xy & (z > zj + gap) & (z <= self.talus_top(x, y)) & (x <= x_tn)
        tal_in &= z >= zj + gap

        # ---- navicular and TN cartilage ----
        tn_reg = (y >= 1.0) & (y <= 24.0) & (z >= 31.0) & (z <= 54.0)
        tn_cart = tn_reg & (x > x_tn) & (x <= x_tn + f.cart_t) & tal_xy
        nav = tn_reg & (x > x_tn + f.cart_t) & (x <= 92.0) & (np.abs(y - 12.5) <= 11.5 - 0.2 * np.abs(z - 43.5))

        # ---- cuboid and CC cartilage (move with the calcaneus: use yc) ----
        cc_reg = (yc >= -14.5) & (yc <= -0.5) & (z >= 12.0) & (z <= 29.0)
        cc_cart = cc_reg & (x > x_cc) & (x <= x_cc + f.cart_t)
        cub = cc_reg & (x > x_cc + f.cart_t) & (x <= 92.0)

        # ---- ligaments (solids, atlas convention) ----
        # plantar calcaneonavicular (spring) ligament: sustentaculum -> navicular, beneath the talar head
        spring = (x >= 62.0) & (x <= 86.0) & (y >= self.f.sust_y - 1.0) & (y <= self.f.sust_y + 7.0) & \
                 (z >= 29.0) & (z <= 33.0)
        # navicular-cuboid (and bifurcate) ligament between the two bones
        navcub = (x >= 80.0) & (x <= 92.0) & (yc >= -0.5) & (y <= 1.0) & (z >= 26.0) & (z <= 33.0)
        ligs = (spring | navcub)

        # ---- coalition, defects, fusion, interposition in the gap ----
        bridge, fibr = self.coalition_xy(x, y)
        dm, dp = self.defect_xy(x, y)
        gap_lab = np.where(post, C_POST, np.where(mid, C_MID, np.where(ant, C_ANT, VOID))).astype(np.int8)
        if a.fusion:
            gap_lab[:] = FUSE
        elif a.resect:
            # complete resection: bridge removed; facet defect (dm|dp) has no cartilage
            hole = bridge | fibr | dm | dp
            gap_lab = np.where(hole, INTERP if a.interp else VOID, gap_lab).astype(np.int8)
        else:
            gap_lab = np.where(bridge, COAL, gap_lab).astype(np.int8)
            gap_lab = np.where(fibr, FIB, gap_lab).astype(np.int8)
            # a pre-operative defect sweep on a coalition-free foot (atlas-style)
            gap_lab = np.where(dm | dp, VOID, gap_lab).astype(np.int8)

        # ---- compose (later assignments win) ----
        lab[ligs] = LIG
        lab[cal_in] = CALC
        lab[cub] = CUB
        lab[cc_cart] = C_CC
        lab[tal_in] = TALUS
        lab[nav] = NAV
        lab[tn_cart] = C_TN
        g = in_gap & (gap_lab != VOID)
        lab[g] = gap_lab[g]
        return lab


# ---------------- grid & voxelisation ----------------
def _segments(segs, f):
    e = [segs[0][0]]
    for a, b, h in segs:
        n = max(1, int(round((b - a) / (h * f))))
        e += list(np.linspace(a, b, n + 1)[1:])
    return np.array(e)


def make_grid_edges(fine=1.0):
    """Rectilinear grid edges: 2 mm in x,y; 1 mm in the subtalar band (z) and TN/CC zone (x).
    `fine` scales every spacing (fine < 1 refines the whole grid, used for the convergence study)."""
    xe = _segments([(0, 74, 2.0), (74, 92, 1.0), (92, 96, 2.0)], fine)
    ye = _segments([(-18, 30, 2.0)], fine)
    ze = _segments([(0, 24, 3.0), (24, 26, 2.0), (26, 42, 1.0), (42, 63, 3.0)], fine)
    return xe, ye, ze


def voxelise(geom: FootGeom, grid, ss=(3, 3, 3)):
    """Partial-volume label fractions: returns (ncell, nlabels) float32 fractions."""
    nx, ny, nz = grid.shape
    ox = (np.arange(ss[0]) + 0.5) / ss[0]
    oy = (np.arange(ss[1]) + 0.5) / ss[1]
    oz = (np.arange(ss[2]) + 0.5) / ss[2]
    X = grid.xe[:-1, None] + grid.dx[:, None] * ox[None, :]   # (nx, sx)
    Y = grid.ye[:-1, None] + grid.dy[:, None] * oy[None, :]
    Z = grid.ze[:-1, None] + grid.dz[:, None] * oz[None, :]
    XX = X[:, None, None, :, None, None]
    YY = Y[None, :, None, None, :, None]
    ZZ = Z[None, None, :, None, None, :]
    XX, YY, ZZ = np.broadcast_arrays(XX, YY, ZZ)
    lab = geom.classify(XX, YY, ZZ)                             # (nx,ny,nz,sx,sy,sz)
    lab = lab.reshape(nx * ny * nz, -1)
    nl = len(LABEL_NAMES)
    frac = np.zeros((lab.shape[0], nl), np.float32)
    for l in range(nl):
        frac[:, l] = (lab == l).mean(1)
    return frac


def materials_from_fractions(frac, mat_override=None, mixing="reuss"):
    """Partial-volume mixing.  In cells that contain articular cartilage the modulus is the series (Reuss)
    average of the occupied part scaled by the occupied fraction, which preserves the through-thickness
    compliance of thin articular layers cut by the grid; other cells use volume (Voigt) averaging.
    Poisson's ratio is volume averaged."""
    mats = dict(MAT)
    if mat_override:
        mats.update(mat_override)
    n = frac.shape[0]
    occ = np.zeros(n); inv = np.zeros(n); Ev = np.zeros(n); NU = np.zeros(n)
    for l, m in LABEL_MAT.items():
        e, nu = mats[m]
        occ += frac[:, l]
        inv += frac[:, l] / e
        Ev += frac[:, l] * e
        NU += frac[:, l] * nu
    safe = np.maximum(occ, 1e-12)
    if mixing == "reuss":
        # series mixing only where an articular layer is present; volume (Voigt) mixing elsewhere
        cart = frac[:, [C_ANT, C_MID, C_POST, C_TN, C_CC]].sum(1) > 1e-6
        E = np.where(cart & (occ > 0), occ * (safe / np.maximum(inv, 1e-30)), Ev)
    else:
        E = Ev
    nu = np.where(occ > 0, NU / safe, 0.3)
    return E, nu, occ
