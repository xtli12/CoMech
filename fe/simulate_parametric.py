"""One FE simulation: Foot + Action -> fields + effect descriptors."""
import numpy as np, time
from fe.geometry import *
from fe.solver import Grid, lame, solve, cell_stress, face_node_areas, assemble, ortho_coeffs

BODY_LOAD = 360.0       # N, single-leg stance of a ~36 kg child (300 N tibia + 60 N fibula in the atlas)
K_HEEL = (0.02, 0.02, 0.12)      # N/mm^3 heel-pad foundation (shear, shear, normal)
K_NAV = [0.0075, 0.0075, 0.03]   # N/mm^3 medial-column foundation (inferior face of navicular)
K_CUB = [0.075, 0.075, 0.3]      # N/mm^3 lateral-column foundation (inferior face of cuboid)
K_ANT = [0.01, 0.01, 0.01]       # N/mm^3 weak restraint on anterior faces (distal joints)
K_MORT = 0.0                     # N/mm^3 optional distributed medial-lateral mortise springs (off)
K_MORT_X = 300.0                 # N/mm restraint of the mean x and y trochlear displacement (mortise)
RIGID_DEFLATION = False          # per-bone rigid-body coarse correction in the preconditioner
OCC_MIN = 0.15                   # cells with smaller occupied fraction are treated as void
JOINT_SHEAR = 1.0                # through-thickness shear factor of subtalar facet cartilage (1 = bonded)
LOAD_X = [30.0, 54.0]            # x-window of the tibial load on the trochlea
LOAD_Y = [-3.0, 21.0]            # y-window (calibrated: middle/posterior mean-stress ratio ~ atlas)
HEEL_HALFWIDTH = None            # half-width (mm) of the heel-pad contact strip; None = whole tuberosity
HEEL_XMAX = 34.0                 # heel-pad contact: calcaneal cells of the lowest layer posterior to this x

_GRID_CACHE = {}


def get_grid(fine=1.0):
    if fine not in _GRID_CACHE:
        _GRID_CACHE[fine] = Grid(*make_grid_edges(fine))
    return _GRID_CACHE[fine]


CART_LABELS = [C_ANT, C_MID, C_POST, C_TN, C_CC]


def active_cells(frac):
    """Cells treated as solid.  Cells containing articular cartilage keep partial-volume treatment
    (occupancy > OCC_MIN) so that facet defects vary smoothly; all other cells follow the majority rule
    (occupancy >= 0.5), which prevents thin bone slivers on either side of a joint gap from touching
    diagonally and bridging it."""
    occ = frac[:, 1:].sum(1)
    cart = frac[:, CART_LABELS].sum(1)
    return np.where(cart > 0.05, occ > OCC_MIN, occ >= 0.5)


def region_labels(frac):
    """Dominant label per active cell (void otherwise)."""
    lab = frac[:, 1:].argmax(1) + 1
    lab[~active_cells(frac)] = VOID
    return lab


def canonical(xyz, s):
    """World -> anatomical (unscaled) coordinates; the foot is scaled isotropically about (35, 0, 35)."""
    c = xyz.copy()
    c[:, 0] = 35.0 + (xyz[:, 0] - 35.0) / s
    c[:, 1] = xyz[:, 1] / s
    c[:, 2] = 35.0 + (xyz[:, 2] - 35.0) / s
    return c


def boundary(grid, lab, frac, act, foot):
    """Return springs and loads.  All anatomical windows are defined in canonical (unscaled) coordinates so
    that they follow the foot when it is scaled."""
    lab3 = lab.reshape(grid.shape)
    occ3 = (lab3 != VOID)
    xyz = grid.cell_xyz
    cxyz = canonical(xyz, foot.scale)
    dn = np.zeros(grid.shape, bool); dn[:, :, 1:] = occ3[:, :, :-1]          # occupied cell below
    up = np.zeros(grid.shape, bool); up[:, :, :-1] = occ3[:, :, 1:]          # occupied cell above
    fw = np.zeros(grid.shape, bool); fw[:-1] = occ3[1:]                      # occupied cell anterior
    dn, up, fw = dn.ravel(), up.ravel(), fw.ravel()
    # heel pad: inferior faces of the lowest 3 mm of the calcaneal tuberosity (canonical x < HEEL_XMAX)
    cal = lab == CALC
    zmin = cxyz[cal, 2].min()
    heel = np.nonzero(cal & ~dn & (cxyz[:, 2] < zmin + 2.9) & (cxyz[:, 0] < HEEL_XMAX))[0]
    if HEEL_HALFWIDTH is not None and len(heel):
        w = grid.cell_vol[heel]
        yc_ = (xyz[heel, 1] * w).sum() / w.sum()
        heel = heel[np.abs(xyz[heel, 1] - yc_) <= HEEL_HALFWIDTH]
    hn, ha = face_node_areas(grid, heel, "z-")
    # medial and lateral columns: inferior faces of navicular / cuboid (ground through the forefoot)
    nav_b = np.nonzero((lab == NAV) & ~dn)[0]
    cub_b = np.nonzero((lab == CUB) & ~dn)[0]
    nn, na = face_node_areas(grid, nav_b, "z-")
    cn, ca = face_node_areas(grid, cub_b, "z-")
    # weak anterior restraint (distal joints): anterior faces of navicular and cuboid
    fant = np.nonzero(((lab == NAV) | (lab == CUB)) & ~fw & (cxyz[:, 0] > 88.0))[0]
    an, aa = face_node_areas(grid, fant, "x+")
    springs = [(hn, ha, np.array(K_HEEL)), (nn, na, np.array(K_NAV)), (cn, ca, np.array(K_CUB)),
               (an, aa, np.array(K_ANT))]
    fa = np.r_[na, ca]
    # load: superior faces of the trochlea within the (canonical) load window
    top = np.nonzero((lab == TALUS) & ~up & (cxyz[:, 0] > LOAD_X[0]) & (cxyz[:, 0] < LOAD_X[1]) &
                     (cxyz[:, 1] > LOAD_Y[0]) & (cxyz[:, 1] < LOAD_Y[1]) & (cxyz[:, 2] > 45))[0]
    tn, ta = face_node_areas(grid, top, "z+")
    F = BODY_LOAD * act.load
    p = F / ta.sum()
    loads = [(tn, ta, np.array([0.0, 0.0, -p]))]
    # ankle mortise: penalty on the mean fore-aft and medial-lateral displacement of the loaded surface,
    # leaving sagittal rotation (the ankle hinge) and vertical motion free.
    if K_MORT > 0:
        springs.append((tn, ta, np.array([0.0, K_MORT, 0.0])))
    penalties = [(tn, ta, 0, K_MORT_X), (tn, ta, 1, K_MORT_X)]
    return springs, loads, dict(heel_area=ha.sum(), fore_area=fa.sum(), load_area=ta.sum(), heel_cells=heel,
                                penalties=penalties)


def rigid_fit(X, U):
    """Least-squares small rigid motion u = t + w x (X - c). Returns t, w, c."""
    c = X.mean(0)
    r = X - c
    # u_i = t + w x r_i = t - [r_i]x w
    A = np.zeros((3 * len(X), 6))
    A[0::3, 0] = 1; A[1::3, 1] = 1; A[2::3, 2] = 1
    A[0::3, 4] = r[:, 2]; A[0::3, 5] = -r[:, 1]
    A[1::3, 3] = -r[:, 2]; A[1::3, 5] = r[:, 0]
    A[2::3, 3] = r[:, 1]; A[2::3, 4] = -r[:, 0]
    sol, *_ = np.linalg.lstsq(A, U.ravel(), rcond=None)
    return sol[:3], sol[3:], c


def facet_forces(grid, frac, sig, labels=None):
    """Vertical force (N) transmitted through each facet: for every (x,y) column, average -szz over
    the cartilage thickness (weights = cartilage fraction * dz) times the column area."""
    labels = labels or [("ant", C_ANT), ("mid", C_MID), ("post", C_POST)]
    nx, ny, nz = grid.shape
    A = (grid.dx[:, None] * grid.dy[None, :])
    out = {}
    for k, L in labels:
        w = (frac[:, L].reshape(grid.shape) * grid.dz[None, None, :])
        tw = w.sum(2)
        s = (-sig[:, 2].reshape(grid.shape) * w).sum(2)
        col = tw > 1e-6
        out[k] = float((s[col] / tw[col] * A[col] * np.minimum(tw[col] / 2.0, 1.0)).sum())
    return out


EFFECT_NAMES = ["ant_p99", "ant_mean", "mid_p99", "mid_mean", "post_p99", "post_mean", "tn_p99", "cc_p99",
                "share_ant", "share_mid", "share_post", "talus_dz", "talus_dy", "talus_evert", "nav_drop",
                "heel_share", "tc_p99"]


def effects(grid, lab, frac, vm, sig, U, cells_vm_idx, info_bc):
    """Clinically interpretable effect descriptors from one solution."""
    d = {}
    vol = grid.cell_vol
    def stats(L):
        m = (lab == L)
        if m.sum() == 0:
            return 0.0, 0.0
        v = vm[m]; w = vol[m] * frac[m, L]
        return float(np.percentile(v, 99)), float((v * w).sum() / w.sum())
    for k, L in [("ant", C_ANT), ("mid", C_MID), ("post", C_POST)]:
        d[k + "_p99"], d[k + "_mean"] = stats(L)
    d["tn_p99"], _ = stats(C_TN)
    d["cc_p99"], _ = stats(C_CC)
    # vertical load carried through each facet: thickness-averaged -szz over every column of the facet
    sh = facet_forces(grid, frac, sig)
    tot = sum(max(v, 0) for v in sh.values()) + 1e-9
    for k in sh:
        d["share_" + k] = float(max(sh[k], 0) / tot)
    # kinematics in calcaneus-fixed frame
    nodes_c = np.unique(grid.cell_nodes[lab == CALC])
    t, w, c = rigid_fit(grid.node_xyz[nodes_c], U[nodes_c])
    Urel = U - (t[None, :] + np.cross(w[None, :], grid.node_xyz - c[None, :]))
    nodes_t = np.unique(grid.cell_nodes[lab == TALUS])
    tt, wt, ct = rigid_fit(grid.node_xyz[nodes_t], Urel[nodes_t])
    d["talus_dz"] = float(tt[2]); d["talus_dy"] = float(tt[1])
    d["talus_evert"] = float(np.rad2deg(wt[0]))       # rotation about the long (x) axis
    nodes_n = np.unique(grid.cell_nodes[lab == NAV])
    d["nav_drop"] = float(-np.nanmean(Urel[nodes_n, 2]))
    d["heel_share"] = float(info_bc.get("heel_share", np.nan))
    # talocalcaneal complex: everything occupied in the subtalar band (bone, cartilage, bridge)
    xyz = grid.cell_xyz
    band = (lab != VOID) & (xyz[:, 2] > 30) & (xyz[:, 2] < 42) & (xyz[:, 0] > 18) & (xyz[:, 0] < 70)
    d["tc_p99"] = float(np.percentile(vm[band], 99))
    cm = lab == COAL
    d["coal_p99"] = float(np.percentile(vm[cm], 99)) if cm.sum() > 3 else 0.0
    d["coal_max"] = float(vm[cm].max()) if cm.sum() > 0 else 0.0
    return d, Urel


def simulate(foot: Foot, act: Action, fine=1.0, mat_override=None, return_fields=True, method="gmg", tol=1e-7,
             frac_hook=None, U0=None):
    grid = get_grid(fine)
    t0 = time.time()
    g = FootGeom(foot, act)
    frac = voxelise(g, grid)
    if frac_hook is not None:
        frac = frac_hook(frac)
    t1 = time.time()
    E, nu, occ = materials_from_fractions(frac, mat_override)
    lab = region_labels(frac)
    active = active_cells(frac)
    lam, mu = lame(E, nu)
    # lubricated subtalar articulations: cartilage of the three facets carries little through-thickness shear.
    # The shear factor is applied in proportion to the facet-cartilage fraction of each cell.
    fcart = frac[:, C_ANT] + frac[:, C_MID] + frac[:, C_POST]
    coeffs = ortho_coeffs(lam, mu)
    if JOINT_SHEAR < 1.0:
        # mix: cartilage part with reduced shear, remainder isotropic
        Ec, nuc = MAT["cart"]; lc, mc = lame(Ec, nuc)
        coeffs[:, 5] = coeffs[:, 5] - fcart * mc * (1 - JOINT_SHEAR)
    springs, loads, bcinfo = boundary(grid, lab, frac, act, foot)
    groups = [np.unique(grid.cell_nodes[lab == L]) for L in (TALUS, CALC, NAV, CUB)] if RIGID_DEFLATION else None
    U, info = solve(grid, lam, mu, active, springs, loads, method=method, tol=tol, penalties=bcinfo["penalties"],
                    coeffs=coeffs, node_groups=groups, U0=U0)
    t2 = time.time()
    cells = np.arange(grid.ncell)
    sig = np.zeros((grid.ncell, 6)); vm = np.zeros(grid.ncell)
    ac = np.nonzero(active)[0]
    Uz = np.nan_to_num(U)
    s, v, _ = cell_stress(grid, lam, mu, Uz, ac, coeffs)
    sig[ac] = s; vm[ac] = v
    # heel reaction share
    hn, ha, kh = springs[0]
    Rheel = (kh[2] * ha * Uz[hn, 2]).sum()
    bcinfo["heel_share"] = -Rheel / (BODY_LOAD * act.load)
    eff, Urel = effects(grid, lab, frac, vm, sig, Uz, None, bcinfo)
    t3 = time.time()
    out = dict(eff=eff, timing=(t1 - t0, t2 - t1, t3 - t2), iters=info.get("iters"), ndof=info["ndof"])
    if return_fields:
        out.update(frac=frac, lab=lab, vm=vm, sig=sig, U=Uz, Urel=np.nan_to_num(Urel), active=active)
    return out
