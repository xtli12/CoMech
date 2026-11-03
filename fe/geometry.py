"""Real-bone CoMech engine: geometry interface (Claude, 2026-10-29).

Drop-in replacement of the parametric fe/geometry.py (kept as fe/geometry_parametric.py): the same Foot and Action
parameters, implemented on the bones of the FE atlas (rb/rbgeom.py), and the same label constants for the analysis
grid, extended with the tissues of the full foot (tibia, fibula, ankle cartilage, other bones and cartilage, soft
tissue, fascia).  Experiments and learning code import this module unchanged."""
import os, sys
import numpy as np
_RB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rb")
if _RB not in sys.path:
    sys.path.insert(0, _RB)
import rbgeom as _G
import rbsim as _RS
from rbgeom import Foot, Action
from rbsim import (VOID, TALUS, CALC, NAV, CUB, C_ANT, C_MID, C_POST, C_TN, C_CC, COAL, FIB, INTERP, FUSE, LIG,
                   TIB, FIBU, C_ANKLE, OBONE, C_OTHER, SOFT, FASC, LABEL_NAMES)

COAL_TYPES = ["none", "I", "II", "III", "IV"]
ROZ_TYPES = ["none", "R1", "R2", "R3", "R4", "R5", "unclass"]
ENGINE = "real-bone"
# analysis grid and input channels of the learning models
ML_PATCH = _RS.ML_PATCH
ML_BOX = _RS.ML_BOX
ML_CH = [[TALUS, CALC, NAV, CUB, FUSE, TIB, FIBU, OBONE], [C_ANT, C_MID, C_POST, C_TN, C_CC, C_ANKLE, C_OTHER],
         [COAL, FIB], [SOFT, FASC], [INTERP]]


class FootGeom:
    """Foot + action (geometry is evaluated by rbgeom.label_points)."""

    def __init__(self, foot, act=None):
        self.f = foot
        self.a = act or Action()

    def classify_bodies(self, X, Y, Z):
        P = np.column_stack([np.ravel(X), np.ravel(Y), np.ravel(Z)])
        return _G.label_points(P, self.f, self.a).reshape(np.shape(X))

    def top_view(self, xs, ys, zlo=30.0, zhi=90.0, dz=0.5):
        """Top-view footprints (world x, y) of the subtalar facets, the coalition and the facet defect: a column is
        marked when it contains the tissue anywhere between zlo and zhi."""
        zs = np.arange(zlo, zhi, dz)
        L = _G.label_lattice(np.asarray(xs, float), np.asarray(ys, float), zs, self.f, self.a)
        out = {"ant": (L == _G.CART["ant"]).any(2), "mid": (L == _G.CART["mid"]).any(2), "post": (L == _G.CART["post"]).any(2),
               "coal": ((L == _G.COAL) | (L == _G.FIB)).any(2)}
        return out


def make_grid_edges(fine=1.0):
    return _RS.ml_edges(fine)


def voxelise(geom, grid, ss=(3, 3, 3)):
    """Label fractions (ncell, 22) of the analysis grid cells."""
    return _RS.ml_fractions(geom.f, geom.a, grid, ss=ss[0])


def region_labels(frac):
    return _RS.ml_labels(frac)


def coalition_ratio(foot, h=0.5):
    """Coalition-to-posterior-facet area ratio (Wilde) and facet involvement from top-view footprints (world frame),
    as exp/cohort.coalition_ratio of the parametric engine."""
    from dataclasses import replace
    T = _G.Template.get()
    fp, fm = T.facets["post"], T.facets["mid"]
    lo = np.minimum(fp["xyz_lo"], fm["xyz_lo"]) - 15; hi = np.maximum(fp["xyz_hi"], fm["xyz_hi"]) + 15
    s = foot.scale
    xs = np.arange(T.c0[0] + s * (lo[0] - T.c0[0]), T.c0[0] + s * (hi[0] - T.c0[0]), h)
    ys = np.arange(T.c0[1] + s * (lo[1] - T.c0[1]) - 12, T.c0[1] + s * (hi[1] - T.c0[1]) + 12, h)
    z0, z1 = s * (lo[2] - 5), s * (hi[2] + 5)
    pre = FootGeom(foot, Action()).top_view(xs, ys, z0, z1)
    intact = FootGeom(replace(foot, ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0), Action()).top_view(xs, ys, z0, z1)
    br = pre["coal"]; mid, post = intact["mid"], intact["post"]
    a = h * h
    return dict(ratio=float(br.sum() / max(post.sum(), 1)), cov_mid_true=float((br & mid).sum() / max(mid.sum(), 1)),
                cov_post_true=float((br & post).sum() / max(post.sum(), 1)), area_coal=float(br.sum() * a),
                area_post=float(post.sum() * a))


def subtalar_band(xyz, foot=None, act=None, dmax=6.0):
    """Cells of the talocalcaneal complex region: within dmax mm of both the talus and the calcaneus (template frame)."""
    T = _G.Template.get()
    ps, _, _ = _G.inverse(T, foot or Foot(), act or Action(), np.asarray(xyz, float))
    return (T.sub_lookup(T.dt, ps) <= dmax) & (T.sub_lookup(T.dc, ps) <= dmax)


def map_axes(foot, h=0.25):
    """x, y axes (world frame, mm) of the top-view maps of the subtalar joint."""
    T = _G.Template.get()
    lo = np.minimum.reduce([T.facets[k]["xyz_lo"] for k in T.facets]) - 8
    hi = np.maximum.reduce([T.facets[k]["xyz_hi"] for k in T.facets]) + 8
    s = foot.scale
    xs = np.arange(T.c0[0] + s * (lo[0] - T.c0[0]), T.c0[0] + s * (hi[0] - T.c0[0]), h)
    ys = np.arange(T.c0[1] + s * (lo[1] - T.c0[1]), T.c0[1] + s * (hi[1] - T.c0[1]), h)
    return xs, ys


def z_ref_align():
    """Height (mm, template) about which hindfoot alignment shears the heel: 1 degree moves the ground contact by
    z_ref tan(1 deg)."""
    return float(_G.Template.get().z_ref)
