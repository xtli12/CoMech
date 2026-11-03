"""Scenes of engine feet for the rendered figures: SDF objects from the analytic geometry at 0.3 mm, coloured by
FE (high-resolution solutions) or CoMech (continuous neural-field queries) von Mises stress."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from render3d import *
from hiresgeo import label_volume
from fe.geometry import *
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

BONE = (0.89, 0.86, 0.80); BONE2 = (0.83, 0.81, 0.77)
COAL_COL = (0.714, 0.263, 0.259)   # figures4papers red (#B64342), the coalition colour of every figure
from figstyle import STRESS, DISP, ERR   # sequential colour maps of the house style (figstyle.py)
from rbview import SIGMA, ISO, NSIGMA, FACET_C, FACET_SCALE   # engine-dependent view constants (Claude, 2026-11-01)


def sdf_vol(mask, lo, h, smooth=1.6):
    return Volume(sdf_from_mask(mask, h, smooth=smooth), lo, h)


def crop_box(mask, lo, h, pad=3.0):
    idx = np.argwhere(mask)
    a = np.maximum(idx.min(0) - int(pad / h), 0); b = np.minimum(idx.max(0) + int(pad / h) + 1, np.array(mask.shape))
    return tuple(slice(i, j) for i, j in zip(a, b)), lo + a * h


def obj_from(L, labels, lo, h, sigma=1.4, **kw):
    m = np.isin(L, labels)
    sl, o = crop_box(m, lo, h)
    return Obj(kw.pop("name", str(labels)), SmoothSurface(m[sl], o, h, sigma=sigma), **kw)


GROUPS = {"calc": [CALC, FUSE], "nav": [NAV], "cub": [CUB], "cart": [C_ANT, C_MID, C_POST], "coal": [COAL, FIB], "talus": [TALUS]}


def engine_scene(foot, act, field=None, vmax=1.6, talus_lift=None, talus_alpha=0.18, show=("calc", "cart", "coal", "nav", "cub", "talus"),
                 h=0.3, defect_mask_from=None):
    from hiresgeo import occupancy_volumes
    occ, lo, h = occupancy_volumes(foot, act, {k: GROUPS[k] for k in show}, h=h)
    def mk(k, sigma, **kw):
        m = occ[k]
        sl, o = crop_box(m > 0.02, lo, h)
        return Obj(k, SmoothSurface(m[sl], o, h, sigma=sigma, iso=ISO.get(k, 0.5), nsigma=NSIGMA.get(k)), **kw)
    objs = []
    if "calc" in show:
        objs.append(mk("calc", SIGMA["bone"], color=BONE))
    if "nav" in show:
        objs.append(mk("nav", SIGMA["bone"], color=BONE2))
    if "cub" in show:
        objs.append(mk("cub", SIGMA["bone"], color=BONE2))
    if "cart" in show:
        o = mk("cart", SIGMA["cart"], color=(0.55, 0.70, 0.85))
        if field is not None:
            o.scalar, o.cmap, o.vmin, o.vmax = field, STRESS, 0.0, vmax
        o.spec = 0.35; objs.append(o)
    if "coal" in show and (occ["coal"] > 0.5).any():
        objs.append(mk("coal", SIGMA["coal"], color=COAL_COL))
    if "talus" in show:
        o = mk("talus", SIGMA["bone"], color=(0.80, 0.84, 0.90), alpha=talus_alpha)
        if talus_lift:
            o.sdf.o = o.sdf.o + np.array(talus_lift)
        objs.append(o)
    return objs, occ, lo, h


def fe_field(npz, labels=(C_ANT, C_MID, C_POST)):
    z = np.load(npz, allow_pickle=True)
    lab = z["lab"]; valid = np.isin(lab, labels)
    return field_sampler(z["vm"], z["xe"], z["ye"], z["ze"], valid=valid), json.loads(str(z["eff"]))


def comech_field(pred, pre, act, norm):
    from nfquery import query
    def f(P):
        y = query(pred, pre, act, P)
        vm, _ = norm.fields_inv(y)
        return vm.astype(np.float32)
    return f


def to_rgba(img, alpha, bg=(1, 1, 1)):
    a = alpha[..., None]
    return np.concatenate([img, alpha[..., None]], -1)
