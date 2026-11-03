"""Shared helpers for the CoMech experiments."""
import os, sys, json, glob, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import *
from ml.evaluate import load_model, Predictor
from fe.geometry import *
from fe.simulate import EFFECT_NAMES

EI = {n: i for i, n in enumerate(EFFECT_NAMES)}
TAU = 0.20          # tolerance: residual-facet mean stress may rise by < 20% (reproduces both atlas tipping points)
MODELS = os.environ.get("COMECH_MODELS", "results/models")


def model_path(name):
    return f"{MODELS}/{name}.pkl"


def get_pred(name):
    pt = f"{MODELS}/{name}.pt"
    if os.path.exists(pt):                      # PyTorch model of the GPU scaling experiment (gpu/)
        from gpu.tpredict import TorchPredictor
        return TorchPredictor(pt)
    return Predictor(load_model(model_path(name)))


def needs_post(name):
    """Does the model receive the explicit post-operative geometry (geometry-to-field models)?"""
    if name in ("geo", "fno"):
        return True
    js = f"{MODELS}/{name}.json"
    return os.path.exists(js) and json.load(open(js))["kind"] in ("gino", "geofno")


def geo_of(foot, act=None):
    fr = voxelise(FootGeom(foot, act or Action()), GRID)
    return channels(np.round(fr * 255).astype(np.uint8)), fr


def defect_fraction(fr, ref):
    v = GRID.cell_vol
    return {n: float(max(0.0, 1 - (fr[:, L] * v).sum() / (ref[:, L] * v).sum())) for n, L in [("mid", C_MID), ("post", C_POST)]}


def tolerance(d, y, tau=TAU, ref=None):
    """Largest defect at which the relative rise of y stays below tau (linear interpolation); censored at max(d)
    if never exceeded.  d ascending, y raw values.  The rise is measured against y[0] (the defect-free value of the
    same curve) or, if given, against ref (e.g. the defect-free joint without correction)."""
    d = np.asarray(d, float); y = np.asarray(y, float)
    r = y / (y[0] if ref is None else ref) - 1.0
    if r[0] >= tau:
        return float(d[0])
    for i in range(1, len(d)):
        if r[i] >= tau:
            if r[i - 1] >= tau:
                return float(d[i - 1])
            return float(d[i - 1] + (tau - r[i - 1]) / (r[i] - r[i - 1]) * (d[i] - d[i - 1]))
    return float(d[-1])


def boot_ci(x, n=2000, seed=0, stat=np.mean):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(len(x), size=len(x))]) for _ in range(n)]
    return float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def act_vec(resect=0, d_mid=0.0, d_post=0.0, osteo=0.0, interp=0, fusion=0, load=1.0):
    return np.array([resect, d_mid, d_post, osteo / 10.0, interp, fusion, load - 1.0], np.float32)


def _deep_update(old, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(old.get(k), dict):
            _deep_update(old[k], v)
        else:
            old[k] = v
    return old


def save_merged(path, new, **kw):
    """Write results, merging with an existing file so that models can be evaluated in separate runs."""
    if os.path.exists(path):
        try:
            new = _deep_update(json.load(open(path)), json.loads(json.dumps(new)))
        except (OSError, ValueError):
            pass
    json.dump(new, open(path, "w"), **kw)
