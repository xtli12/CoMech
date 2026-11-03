"""High-resolution label volumes of the real-bone engine for rendering (Claude, 2026-10-29): body labels of the atlas
foot (rb/rbgeom.label_points, 0.5-mm template raster with the foot's warps and edits) mapped to the analysis labels of
fe/geometry.  Same interface as the parametric version (hiresgeo_parametric.py)."""
import os, sys, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fe.geometry import *
import rbgeom as _G, rbsim as _RS

LO = np.array([b[0] for b in ML_BOX], float); HI = np.array([b[1] for b in ML_BOX], float)
_MAP = None


def _labels(foot, act, X, Y, Z):
    global _MAP
    if _MAP is None:
        _MAP = _RS.body_to_ml()
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return _MAP[_G.label_points(P, foot, act)].reshape(X.shape)


def _lattice(foot, act, xs, ys, zs):
    global _MAP
    if _MAP is None:
        _MAP = _RS.body_to_ml()
    return _MAP[_G.label_lattice(xs, ys, zs, foot, act)]


def label_volume(foot, act, h=0.3, lo=LO, hi=HI, chunk_x=24):
    xs = np.arange(lo[0] + h / 2, hi[0], h); ys = np.arange(lo[1] + h / 2, hi[1], h); zs = np.arange(lo[2] + h / 2, hi[2], h)
    return _lattice(foot, act, xs, ys, zs).astype(np.int8), lo, h


def occupancy_volumes(foot, act, groups, h=0.3, ssz=2, lo=LO, hi=HI, chunk_x=24):
    """Partial-volume occupancy of label groups (supersampled ssz times along z)."""
    xs = np.arange(lo[0] + h / 2, hi[0], h); ys = np.arange(lo[1] + h / 2, hi[1], h); zs = np.arange(lo[2] + h / 2, hi[2], h)
    occ = {k: np.zeros((len(xs), len(ys), len(zs)), np.float32) for k in groups}
    offs = (np.arange(ssz) + 0.5) / ssz - 0.5
    for dz in offs:
        L = _lattice(foot, act, xs, ys, zs + dz * h)
        for k, labs in groups.items():
            occ[k] += np.isin(L, labs).astype(np.float32) / ssz
    return occ, lo, h


if __name__ == "__main__":
    t = time.time(); L, lo, h = label_volume(Foot(), Action(), h=0.5); print(L.shape, time.time() - t, np.bincount(L.ravel() + 0))
