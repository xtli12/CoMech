"""High-resolution label volumes of the parametric hindfoot for rendering (analytic geometry, uniform grid)."""
import os, sys, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fe.geometry import *

LO = np.array([0.0, -18.0, 0.0]); HI = np.array([96.0, 30.0, 63.0])


def label_volume(foot, act, h=0.3, lo=LO, hi=HI, chunk_x=24):
    xs = np.arange(lo[0] + h / 2, hi[0], h); ys = np.arange(lo[1] + h / 2, hi[1], h); zs = np.arange(lo[2] + h / 2, hi[2], h)
    g = FootGeom(foot, act)
    out = np.zeros((len(xs), len(ys), len(zs)), np.int8)
    for i0 in range(0, len(xs), chunk_x):
        X, Y, Z = np.meshgrid(xs[i0:i0 + chunk_x], ys, zs, indexing="ij")
        out[i0:i0 + chunk_x] = g.classify(X, Y, Z)
    return out, lo, h


if __name__ == "__main__":
    t = time.time(); L, lo, h = label_volume(Foot(), Action()); print(L.shape, time.time() - t, np.bincount(L.ravel() + 0))


def occupancy_volumes(foot, act, groups, h=0.3, ssz=4, lo=LO, hi=HI, chunk_x=24):
    """Partial-volume occupancy of label groups, supersampled ssz times along z (the direction in which the
    articular surfaces are shallow), so that the 0.5-level of the occupancy has no terraces."""
    xs = np.arange(lo[0] + h / 2, hi[0], h); ys = np.arange(lo[1] + h / 2, hi[1], h); zs = np.arange(lo[2] + h / 2, hi[2], h)
    g = FootGeom(foot, act)
    occ = {k: np.zeros((len(xs), len(ys), len(zs)), np.float32) for k in groups}
    offs = (np.arange(ssz) + 0.5) / ssz - 0.5
    for i0 in range(0, len(xs), chunk_x):
        for dz in offs:
            X, Y, Z = np.meshgrid(xs[i0:i0 + chunk_x], ys, zs + dz * h, indexing="ij")
            L = g.classify(X, Y, Z)
            for k, labs in groups.items():
                occ[k][i0:i0 + chunk_x] += np.isin(L, labs).astype(np.float32) / ssz
    return occ, lo, h
