"""Continuous queries of the CoMech neural-field decoder at arbitrary points (not only FE cell centres).

The decoder Phi(x | s') reads the latent token grid by trilinear interpolation, Fourier features of the normalised
position and the pre-operative material channels at the query point.  At cell centres this reproduces the
predictions used throughout the paper; between centres the token features and material channels are interpolated
trilinearly, so the predicted field is continuous and can be sampled at any resolution for rendering."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import GRID, PATCH, TOK, NTOK, NCH, GSHAPE, fourier
from ml import ad
from ml.ad import T, concat, spmm
import scipy.sparse as sp
from scipy import ndimage

import fe.geometry as _FG
# position normalisation of the decoder's Fourier features: the box of ml/data.cell_norm_coords (fe.geometry.ML_BOX);
# identical to the former constants [0, 96] x [-18, 30] x [0, 63] for the parametric engine (Claude, 2026-11-01)
_BOX = getattr(_FG, "ML_BOX", ((0, 96), (-18, 30), (0, 63)))
LO = np.array([b[0] for b in _BOX], float); HI = np.array([b[1] for b in _BOX], float)


def cont_index(P):
    """Fractional cell-centre index of points on the (rectilinear) training grid."""
    return np.stack([np.interp(P[:, 0], GRID.xc, np.arange(GRID.nx)), np.interp(P[:, 1], GRID.yc, np.arange(GRID.ny)),
                     np.interp(P[:, 2], GRID.zc, np.arange(GRID.nz))], 1)


def token_weights(P):
    idx = cont_index(P) + 0.5
    t = idx / np.array(PATCH) - 0.5
    t0 = np.floor(t).astype(int); w = t - t0
    tids, wts = [], []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                ti = np.clip(t0[:, 0] + a, 0, TOK[0] - 1); tj = np.clip(t0[:, 1] + b, 0, TOK[1] - 1); tk = np.clip(t0[:, 2] + c, 0, TOK[2] - 1)
                tids.append((ti * TOK[1] + tj) * TOK[2] + tk)
                wts.append((w[:, 0] if a else 1 - w[:, 0]) * (w[:, 1] if b else 1 - w[:, 1]) * (w[:, 2] if c else 1 - w[:, 2]))
    return np.stack(tids, 1), np.stack(wts, 1).astype(np.float32)


def material_at(pre, P):
    """pre: (NCH, nx, ny, nz) channels on the training grid; trilinear interpolation at points."""
    ci = cont_index(P).T
    return np.stack([ndimage.map_coordinates(pre[c], ci, order=1, mode="nearest") for c in range(NCH)], 1).astype(np.float32)


def query(pred, pre, act, P, chunk=20000):
    """Normalised field outputs (Nq, n_out) of a CoMech predictor at points P (mm) for one action."""
    m = pred.m
    s1 = pred.latent(pre, act[None])                       # (1, NTOK, D)
    pre4 = pre.reshape(NCH, *GSHAPE)
    out = []
    for c0 in range(0, len(P), chunk):
        Q = P[c0:c0 + chunk]; n = len(Q)
        tid, tw = token_weights(Q)
        W = sp.csr_matrix((tw.ravel(), (np.repeat(np.arange(n), 8), tid.ravel())), shape=(n, NTOK))
        f = spmm(W, s1.reshape(NTOK, m.D)).reshape(1, n, m.D)
        uvw = ((Q - LO) / (HI - LO)).astype(np.float32)
        x = concat([f, T(fourier(uvw)[None]), T(material_at(pre4, Q)[None])], -1)
        out.append(m.MLP("dec", x, m.ndec).data[0])
    return np.concatenate(out, 0)
