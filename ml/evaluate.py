"""Prediction wrappers and field/effect metrics for trained models."""
import os, sys, time, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import ad
from ml.ad import T
from ml.data import *
from ml.models import CoMech, Geo2Field, DeepONet, FNO
from ml.train import make_model, load_only, pooled_geo, patchify, ROLL_OST_ONLY, joint_mask
from fe.geometry import *
from fe.simulate import EFFECT_NAMES

ART = [C_ANT, C_MID, C_POST, C_TN, C_CC]


def load_model(path):
    d = pickle.load(open(path, "rb"))
    m = make_model(d["kind"], d["cfg"]["seed"])
    m.load(d["state"])
    m.kind = d["kind"]; m.cfg = d["cfg"]
    for v in m.p.values():
        v.req = False
    return m


class Predictor:
    """Uniform interface: given pre-op geometry (C,X,Y,Z), optional post-op geometries (n,C,X,Y,Z) and
    actions (n,7), predict normalised fields at cells (n,Nq) and normalised effects (n,17)."""

    def __init__(self, model):
        self.m = model
        self.kind = model.kind

    def latent(self, pre, acts):
        """CoMech only: post-action latent states (n, NTOK, D) as a tensor."""
        m = self.m
        s0 = m.encode(patchify(pre[None]))
        s = ad.take(s0, np.zeros(len(acts), int), 0)
        return m.step(s, m.act_enc(acts))

    def effects(self, pre, acts, post=None):
        m = self.m
        if self.kind == "comech":
            return m.effect(self.latent(pre, acts)).data
        cells = np.zeros((len(acts), 1), int)
        return self.fields_and_effects(pre, acts, cells, post)[1]

    def fields_and_effects(self, pre, acts, cells, post=None, chunk=4096):
        m, n = self.m, len(acts)
        cells = np.asarray(cells)
        if cells.ndim == 1:
            cells = np.repeat(cells[None], n, 0)
        mat_pre = pre.reshape(NCH, -1)[:, cells].transpose(1, 2, 0)
        outs, eff = [], None
        if self.kind == "comech":
            s1 = self.latent(pre, acts)
            eff = m.effect(s1).data
            for c0 in range(0, cells.shape[1], chunk):
                outs.append(m.decode(s1, cells[:, c0:c0 + chunk], mat_pre[:, c0:c0 + chunk]).data)
        elif self.kind == "actcat":
            s = m.encode(patchify(np.repeat(pre[None], n, 0)), acts)
            eff = m.effect(s).data
            for c0 in range(0, cells.shape[1], chunk):
                outs.append(m.decode(s, cells[:, c0:c0 + chunk], mat_pre[:, c0:c0 + chunk]).data)
        elif self.kind == "geo":
            s = m.encode(patchify(post), load_only(acts))
            eff = m.effect(s).data
            mat_post = np.stack([post[i].reshape(NCH, -1)[:, cells[i]].T for i in range(n)])
            for c0 in range(0, cells.shape[1], chunk):
                outs.append(m.decode(s, cells[:, c0:c0 + chunk], mat_post[:, c0:c0 + chunk]).data)
        elif self.kind == "deeponet":
            g = np.repeat(pooled_geo(pre[None]), n, 0)
            for c0 in range(0, cells.shape[1], chunk):
                y, e = m.forward(g, acts, cells[:, c0:c0 + chunk], mat_pre[:, c0:c0 + chunk])
                outs.append(y.data); eff = e.data
        elif self.kind == "fno":
            y, e = m.forward(post, load_only(acts), cells)
            outs.append(y.data); eff = e.data
        return np.concatenate(outs, 1), eff

    def rollout2(self, pre, a1, a2_ost, cells):
        """CoMech two-step roll-out: resection, then osteotomy-only action."""
        m = self.m
        s1 = self.latent(pre, a1)
        s2 = m.step(s1, m.act_enc(a2_ost))
        mat_pre = pre.reshape(NCH, -1)[:, cells].transpose(1, 2, 0)
        return m.decode(s2, cells, mat_pre).data, m.effect(s2).data


# ---------------- metrics ----------------
def field_metrics(vm_p, u_p, vm_t, u_t, lab):
    """All arrays over the occupied cells of one solution (post-operative FE labels)."""
    out = {}
    out["stress_l2"] = float(np.linalg.norm(vm_p - vm_t) / np.linalg.norm(vm_t))
    art = np.isin(lab, ART)
    out["stress_l2_art"] = float(np.linalg.norm(vm_p[art] - vm_t[art]) / np.linalg.norm(vm_t[art]))
    out["disp_l2"] = float(np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t))
    # hotspot: top-5% von Mises among articular cartilage cells
    va, vp = vm_t[art], vm_p[art]
    k = max(1, int(round(0.05 * len(va))))
    ta = set(np.argsort(-va)[:k]); tp = set(np.argsort(-vp)[:k])
    out["hot_iou"] = len(ta & tp) / len(ta | tp)
    for name, L in [("mid", C_MID), ("post", C_POST)]:
        m = lab == L
        if m.sum() >= 3:
            pt, pp = np.percentile(vm_t[m], 99), np.percentile(vm_p[m], 99)
            mt, mp = vm_t[m].mean(), vm_p[m].mean()
            out[f"{name}_peak_err"] = float(abs(pp - pt) / pt)
            out[f"{name}_mean_err"] = float(abs(mp - mt) / mt)
    return out


def evaluate_split(pred, S, norm, max_feet=None, chunk=4096):
    """Field and effect metrics for every (foot, action) of a split."""
    rows = []
    nf = S.nf if max_feet is None else min(S.nf, max_feet)
    for f in range(nf):
        pre = S.geo_pre[f].astype(np.float32)
        post = S.geo_post[f].astype(np.float32) if (S.geo_post is not None and pred.kind in ("geo", "fno")) else None
        acts = S.act[f]
        for k in range(S.K):
            lab = S.lab[f, k]
            cells = np.nonzero(lab != VOID)[0]
            t0 = time.time()
            y, e = pred.fields_and_effects(pre, acts[k:k + 1], cells, post[k:k + 1] if post is not None else None)
            dt = time.time() - t0
            vm_p, u_p = norm.fields_inv(y[0])
            vm_t = S.vm[f, k, cells].astype(np.float32); u_t = S.uc[f, k, cells].astype(np.float32)
            r = field_metrics(vm_p, u_p, vm_t, u_t, lab[cells])
            et = norm.eff(S.eff_raw[f, k][None])[0]
            r["eff_rmse"] = float(np.sqrt(((e[0] - et) ** 2).mean()))
            r.update(f=f, k=k, meta=S.metas[f][k], ctype=S.ctype[f], t=dt)
            rows.append(r)
    return rows
