"""Predictor for the PyTorch models with the same interface as ml.evaluate.Predictor, so that every experiment
script in exp/ evaluates the GPU models unchanged (exp.common.get_pred loads <name>.pt when it exists)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import NCH
from gpu.tmodels import build, patchify

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TL:
    """Minimal tensor wrapper exposing .data (numpy) and .shape, like ml.ad.T."""

    def __init__(self, t):
        self.t = t

    @property
    def data(self):
        return self.t.detach().float().cpu().numpy()

    @property
    def shape(self):
        return tuple(self.t.shape)


def _t(x, dt=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dt, device=DEV)


class _Shim:
    """The model methods used directly by exp/e_taxonomy.py (encode, act_enc, step, effect, rep)."""

    def __init__(self, m):
        self.m = m

    @torch.no_grad()
    def encode(self, patches, a=None):
        return TL(self.m.encode(_t(patches), None if a is None else _t(a)))

    @torch.no_grad()
    def act_enc(self, a):
        return TL(self.m.act_enc(_t(a)))

    @torch.no_grad()
    def step(self, s, u):
        return TL(self.m.step(s.t, u.t))

    @torch.no_grad()
    def effect(self, s):
        return TL(self.m.effect(s.t))

    @torch.no_grad()
    def rep(self, s):
        return TL(self.m.rep(s.t))


class TorchPredictor:
    def __init__(self, path):
        d = torch.load(path, map_location=DEV, weights_only=False)
        self.cfg = d["cfg"]
        k = self.cfg["kind"]
        self.net = build(k, self.cfg["model"]).to(DEV).eval()
        self.net.load_state_dict(d["state"])
        self.kind = {"comech": "comech", "actcat": "actcat"}.get(k, "geo")     # "geo": needs post-op geometry
        self.arch = k
        self.m = _Shim(self.net)

    # ---- helpers
    def _latent_t(self, pre, acts):
        s0 = self.net.encode(patchify(_t(pre)[None]))
        s = s0.expand(len(acts), -1, -1)
        return self.net.step(s, self.net.act_enc(_t(acts)))

    @torch.no_grad()
    def latent(self, pre, acts):
        return TL(self._latent_t(pre, acts))

    @torch.no_grad()
    def effects(self, pre, acts, post=None, chunk=None):
        chunk = chunk or (64 if self.kind in ("comech", "actcat") else 4)
        out = []
        for c0 in range(0, len(acts), chunk):
            a = acts[c0:c0 + chunk]
            if self.kind == "comech":
                out.append(self.net.effect(self._latent_t(pre, a)).float().cpu().numpy())
            elif self.kind == "actcat":
                s = self.net.encode(patchify(_t(pre)[None].expand(len(a), -1, -1, -1, -1)), _t(a))
                out.append(self.net.effect(s).float().cpu().numpy())
            else:
                cells = np.zeros((len(a), 1), np.int64)
                pp = post[c0:c0 + chunk]
                mat = np.stack([pp[i].reshape(NCH, -1)[:, cells[i]].T for i in range(len(a))])
                _, e = self.net(_t(pp), _t(a)[:, 6], _t(cells, torch.long), _t(mat))
                out.append(e.float().cpu().numpy())
        return np.concatenate(out, 0)

    @torch.no_grad()
    def fields_and_effects(self, pre, acts, cells, post=None, chunk=8192):
        n = len(acts)
        cells = np.asarray(cells)
        if cells.ndim == 1:
            cells = np.repeat(cells[None], n, 0)
        outs = []
        if self.kind in ("comech", "actcat"):
            mat_pre = pre.reshape(NCH, -1)[:, cells].transpose(1, 2, 0)
            if self.kind == "comech":
                s = self._latent_t(pre, acts)
            else:
                s = self.net.encode(patchify(_t(pre)[None].expand(n, -1, -1, -1, -1)), _t(acts))
            eff = self.net.effect(s).float().cpu().numpy()
            for c0 in range(0, cells.shape[1], chunk):
                outs.append(self.net.decode(s, _t(cells[:, c0:c0 + chunk], torch.long),
                                            _t(mat_pre[:, c0:c0 + chunk])).float().cpu().numpy())
        else:
            mat_post = np.stack([post[i].reshape(NCH, -1)[:, cells[i]].T for i in range(n)])
            eff = None
            for i in range(n):
                ys = []
                for c0 in range(0, cells.shape[1], chunk):
                    y, e = self.net(_t(post[i:i + 1]), _t(acts[i:i + 1])[:, 6], _t(cells[i:i + 1, c0:c0 + chunk], torch.long),
                                    _t(mat_post[i:i + 1, c0:c0 + chunk]))
                    ys.append(y.float().cpu().numpy())
                outs.append(np.concatenate(ys, 1))
                eff = e.float().cpu().numpy() if eff is None else np.concatenate([eff, e.float().cpu().numpy()], 0)
            return np.concatenate(outs, 0), eff
        return np.concatenate(outs, 1), eff

    @torch.no_grad()
    def rollout2(self, pre, a1, a2_ost, cells):
        s1 = self._latent_t(pre, a1)
        s2 = self.net.step(s1, self.net.act_enc(_t(a2_ost)))
        mat_pre = pre.reshape(NCH, -1)[:, cells].transpose(1, 2, 0)
        y = self.net.decode(s2, _t(cells, torch.long), _t(mat_pre))
        return y.float().cpu().numpy(), self.net.effect(s2).float().cpu().numpy()
