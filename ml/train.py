"""Training for CoMech and baselines on TCC-MechBank."""
import os, sys, time, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import ad
from ml.ad import T
from ml.data import *
from ml.models import CoMech, Geo2Field, DeepONet, FNO, NACT
from fe.geometry import VOID

ROLL_OST_ONLY = lambda a: np.array([0, 0, 0, a[3], 0, 0, 0], np.float32)


def joint_mask(lab, cells=None):
    """Cells of the articular / surgical region (cartilage, bridge, interposition, subtalar band)."""
    band = _BAND if cells is None else _BAND[cells]
    return (np.isin(lab, JOINT_LABELS) | band) & (lab != VOID)


def _subtalar_band():
    import fe.geometry as FG
    if hasattr(FG, "subtalar_band"):                 # real-bone engine: within 6 mm of both talus and calcaneus
        return FG.subtalar_band(GRID.cell_xyz)
    xyz = GRID.cell_xyz; z = xyz[:, 2]; x = xyz[:, 0]
    return (z > 30) & (z < 42) & (x > 18) & (x < 70)


_BAND = _subtalar_band()


class Pairs:
    """Within-foot action pairs for the effect-swap (equivalent) and action-sensitivity (neighbour) losses."""

    def __init__(self, S, norm, eps=None, rho=0.25, eps_q=0.12):
        Z = norm.eff(S.eff_raw)                                   # (nf, K, 17)
        self.eq, self.nb = [], []
        dists = []
        for f in range(S.nf):
            for i in range(S.K):
                for j in range(i + 1, S.K):
                    dists.append(np.sqrt(((Z[f, i] - Z[f, j]) ** 2).mean()))
        self.eps = eps if eps is not None else float(np.quantile(dists, eps_q))
        self.rho = rho
        for f in range(S.nf):
            e, n = [], []
            for i in range(S.K):
                for j in range(i + 1, S.K):
                    de = np.sqrt(((Z[f, i] - Z[f, j]) ** 2).mean())
                    da = np.linalg.norm(S.act[f, i] - S.act[f, j])
                    if de <= self.eps:
                        e.append((i, j))
                    elif da <= rho:
                        n.append((i, j))
            self.eq.append(e); self.nb.append(n)


class Batcher:
    def __init__(self, S, norm, pairs, rng, Bf=4, Kb=5, Nq=1024, joint_frac=0.5):
        self.S, self.norm, self.pairs, self.rng = S, norm, pairs, rng
        self.Bf, self.Kb, self.Nq, self.jf = Bf, Kb, Nq, joint_frac
        self.occ_idx, self.joint_idx = [], []
        for f in range(S.nf):
            oi, ji = [], []
            for k in range(S.K):
                lab = S.lab[f, k]
                oi.append(np.nonzero(lab != VOID)[0].astype(np.int32)); ji.append(np.nonzero(joint_mask(lab))[0].astype(np.int32))
            self.occ_idx.append(oi); self.joint_idx.append(ji)
        self.roll = [(m.index("roll_a"), m.index("roll_b")) if "roll_a" in m else None for m in S.metas]
        self.order = rng.permutation(S.nf); self.pos = 0

    def sample_actions(self, f):
        rng = self.rng
        chosen = []
        if self.roll[f] is not None and rng.random() < 0.5:
            chosen += list(self.roll[f])
        if self.pairs is not None:
            if self.pairs.eq[f] and rng.random() < 0.7:
                chosen += list(self.pairs.eq[f][rng.integers(len(self.pairs.eq[f]))])
            if self.pairs.nb[f] and rng.random() < 0.7:
                chosen += list(self.pairs.nb[f][rng.integers(len(self.pairs.nb[f]))])
        chosen = list(dict.fromkeys(chosen))[: self.Kb]
        rest = [k for k in rng.permutation(self.S.K) if k not in chosen]
        chosen += rest[: self.Kb - len(chosen)]
        return chosen

    def cells(self, f, k):
        nj = int(self.Nq * self.jf)
        ji, oi = self.joint_idx[f][k], self.occ_idx[f][k]
        a = ji[self.rng.integers(len(ji), size=nj)] if len(ji) else oi[self.rng.integers(len(oi), size=nj)]
        b = oi[self.rng.integers(len(oi), size=self.Nq - nj)]
        return np.concatenate([a, b])

    def next(self):
        if self.pos + self.Bf > len(self.order):
            self.order = self.rng.permutation(self.S.nf); self.pos = 0
        feet = self.order[self.pos:self.pos + self.Bf]; self.pos += self.Bf
        S, norm = self.S, self.norm
        tr = []                                # (foot slot, foot, action)
        for bi, f in enumerate(feet):
            for k in self.sample_actions(f):
                tr.append((bi, f, k))
        cells = np.stack([self.cells(f, k) for _, f, k in tr])
        y = np.stack([norm.targets(S.vm[f, k, c], S.uc[f, k, c]) for (_, f, k), c in zip(tr, cells)])
        jm = np.stack([joint_mask(S.lab[f, k][c], c) for (_, f, k), c in zip(tr, cells)])
        w = np.where(jm, 3.0, 1.0).astype(np.float32)
        pre = S.geo_pre[feet].astype(np.float32)                      # (Bf, C, X, Y, Z)
        mat_pre = np.stack([pre[bi].reshape(NCH, -1)[:, c].T for (bi, _, _), c in zip(tr, cells)])
        if S.geo_post is not None:
            post = np.stack([S.geo_post[f, k].astype(np.float32) for _, f, k in tr])
            mat_post = np.stack([post[i].reshape(NCH, -1)[:, c].T for i, c in enumerate(cells)])
        else:
            post = mat_post = np.zeros((len(tr), 1), np.float32)       # not used by this model
        a = np.stack([S.act[f, k] for _, f, k in tr])
        eff = norm.eff(np.stack([S.eff_raw[f, k] for _, f, k in tr]))
        return dict(feet=feet, tr=tr, cells=cells, y=y, w=w, pre=pre, mat_pre=mat_pre, post=post,
                    mat_post=mat_post, a=a, eff=eff)


def load_only(a):
    b = np.zeros_like(a); b[:, 6] = a[:, 6]
    return b


# ---------------- forward passes ----------------
def fwd_comech(m, b):
    s0 = m.encode(patchify(b["pre"]))
    slot = np.array([t[0] for t in b["tr"]])
    s = ad.take(s0, slot, 0)
    u = m.act_enc(b["a"])
    s1 = m.step(s, u)
    y = m.decode(s1, b["cells"], b["mat_pre"])
    e = m.effect(s1)
    return y, e, s1, s0


def fwd_actcat(m, b):
    pre = b["pre"]; slot = np.array([t[0] for t in b["tr"]])
    s = m.encode(patchify(pre[slot]), b["a"])
    return m.decode(s, b["cells"], b["mat_pre"]), m.effect(s)


def fwd_geo(m, b):
    s = m.encode(patchify(b["post"]), load_only(b["a"]))
    return m.decode(s, b["cells"], b["mat_post"]), m.effect(s)


def pooled_geo(pre):
    B = pre.shape[0]
    return patchify(pre).reshape(B, NTOK, NCH, -1).mean(-1).reshape(B, -1)


def fwd_deeponet(m, b):
    slot = np.array([t[0] for t in b["tr"]])
    return m.forward(pooled_geo(b["pre"])[slot], b["a"], b["cells"], b["mat_pre"])


def fwd_fno(m, b):
    return m.forward(b["post"], load_only(b["a"]), b["cells"])


def mse(a, b, w=None):
    d = (a - T(b)) ** 2
    if w is not None:
        d = d * T(w[..., None])
    return d.mean()


# ---------------- training ----------------
def make_model(kind, seed):
    if kind in ("comech",):
        return CoMech(seed=seed)
    if kind == "actcat":
        return CoMech(seed=seed, act_concat=True)
    if kind == "geo":
        return Geo2Field(seed=seed)
    if kind == "deeponet":
        return DeepONet(seed=seed)
    if kind == "fno":
        return FNO(seed=seed)
    raise ValueError(kind)


def sub_batch(b, idx):
    """Transitions `idx` of a batch (for gradient accumulation in the baselines)."""
    c = dict(b)
    for k in ["cells", "y", "w", "post", "mat_pre", "mat_post", "a", "eff"]:
        c[k] = b[k][idx]
    c["tr"] = [b["tr"][i] for i in idx]
    return c


def train(kind, S, norm, pairs, steps=3000, seed=0, lr=1e-3, swap=1.0, sens=1.0, roll=1.0, w_eff=1.0,
          mu_frac=0.5, log_every=50, out=None, val=None, Bf=4, Kb=5, Nq=1024, micro=1, init=None):
    # init: a saved model dict; training continues from its weights with a warm restart of the cosine schedule
    # (fresh Adam moments, 100-step warm-up) and a different batch stream
    rng = np.random.default_rng(seed + 100 + (1000 if init is not None else 0))
    m = make_model(kind, seed)
    if init is not None:
        m.load(init["state"])
    opt = ad.Adam(m.p, lr=lr, wd=0.0)
    bat = Batcher(S, norm, pairs if kind == "comech" else pairs, rng, Bf=Bf, Kb=Kb, Nq=Nq)
    hist = []
    t0 = time.time()
    eps = pairs.eps if pairs is not None else 0.1
    for it in range(steps):
        lr_t = lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * it / steps)))
        if it < 100:
            lr_t *= (it + 1) / 100
        b = bat.next()
        logs = {}
        if kind == "comech":
            y, e, s1, s0 = fwd_comech(m, b)
            L = mse(y, b["y"], b["w"]); logs["field"] = float(L.data)
            Le = mse(e, b["eff"]); logs["eff"] = float(Le.data)
            L = L + w_eff * Le
            # ---- multi-step roll-out: resection then osteotomy ----
            tr = b["tr"]
            if roll > 0:
                ri = []
                for i, (bi, f, k) in enumerate(tr):
                    if bat.roll[f] is not None and k == bat.roll[f][0]:
                        j = [jj for jj, (bj, fj, kj) in enumerate(tr) if fj == f and kj == bat.roll[f][1]]
                        if j:
                            ri.append((i, j[0]))
                if ri:
                    ia = np.array([p[0] for p in ri]); ib = np.array([p[1] for p in ri])
                    s_a = ad.take(s1, ia, 0)
                    u2 = m.act_enc(np.stack([ROLL_OST_ONLY(b["a"][j]) for j in ib]))
                    s_ab = m.step(s_a, u2)
                    y2 = m.decode(s_ab, b["cells"][ib], b["mat_pre"][ib])
                    e2 = m.effect(s_ab)
                    Lr = mse(y2, b["y"][ib], b["w"][ib]) + w_eff * mse(e2, b["eff"][ib])
                    logs["roll"] = float(Lr.data); L = L + roll * Lr
            # ---- effect-swap and action-sensitivity ----
            Z = b["eff"]
            eq, nb = [], []
            for i in range(len(tr)):
                for j in range(i + 1, len(tr)):
                    if tr[i][1] != tr[j][1]:
                        continue
                    de = np.sqrt(((Z[i] - Z[j]) ** 2).mean())
                    da = np.linalg.norm(b["a"][i] - b["a"][j])
                    if de <= eps:
                        eq.append((i, j))
                    elif da <= pairs.rho:
                        nb.append((i, j))
            if swap > 0 and eq:
                ia = np.array([p[0] for p in eq]); ib = np.array([p[1] for p in eq])
                ea, eb = ad.take(e, ia, 0), ad.take(e, ib, 0)
                Ls = ((ea - eb) ** 2).mean() + mse(ea, Z[ib]) + mse(eb, Z[ia])
                logs["swap"] = float(Ls.data); L = L + swap * Ls
            if sens > 0 and nb:
                ia = np.array([p[0] for p in nb]); ib = np.array([p[1] for p in nb])
                ea, eb = ad.take(e, ia, 0), ad.take(e, ib, 0)
                def dist(x, z):
                    return ad.sqrt(((x - T(z)) ** 2).mean(1) + 1e-8)
                mu = mu_frac * eps
                h1 = ad.relu(dist(ea, Z[ia]) - dist(ea, Z[ib]) + mu)
                h2 = ad.relu(dist(eb, Z[ib]) - dist(eb, Z[ia]) + mu)
                Ln = (h1 + h2).mean()
                logs["sens"] = float(Ln.data); L = L + sens * Ln
        else:
            fw = dict(actcat=fwd_actcat, geo=fwd_geo, deeponet=fwd_deeponet, fno=fwd_fno)[kind]
            n = len(b["tr"])
            chunks = np.array_split(np.arange(n), micro)
            opt.zero()
            lf = le = 0.0
            for ch in chunks:          # gradient accumulation: identical to one batch of n transitions
                bc = sub_batch(b, ch) if micro > 1 else b
                y, e = fw(m, bc)
                Lf = mse(y, bc["y"], bc["w"]); Le = mse(e, bc["eff"])
                Lc = (Lf + w_eff * Le) * (len(ch) / n)
                Lc.backward()
                lf += float(Lf.data) * len(ch) / n; le += float(Le.data) * len(ch) / n
            logs["field"] = lf; logs["eff"] = le
            L = T(np.float32(lf + w_eff * le))
        if kind == "comech":
            opt.zero()
            L.backward()
        gn = opt.step(lr_t)
        logs["loss"] = float(L.data); logs["gn"] = float(gn)
        hist.append(logs)
        if (it + 1) % log_every == 0:
            avg = {k: np.mean([h[k] for h in hist[-log_every:] if k in h]) for k in hist[-1]}
            print(kind, seed, it + 1, f"{time.time() - t0:.0f}s", {k: round(v, 4) for k, v in avg.items()}, flush=True)
    if out:
        with open(out, "wb") as fh:
            cfg = dict(steps=steps, seed=seed, lr=lr, swap=swap, sens=sens, roll=roll, eps=eps,
                       rho=pairs.rho if pairs else None, Bf=Bf, Kb=Kb, Nq=Nq, train_time=time.time() - t0)
            if init is not None:
                c0 = init["cfg"]
                cfg.update(phases=[dict(steps=c0["steps"], lr=c0["lr"]), dict(steps=steps, lr=lr)],
                           steps=c0["steps"] + steps, train_time=c0["train_time"] + cfg["train_time"])
                hist = init["hist"] + hist
            pickle.dump(dict(kind=kind, state=m.state(), hist=hist, cfg=cfg), fh)
    return m, hist
