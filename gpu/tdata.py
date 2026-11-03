"""Batch sampling for GPU training: the same batch composition as ml.train.Batcher (roll-out pairs, effect-equivalent
pairs and neighbouring pairs, joint-region-weighted query cells), on memory-mapped caches.  Pure numpy."""
import os, sys, threading, queue, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import NCH, GRID
from ml.train import joint_mask
from fe.geometry import VOID


class Pairs:
    """Within-foot effect-equivalent (|dZ| <= eps) and neighbouring (|da| <= rho, |dZ| > eps) action pairs."""

    def __init__(self, C, norm, eps, rho=0.25):
        Z = norm.eff(C.eff_all)                                  # (nf, K, 17)
        A = C.act_all
        self.eps, self.rho = eps, rho
        self.eq, self.nb = [], []
        K = Z.shape[1]
        for f in range(C.nf):
            dz = np.sqrt(((Z[f][:, None] - Z[f][None]) ** 2).mean(-1))
            da = np.linalg.norm(A[f][:, None] - A[f][None], axis=-1)
            e = [(i, j) for i in range(K) for j in range(i + 1, K) if dz[i, j] <= eps]
            n = [(i, j) for i in range(K) for j in range(i + 1, K) if dz[i, j] > eps and da[i, j] <= rho]
            self.eq.append(e); self.nb.append(n)


class Sampler:
    def __init__(self, C, norm, pairs, seed, Bf=16, Kb=5, Nq=4096, joint_frac=0.5, need_post=False):
        self.C, self.norm, self.pairs = C, norm, pairs
        self.rng = np.random.default_rng(seed)
        self.Bf, self.Kb, self.Nq, self.jf, self.need_post = Bf, Kb, Nq, joint_frac, need_post
        self.roll = [(m.index("roll_a"), m.index("roll_b")) if "roll_a" in m else None for m in C.metas]
        self.order = self.rng.permutation(C.nf); self.pos = 0
        self.Zall = norm.eff(C.eff_all)

    def sample_actions(self, f):
        rng = self.rng
        chosen = []
        if self.roll[f] is not None and rng.random() < 0.5:
            chosen += list(self.roll[f])
        if self.pairs.eq[f] and rng.random() < 0.7:
            chosen += list(self.pairs.eq[f][rng.integers(len(self.pairs.eq[f]))])
        if self.pairs.nb[f] and rng.random() < 0.7:
            chosen += list(self.pairs.nb[f][rng.integers(len(self.pairs.nb[f]))])
        chosen = list(dict.fromkeys(chosen))[: self.Kb]
        rest = [k for k in rng.permutation(self.C.K) if k not in chosen]
        return chosen + rest[: self.Kb - len(chosen)]

    def cells(self, lab):
        occ = np.nonzero(lab != VOID)[0]
        ji = np.nonzero(joint_mask(lab))[0]
        nj = int(self.Nq * self.jf)
        a = ji[self.rng.integers(len(ji), size=nj)] if len(ji) else occ[self.rng.integers(len(occ), size=nj)]
        b = occ[self.rng.integers(len(occ), size=self.Nq - nj)]
        return np.concatenate([a, b])

    def next(self):
        C, norm = self.C, self.norm
        if self.pos + self.Bf > len(self.order):
            self.order = self.rng.permutation(C.nf); self.pos = 0
        feet = self.order[self.pos:self.pos + self.Bf]; self.pos += self.Bf
        tr = [(bi, int(f), int(k)) for bi, f in enumerate(feet) for k in self.sample_actions(f)]
        n = len(tr)
        cells = np.zeros((n, self.Nq), np.int64); y = np.zeros((n, self.Nq, 4), np.float32)
        w = np.zeros((n, self.Nq), np.float32)
        for i, (_, f, k) in enumerate(tr):
            lab = np.asarray(C.get("lab", f)[k])
            c = self.cells(lab); cells[i] = c
            vm = np.asarray(C.get("vm", f)[k])[c]; uc = np.asarray(C.get("uc", f)[k])[c]
            y[i] = norm.targets(vm, uc)
            w[i] = np.where(joint_mask(lab[c], c), 3.0, 1.0)
        pre = np.stack([np.asarray(C.get("geo", int(f))[0], np.float32) for f in feet])        # (Bf, C, X, Y, Z)
        slot = np.array([t[0] for t in tr])
        mat_pre = np.stack([pre[s].reshape(NCH, -1)[:, c].T for s, c in zip(slot, cells)])
        out = dict(tr=tr, slot=slot, cells=cells, y=y, w=w, pre=pre, mat_pre=mat_pre,
                   a=np.stack([C.act_all[f, k] for _, f, k in tr]).astype(np.float32),
                   eff=np.stack([self.Zall[f, k] for _, f, k in tr]).astype(np.float32))
        if self.need_post:
            post = np.stack([np.asarray(C.get("geo", f)[k], np.float32) for _, f, k in tr])
            out["post"] = post
            out["mat_post"] = np.stack([post[i].reshape(NCH, -1)[:, c].T for i, c in enumerate(cells)])
        # roll-out, equivalent and neighbouring pairs inside the batch (same rules as ml.train.train)
        ri, eq, nb = [], [], []
        for i, (_, f, k) in enumerate(tr):
            if self.roll[f] is not None and k == self.roll[f][0]:
                j = [jj for jj, (_, fj, kj) in enumerate(tr) if fj == f and kj == self.roll[f][1]]
                if j:
                    ri.append((i, j[0]))
        Z, A = out["eff"], out["a"]
        for i in range(n):
            for j in range(i + 1, n):
                if tr[i][1] != tr[j][1]:
                    continue
                de = np.sqrt(((Z[i] - Z[j]) ** 2).mean()); da = np.linalg.norm(A[i] - A[j])
                if de <= self.pairs.eps:
                    eq.append((i, j))
                elif da <= self.pairs.rho:
                    nb.append((i, j))
        out["roll"] = np.array(ri, np.int64).reshape(-1, 2)
        out["eq"] = np.array(eq, np.int64).reshape(-1, 2)
        out["nb"] = np.array(nb, np.int64).reshape(-1, 2)
        return out


class Prefetch:
    """Background threads that keep a few batches ready."""

    def __init__(self, make_sampler, n_workers=3, depth=6):
        self.q = queue.Queue(depth)
        self.stop = False
        self.threads = []
        for w in range(n_workers):
            s = make_sampler(w)
            t = threading.Thread(target=self._run, args=(s,), daemon=True); t.start(); self.threads.append(t)

    def _run(self, s):
        while not self.stop:
            self.q.put(s.next())

    def next(self):
        return self.q.get()
