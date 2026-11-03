"""Convert TCC-MechBank splits into memory-mapped arrays for fast GPU training (same arrays as ml.data.Split).

    python3 gpu/cache.py train train_x val test ood
"""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import channels, intact_volumes, GRID, GSHAPE, NCH
from fe.geometry import C_MID, C_POST, Foot
from fe.simulate import EFFECT_NAMES

ROOT = "data/bank"
OUT = "data/cache"


def foot_arrays(fn, cache):
    m = json.load(open(fn))
    d = np.load(fn.replace(".json", ".npz"))
    fr = d["frac"]                                               # (K, ncell, 15) uint8
    key = os.path.basename(fn)
    if key in cache:
        ref_mid, ref_post = cache[key]
    else:
        ref_mid, ref_post = intact_volumes(m["foot"]); cache[key] = [ref_mid, ref_post]
    geo = channels(fr).astype(np.float16)                        # (K, C, X, Y, Z)
    fr32 = fr.astype(np.float32) / 255.0
    vol = GRID.cell_vol[None, :]
    vmid = (fr32[:, :, C_MID] * vol).sum(1); vpost = (fr32[:, :, C_POST] * vol).sum(1)
    dm = np.clip(1 - vmid / ref_mid, 0, 1); dp = np.clip(1 - vpost / ref_post, 0, 1)
    acts = []
    for k, a in enumerate(m["actions"]):
        A = a["act"]
        acts.append([A["resect"], dm[k] if A["resect"] else 0.0, dp[k] if A["resect"] else 0.0,
                     A["osteo"] / 10.0, A["interp"], A["fusion"], A["load"] - 1.0])
    eff = [[a["eff"][n] for n in EFFECT_NAMES] for a in m["actions"]]
    meta = dict(fid=m["fid"], metas=[a["meta"] for a in m["actions"]], ctype=m["foot"]["ctype"],
                rtype=m["foot"]["rtype"], foot=m["foot"])
    return geo, d["vm"], d["ucell"], d["lab"], np.array(acts, np.float32), np.array(eff), meta


def build(split):
    files = sorted(glob.glob(f"{ROOT}/{split}/foot_*.json"))
    files = [f for f in files if os.path.exists(f.replace(".json", ".npz"))]
    out = f"{OUT}/{split}"
    if os.path.exists(f"{out}/meta.json") and json.load(open(f"{out}/meta.json"))["nf"] == len(files):
        print("cache up to date", split, len(files), flush=True); return
    os.makedirs(out, exist_ok=True)
    nf = len(files); K = 8; nc = GRID.ncell
    mm = lambda name, shape, dt: np.lib.format.open_memmap(f"{out}/{name}.npy", mode="w+", dtype=dt, shape=shape)
    G = mm("geo", (nf, K, NCH) + GSHAPE, np.float16)
    VM = mm("vm", (nf, K, nc), np.float16)
    UC = mm("uc", (nf, K, nc, 3), np.float16)
    LB = mm("lab", (nf, K, nc), np.int8)
    AC = np.zeros((nf, K, 7), np.float32); EF = np.zeros((nf, K, 17))
    cfn = f"{ROOT}/{split}/intact_volumes.json"
    cache = json.load(open(cfn)) if os.path.exists(cfn) else {}
    metas = []
    for i, fn in enumerate(files):
        g, vm, uc, lab, a, e, meta = foot_arrays(fn, cache)
        assert g.shape[0] == K, (fn, g.shape)
        G[i] = g; VM[i] = vm; UC[i] = uc; LB[i] = lab; AC[i] = a; EF[i] = e; metas.append(meta)
        if (i + 1) % 100 == 0:
            print(split, i + 1, "/", nf, flush=True)
    for X in (G, VM, UC, LB):
        X.flush()
    np.save(f"{out}/act.npy", AC); np.save(f"{out}/eff.npy", EF)
    json.dump(cache, open(cfn, "w"))
    json.dump(dict(nf=nf, K=K, feet=metas), open(f"{out}/meta.json", "w"))
    print("cache built", split, nf, flush=True)


class Cache:
    """Concatenation of the first n feet of several cached splits (memory-mapped, read-only)."""

    def __init__(self, splits, n=None):
        parts = []
        for s in splits:
            d = f"{OUT}/{s}"
            meta = json.load(open(f"{d}/meta.json"))
            parts.append((d, meta))
        self.geo, self.vm, self.uc, self.lab, self.act, self.eff, self.feet = [], [], [], [], [], [], []
        left = n if n is not None else 10 ** 9
        for d, meta in parts:
            take = min(meta["nf"], left)
            if take <= 0:
                break
            self.geo.append(np.load(f"{d}/geo.npy", mmap_mode="r")[:take])
            self.vm.append(np.load(f"{d}/vm.npy", mmap_mode="r")[:take])
            self.uc.append(np.load(f"{d}/uc.npy", mmap_mode="r")[:take])
            self.lab.append(np.load(f"{d}/lab.npy", mmap_mode="r")[:take])
            self.act.append(np.load(f"{d}/act.npy")[:take]); self.eff.append(np.load(f"{d}/eff.npy")[:take])
            self.feet += meta["feet"][:take]
            left -= take
        self.nf = len(self.feet)
        if n is not None and self.nf < n:
            raise RuntimeError(f"only {self.nf} feet cached, {n} requested")
        self.K = 8
        # index: global foot -> (part, local index)
        self.index = [(p, i) for p, g in enumerate(self.geo) for i in range(len(g))]
        self.act_all = np.concatenate(self.act); self.eff_all = np.concatenate(self.eff)
        self.metas = [f["metas"] for f in self.feet]

    def get(self, name, f):
        p, i = self.index[f]
        return getattr(self, name)[p][i]


if __name__ == "__main__":
    for s in sys.argv[1:]:
        if glob.glob(f"{ROOT}/{s}/foot_*.npz"):
            build(s)
