"""Virtual cohort mirroring the three-centre cohort (135 feet; anatomy-based types I-IV 44/15/69/7) and a
type-balanced coalition set for the label-free taxonomy test.  Only geometry is generated here (no FE)."""
import os, sys, json, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import asdict, replace
from fe.geometry import *
import exp.bank as B
from ml.data import channels, GRID

OUT = "data/cohort"


def sample_typed(rng, ctype, rtype=None):
    while True:
        f = B.sample_foot(rng, p_intact=0.0)
        if f.ctype == ctype and (rtype is None or f.rtype == rtype):
            return f


def coalition_ratio(foot):
    """Coalition-to-posterior-facet area ratio (Wilde) and facet involvement (top-view footprints; real-bone engine:
    fe.geometry.coalition_ratio)."""
    import fe.geometry as FG
    return FG.coalition_ratio(foot)


def child_defect(foot):
    """Effective facet defect left by complete resection (fraction of intact facet cartilage volume)."""
    fr_int = voxelise(FootGeom(replace(foot, ctype="none", rtype="none"), Action()), GRID)
    fr_res = voxelise(FootGeom(foot, Action(resect=1)), GRID)
    v = GRID.cell_vol
    out = {}
    for n, L in [("mid", C_MID), ("post", C_POST)]:
        vi = (fr_int[:, L] * v).sum(); vr = (fr_res[:, L] * v).sum()
        out[f"d_{n}"] = float(max(0.0, 1 - vr / vi))
    return out, fr_int, fr_res


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    # ---- virtual cohort, composition of the real cohort ----
    comp = [("I", 44), ("II", 15), ("III", 69), ("IV", 7)]
    roz_I = {"R1": 20, "R2": 18, "R3": 7, "R4": 3}
    feet = []
    for t, n in comp:
        if t == "I":
            keys = list(roz_I); p = np.array([roz_I[k] for k in keys], float); p /= p.sum()
            cnt = np.floor(p * n).astype(int); cnt[np.argsort(-(p * n - cnt))[: n - cnt.sum()]] += 1
            for k, c in zip(keys, cnt):
                feet += [sample_typed(rng, "I", k) for _ in range(c)]
        else:
            feet += [sample_typed(rng, t) for _ in range(n)]
    recs, pre, intact, res = [], [], [], []
    for i, f in enumerate(feet):
        cr = coalition_ratio(f)
        d, fr_int, fr_res = child_defect(f)
        fr_pre = voxelise(FootGeom(f, Action()), GRID)
        pre.append(channels(np.round(fr_pre * 255).astype(np.uint8)).astype(np.float16))
        intact.append(channels(np.round(fr_int * 255).astype(np.uint8)).astype(np.float16))
        res.append(channels(np.round(fr_res * 255).astype(np.uint8)).astype(np.float16))
        recs.append(dict(i=i, foot=asdict(f), **cr, **d))
        if i % 20 == 0:
            print("cohort", i, flush=True)
    np.savez_compressed(f"{OUT}/cohort_geo.npz", pre=np.stack(pre), intact=np.stack(intact), res=np.stack(res))
    json.dump(recs, open(f"{OUT}/cohort.json", "w"))
    # ---- type-balanced coalition set for the taxonomy test ----
    rng = np.random.default_rng(7)
    tax = []
    for r in ["R1", "R2", "R3", "R4"]:
        tax += [sample_typed(rng, "I", r) for _ in range(15)]
    for t in ["II", "III", "IV"]:
        tax += [sample_typed(rng, t) for _ in range(60)]
    geo = []
    for i, f in enumerate(tax):
        fr = voxelise(FootGeom(f, Action()), GRID)
        geo.append(channels(np.round(fr * 255).astype(np.uint8)).astype(np.float16))
        if i % 40 == 0:
            print("tax", i, flush=True)
    np.savez_compressed(f"{OUT}/taxonomy_geo.npz", pre=np.stack(geo))
    json.dump([asdict(f) for f in tax], open(f"{OUT}/taxonomy.json", "w"))


if __name__ == "__main__":
    main()
