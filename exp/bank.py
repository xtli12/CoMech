"""TCC-MechBank (open-source analogue): virtual feet x surgical actions, solved with the voxel FE engine.

Each foot gets K actions: the pre-operative state (no surgery) plus complete resections that leave facet
defects of sampled extent (emulating coalitions of different size), optional calcaneal osteotomy,
interposition, the fusion branch and a load multiplier.  Every resection is complete; the defect is the
coalition footprint grown to the sampled extent.  For roll-out training each foot also contains a
resection-only action and the same resection followed by an osteotomy.
"""
import os, sys, json, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import asdict
from fe.geometry import Foot, Action
import fe.simulate as S

ROOT = "data/bank"
# real three-centre cohort (anatomy-based types I-IV) and Rozansky I-IV within middle-facet coalitions
COHORT_TYPES = {"I": 44, "II": 15, "III": 69, "IV": 7}
COHORT_ROZ_I = {"R1": 20, "R2": 18, "R3": 7, "R4": 3}


def _mix(rng, counts, w_uniform=0.5):
    keys = list(counts); p = np.array([counts[k] for k in keys], float); p /= p.sum()
    p = (1 - w_uniform) * p + w_uniform / len(keys)
    return keys[rng.choice(len(keys), p=p)]


def sample_foot(rng, ood=False, p_intact=0.15):
    U = rng.uniform
    if not ood:
        sc, al = U(0.86, 1.04), U(-12, 12)
    else:   # out-of-range anatomy and alignment (never seen in training)
        sc = U(0.80, 0.86) if rng.random() < 0.5 else U(1.04, 1.08)
        al = U(12, 16) * (1 if rng.random() < 0.5 else -1)
    f = dict(scale=sc, post_a=12.0 * U(0.85, 1.15), post_b=9.5 * U(0.85, 1.15), mid_a=6.5 * U(0.8, 1.2),
             mid_b=4.5 * U(0.8, 1.2), ant_a=4.5 * U(0.8, 1.2), ant_b=4.0 * U(0.8, 1.2),
             ant_mode=int(rng.choice(3, p=[0.6, 0.25, 0.15])), sust_y=U(9.5, 12.5), post_dome=U(0.5, 2.5),
             cart_t=U(1.6, 2.4), align=al, seed_ang=U(-20, 20))
    if rng.random() < p_intact:
        f.update(ctype="none", rtype="none")
    else:
        t = _mix(rng, COHORT_TYPES)
        f["ctype"] = t
        if t == "I":
            r = _mix(rng, COHORT_ROZ_I)
            f["rtype"] = r
            f["cov_mid"] = U(0.5, 1.0) if r == "R4" else U(0.2, 0.6)
            if r == "R3":
                f["fib_frac"] = U(0.3, 0.7)
        elif t == "II":
            f.update(rtype="R5", cov_post=U(0.15, 0.6))
        elif t == "III":
            f.update(rtype="R5")
        else:
            f.update(rtype="unclass", cov_mid=U(0.3, 0.8), cov_post=U(0.15, 0.5))
    return Foot(**f)


def sample_actions(rng, foot, K=8):
    U = rng.uniform
    acts, meta = [], []
    acts.append(Action()); meta.append("preop")
    facets = {"I": ["mid"], "II": ["post"], "IV": ["mid", "post"], "III": [], "none": ["mid", "post"]}[foot.ctype]

    def resection():
        d_mid = d_post = 0.0
        if foot.ctype == "none":
            f = facets[rng.integers(2)]
            if f == "mid":
                d_mid = U(0.05, 0.75)
            else:
                d_post = U(0.05, 0.75)
        else:
            if "mid" in facets and rng.random() < 0.7:
                d_mid = U(foot.cov_mid, max(foot.cov_mid, min(foot.cov_mid + 0.35, 0.85))) if rng.random() < 0.6 else 0.0
            if "post" in facets and rng.random() < 0.7:
                d_post = U(foot.cov_post, max(foot.cov_post, min(foot.cov_post + 0.35, 0.85))) if rng.random() < 0.6 else 0.0
            if foot.ctype == "III" and rng.random() < 0.5:
                d_mid, d_post = U(0, 0.25), U(0, 0.2)
        return dict(resect=1, d_mid=d_mid, d_post=d_post, interp=int(rng.random() < 0.5))

    # resection only, then the same resection followed by an osteotomy (roll-out pair)
    r0 = resection()
    acts.append(Action(**r0, osteo=0.0, load=1.0)); meta.append("roll_a")
    acts.append(Action(**r0, osteo=float(U(-10, 10)), load=1.0)); meta.append("roll_b")
    while len(acts) < K:
        if rng.random() < 0.06:
            acts.append(Action(resect=1, fusion=1, osteo=float(U(-10, 10)) if rng.random() < 0.5 else 0.0,
                               load=float(U(1, 2)) if rng.random() < 0.3 else 1.0)); meta.append("fusion")
            continue
        r = resection()
        # neighbouring actions: small perturbation of a previous resection (for the sensitivity margin)
        if rng.random() < 0.3 and len(acts) > 1:
            prev = asdict(acts[int(rng.integers(1, len(acts)))])
            if not prev["fusion"]:
                r = dict(resect=1, d_mid=float(np.clip(prev["d_mid"] + (U(-0.08, 0.08) if prev["d_mid"] > 0 else 0), 0, 0.85)),
                         d_post=float(np.clip(prev["d_post"] + (U(-0.08, 0.08) if prev["d_post"] > 0 else 0), 0, 0.85)),
                         interp=prev["interp"])
                acts.append(Action(**r, osteo=float(np.clip(prev["osteo"] + U(-1.5, 1.5), -12, 12)), load=prev["load"]))
                meta.append("neighbour"); continue
        acts.append(Action(**r, osteo=float(U(-10, 10)) if rng.random() < 0.5 else 0.0,
                           load=float(U(1, 2)) if rng.random() < 0.3 else 1.0)); meta.append("random")
    return acts, meta


def solve_foot(job):
    fid, split, foot, acts, meta = job
    d = f"{ROOT}/{split}"
    if os.path.exists(f"{d}/foot_{fid:04d}.json"):
        return fid
    recs = []
    vm_l, uc_l, lab_l, fr_l = [], [], [], []
    U0 = None
    for k, act in enumerate(acts):
        t = time.time()
        r = S.simulate(foot, act, tol=1e-5, U0=U0)
        if k == 0:
            U0 = r["U"]          # warm start every action of this foot from the pre-operative solution
        grid = S.get_grid()
        ucell = r["Urel"][grid.cell_nodes].mean(1)
        vm_l.append(r["vm"].astype(np.float16)); uc_l.append(ucell.astype(np.float16))
        lab_l.append(r["lab"].astype(np.int8)); fr_l.append(np.round(r["frac"] * 255).astype(np.uint8))
        if not (0.3 < r["eff"]["heel_share"] < 0.9):
            print("QC WARNING heel share", fid, k, r["eff"]["heel_share"], flush=True)
        recs.append(dict(k=k, meta=meta[k], act=asdict(act), eff=r["eff"], iters=r["iters"], time=time.time() - t,
                         fe={kk: v for kk, v in (r.get("res") or {}).items() if kk in ("F_on_talus", "F_on_calcaneus", "ground_Fz", "ndof", "t_prep_s", "t_solve_s", "solver")}))
    np.savez_compressed(f"{d}/foot_{fid:04d}.npz", vm=np.stack(vm_l), ucell=np.stack(uc_l), lab=np.stack(lab_l),
                        frac=np.stack(fr_l))
    with open(f"{d}/foot_{fid:04d}.json", "w") as fh:
        json.dump(dict(fid=fid, split=split, foot=asdict(foot), actions=recs), fh)
    print(split, fid, round(sum(r["time"] for r in recs), 1), flush=True)
    return fid


def make_jobs(split, n, seed, K=8, ood=False, start=0):
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n):
        foot = sample_foot(rng, ood=ood)
        acts, meta = sample_actions(rng, foot, K)
        jobs.append((start + i, split, foot, acts, meta))
    return jobs


if __name__ == "__main__":
    from multiprocessing import Pool
    split = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3])
    ood = split == "ood"
    os.makedirs(f"{ROOT}/{split}", exist_ok=True)
    jobs = make_jobs(split, n, seed, ood=ood)
    with Pool(2) as p:
        for _ in p.imap_unordered(solve_foot, jobs, chunksize=1):
            pass
