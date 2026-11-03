"""FE ground truth for the tipping-point, exchange-rate and planning experiments (held-out test feet)."""
import os, sys, json, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import asdict, replace
from fe.geometry import Foot, Action
import fe.simulate as S
import exp.bank as B

ROOT = "data/sweeps"
DEFS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
PLAN_T = [-10.0, -5.0, 0.0, 5.0, 10.0]


def test_feet():
    return [j[2] for j in B.make_jobs("test", 50, 13)]


def intact(foot):
    return replace(foot, ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0)


def jobs():
    feet = test_feet()
    J = []
    # (1) defect sweeps on the coalition-free version of 20 test feet
    for i in range(20):
        f = intact(feet[i])
        J.append(("sweep", i, "none", 0.0, f, Action(resect=1)))
        for facet in ("mid", "post"):
            for d in DEFS[1:]:
                a = Action(resect=1, d_mid=d) if facet == "mid" else Action(resect=1, d_post=d)
                J.append(("sweep", i, facet, d, f, a))
    # (2) exchange rate: posterior sweeps after a 5 mm medialising / lateralising osteotomy, 6 feet
    for i in range(6):
        f = intact(feet[i])
        for t in (5.0, -5.0):
            for d in DEFS:
                J.append(("xrate", i, f"post_t{t:+.0f}", d, f, Action(resect=1, d_post=d, osteo=t)))
    # (3) planning: complete resection (defect = coalition footprint) + adjunct grid, 16 coalition feet
    n = 0
    for i, f in enumerate(feet):
        if f.ctype == "none" or n >= 16:
            continue
        n += 1
        for t in PLAN_T:
            for ip in (0, 1):
                J.append(("plan", i, f"t{t:+.0f}_i{ip}", 0.0, f, Action(resect=1, osteo=t, interp=ip)))
        J.append(("plan", i, "fusion", 0.0, f, Action(resect=1, fusion=1)))
    return J


def run(j):
    kind, i, tag, d, foot, act = j
    fn = f"{ROOT}/{kind}_{i:03d}_{tag}_{d:.2f}.json"
    if os.path.exists(fn):
        return
    t = time.time()
    r = S.simulate(foot, act, tol=1e-5)
    grid = S.get_grid()
    rec = dict(kind=kind, i=i, tag=tag, d=d, foot=asdict(foot), act=asdict(act), eff=r["eff"], time=time.time() - t,
               fe={kk: v for kk, v in (r.get("res") or {}).items() if kk in ("F_on_talus", "F_on_calcaneus", "ground_Fz")})
    if kind == "plan" or (kind == "sweep" and i < 4):
        np.savez_compressed(fn.replace(".json", ".npz"), vm=r["vm"].astype(np.float16),
                            ucell=r["Urel"][grid.cell_nodes].mean(1).astype(np.float16), lab=r["lab"].astype(np.int8),
                            frac=np.round(r["frac"] * 255).astype(np.uint8))
    json.dump(rec, open(fn, "w"))
    print(kind, i, tag, d, round(time.time() - t, 1), flush=True)


if __name__ == "__main__":
    from multiprocessing import Pool
    while os.path.exists("data/HOLD_SWEEPS"):      # released once model training has finished
        time.sleep(60)
    os.makedirs(ROOT, exist_ok=True)
    J = jobs()
    print("jobs", len(J), flush=True)
    with Pool(2) as p:
        for _ in p.imap_unordered(run, J, chunksize=1):
            pass
