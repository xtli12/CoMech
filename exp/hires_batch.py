"""Refined-grid FE re-solve of the held-out bank (for a many-core server).

Every configuration of the held-out test split (40 feet x 8 actions) is solved again on a grid refined by `fine`
(0.5 = twofold in every direction, about 8x the cells and ~10x the time of the standard grid).  The outputs give a
mesh-converged reference for the field metrics and the rendered maps.  One process per core; resumable.

    python exp/hires_batch.py 0.5 test 32        # fine, split, processes
"""
import os, sys, json, glob, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fe.geometry import Foot, Action
import fe.simulate as S


def job(args):
    fn, k, fine, out = args
    m = json.load(open(fn)); a = m["actions"][k]
    dst = f"{out}/{m['fid']:04d}_{k}.npz" if isinstance(m["fid"], int) else f"{out}/{m['fid']}_{k}.npz"
    if os.path.exists(dst):
        return
    t = time.time()
    r = S.simulate(Foot(**m["foot"]), Action(**a["act"]), fine=fine, tol=1e-6)
    g = S.get_grid(fine)
    np.savez_compressed(dst, vm=r["vm"].astype(np.float32), U=r["U"].astype(np.float32), lab=r["lab"].astype(np.int8),
                        xe=g.xe, ye=g.ye, ze=g.ze, eff=json.dumps(r["eff"]), eff_std=json.dumps(a["eff"]), ndof=r["ndof"])
    print(os.path.basename(dst), round(time.time() - t, 1), "s", r["ndof"], flush=True)


if __name__ == "__main__":
    fine = float(sys.argv[1]); split = sys.argv[2]; nproc = int(sys.argv[3])
    out = f"data/hires_{split}_f{fine:.2f}"; os.makedirs(out, exist_ok=True)
    J = []
    for fn in sorted(glob.glob(f"data/bank/{split}/foot_*.json")):
        K = len(json.load(open(fn))["actions"])
        J += [(fn, k, fine, out) for k in range(K)]
    print("jobs", len(J), flush=True)
    from multiprocessing import Pool
    with Pool(nproc) as p:
        list(p.imap_unordered(job, J, chunksize=1))
    print("HIRES_DONE", flush=True)
