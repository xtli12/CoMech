"""FE mechanical fingerprints for the label-free taxonomy test: pre-operative solution of each of the 240
type-balanced coalitions and of its coalition-free twin (480 solves; effect descriptors only)."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from fe.geometry import Foot, Action
import fe.simulate as S

ROOT = "data/sweeps_tax"


def run(j):
    i, kind = j
    fn = f"{ROOT}/{i:03d}_{kind}.json"
    if os.path.exists(fn):
        return
    f = Foot(**json.load(open("data/cohort/taxonomy.json"))[i])
    if kind == "twin":
        f = replace(f, ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0)
    t = time.time()
    r = S.simulate(f, Action(), tol=1e-5)
    json.dump(dict(i=i, kind=kind, eff=r["eff"], time=time.time() - t), open(fn, "w"))
    print(i, kind, round(time.time() - t, 1), flush=True)


if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs(ROOT, exist_ok=True)
    J = [(i, k) for i in range(240) for k in ("coal", "twin")]
    with Pool(int(os.environ.get("NPROC", "2"))) as p:
        for _ in p.imap_unordered(run, J, chunksize=1):
            pass
