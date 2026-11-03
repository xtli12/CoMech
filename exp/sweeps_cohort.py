"""FE ground truth for the virtual cohort: defect sweeps on the coalition-free twin of every foot whose complete
resection leaves a facet defect (type I: middle facet; type II: posterior; type IV: both; type III leaves none)."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from fe.geometry import Foot, Action
import fe.simulate as S

ROOT = "data/sweeps_cohort"
DEFS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
FACETS = {"I": ["mid"], "II": ["post"], "IV": ["mid", "post"], "III": []}


def jobs():
    C = json.load(open("data/cohort/cohort.json"))
    J = []
    for c in C:
        fac = FACETS[c["foot"]["ctype"]]
        if not fac:
            continue
        J.append((c["i"], "none", 0.0))
        for f in fac:
            J += [(c["i"], f, d) for d in DEFS]
    return J, {c["i"]: c for c in C}


def run(j):
    i, fac, d = j
    fn = f"{ROOT}/{i:03d}_{fac}_{d:.2f}.json"
    if os.path.exists(fn):
        return
    c = CO[i]
    f = replace(Foot(**c["foot"]), ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0)
    a = Action(resect=1, d_mid=d if fac == "mid" else 0.0, d_post=d if fac == "post" else 0.0)
    t = time.time()
    r = S.simulate(f, a, tol=1e-5)
    json.dump(dict(i=i, fac=fac, d=d, eff=r["eff"], time=time.time() - t), open(fn, "w"))
    print(i, fac, d, round(time.time() - t, 1), flush=True)


J, CO = jobs()
if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs(ROOT, exist_ok=True)
    print("jobs", len(J), flush=True)
    with Pool(int(os.environ.get("NPROC", "1"))) as p:
        for _ in p.imap_unordered(run, J, chunksize=1):
            pass
    print("COHORT_FE_DONE", flush=True)
