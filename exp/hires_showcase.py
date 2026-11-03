"""High-resolution FE solutions of the showcase cases for the rendered figures (grid spacing x fine)."""
import os, sys, json, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from fe.geometry import Foot, Action
import fe.simulate as S

OUT = "data/hires"


def cases():
    M = json.load(open("results/eval_maps.json"))["child"]
    child = Foot(**M["foot"]); t, ip, fu = M["choice"]
    C = {"child_pre": (child, Action()), "child_resect": (child, Action(resect=1)),
         "child_plan": (child, Action(resect=1, osteo=float(t), interp=int(ip), fusion=int(fu)))}
    p09 = Foot(scale=0.95, align=0.0, ctype="I", rtype="R3", cov_mid=0.85, fib_frac=0.5)
    C.update({"P09L_pre": (p09, Action()), "P09L_resect": (p09, Action(resect=1)),
              "ref_intact": (Foot(), Action())})
    return C


def run(tag, fine):
    fn = f"{OUT}/{tag}_f{fine:.2f}.npz"
    if os.path.exists(fn):
        return
    f, a = cases()[tag]
    t = time.time()
    r = S.simulate(f, a, fine=fine, tol=1e-6)
    g = S.get_grid(fine)
    np.savez_compressed(fn, vm=r["vm"].astype(np.float32), sig=r["sig"].astype(np.float32), U=r["U"].astype(np.float32),
                        Urel=r["Urel"].astype(np.float32), lab=r["lab"].astype(np.int8), xe=g.xe, ye=g.ye, ze=g.ze,
                        eff=json.dumps(r["eff"]), timing=np.array(r["timing"]), ndof=r["ndof"])
    print(tag, fine, round(time.time() - t, 1), "s", r["ndof"], "dof", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fine = float(sys.argv[1]); tags = sys.argv[2].split(",")
    for tag in tags:
        run(tag, fine)
