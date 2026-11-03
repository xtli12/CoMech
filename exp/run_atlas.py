"""Run the atlas-analog series on the reference foot with the open-source engine."""
import os, sys, json, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from exp.scenarios import atlas_scenarios
import fe.simulate as S
from dataclasses import asdict

OUT = "results/atlas_analog"


def run(i):
    grp, name, foot, act = atlas_scenarios()[i]
    t = time.time()
    r = S.simulate(foot, act, tol=1e-6)
    np.savez_compressed(f"{OUT}/fields_{i:02d}.npz", vm=r["vm"].astype(np.float32), U=r["Urel"].astype(np.float32),
                        lab=r["lab"].astype(np.int8))
    rec = dict(i=i, group=grp, name=name, foot=asdict(foot), act=asdict(act), eff=r["eff"], iters=r["iters"],
               time=time.time() - t, fe=r.get("res"))
    with open(f"{OUT}/rec_{i:02d}.json", "w") as f:
        json.dump(rec, f)
    print(i, name, round(time.time() - t, 1), flush=True)
    return rec


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n = len(atlas_scenarios())
    todo = [i for i in range(n) if not os.path.exists(f"{OUT}/rec_{i:02d}.json")]
    with Pool(2) as p:
        p.map(run, todo, chunksize=1)
