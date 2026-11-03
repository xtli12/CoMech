"""Atlas-analog normal foot and defect series with the atlas's defect convention -- the cartilage removed from a facet
replaced by soft tissue bonded to both bones -- instead of an empty defect (Claude, 2026-10-29).  Writes
results/atlas_analog_softfill/rec_<i>.json in the format of exp/run_atlas.py (no fields)."""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from dataclasses import asdict
OUT = "results/atlas_analog_softfill"


def _init(counter):
    with counter.get_lock():
        i = counter.value; counter.value += 1
    os.environ["COMECH_GPU"] = str(i % 4)


def run(i):
    from exp.scenarios import atlas_scenarios
    import fe.simulate  # noqa: F401  (puts rb/ on the path)
    import rbsim as RS
    grp, name, foot, act = atlas_scenarios()[i]
    fn = f"{OUT}/rec_{i:02d}.json"
    if os.path.exists(fn):
        return i
    t = time.time()
    r = RS.simulate(foot, act, tol=1e-6, defect_fill="soft", return_fields=False)
    rec = dict(i=i, group=grp, name=name, foot=asdict(foot), act=asdict(act), eff=r["eff"], iters=r["iters"],
               time=time.time() - t, fe=r.get("res"), defect_fill="soft")
    json.dump(rec, open(fn, "w"), default=float)
    print(i, name, round(time.time() - t, 1), flush=True)
    return i


if __name__ == "__main__":
    import multiprocessing as mp
    from exp.scenarios import atlas_scenarios
    os.makedirs(OUT, exist_ok=True)
    todo = [i for i, (g, n, f, a) in enumerate(atlas_scenarios()) if g == "defect" or n == "Normal"]
    ctx = mp.get_context("spawn"); c = ctx.Value("i", 0)
    with ctx.Pool(int(os.environ.get("NPROC", "4")), initializer=_init, initargs=(c,)) as p:
        list(p.imap_unordered(run, todo))
