"""Rendered subtalar stress maps of the virtual child (Fig. 7a): superior orthographic view of the calcaneal
articular surfaces (talus removed), cartilage coloured by von Mises stress from (i) the CoMech ensemble queried
continuously as a neural field and (ii) a refined FE solution (grid spacing halved, nodal-averaged stresses).
Saves image arrays with their millimetre extents to results/renders/."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.chdir(ROOT)
from scene_engine import *
from exp.common import get_pred, geo_of, defect_fraction, act_vec
from ml.data import Norm
from dataclasses import replace

OUT = os.environ.get("RENDER_OUT", "results/renders")
SHOW = ("calc", "cart")


def ens_field(preds, pre, act, norm):
    from nfquery import query
    def f(P):
        ys = np.mean([query(p, pre, act, P) for p in preds], 0)
        vm, _ = norm.fields_inv(ys)
        return vm.astype(np.float32)
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    M = json.load(open("results/eval_maps.json"))["child"]
    foot = Foot(**M["foot"]); t, ip, fu = M["choice"]
    acts = {"resect": Action(resect=1), "plan": Action(resect=1, osteo=float(t), interp=int(ip), fusion=int(fu))}
    norm = Norm.cached(); preds = [get_pred(n) for n in ["comech_s0", "comech_s1", "comech_s2"]]
    pre = geo_of(foot, Action())[0].astype(np.float32)
    twin = replace(foot, ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0)
    ref_fr = geo_of(twin, Action(resect=1))[1]
    W, H, ortho = 760, 560, 31.0
    cx, cy = FACET_C; ortho *= FACET_SCALE      # engine-dependent view (figsrc/rbview.py)
    cam = dict(eye=(cx, cy, 200.0), target=(cx, cy, 0.0), up=(0, 1, 0), W=W, H=H, ortho=ortho, ss=2, lights="camera")
    ext = [cx - ortho, cx + ortho, cy - ortho * H / W, cy + ortho * H / W]
    for tag, act in acts.items():
        fr = geo_of(foot, act)[1]; d = defect_fraction(fr, ref_fr)
        a = act_vec(1, d["mid"], d["post"], act.osteo, act.interp, act.fusion)
        for src in ["comech", "fe", "fefine"]:
            fn = f"{OUT}/child_{tag}_{src}.npz"
            if os.path.exists(fn):
                continue
            t0 = time.time()
            if src == "comech":
                field = ens_field(preds, pre, a, norm)
            elif src == "fe":
                field, _ = fe_field(f"data/hires/child_{tag}_f1.00.npz")
            else:
                field, _ = fe_field(f"data/hires/child_{tag}_f0.50.npz")
            objs, L, lo, h = engine_scene(foot, act, field=field, vmax=1.6, show=SHOW)
            for o in objs:
                if o.name in ("calc", "nav", "cub"):
                    o.color = np.array((0.90, 0.89, 0.87)); o.spec = 0.05; o.rim = 0.05
            img, al = render(objs, **cam)
            np.savez_compressed(fn, img=img.astype(np.float32), alpha=al.astype(np.float32), ext=np.array(ext))
            print(tag, src, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
