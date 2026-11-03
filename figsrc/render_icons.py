"""Small renderings of engine feet for the schematic figures (Fig. 4a): superior views of the calcaneal articular
surfaces of virtual feet (bank sampler, intact) and of the reference foot with the four anatomy-based coalition
types (atlas-analogue morphologies). Pure geometry, no fields. (oblique(): optional view, not used in the figures.)"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.chdir(ROOT)
from scene_engine import *
from exp.bank import sample_foot
from exp.scenarios import COAL_MORPH

OUT = os.environ.get("RENDER_OUT", "results/renders")


def superior(foot, act, name, show=("calc", "cart", "coal"), W=420, H=310, ortho=31.0):
    fn = f"{OUT}/{name}.npz"
    if os.path.exists(fn):
        return
    objs, L, lo, h = engine_scene(foot, act, field=None, show=show)
    for o in objs:
        if o.name == "cart":
            o.color = np.array((0.62, 0.78, 0.90))
        if o.name in ("calc", "nav", "cub"):
            o.color = np.array((0.90, 0.89, 0.87)); o.spec = 0.05; o.rim = 0.05
    cx, cy = FACET_C; ortho *= FACET_SCALE      # engine-dependent view (figsrc/rbview.py)
    img, al = render(objs, eye=(cx, cy, 200.0), target=(cx, cy, 0.0), up=(0, 1, 0), W=W, H=H, ortho=ortho, ss=2,
                     lights="camera")
    np.savez_compressed(fn, img=img.astype(np.float32), alpha=al.astype(np.float32))
    print("saved", name, flush=True)


def oblique(foot, act, name, W=520, H=400):
    fn = f"{OUT}/{name}.npz"
    if os.path.exists(fn):
        return
    objs, L, lo, h = engine_scene(foot, act, field=None, show=("calc", "nav", "cub", "cart", "coal", "talus"),
                                  talus_alpha=0.55)
    for o in objs:
        if o.name in ("calc", "nav", "cub"):
            o.color = np.array((0.90, 0.88, 0.84)); o.spec = 0.08
    img, al = render(objs, **__import__('rbview').OBLIQUE, up=(0, 0, 1), W=W, H=H, fov=26,
                     ss=2)
    np.savez_compressed(fn, img=img.astype(np.float32), alpha=al.astype(np.float32))
    print("saved", name, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["feet", "coal", "child"]
    if "oblique" in which:
        oblique(Foot(**COAL_MORPH["Type I"]), Action(), "icon_oblique_typeI")
    if "feet" in which:
        rng = np.random.default_rng(7)
        for k in range(3):
            superior(sample_foot(rng, p_intact=1.0), Action(), f"icon_foot{k + 1}", show=("calc", "cart"))
    if "child" in which:
        # the virtual child of Fig. 7a (type II coalition, 8 deg varus): pre-operative and after complete resection
        import json as _json
        ch = _json.load(open("results/eval_maps.json"))["child"]
        foot = Foot(**ch["foot"])
        superior(foot, Action(), "icon_child_pre")
        superior(foot, Action(resect=1), "icon_child_resect")
    if "coal" in which:
        superior(Foot(**COAL_MORPH["Normal"]), Action(), "icon_coal_none")
        for t in ["Type I", "Type II", "Type III", "Type IV"]:
            superior(Foot(**COAL_MORPH[t]), Action(), "icon_coal_" + t.split()[1])
