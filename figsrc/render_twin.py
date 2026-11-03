"""Point twin of P05-R (Fig. 8): engine foot with the coalition read from CT; superior view of the calcaneal
articular surfaces before resection (coalition footprint shown) and after complete resection (FE von Mises stress
on the residual cartilage; the defect is bare bone)."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.chdir(ROOT)
from scene_engine import *
from exp.e_ct_twins import read_csv
from exp.e_ct_twins_fe import point_twin
import fe.simulate as FS

OUT = os.environ.get("RENDER_OUT", "results/renders")


def main(foot_id="P05-R"):
    os.makedirs(OUT, exist_ok=True)
    fn = f"data/hires/twin_{foot_id}_resect.npz"
    # real-bone engine: the server's high-resolution job writes twin_<id>_resect_f1.00.npz (analysis grid, as the
    # parametric file) and _f0.50.npz, each with the twin's foot; the CT readings are then not needed (Claude, 2026-11-01)
    alt = f"data/hires/twin_{foot_id}_resect_f{os.environ.get('TWIN_RES', '1.00')}.npz"
    if not os.path.exists(fn) and os.path.exists(alt):
        fn = alt
    z = np.load(fn, allow_pickle=True) if os.path.exists(fn) else None
    if z is not None and "foot" in z.files:
        f = Foot(**json.loads(str(z["foot"])))
    else:
        r = [x for x in read_csv("data/ct/ct_readings.csv") if x["foot"] == foot_id][0]
        f = point_twin(r)
    if not os.path.exists(fn):
        res = FS.simulate(f, Action(resect=1), tol=1e-6); g = FS.get_grid()
        np.savez_compressed(fn, vm=res["vm"].astype(np.float32), lab=res["lab"].astype(np.int8), xe=g.xe, ye=g.ye, ze=g.ze,
                            eff=json.dumps(res["eff"]))
    W, H, ortho = 760, 560, 31.0
    cx, cy = FACET_C; ortho *= FACET_SCALE      # engine-dependent view (figsrc/rbview.py)
    cam = dict(eye=(cx, cy, 200.0), target=(cx, cy, 0.0), up=(0, 1, 0), W=W, H=H, ortho=ortho, ss=2, lights="camera")
    ext = [cx - ortho, cx + ortho, cy - ortho * H / W, cy + ortho * H / W]
    for tag in ["pre", "resect"]:
        out = f"{OUT}/twin_{foot_id}_{tag}.npz"
        if os.path.exists(out):
            continue
        t0 = time.time()
        if tag == "pre":
            objs, L, lo, h = engine_scene(f, Action(), field=None, show=("calc", "cart", "coal"))
            for o in objs:
                if o.name == "cart":
                    o.color = np.array((0.62, 0.76, 0.88))
        else:
            fld, eff = fe_field(fn)
            objs, L, lo, h = engine_scene(f, Action(resect=1), field=fld, vmax=1.6, show=("calc", "cart"))
        for o in objs:
            if o.name == "calc":
                o.color = np.array((0.90, 0.89, 0.87)); o.spec = 0.05; o.rim = 0.05
        img, al = render(objs, **cam)
        np.savez_compressed(out, img=img.astype(np.float32), alpha=al.astype(np.float32), ext=np.array(ext))
        print(tag, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
