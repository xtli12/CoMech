"""Rendered field maps of the held-out foot of Fig. 5a: superior view of the calcaneal articular cartilage coloured by
von Mises stress and inferior view of the talus coloured by talar displacement (calcaneus-fixed frame), from FE
(engine grid, nodal-averaged) and from CoMech queried continuously, and their absolute difference."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.chdir(ROOT)
from scene_engine import *
from exp.common import get_pred, geo_of
from ml.data import Norm, GRID
from nfquery import query
import matplotlib.pyplot as plt

OUT = os.environ.get("RENDER_OUT", "results/renders")


def main():
    os.makedirs(OUT, exist_ok=True)
    M = np.load("results/map_heldout.npz", allow_pickle=True)
    foot = Foot(**json.loads(str(M["foot"]))); a = M["act"].astype(np.float32)
    act = Action(resect=1, d_mid=float(a[1]), d_post=float(a[2]))
    cells = M["cells"]; lab = M["lab"]
    norm = Norm.cached(); pred = get_pred("comech_s0")
    pre = geo_of(foot, Action())[0].astype(np.float32)
    xe, ye, ze = GRID.xe, GRID.ye, GRID.ze
    full = lambda v: (lambda F: (F.__setitem__(cells, v), F)[1])(np.zeros(GRID.ncell, np.float32))
    cart = np.isin(lab, [C_ANT, C_MID, C_POST]); talm = lab == TALUS
    fe_s = field_sampler(full(M["vm_fe"]), xe, ye, ze, valid=cart)
    um = np.linalg.norm(M["u_fe"], axis=1)
    fe_u = field_sampler(full(um), xe, ye, ze, valid=talm)

    def cm_s(P):
        vm, _ = norm.fields_inv(query(pred, pre, a, P)); return vm.astype(np.float32)

    def cm_u(P):
        _, u = norm.fields_inv(query(pred, pre, a, P)); return np.linalg.norm(u, axis=1).astype(np.float32)
    vmax = float(np.percentile(M["vm_fe"][np.isin(lab[cells], [C_ANT, C_MID, C_POST])], 99))
    umax = float(np.percentile(um[lab[cells] == TALUS], 99))
    W, H, ortho = 700, 520, 30.0
    cx, cy = FACET_C; ortho *= FACET_SCALE      # engine-dependent view (figsrc/rbview.py)
    top = dict(eye=(cx, cy, 200.0), target=(cx, cy, 0.0), up=(0, 1, 0), W=W, H=H, ortho=ortho, ss=2, lights="camera")
    import rbview
    tx, ty = rbview.TALUS_C; tortho = 30.0 * rbview.TALUS_SCALE
    bot = dict(eye=(tx, ty, -200.0), target=(tx, ty, 0.0), up=(0, -1, 0), W=W, H=H, ortho=tortho, ss=2, lights="camera")
    jobs = [("stress_fe", fe_s, vmax, STRESS, top, ("calc", "cart")), ("stress_cm", cm_s, vmax, STRESS, top, ("calc", "cart")),
            ("stress_err", lambda P: np.abs(cm_s(P) - fe_s(P)), vmax / 2, ERR, top, ("calc", "cart")),
            ("disp_fe", fe_u, umax, DISP, bot, ("talus",)), ("disp_cm", cm_u, umax, DISP, bot, ("talus",)),
            ("disp_err", lambda P: np.abs(cm_u(P) - fe_u(P)), umax / 2, ERR, bot, ("talus",))]
    meta = dict(vmax=vmax, umax=umax, ext_top=[cx - ortho, cx + ortho, cy - ortho * H / W, cy + ortho * H / W],
                ext_bot=[tx - tortho, tx + tortho, ty - tortho * H / W, ty + tortho * H / W])
    for name, fld, vm_, cmap, cam, show in jobs:
        fn = f"{OUT}/heldout_{name}.npz"
        if os.path.exists(fn):
            continue
        t0 = time.time()
        objs, L, lo, h = engine_scene(foot, act, field=None, show=show, talus_alpha=1.0)
        for o in objs:
            if o.name in ("cart", "talus"):
                o.scalar, o.cmap, o.vmin, o.vmax = fld, cmap, 0.0, vm_
                o.color = np.array((0.90, 0.89, 0.87)) if o.name == "talus" else o.color
            if o.name == "calc":
                o.color = np.array((0.90, 0.89, 0.87)); o.spec = 0.05; o.rim = 0.05
        img, al = render(objs, **cam)
        np.savez_compressed(fn, img=img.astype(np.float32), alpha=al.astype(np.float32), **{k: np.array(v) for k, v in meta.items()})
        print(name, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
