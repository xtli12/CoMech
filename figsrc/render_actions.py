"""Renderings of the surgical actions of the data engine on the virtual child of Fig. 7a (type II coalition), for
the action panels of Figs. 1b and 4a: pre-operative coalition, complete resection (facet defect), calcaneal
osteotomy, interposition, subtalar fusion and load.  Pure engine geometry (no fields), one orthographic camera for
all actions (near-superior view from posteromedial, anterior to the right) so that the panels differ only by the
action.  The talus is removed except for the load panel, where it is seated on the calcaneus.  Each npz stores the
image, its alpha and the camera, so that figure scripts can project 3D points (cut line, coalition footprint, load
arrows) onto the picture.  Run:  python figsrc/render_actions.py   (about 20 s per action)."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from render3d import Obj, SmoothSurface, render
from hiresgeo import occupancy_volumes
from fe.geometry import *
from scene_engine import crop_box, COAL_COL
import rbview

OUT = os.environ.get("RENDER_OUT", os.path.join(ROOT, "results", "renders"))
GROUPS = {"calc": [CALC], "cart": [C_ANT, C_MID, C_POST], "coal": [COAL, FIB], "interp": [INTERP],
          "fuse": [FUSE], "talus": [TALUS]}
COLOR = {"calc": (0.90, 0.89, 0.87), "cart": (0.62, 0.78, 0.90), "coal": COAL_COL, "interp": (0.86, 0.50, 0.04),
         "fuse": (0.58, 0.60, 0.63), "talus": (0.84, 0.82, 0.78)}
SIGMA = {"calc": rbview.SIGMA["bone"], "talus": rbview.SIGMA["bone"], "cart": rbview.SIGMA["cart"], "coal": rbview.SIGMA["coal"],
         "interp": rbview.SIGMA["interp"], "fuse": rbview.SIGMA["fuse"]}
TILT, ROLL = 20.0, 0.0                    # camera tilted from the superior view toward lateral (deg)
TARGET = rbview.ACT_TARGET.copy(); DIST = 220.0
UP = (0.0, 1.0, 0.0)                      # medial up, anterior to the right (as in Figs. 2b, 5a, 7a, 8b)
W, H, ORTHO = 440, 250, rbview.ACT_ORTHO
TALUS_ALPHA = 1.0


def camera():
    tl = np.deg2rad(TILT)
    d = np.array([0.0, -np.sin(tl), np.cos(tl)])
    return TARGET + DIST * d, TARGET


def project(P, eye=None, target=None, up=UP, w=W, h=H, ortho=ORTHO):
    """Pixel coordinates (col, row) of 3D points for the orthographic camera of render3d.camera_rays."""
    if eye is None:
        eye, target = camera()
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    P = np.atleast_2d(P) - eye
    x = P @ r / ortho; y = P @ u / ortho
    return np.c_[(x / 2 + 0.5) * w, (0.5 - y * w / (2 * h)) * h]


def scene(foot, act, show, h=0.3):
    occ, lo, h = occupancy_volumes(foot, act, {k: GROUPS[k] for k in show}, h=h)
    objs = []
    for k in show:
        m = occ[k]
        if not (m > 0.5).any():
            continue
        sl, o = crop_box(m > 0.02, lo, h)
        ob = Obj(k, SmoothSurface(m[sl], o, h, sigma=SIGMA[k], iso=rbview.ISO.get(k, 0.5), nsigma=rbview.NSIGMA.get(k)), color=COLOR[k], alpha=TALUS_ALPHA if k == "talus" else 1.0)
        ob.spec = 0.06 if k in ("calc", "talus") else 0.30
        ob.rim = 0.06 if k in ("calc", "talus") else 0.12
        objs.append(ob)
    return objs


def lights(eye, target, up=UP):
    """Camera-relative lighting: key from the upper left, a lateral fill that keeps the side walls readable, and a
    weak frontal light."""
    e_, t_, u_ = (np.asarray(v, float) for v in (eye, target, up))
    f_ = t_ - e_; f_ /= np.linalg.norm(f_); r_ = np.cross(f_, u_); r_ /= np.linalg.norm(r_); uu = np.cross(r_, f_)
    key = -f_ * 0.85 - r_ * 0.45 + uu * 0.55
    fill = -f_ * 0.55 + r_ * 0.25 - uu * 0.80
    front = -f_
    return [(key, (1.0, 0.97, 0.93), 0.78, True), (fill, (0.90, 0.93, 1.0), 0.42, False),
            (front, (1.0, 1.0, 1.0), 0.18, False)]


def child_foot():
    return Foot(**json.load(open(os.path.join(ROOT, "results", "eval_maps.json")))["child"]["foot"])


ACTIONS = {
    "pre": (Action(), ("calc", "cart", "coal")),
    "resect": (Action(resect=1), ("calc", "cart")),
    "osteo": (Action(resect=1, osteo=10.0), ("calc", "cart")),
    "interp": (Action(resect=1, interp=1), ("calc", "cart", "interp")),
    "fusion": (Action(resect=1, fusion=1), ("calc", "cart", "fuse")),
    "load": (Action(resect=1, load=1.5), ("calc", "cart", "talus")),
}


def main(which=None):
    os.makedirs(OUT, exist_ok=True)
    foot = child_foot()
    eye, tg = camera()
    for name, (act, show) in ACTIONS.items():
        if which and name not in which:
            continue
        fn = f"{OUT}/act_{name}.npz"
        if os.path.exists(fn):
            continue
        objs = scene(foot, act, show)
        img, al = render(objs, eye=eye, target=tg, up=UP, W=W, H=H, ortho=ORTHO, ss=2, lights=lights(eye, tg))
        np.savez_compressed(fn, img=img.astype(np.float32), alpha=al.astype(np.float32), eye=eye, target=tg,
                            ortho=ORTHO, act=json.dumps(act.__dict__))
        print("saved", fn, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
