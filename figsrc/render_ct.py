"""Rendered reconstructions of a real foot from its CT export (P05-R, zoomed series): assembled medial view and an
exploded view in which the talus is lifted off the calcaneus and the facing articular surfaces are coloured by the
gap between the segmented talus and calcaneus (no visible joint space = contact of the segmented surfaces).
The export has no pixel spacing; millimetres are nominal (calcaneus scaled to 73 mm)."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from render3d import *
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap

SEG = os.path.join(os.environ.get("CT_READ_DIR", "data/ct/read"), "seg39c.npy")
OUT = os.path.join(ROOT, "results/renders")
GAP = LinearSegmentedColormap.from_list("gap", ["#b2281b", "#e8602c", "#f5b041", "#d9e8c9", "#69b7c4", "#2c7fb8"])
IVORY = (0.95, 0.91, 0.83); IVORY2 = (0.92, 0.89, 0.83); GHOST = (0.80, 0.85, 0.92)


def load(calc_len_mm=73.0):
    seg = np.load(SEG)
    hv = 0.6                                                    # px per voxel (anatomical volume)
    cl = np.ptp(np.argwhere(seg == 2)[:, 0]) * hv               # calcaneal length in px
    p = calc_len_mm / cl                                         # nominal mm per px
    return seg, hv * p


def surf(mask, h, sigma=1.6, ds=1, pad=6):
    idx = np.argwhere(mask); a = np.maximum(idx.min(0) - pad, 0); b = np.minimum(idx.max(0) + pad + 1, np.array(mask.shape))
    sub = mask[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    if ds > 1:
        sub = ndimage.zoom(sub.astype(np.float32), 1 / ds, order=1) > 0.5
    return SmoothSurface(sub, a * h, h * ds, sigma=sigma)


def joint_field(J, other_dist, h, shift=(0, 0, 0), near=3.0):
    """f(P): joint-line contrast J (black top-hat of the CT, max over 7 voxels within the talocalcaneal band) at P,
    for surface points within `near` voxels of the opposing bone; NaN elsewhere (bone colour)."""
    VJ = Volume(J.astype(np.float32), (0, 0, 0), h); VD = Volume(other_dist.astype(np.float32), (0, 0, 0), h)
    sh = np.asarray(shift, float)

    def f(P):
        Q = P - sh
        v = VJ.sample(Q, box_dist=False); d = VD.sample(Q, box_dist=False)
        v[d > near] = np.nan
        return v
    return f


def main(views=("medial", "exploded", "lateral")):
    os.makedirs(OUT, exist_ok=True)
    seg, h = load()
    tal, cal = seg == 1, seg == 2
    c = np.argwhere(tal | cal).mean(0) * h
    t0 = time.time()
    S = {"talus": surf(tal, h), "calc": surf(cal, h), "nav": surf(seg == 3, h), "mid": surf(np.isin(seg, [4, 6]), h, ds=2),
         "tib": surf(seg == 5, h, ds=2), "fib": surf(seg == 7, h, ds=2)}
    print("surfaces", round(time.time() - t0, 1), flush=True)
    J = np.load(os.path.join(os.environ.get("CT_READ_DIR", "data/ct/read"), "jv39s.npy")).astype(np.float32)
    dT = ndimage.distance_transform_edt(~tal); dC = ndimage.distance_transform_edt(~cal)
    for view in views:
        fn = f"{OUT}/ct_P05R_{view}.npz"
        if os.path.exists(fn):
            continue
        t0 = time.time()
        if view in ("medial", "lateral"):
            objs = [Obj("talus", S["talus"], color=IVORY), Obj("calc", S["calc"], color=IVORY2), Obj("nav", S["nav"], color=IVORY2),
                    Obj("mid", S["mid"], color=(0.88, 0.86, 0.82)),
                    Obj("tib", S["tib"], color=GHOST, alpha=0.20), Obj("fib", S["fib"], color=GHOST, alpha=0.20)]
            sgn = 1 if view == "medial" else -1
            eye = c + np.array([-25, sgn * 280, 70]); tgt = c + np.array([-6, 0, -2]); up = (0, 0, 1); ortho = 84
        else:
            # superior view of the calcaneus with the talus removed (same convention as the engine maps:
            # anterior to the right, medial up); the surface facing the talus is coloured by joint-line contrast
            cc = np.argwhere(cal).mean(0) * h
            objs = [Obj("calc", surf(cal, h, sigma=2.2), color=(0.90, 0.89, 0.87), scalar=joint_field(J, dT, h, near=5.0), cmap=GAP,
                        vmin=0, vmax=60, spec=0.05, rim=0.05)]
            eye = cc + np.array([0, 0, 260]); tgt = cc; up = (0, 1, 0); ortho = 46
        img, al = render(objs, eye=eye, target=tgt, up=up, W=900, H=640, ortho=ortho, ss=2, lights="camera")
        np.savez_compressed(fn, img=img.astype(np.float32), alpha=al.astype(np.float32), h=h)
        print(view, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("medial", "exploded", "lateral"))
