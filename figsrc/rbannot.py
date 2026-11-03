"""Projected annotations of the action renderings for the real-bone engine (Claude, 2026-11-01).

The parametric figure code draws the coalition outline, the osteotomy cut and the load arrows from the analytic
geometry (FootGeom.coalition_xy / z_joint, fe.geometry.x_cut, canonical coordinates scaled by world_mm).  The
real-bone engine has no analytic surfaces, so the same elements are measured on its label raster (the geometry the
renderings are made from).  Every function returns world millimetres (x anterior, y medial, z up)."""
import numpy as np
from scipy import ndimage
import fe.geometry as FG
from fe.geometry import Action, FootGeom, CALC, TALUS, COAL, FIB, C_ANT, C_MID, C_POST
import rbgeom as _G
from hiresgeo import _labels, _lattice

VIEW = np.array([0.0, -np.sin(np.deg2rad(20.0)), np.cos(np.deg2rad(20.0))])   # towards the camera of render_actions


def _T():
    return _G.Template.get()


def _contours(xs, ys, mask):
    """Closed iso-lines (x, y) of a boolean mask on the axes xs, ys."""
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    cs = plt.contour(xs, ys, mask.T.astype(float), levels=[0.5])
    segs = [s for s in cs.allsegs[0] if len(s) > 2]
    plt.close(fig)
    return segs


def _column_z(xs, ys, zs, m3, top=True):
    """Height of the highest (top=True) or lowest voxel of m3 (nx, ny, nz) in every column, NaN where empty."""
    any_ = m3.any(2)
    if top:
        k = m3.shape[2] - 1 - np.argmax(m3[:, :, ::-1], 2)
    else:
        k = np.argmax(m3, 2)
    z = np.where(any_, zs[k], np.nan)
    return z


def _fill_nearest(z):
    """NaN cells take the value of the nearest valid cell."""
    bad = np.isnan(z)
    if not bad.any() or bad.all():
        return z
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def _sample(xs, ys, Z, x, y):
    i = np.interp(x, xs, np.arange(len(xs))); j = np.interp(y, ys, np.arange(len(ys)))
    return ndimage.map_coordinates(Z, [i, j], order=1, mode="nearest")


def _joint_lattice(foot, act, h=0.5):
    """Labels of the subtalar region of the foot (world lattice) and its axes."""
    xs, ys = FG.map_axes(foot, h)
    T = _T(); s = foot.scale
    zl = min(T.facets[k]["xyz_lo"][2] for k in T.facets) - 12.0
    zh = max(T.facets[k]["xyz_hi"][2] for k in T.facets) + 12.0
    zs = np.arange(s * zl, s * zh, h)
    return xs, ys, zs, _lattice(foot, act, xs, ys, zs)


def coalition_outline(foot, h=0.25):
    """Outline(s) of the pre-operative coalition footprint, placed on the calcaneal side of the bridge (the surface
    left after resection), as (n, 3) arrays."""
    xs, ys = FG.map_axes(foot, h)
    tv = FootGeom(foot, Action()).top_view(xs, ys)
    xl, yl, zl, L = _joint_lattice(foot, Action())
    zb = _fill_nearest(_column_z(xl, yl, zl, np.isin(L, [COAL, FIB]), top=False))
    out = []
    for seg in _contours(xs, ys, tv["coal"]):
        z = _sample(xl, yl, zb, seg[:, 0], seg[:, 1])
        out.append(np.c_[seg, z])
    return out


def mask_centroid(foot, which, exclude_coal=False, h=0.25):
    """Centroid of a top-view footprint ("coal", "post", "mid", "ant") at the top of that tissue."""
    from dataclasses import replace
    xs, ys = FG.map_axes(foot, h)
    tv = FootGeom(foot, Action()).top_view(xs, ys)
    if which == "coal":
        m = tv["coal"]; labs = [COAL, FIB]; g = foot
    else:
        intact = replace(foot, ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0)
        m = FootGeom(intact, Action()).top_view(xs, ys)[which]
        if exclude_coal:
            m = m & ~tv["coal"]
        labs = [{"post": C_POST, "mid": C_MID, "ant": C_ANT}[which]]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xm, ym = float(X[m].mean()), float(Y[m].mean())
    xl, yl, zl, L = _joint_lattice(foot, Action())
    zt = _fill_nearest(_column_z(xl, yl, zl, np.isin(L, labs), top=True))
    return np.array([xm, ym, float(_sample(xl, yl, zt, [xm], [ym])[0])])


def _x_cut_world(foot, z):
    T = _T(); s = foot.scale
    return T.c0[0] + s * (T.x_cut(np.asarray(z, float) / s) - T.c0[0])


def osteotomy_lines(foot, act, h=0.3):
    """Visible trace of the osteotomy plane on the calcaneus (the parts of the section outline that face the camera
    of the action renderings: the superior and lateral surfaces) and the translation arrow on top of the tuberosity
    (from the tuberosity's pre-operative position to the translated one).  Returns (list of (n, 3), (2, 3))."""
    T = _T(); s = foot.scale
    ys = np.arange(-40.0, 50.0, h); zs = np.arange(0.0, s * T.z_frag_top, h)
    Yg, Zg = np.meshgrid(ys, zs, indexing="ij")
    Xg = _x_cut_world(foot, Zg) + 0.3                      # static side of the cut
    L = _labels(foot, act, Xg, Yg, Zg)
    m = ndimage.binary_opening(L == CALC, iterations=1)
    lab, n = ndimage.label(m)
    if n > 1:                                              # keep the calcaneus body only
        m = lab == (1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
    sm = ndimage.gaussian_filter(m.astype(float), 1.5)
    gy, gz = np.gradient(sm, h, h)
    lines = []
    for seg in _contours(ys, zs, m):
        ny = -_sample(ys, zs, gy, seg[:, 0], seg[:, 1]); nz = -_sample(ys, zs, gz, seg[:, 0], seg[:, 1])
        vis = (ny * VIEW[1] + nz * VIEW[2]) > 0.15 * np.hypot(ny, nz)
        P = np.c_[_x_cut_world(foot, seg[:, 1]), seg[:, 0], seg[:, 1]]
        # split into visible runs
        run = []
        for p, v in zip(P, vis):
            if v:
                run.append(p)
            elif len(run) > 3:
                lines.append(np.array(run)); run = []
            else:
                run = []
        if len(run) > 3:
            lines.append(np.array(run))
    # translation arrow: 10 mm behind the cut at the tuberosity's top, centred on the translated fragment
    t = float(act.osteo)
    xa = float(_x_cut_world(foot, s * T.z_cut0)) - 12.0 * s
    yy = np.arange(-40.0, 50.0, h); zz = np.arange(0.0, s * T.z_frag_top, h)
    Y2, Z2 = np.meshgrid(yy, zz, indexing="ij")
    L2 = _labels(foot, act, np.full_like(Y2, xa), Y2, Z2) == CALC
    cols = L2.any(1)
    if cols.any():
        ztop = np.where(cols, zz[L2.shape[1] - 1 - np.argmax(L2[:, ::-1], 1)], -np.inf)
        zmax = ztop.max(); top = ztop > zmax - 4.0
        yc = float(yy[top].mean())
    else:
        zmax, yc = s * T.z_cut0, 0.0
    arrow = np.array([[xa, yc - t, zmax + 1.5], [xa, yc, zmax + 1.5]])
    return lines, arrow


def load_arrows(foot, act, n=3, length=15.0, gap=1.5, h=0.5):
    """Downward arrows onto the talar dome: n positions spread over the part of the dome within 3 mm of its top."""
    T = _T(); s = foot.scale
    xs = np.arange(-110.0, 20.0, h); ys = np.arange(-40.0, 50.0, h); zs = np.arange(20.0, 140.0, h)
    L = _lattice(foot, act, xs, ys, zs) == TALUS
    zt = _column_z(xs, ys, zs, L, top=True)
    zmax = np.nanmax(zt)
    dome = zt > zmax - 3.0
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x0, x1 = np.percentile(X[dome], [15, 85]); yc = float(np.median(Y[dome]))
    out = []
    for xx in np.linspace(x0, x1, n):
        i = np.argmin(np.abs(xs - xx)); j = np.argmin(np.abs(ys - yc))
        col = zt[i, max(j - 2, 0):j + 3]
        z0 = float(np.nanmax(col)) if np.isfinite(col).any() else zmax
        out.append(np.array([[xx, yc, z0 + gap + length], [xx, yc, z0 + gap]]))
    return out
