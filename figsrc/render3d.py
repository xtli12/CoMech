"""Small NumPy/SciPy renderer for publication figures: sphere tracing of smoothed signed-distance fields,
Blinn-Phong shading with soft shadows, ambient occlusion, rim light, translucent layers and per-surface colour
mapping of scalar fields (FE or neural-field stress).  No OpenGL needed.

Coordinates are millimetres.  Each object is a signed-distance volume (negative inside) on a uniform grid."""
import numpy as np
from scipy import ndimage


def sdf_from_mask(mask, h, smooth=1.0):
    """Signed distance (mm, negative inside) of a binary mask on a uniform grid of spacing h.  Far from the surface
    the Euclidean distance transform is used; within two voxels of the surface the distance is taken from a
    Gaussian-smoothed occupancy (sigma = smooth voxels), which places the surface with sub-voxel accuracy and
    removes the staircase of the voxel boundary."""
    m = mask.astype(bool)
    if not m.any():
        return np.full(m.shape, 1e3, np.float32)
    d_out = ndimage.distance_transform_edt(~m) * h
    d_in = ndimage.distance_transform_edt(m) * h
    s = (d_out - d_in).astype(np.float32)
    s[m] += 0.5 * h; s[~m] -= 0.5 * h
    if smooth:
        occ = ndimage.gaussian_filter(m.astype(np.float32), smooth)
        near = (0.5 - occ) * smooth * np.sqrt(2 * np.pi) * h
        w = np.clip(1.0 - (np.abs(s) - 1.5 * h) / (1.5 * h), 0, 1)       # 1 within 1.5 voxels, 0 beyond 3 voxels
        s = (w * near + (1 - w) * ndimage.gaussian_filter(s, 0.7)).astype(np.float32)
    return s


def sdf_from_occupancy(occ, h, iso=0.5, smooth=1.0):
    """SDF from a soft occupancy field (e.g. partial-volume fractions or a smoothed CT bone probability)."""
    if smooth:
        occ = ndimage.gaussian_filter(occ.astype(np.float32), smooth)
    return sdf_from_mask(occ > iso, h, smooth=smooth)


class Volume:
    """A scalar volume on a uniform grid: origin (mm, 3), spacing h (mm)."""

    def __init__(self, data, origin, h):
        self.data = np.ascontiguousarray(data, np.float32); self.o = np.asarray(origin, float); self.h = float(h)

    def sample(self, P, cval=1e3, order=1, box_dist=True):
        idx = ((P - self.o) / self.h).T
        v = ndimage.map_coordinates(self.data, idx, order=order, mode="nearest").astype(np.float32)
        if box_dist:
            # outside the grid: distance to the grid box is a lower bound of the distance to any surface inside it
            hi = self.o + (np.array(self.data.shape) - 1) * self.h
            q = np.maximum(np.maximum(self.o - P, P - hi), 0)
            db = np.linalg.norm(q, axis=1).astype(np.float32)
            out = db > 0
            v[out] = np.maximum(db[out] + self.h, v[out] if cval is None else db[out] + self.h)
        return v

    def coarse(self, P):
        if hasattr(self, "coarse_data"):
            idx = ((P - self.o) / self.h).T
            v = ndimage.map_coordinates(self.coarse_data, idx, order=1, mode="nearest").astype(np.float32)
            hi = self.o + (np.array(self.data.shape) - 1) * self.h
            q = np.maximum(np.maximum(self.o - P, P - hi), 0); db = np.linalg.norm(q, axis=1).astype(np.float32)
            out = db > 0; v[out] = np.maximum(db[out] + self.h, v[out])
            return v
        return self.sample(P)

    def grad(self, P, eps=None):
        e = eps or self.h * 0.75
        g = np.zeros_like(P)
        for k in range(3):
            d = np.zeros(3); d[k] = e
            g[:, k] = self.sample(P + d) - self.sample(P - d)
        n = np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
        return g / n


class SmoothSurface(Volume):
    """Implicit surface 0.5-level of a Gaussian-smoothed occupancy (sub-voxel, no staircase), with the Euclidean
    distance of the binary mask as a conservative step bound for sphere tracing far from the surface."""

    def __init__(self, mask, origin, h, sigma=1.6, iso=0.5, nsigma=None):
        # iso < 0.5 keeps thin layers (e.g. 1-mm cartilage on a 0.5-mm raster) closed at a smoothing that removes the
        # raster steps; the surface moves outwards by about (0.5 - iso) * sigma * sqrt(2 pi) * h (Claude, 2026-11-01)
        self.iso = float(iso)
        soft = mask.dtype.kind == "f"
        m = mask > self.iso if soft else mask.astype(bool)
        d = (ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)) * h
        super().__init__(d.astype(np.float32), origin, h)
        self.occ = np.ascontiguousarray(ndimage.gaussian_filter(mask.astype(np.float32), sigma), np.float32)
        self.scale = sigma * np.sqrt(2 * np.pi) * h          # occupancy slope -> distance near the surface
        self.band = (2.5 + sigma) * h
        self.coarse_data = np.ascontiguousarray(ndimage.gaussian_filter(self.data, 1.5), np.float32)
        # nsigma: shading normals from a more strongly smoothed occupancy (removes the shading ripples of a raster
        # staircase on thin layers without moving or opening the surface; Claude, 2026-11-01)
        g = np.gradient(ndimage.gaussian_filter(self.occ, 0.7) if nsigma is None else
                        ndimage.gaussian_filter(mask.astype(np.float32), nsigma))
        self.G = [np.ascontiguousarray(-gi, np.float32) for gi in g]   # outward normal field (unnormalised)

    def implicit(self, P):
        idx = ((P - self.o) / self.h).T
        return ((self.iso - ndimage.map_coordinates(self.occ, idx, order=1, mode="constant", cval=0.0)) * self.scale).astype(np.float32)

    def grad(self, P, eps=None):
        idx = ((P - self.o) / self.h).T
        g = np.stack([ndimage.map_coordinates(G, idx, order=1, mode="nearest") for G in self.G], 1)
        return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)

    def sample(self, P, cval=1e3, order=1, box_dist=True):
        d = Volume.sample(self, P, cval, order, box_dist)
        nb = np.abs(d) < self.band
        if nb.any():
            d[nb] = self.implicit(P[nb])
        return d


class Obj:
    def __init__(self, name, sdf: Volume, color=(0.85, 0.82, 0.76), alpha=1.0, scalar=None, cmap=None, vmin=0, vmax=1,
                 spec=0.25, shininess=40, rim=0.25, clip=None):
        self.name, self.sdf, self.color, self.alpha = name, sdf, np.array(color, float), alpha
        self.scalar, self.cmap, self.vmin, self.vmax = scalar, cmap, vmin, vmax
        self.spec, self.shin, self.rim = spec, shininess, rim
        self.clip = clip          # optional half-space (point, normal): only the side n.(x-p) <= 0 is kept


def _clip_sdf(o, P, d):
    if o.clip is None:
        return d
    p, n = o.clip
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    return np.maximum(d, (P - np.asarray(p, float)) @ n)


def camera_rays(eye, target, up, W, H, fov=None, ortho=None):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    j, i = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x = (j / W - 0.5) * 2; y = -(i / H - 0.5) * 2 * H / W
    if ortho:
        O = eye[None] + (x.ravel()[:, None] * r + y.ravel()[:, None] * u) * ortho
        D = np.repeat(f[None], W * H, 0)
    else:
        t = np.tan(np.deg2rad(fov or 20) / 2)
        D = f[None] + (x.ravel()[:, None] * r + y.ravel()[:, None] * u) * t
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        O = np.repeat(eye[None], W * H, 0)
    return O, D, f


def trace(o: Obj, O, D, tmax, steps=260, eps=0.02):
    """Sphere tracing with a minimum step and bisection refinement at the first sign change; returns hit distance."""
    n = len(O); t = np.zeros(n, np.float32); hit = np.full(n, np.inf, np.float32)
    act = np.arange(n)
    hmin = o.sdf.h * 0.3
    dprev = np.full(n, 1e3, np.float32)
    for _ in range(steps):
        if len(act) == 0:
            break
        P = O[act] + D[act] * t[act, None]
        d = _clip_sdf(o, P, o.sdf.sample(P))
        done = d < eps
        if done.any():
            a = act[done]
            # bisection between the previous point (outside) and the current one
            lo = np.maximum(t[a] - np.maximum(dprev[a], hmin) * 0.9, 0); hi = t[a].copy()
            for _b in range(6):
                mid = 0.5 * (lo + hi)
                dm = _clip_sdf(o, O[a] + D[a] * mid[:, None], o.sdf.sample(O[a] + D[a] * mid[:, None]))
                ins = dm < 0
                hi = np.where(ins, mid, hi); lo = np.where(ins, lo, mid)
            hit[a] = hi
        dprev[act] = d
        t[act] += np.maximum(d, hmin) * 0.9
        keep = (~done) & (t[act] < tmax[act])
        act = act[keep]
    return hit


def soft_shadow(objs, P, L, k=10.0, tmax=120.0, steps=48):
    res = np.ones(len(P), np.float32); t = np.full(len(P), 0.8, np.float32)
    act = np.arange(len(P))
    for _ in range(steps):
        if len(act) == 0:
            break
        Q = P[act] + L[None] * t[act, None]
        d = np.min(np.stack([_clip_sdf(o, Q, o.sdf.coarse(Q)) for o in objs if o.alpha >= 0.99] or [np.full(len(Q), 1e3)]), 0)
        res[act] = np.minimum(res[act], np.clip(k * d / t[act], 0, 1))
        t[act] += np.clip(d, 0.3, 6.0)
        keep = (d > 0.01) & (t[act] < tmax)
        res[act[~keep & (d <= 0.01)]] = 0.0
        act = act[keep]
    return res


def ambient_occlusion(objs, P, N, dists=(1.2, 2.5, 4.5, 8.0)):
    ao = np.zeros(len(P), np.float32)
    for i, dd in enumerate(dists):
        Q = P + N * dd
        d = np.min(np.stack([_clip_sdf(o, Q, o.sdf.coarse(Q)) for o in objs if o.alpha >= 0.99]), 0)
        ao += (dd - np.clip(d, -dd, dd)) / dd * (0.5 ** i)
    return np.clip(1 - 0.55 * ao, 0.25, 1.0)


def shade(o, objs, P, D, lights, ambient=0.33, shadows=True, ao=True):
    N = o.sdf.grad(P)
    if o.clip is not None:          # flat cut face where the clip plane is active
        p, n = o.clip; n = np.asarray(n, float) / np.linalg.norm(n)
        oncut = np.abs((P - np.asarray(p, float)) @ n) < o.sdf.h * 0.8
        N[oncut] = n
    if o.scalar is not None:
        v = (o.scalar(P) - o.vmin) / (o.vmax - o.vmin)
        base = o.cmap(np.clip(v, 0, 1))[:, :3]
        nanm = ~np.isfinite(v)
        base[nanm] = o.color
    else:
        base = np.repeat(o.color[None], len(P), 0)
    V = -D
    aoterm = ambient_occlusion(objs, P, N) if ao else np.ones(len(P), np.float32)
    col = base * ambient * aoterm[:, None]
    for (Ld, Lc, Li, cast) in lights:
        Ld = np.asarray(Ld, float); Ld /= np.linalg.norm(Ld)
        lam = np.clip(N @ Ld, 0, 1)
        Hh = Ld[None] + V; Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
        sp = np.clip((N * Hh).sum(1), 0, 1) ** o.shin
        sh = soft_shadow(objs, P + N * 1.5, Ld) if (shadows and cast) else 1.0
        col += (base * lam[:, None] * Li + o.spec * sp[:, None] * Li) * (np.asarray(Lc)[None]) * (sh[:, None] if np.ndim(sh) else sh)
    rim = (1 - np.clip((N * V).sum(1), 0, 1)) ** 3
    col += o.rim * rim[:, None] * 0.6
    return np.clip(col, 0, 1), N


def render(objs, eye, target, up=(0, 0, 1), W=900, H=700, fov=None, ortho=None, bg=(1, 1, 1), lights=None,
           shadows=True, ao=True, ss=2, chunk=400000):
    """Opaque objects are composited by depth; translucent objects (alpha<1) are layered in front with a
    Fresnel-like edge opacity.  Returns an (H, W, 3) float image and a (H, W) alpha mask."""
    if lights is None or lights == "camera":
        # camera-relative three-point lighting: key from upper left, fill from right, top light
        e_, t_, u_ = (np.asarray(v, float) for v in (eye, target, up))
        f_ = t_ - e_; f_ /= np.linalg.norm(f_); r_ = np.cross(f_, u_); r_ /= np.linalg.norm(r_); uu = np.cross(r_, f_)
        key = -f_ * 0.8 - r_ * 0.55 + uu * 0.75; fill = -f_ * 0.7 + r_ * 0.7 + uu * 0.1; top = uu * 1.0 - f_ * 0.2
        lights = [(key, (1.0, 0.97, 0.92), 0.80, True), (fill, (0.88, 0.92, 1.0), 0.32, False), (top, (1.0, 1.0, 1.0), 0.22, False)]
    Ws, Hs = W * ss, H * ss
    O, D, fwd = camera_rays(eye, target, up, Ws, Hs, fov=fov, ortho=ortho * 1.0 if ortho else None)
    img = np.ones((Ws * Hs, 3), np.float32) * np.array(bg, np.float32)
    alpha = np.zeros(Ws * Hs, np.float32)
    tmax = np.full(Ws * Hs, 1e4, np.float32)
    # bounding sphere of the scene to bound the march
    los = np.min([o.sdf.o for o in objs], 0); his = np.max([o.sdf.o + np.array(o.sdf.data.shape) * o.sdf.h for o in objs], 0)
    c = (los + his) / 2; R = np.linalg.norm(his - los) / 2
    oc = O - c; b = (oc * D).sum(1); disc = b * b - ((oc * oc).sum(1) - R * R)
    valid = disc > 0
    t0 = np.where(valid, np.maximum(-b - np.sqrt(np.maximum(disc, 0)), 0), 0); t1 = np.where(valid, -b + np.sqrt(np.maximum(disc, 0)), 0)
    O = O + D * t0[:, None]; tmax = (t1 - t0).astype(np.float32); tmax[~valid] = 0
    depth = np.full(Ws * Hs, np.inf, np.float32); owner = np.full(Ws * Hs, -1)
    opaque = [k for k, o in enumerate(objs) if o.alpha >= 0.99]
    for k in opaque:
        hk = trace(objs[k], O, D, tmax)
        better = hk < depth; depth[better] = hk[better]; owner[better] = k
    for k in opaque:
        m = np.nonzero(owner == k)[0]
        for c0 in range(0, len(m), chunk):
            mm = m[c0:c0 + chunk]
            P = O[mm] + D[mm] * depth[mm, None]
            col, _ = shade(objs[k], objs, P, D[mm], lights, shadows=shadows, ao=ao)
            img[mm] = col
        alpha[m] = 1.0
    for k, o in enumerate(objs):
        if o.alpha >= 0.99:
            continue
        hk = trace(o, O, D, np.minimum(tmax, np.where(np.isfinite(depth), depth, tmax)))
        m = np.nonzero(np.isfinite(hk) & (hk < depth))[0]
        for c0 in range(0, len(m), chunk):
            mm = m[c0:c0 + chunk]
            P = O[mm] + D[mm] * hk[mm, None]
            col, N = shade(o, objs, P, D[mm], lights, shadows=False, ao=False)
            fres = (1 - np.abs((N * D[mm]).sum(1))) ** 2.5
            a = np.clip(o.alpha + (1 - o.alpha) * fres * 0.85, 0, 1)[:, None]
            img[mm] = a * col + (1 - a) * img[mm]
            alpha[mm] = np.maximum(alpha[mm], a[:, 0])
    img = img.reshape(Hs, Ws, 3); alpha = alpha.reshape(Hs, Ws)
    if ss > 1:
        img = img.reshape(H, ss, W, ss, 3).mean((1, 3)); alpha = alpha.reshape(H, ss, W, ss).mean((1, 3))
    return img, alpha


def field_sampler(values, xe, ye, ze, valid=None, nodal=True):
    """Return f(P) sampling a cell-wise FE field on a rectilinear grid (edges xe, ye, ze).  With nodal=True the
    cell values of the valid cells are averaged to the nodes (standard FE contour post-processing) and interpolated
    trilinearly between nodes; otherwise cell-centre values are interpolated.  Cells outside `valid` do not
    contribute; nodes without valid cells take the value of the nearest valid node."""
    nx, ny, nz = len(xe) - 1, len(ye) - 1, len(ze) - 1
    V = np.asarray(values, np.float32).reshape(nx, ny, nz)
    vm = np.ones((nx, ny, nz), bool) if valid is None else np.asarray(valid, bool).reshape(nx, ny, nz)
    if nodal:
        S = np.zeros((nx + 1, ny + 1, nz + 1), np.float64); C = np.zeros_like(S)
        Vv = np.where(vm, V, 0.0); w = vm.astype(np.float64)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    S[a:a + nx, b:b + ny, c:c + nz] += Vv; C[a:a + nx, b:b + ny, c:c + nz] += w
        ok = C > 0
        N = np.where(ok, S / np.maximum(C, 1), 0).astype(np.float32)
        if ok.any() and not ok.all():
            _, ind = ndimage.distance_transform_edt(~ok, return_indices=True)
            N = N[tuple(ind)]
        gx, gy, gz = xe, ye, ze; G = N
    else:
        if vm.any() and not vm.all():
            _, ind = ndimage.distance_transform_edt(~vm, return_indices=True)
            V = V[tuple(ind)]
        gx, gy, gz = (0.5 * (e[1:] + e[:-1]) for e in (xe, ye, ze)); G = V

    def f(P):
        ix = np.interp(P[:, 0], gx, np.arange(len(gx))); iy = np.interp(P[:, 1], gy, np.arange(len(gy))); iz = np.interp(P[:, 2], gz, np.arange(len(gz)))
        return ndimage.map_coordinates(G, np.stack([ix, iy, iz]), order=1, mode="nearest").astype(np.float32)
    return f
