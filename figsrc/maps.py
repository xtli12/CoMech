"""Deformation-guided surgical map of the virtual child (Fig. 7a, Fig. 1c): rendered subtalar cartilage
coloured by von Mises stress, the defect at the posterior-facet tolerance (black), the resected coalition footprint
(dashed) and in-plane talar displacement arrows."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
from figstyle import INK, BLUE, INK2

_CACHE = {}


def _grid():
    if "g" not in _CACHE:
        import sys
        sys.path.insert(0, ROOT)
        from fe.simulate import get_grid
        _CACHE["g"] = get_grid()
    return _CACHE["g"]


def child_map(ax, tag, src, uref=None, arrows=True, contours=True, arrow_scale=13, arrow_width=0.006, lw=0.9,
              coal_color=BLUE):
    """Draw one map into ax; returns the arrow reference length (pass it on so all maps share one scale)."""
    from fe.geometry import TALUS
    from scipy.ndimage import binary_dilation
    g = _grid()
    masks = np.load(os.path.join(ROOT, "results/map_child_masks.npz"))
    xs, ys = masks["xs"], masks["ys"]
    Z = np.load(os.path.join(ROOT, f"results/renders/child_{tag}_{src}.npz")); ext = Z["ext"]
    ax.imshow(np.clip(Z["img"], 0, 1), extent=ext, origin="upper", interpolation="none")
    if contours:
        ax.contour(xs, ys, masks["tol_def"].T.astype(float), levels=[0.5], colors=[INK], linewidths=lw)
        ax.contour(xs, ys, masks["coal"].T.astype(float), levels=[0.5], colors=[coal_color], linewidths=lw * 0.9,
                   linestyles="--")
    if arrows and src in ("comech", "fe"):
        Mc = np.load(os.path.join(ROOT, f"results/map_child_{tag}.npz"))
        cells, lab = Mc["cells"], Mc["lab"]
        labc = lab[cells] if len(lab) != len(cells) else lab
        import fe.geometry as _FG
        if hasattr(_FG, "subtalar_band"):     # real-bone engine: talar cells within 4.5 mm of the calcaneus (Claude, 2026-11-01)
            import json as _json
            _ch = _json.load(open(os.path.join(ROOT, "results/eval_maps.json")))["child"]
            tal = (labc == TALUS) & _FG.subtalar_band(g.cell_xyz[cells], foot=_FG.Foot(**_ch["foot"]), dmax=4.5)
        else:
            tal = (labc == TALUS) & (g.cell_xyz[cells, 2] > 36) & (g.cell_xyz[cells, 2] < 40)
        Pp = g.cell_xyz[cells][tal]
        fac = binary_dilation(masks["post"] | masks["mid"] | masks["ant"], iterations=8)
        ix = np.clip(np.searchsorted(xs, Pp[:, 0]), 0, len(xs) - 1); iy = np.clip(np.searchsorted(ys, Pp[:, 1]), 0, len(ys) - 1)
        sel = (np.round(Pp[:, 0]) % 8 == 1) & (np.round(Pp[:, 1]) % 6 == 1) & fac[ix, iy]
        U = (Mc["u_p"] if src == "comech" else Mc["u_fe"])[tal]
        key = np.round(Pp[sel, :2], 1); uk, inv = np.unique(key, axis=0, return_inverse=True)
        Ua = np.stack([U[sel][inv.ravel() == i, :2].mean(0) for i in range(len(uk))])
        if uref is None:
            uref = float(np.percentile(np.linalg.norm(Ua, axis=1), 90)) + 1e-9
        ax.quiver(uk[:, 0], uk[:, 1], Ua[:, 0], Ua[:, 1], color=INK, scale=uref * arrow_scale, scale_units="width",
                  width=arrow_width, headwidth=3.0, headlength=3.2, alpha=0.85, zorder=5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    return uref
