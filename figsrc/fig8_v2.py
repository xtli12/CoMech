"""Figure 8: from a child's CT to a mechanical twin, and the twins of 39 real feet (house style v2).
a, 3D reconstruction of P05-R from its CT export; b, calcaneus with the joint-line contrast of the CT; c, coronal
reformats with the segmentation, the zone without a visible joint line and the preliminary AI-assisted reading;
d, the foot's point twin before and after complete resection (FE stress); e, envelope versus the 50% rule for all 39
feet; f, FE-verified point twins of the 13 type I feet. Supplementary Fig. 10 keeps the per-foot strip.
Images are the renders stored in results/renders (pixels unchanged); the CT-derived renders of a and c, whose sources
fall below 300 ppi at print size, are resampled uniformly (Lanczos) to the output grid, which adds no detail. The coronal
reformats are shown at their true (isotropic) pixel aspect."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from figstyle import *
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from scipy import ndimage
from fig8_ct import load, TYPE_COL, TYPE_MK, OPER

LOWRES = "lanczos"
TCC = os.environ.get("CT_READ_DIR", "data/ct/read")           # CT volumes of the reading (not shipped); the cached tiles are used without them
R = lambda n: np.load(os.path.join(ROOT, f"results/renders/{n}.npz"))
HALO = [pe.withStroke(linewidth=1.8, foreground="white", alpha=0.9)]
OUTLINE = {"talus": (0.90, 0.47, 0.35), "calc": (0.35, 0.59, 0.90), "contact": (0.99, 0.62, 0.0)}   # as drawn in the tiles


def cam_project(eye, target, up, W, H, ortho, P):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    x = (P - eye) @ r / ortho; y = (P - eye) @ u / ortho
    return (x / 2 + 0.5) * W, (0.5 - y * W / (2 * H)) * H


def coronal(A, seg, xx, contact):
    sl = (slice(60, 232), slice(150, 330))
    s = A[xx][sl].T[::-1][:, ::-1]                       # rows z (superior up), columns y flipped -> medial on the left
    T = (seg[xx] == 1)[sl].T[::-1][:, ::-1]; C = (seg[xx] == 2)[sl].T[::-1][:, ::-1]; K = contact[xx][sl].T[::-1][:, ::-1]
    return s, T, C, K


def bbox(alpha, pad):
    al = alpha > 0.02
    rr, cc = np.nonzero(al.any(1))[0], np.nonzero(al.any(0))[0]
    return max(rr[0] - pad, 0), rr[-1] + pad + 1, max(cc[0] - pad, 0), cc[-1] + pad + 1


def compass(cv, x, y, h_lab, v_lab, h_dir=1, v_dir=-1, L=4.0, color=INK2):
    """Small orientation key: a horizontal and a vertical arrow from (x, y) mm with one-letter labels."""
    st = "-|>,head_length=1.8,head_width=0.9"
    cv.arrow(x, y, x + h_dir * L, y, lw=0.5, color=color, arrowstyle=st)
    cv.arrow(x, y, x, y + v_dir * L, lw=0.5, color=color, arrowstyle=st)
    cv.text(x + h_dir * (L + 0.7), y + 0.7, h_lab, fontsize=FS_SMALL, color=color, ha="left" if h_dir > 0 else "right")
    cv.text(x, y + v_dir * (L + 0.9) + (0.0 if v_dir < 0 else 1.6), v_lab, fontsize=FS_SMALL, color=color, ha="center")


def fig8(out):
    T, F = load(); rows = T["rows"]
    cv = Canvas(FIG_W, 133)

    # ---------------- a: reconstruction ----------------
    Zm = R("ct_P05R_medial"); h = float(Zm["h"])
    cv.letter(0, 0.5, "a"); cv.header(4.8, 3.4, "P05-R: CT export to 3D reconstruction")
    r0, r1, c0, c1 = bbox(Zm["alpha"], 6)
    Wa = 49.0; Ha = Wa * (r1 - r0) / (c1 - c0)
    ax = cv.ax(3.0, 7.5, Wa, Ha)
    img = np.dstack([np.clip(Zm["img"], 0, 1), np.clip(Zm["alpha"], 0, 1)])[r0:r1, c0:c1]
    ax.imshow(img, interpolation=LOWRES, extent=(c0, c1, r1, r0))
    CACHE = os.path.join(ROOT, "results/renders/ct_P05R_coronal.npz")
    have_ct = os.path.exists(f"{TCC}/seg39c.npy")
    if have_ct:
        seg = np.load(f"{TCC}/seg39c.npy"); c = np.argwhere(np.isin(seg, [1, 2])).mean(0) * h
        cents = {nm: np.argwhere(seg == L).mean(0) * h for nm, L in [("tibia", 5), ("talus", 1), ("calcaneus", 2),
                                                                       ("navicular", 3)]}
    else:
        Zc = np.load(CACHE); c = Zc["c"]; cents = json.loads(str(Zc["cents"]))
    eye = c + np.array([-25, 280, 70]); tgt = c + np.array([-6, 0, -2]); W, H = Zm["img"].shape[1], Zm["img"].shape[0]
    XS = [110, 150, 210]
    for k, xx in enumerate(XS):
        P = np.array([[xx * h, c[1] + 20, c[2] + 24], [xx * h, c[1] + 20, c[2] - 26]])
        px, py = cam_project(eye, tgt, (0, 0, 1), W, H, 84, P)
        ax.plot(px, py, color=BLUE, lw=0.8, ls=(0, (2.5, 1.5)), solid_capstyle="butt")
        ax.text(px[0], py[0] - 8, str(k + 1), color=BLUE, fontsize=FS, ha="center", va="bottom", fontweight="bold",
                path_effects=HALO)
    for nm, pc_ in cents.items():
        px, py = cam_project(eye, tgt, (0, 0, 1), W, H, 84, np.asarray(pc_)[None])
        if nm == "talus":       # label in the free space, with a leader to the bone
            ax.annotate("Talus", xy=(px[0], py[0]), xytext=(232, 140), fontsize=FS, color=INK, ha="right",
                        va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1.5, shrinkB=0),
                        path_effects=HALO)
            ax.plot(px, py, "o", ms=1.6, color=INK, zorder=5)
            continue
        off = {"tibia": (0, -40), "calcaneus": (60, 60), "navicular": (-10, -30)}[nm]
        ax.text(px[0] + off[0], py[0] + off[1], nm.capitalize(), fontsize=FS, color=INK, ha="center", va="center",
                path_effects=HALO)
    ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.set_axis_off()
    mm = Wa / (c1 - c0)
    compass(cv, 3.0 + (130 - c0) * mm, 7.5 + (52 - r0) * mm, "A", "S", h_dir=-1, v_dir=-1, L=3.4)
    cv.text(3.0 + Wa - 0.5, 10.5, "Medial view", fontsize=FS_SMALL, color=INK2, ha="right")

    # ---------------- b: calcaneus, joint-line visibility ----------------
    Ze = R("ct_P05R_exploded")
    xb = 57.0
    cv.letter(xb, 0.5, "b"); cv.header(xb + 4.8, 3.4, "Calcaneus, talus removed: joint line on CT")
    r0, r1, c0, c1 = bbox(Ze["alpha"], 6)
    Wb = 60.0; Hb = Wb * (r1 - r0) / (c1 - c0)
    ax = cv.ax(xb + 2.0, 14.0, Wb, Hb)
    ax.imshow(np.dstack([np.clip(Ze["img"], 0, 1), np.clip(Ze["alpha"], 0, 1)])[r0:r1, c0:c1], interpolation="none")
    ax.set_axis_off()
    compass(cv, xb + 3.0, 13.5, "A", "M", h_dir=1, v_dir=-1, L=3.4)
    from render_ct import GAP
    yb = 14.0 + Hb + 5.5
    cbar_mm(cv, ScalarMappable(Normalize(0, 60), GAP), xb + 17.0, yb, 30.0, 1.3, "Joint-line contrast (black top-hat)",
            ticks=[0, 20, 40, 60])
    cv.text(xb + 16.0, yb + 1.1, "No line", fontsize=FS_SMALL, color=INK2, ha="right")
    cv.text(xb + 48.0, yb + 1.1, "Line visible", fontsize=FS_SMALL, color=INK2, ha="left")
    cv.text(xb + 2.0, yb + 13.0, "Nominal scale: the export carries no pixel spacing", fontsize=FS_SMALL, color=INK2)

    # ---------------- c: coronal reformats + reading ----------------
    xc = 124.0
    cv.letter(xc, 0.5, "c"); cv.header(xc + 4.8, 3.4, "Coronal reformats at planes 1–3")
    if have_ct:
        A = np.load(f"{TCC}/anat_E39R_c.npy")
        tal, cal = seg == 1, seg == 2
        dT = ndimage.distance_transform_edt(~tal); dC = ndimage.distance_transform_edt(~cal)
        Jv = np.load(f"{TCC}/jv39s.npy").astype(np.float32)
        contact = ((cal & (dT <= 3.0)) | (tal & (dC <= 3.0))) & (Jv < 20)
        tiles = []
        for xx in XS:
            sl_, Tm, Cm, K = coronal(A, seg, xx, contact)
            rgb = np.stack([np.clip(sl_, 0, 255) / 255] * 3, -1)
            for M, col in [(Tm, OUTLINE["talus"]), (Cm, OUTLINE["calc"])]:
                e = M & ~ndimage.binary_erosion(M); rgb[e] = col
            Kd = ndimage.binary_dilation(K, iterations=1); rgb[Kd] = 0.3 * rgb[Kd] + 0.7 * np.array(OUTLINE["contact"])
            tiles.append(rgb.astype(np.float32))
        np.savez_compressed(CACHE, tiles=np.stack(tiles), c=c, xs=np.array(XS),
                            cents=json.dumps({k: list(map(float, v)) for k, v in cents.items()}))
    else:
        tiles = list(np.load(CACHE)["tiles"])
    Wt = 18.8; Ht = Wt * tiles[0].shape[0] / tiles[0].shape[1]
    for k in range(3):
        x = xc + 1.0 + k * (Wt + 0.6)
        ax = cv.ax(x, 12.0, Wt, Ht)
        ax.imshow(tiles[k], interpolation=LOWRES); ax.set_axis_off()
        cv.text(x + Wt / 2, 10.9, str(k + 1), color=BLUE, fontsize=FS, fontweight="bold", ha="center")
    yk = 12.0 + Ht + 3.6
    for j, (key, lab, lw, x) in enumerate([("talus", "Talus", 0.8, 0.0), ("calc", "Calcaneus", 0.8, 12.5),
                                           ("contact", "No joint line", 2.0, 29.0)]):
        cv.line([xc + 1.0 + x, xc + 4.6 + x], [yk - 0.7, yk - 0.7], color=OUTLINE[key], lw=lw)
        cv.text(xc + 5.6 + x, yk, lab, fontsize=FS_SMALL)
    cv.text(xc + 1.0, yk + 2.7, "Medial to the left, superior up", fontsize=FS_SMALL, color=INK2)
    yr_ = yk + 8.2
    cv.text(xc + 1.0, yr_, "Preliminary AI-assisted reading (bone window)", fontsize=FS)
    lines = ["Anatomy-based type I; Rozansky II",
             "Middle facet replaced by a narrowed, sclerotic interface",
             "Posteromedial ossicles",
             "Middle-facet coverage 0.8; confidence medium"]
    for k, s in enumerate(lines):
        cv.text(xc + 1.0, yr_ + 2.7 * (k + 1), s, fontsize=FS_SMALL, color=INK2)
    cv.text(xc + 1.0, yr_ + 2.7 * (len(lines) + 1) + 0.4, "Pending clinician adjudication", fontsize=FS_SMALL,
            color=INK, style="italic")

    # ---------------- d: twin ----------------
    y2 = 69.0
    fr = {r["foot"]: r for r in F["rows"]}["P05-R"]
    cv.letter(0, y2, "d"); cv.header(4.8, y2 + 2.9, "P05-R point twin (engine)")
    Zs = [R("twin_P05-R_pre"), R("twin_P05-R_resect")]
    im0 = Zs[0]["img"]; al = np.max([z["alpha"] for z in Zs], 0)
    sat = np.zeros(al.shape, bool)
    for z in Zs:
        im = z["img"]; sat |= ((im.max(2) - im.min(2)) > 0.25) & (z["alpha"] > 0.5)
    rr, cc = np.nonzero(sat.any(1))[0], np.nonzero(sat.any(0))[0]
    pr, pc = 28, 40
    q0, q1, p0, p1 = max(rr[0] - pr, 0), min(rr[-1] + pr, al.shape[0]), max(cc[0] - pc, 0), min(cc[-1] + pc, al.shape[1])
    Wd = 28.0; Hd = Wd * (q1 - q0) / (p1 - p0)
    for k, (Z, tt) in enumerate(zip(Zs, ["Before resection", "After complete resection"])):
        x = 1.0 + k * (Wd + 1.6)
        ax = cv.ax(x, y2 + 11.0, Wd, Hd)
        ax.imshow(np.dstack([np.clip(Z["img"], 0, 1), np.clip(Z["alpha"], 0, 1)])[q0:q1, p0:p1], interpolation="none")
        ax.set_axis_off()
        cv.text(x + Wd / 2, y2 + 9.4, tt, ha="center", fontsize=FS)
    compass(cv, 2.4, y2 + 11.0 + 4.6, "A", "M", h_dir=1, v_dir=-1, L=3.2)
    yb = y2 + 11.0 + Hd + 2.6
    from matplotlib.patches import Rectangle
    cv.fig.add_artist(Rectangle((cv.fx(2.0), cv.fy(yb + 1.6)), 2.4 / cv.w, 1.6 / cv.h, transform=cv.fig.transFigure,
                                fc=COAL, ec="none"))
    cv.text(5.2, yb + 1.4, "Coalition, as read", fontsize=FS_SMALL)
    cbar_mm(cv, ScalarMappable(Normalize(0, 1.6), STRESS), 1.0 + Wd + 1.6 + 1.0, yb, Wd - 2.0, 1.3,
            "FE von Mises stress (MPa)", ticks=[0, 0.8, 1.6])
    yt = yb + 11.0
    tab = [("Middle-facet coverage (reading)", "0.80"),
           ("Shape; alignment", "engine median; neutral"),
           ("Coalition / posterior facet", f"{100 * fr['ratio']:.0f}%: 50% rule, " +
            ("resect alone" if fr["ratio"] < 0.5 else "do not resect alone")),          # (Claude, 2026-11-01)
           ("Defect left in the middle facet", f"{100 * fr['d_mid']:.0f}%"),
           ("FE tolerance of the middle facet", f"{100 * fr['fe_tol']:.0f}%: " +
            ("beyond the envelope" if fr["d_mid"] > fr["fe_tol"] else "within the envelope"))]
    for k, (a_, b_) in enumerate(tab):
        cv.text(1.0, yt + 2.6 * k, a_, fontsize=FS_SMALL, color=INK2)
        cv.text(31.5, yt + 2.6 * k, b_, fontsize=FS_SMALL, color=INK)

    # ---------------- e: envelope vs 50% rule ----------------
    xe = 64.0
    cv.letter(xe, y2, "e"); cv.header(xe + 4.8, y2 + 2.9, "39 feet: envelope and the 50% rule")
    ax = cv.ax(xe + 11.0, y2 + 15.0, 38.0, 38.0)
    rng = np.random.default_rng(3)
    ax.axvline(50, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for r in rows:
        x = 100 * r["ratio"][1]; y = r["p_within"] + rng.uniform(-0.015, 0.015)
        col = TYPE_COL[r["type"]]
        ax.plot([100 * r["ratio"][0], 100 * r["ratio"][2]], [y, y], color=col, lw=0.5, alpha=0.7, zorder=2)
        ax.plot(x, y, TYPE_MK[r["type"]], ms=3.2, mfc=col if OPER(r["treatment"]) else "white", mec=col, mew=0.7,
                zorder=3)
    ax.text(51.0, 1.07, "50% rule", fontsize=FS_SMALL, color=INK2, ha="left", va="center")
    ax.set_ylim(-0.06, 1.1); ax.set_xlim(0, 60); ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    style(ax)
    ax.set_xlabel("Coalition / posterior facet (%)")
    ax.set_ylabel("Share of twins within the envelope")
    hs = [Line2D([], [], ls="none", marker=TYPE_MK[t], ms=3.2, color=TYPE_COL[t], mew=0.7, label=lab)
          for t, lab in [("I", "Type I"), ("III", "Type III"), ("indeterminate", "Indeterminate")]]
    hs += [Line2D([], [], ls="none", marker="o", ms=3.2, color=INK2, mew=0.7, label="Operated"),
           Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=INK2, mew=0.7, label="Not operated")]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.03, 1.02), ncol=3, handletextpad=0.05,
              columnspacing=0.7, borderaxespad=0.0)

    # ---------------- f: FE point twins ----------------
    xf = 122.0
    cv.letter(xf, y2, "f"); cv.header(xf + 4.8, y2 + 2.9, f"FE-verified point twins, {len(F['rows'])} type I feet")
    ax = cv.ax(xf + 12.0, y2 + 15.0, 47.0, 38.0)
    Rr = sorted(F["rows"], key=lambda r: r["d_mid"])
    for i, r in enumerate(Rr):
        ax.plot([100 * r["fe_tol"], 100 * r["d_mid"]], [i, i], color=BASE, lw=0.6, zorder=1)
        ax.plot(100 * r["fe_tol"], i, "|", ms=5.5, color=FE, mew=1.1, zorder=3)
        ax.plot(100 * r["pred_tol"], i, "x", ms=3.2, color=BLUE, mew=0.8, zorder=4)
        beyond = r["d_mid"] > r["fe_tol"]
        ax.plot(100 * r["d_mid"], i, "o", ms=3.2, mfc=VERM if beyond else "white", mec=VERM, mew=0.7, zorder=5)
    ax.set_yticks(range(len(Rr))); ax.set_yticklabels([r["foot"] for r in Rr], fontsize=FS_SMALL)
    ax.set_ylim(len(Rr) - 0.4, -0.6); ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    style(ax, left=False)
    ax.set_xlabel("Middle facet (%)")
    hs = [Line2D([], [], ls="none", marker="|", ms=5.5, color=FE, mew=1.1, label="FE tolerance"),
          Line2D([], [], ls="none", marker="x", ms=3.2, color=BLUE, mew=0.8, label="CoMech tolerance"),
          Line2D([], [], ls="none", marker="o", ms=3.2, color=VERM, mew=0.7, label="Defect, beyond"),
          Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=VERM, mew=0.7, label="Defect, within")]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.03, 1.02), ncol=2, handletextpad=0.1,
              columnspacing=0.9, borderaxespad=0.0)
    cv.save(out[:-4] if out.endswith(".pdf") else out)
    print("saved", out)


if __name__ == "__main__":
    fig8(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/paper/figures/fig8.pdf")
