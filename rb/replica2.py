"""Voxel replica of an ANSYS atlas model on the CoMech engine solver, version 2 (Claude, 2026-10-28).

Differences from voxel_replica.py / run_replica.py:
  * the voxelisation (body label of every sub-sample) is cached, so material rules can be compared cheaply;
  * material rules ("engine": Reuss in cells with cartilage, Voigt elsewhere, as in the engine;
    "iface": in every cell that lies within one cell of two different bones, or contains cartilage, the soft phases
    (cartilage, soft tissue) are put in series with the hard phase (bone, fascia; Voigt among themselves), so thin
    soft-tissue gaps between bones are no longer bridged by stiff partial-volume cells);
  * the 15 ligament springs of the atlas (COMBIN14, 80 N/mm) can be included (patch-averaged ends);
  * diagnostics: forces that each neighbouring region exerts on the talus and on the calcaneus, ground reaction
    split heel / midfoot / forefoot, facet stresses (cell von Mises and normal pressure)."""
import os, sys, time, json, pickle, numpy as np
import scipy.sparse as sp
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.environ.get("COMECH", os.path.dirname(HERE)))
import voxel_replica as VR
from frame import to_engine
from fe.solver import Grid, lame, face_node_areas, assemble, cell_stress

CL = VR.CLASSES                                     # void cortical cancellous cartilage fascia soft plate
E_CL = np.array([0.0] + [VR.MAT[c][0] for c in CL[1:]])
NU_CL = np.array([0.3] + [VR.MAT[c][1] for c in CL[1:]])

# roles in the normal model (body index of ds.dat); bones = cortical + cancellous body of the same bone
NORMAL_ROLES = {
    "bones": {"calcaneus": (1, 53), "talus": (2, 52), "fibula": (3, 54), "tibia": (4, 13), "navicular": (40, 45),
              "cuboid": (43, 49), "cun_med": (39, 51), "cun_int": (41, 46), "cun_lat": (42, 48)},
    "cart": {"ankle": 5, "syndesmosis": 6, "talofibular": 7, "mid": 8, "tn": 9, "ant": 10, "cc": 11, "post": 12},
}


def bone_groups(cls, roles):
    """Group id (1..) per body for every bone body; named bones from roles, every other bone body on its own."""
    g = {}; names = {}
    k = 1
    for name, bodies in roles["bones"].items():
        for b in bodies:
            g[b] = k
        names[k] = name; k += 1
    skip = set(roles.get("bonded_extra", ()))           # e.g. coalition bars: bonded to the bones they touch
    for b, c in sorted(cls.items()):
        if c in (1, 2) and b not in g and b not in skip:
            g[b] = k; names[k] = f"body{b}"; k += 1
    return g, names


def voxelize(pkl, h_fine, h_coarse, z_cut=150.0, ns=3, box=((-100, 10), (-30, 40), (10, 100)), cache=None):
    if cache and os.path.exists(cache):
        Z = np.load(cache, allow_pickle=True)
        return {k: Z[k] for k in Z.files} | {"cls": json.loads(str(Z["cls_json"]))}
    t0 = time.time()
    T, body, cls, D = VR.load_model(pkl)
    keep = (T[:, :, 2].max(1) > -1.0) & (T[:, :, 2].min(1) < z_cut + 1.0) & \
           (np.vectorize(cls.get)(body) != CL.index("plate"))
    T, body = T[keep], body[keep]
    lo = T.reshape(-1, 3).min(0); hi = T.reshape(-1, 3).max(0)
    (bx0, bx1), (by0, by1), (bz0, bz1) = box
    xe = VR.axis_edges(np.floor(lo[0]) - 1, np.ceil(hi[0]) + 1, bx0, bx1, h_fine, h_coarse)
    ye = VR.axis_edges(np.floor(lo[1]) - 1, np.ceil(hi[1]) + 1, by0, by1, h_fine, h_coarse)
    ze = VR.axis_edges(0.0, z_cut, bz0, bz1, h_fine, h_coarse)
    sub = lambda e: ((e[:-1, None] + (np.arange(ns) + 0.5)[None] / ns * np.diff(e)[:, None])).ravel()
    lab = VR.locate(T, body, sub(xe), sub(ye), sub(ze))
    nx, ny, nz = len(xe) - 1, len(ye) - 1, len(ze) - 1
    L = lab.reshape(nx, ns, ny, ns, nz, ns).transpose(0, 2, 4, 1, 3, 5).reshape(nx * ny * nz, ns ** 3)
    # spring end points (engine frame)
    springs = []
    for s in D["springs"]:
        a, b = s["nodes"]
        name = s["name"].split('"')[1] if '"' in s["name"] else s["name"]
        springs.append({"name": name, "k": float(s["k"]), "a": to_engine(np.array(D["nodes"][a])).tolist(),
                        "b": to_engine(np.array(D["nodes"][b])).tolist()})
    # ground contact footprint: fraction of the bottom face of every bottom cell where the soft tissue lies below
    # z_tol (raster of 0.5 mm); in the atlas, the soft-tissue nodes below about 0.4 mm are the ones bonded to the plate
    z_tol, rr = 0.4, 0.5
    gx = np.arange(xe[0] + rr / 2, xe[-1], rr); gy = np.arange(ye[0] + rr / 2, ye[-1], rr)
    soft_b = [b for b, c in cls.items() if c == CL.index("soft")]
    ms = np.isin(body, soft_b)
    labc = VR.locate(T[ms], body[ms], gx, gy, np.array([z_tol]))[:, :, 0] > 0
    ii = np.clip(np.searchsorted(xe, gx) - 1, 0, nx - 1); jj = np.clip(np.searchsorted(ye, gy) - 1, 0, ny - 1)
    cnt = np.zeros((nx, ny)); hit = np.zeros((nx, ny))
    np.add.at(cnt, (ii[:, None].repeat(len(gy), 1), jj[None, :].repeat(len(gx), 0)), 1)
    np.add.at(hit, (ii[:, None].repeat(len(gy), 1), jj[None, :].repeat(len(gx), 0)), labc)
    contact = (hit / np.maximum(cnt, 1)).astype(np.float32)
    out = dict(xe=xe, ye=ye, ze=ze, L=L.astype(np.int16), ns=ns, cls_json=json.dumps({int(k): int(v) for k, v in cls.items()}),
               springs_json=json.dumps(springs), t_vox=time.time() - t0, contact=contact, contact_area=float(labc.sum() * rr * rr))
    if cache:
        np.savez_compressed(cache, **out)
    out["cls"] = {int(k): int(v) for k, v in cls.items()}
    return out


def fractions(L, cls, nb):
    lut = np.zeros(nb + 1, np.int16)
    for b, c in cls.items():
        lut[b] = c
    C = lut[L]
    return np.stack([(C == k).mean(1) for k in range(len(CL))], 1)          # (ncell, nclass)


def materials(V, roles, rule="iface", nu_soft=0.49):
    cls = {int(k): int(v) for k, v in V["cls"].items()}
    L = V["L"]; nb = max(max(cls), int(L.max()))
    F = fractions(L, cls, nb)
    nu_cl = NU_CL.copy(); nu_cl[CL.index("soft")] = nu_soft
    f = F[:, 1:6]; occ = f.sum(1)
    Ev, nv = E_CL[1:6], nu_cl[1:6]
    with np.errstate(divide="ignore", invalid="ignore"):
        voigt = (f * Ev).sum(1) / occ
        nu = (f * nv).sum(1) / occ
        reuss = occ / (f / Ev).sum(1)
        hard = f[:, [0, 1, 3]].sum(1)                                           # cortical, cancellous, fascia
        Eh = (f[:, [0, 1, 3]] * Ev[[0, 1, 3]]).sum(1) / hard
        soft_series = occ / (f[:, 2] / Ev[2] + f[:, 4] / Ev[4] + np.where(hard > 0, hard / Eh, 0.0))
    cart = F[:, 3] > 0
    nx, ny, nz = len(V["xe"]) - 1, len(V["ye"]) - 1, len(V["ze"]) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        fch = f[:, 2] + hard                                                    # cartilage + hard phase
        Ech = fch / (f[:, 2] / Ev[2] + np.where(hard > 0, hard / Eh, 0.0))      # in series
        cart_par = (fch * Ech + f[:, 4] * Ev[4]) / occ                          # soft tissue alongside (parallel)
    if rule == "engine":
        ser = cart; E = np.where(ser, reuss, voigt)
    else:
        grp, _ = bone_groups(cls, roles)
        lutg = np.zeros(nb + 1, np.int16)
        for b, k in grp.items():
            lutg[b] = k
        G = lutg[L]
        count = np.zeros(nx * ny * nz, np.int16)
        st = np.ones((3, 3, 3), bool)
        for k in np.unique(G[G > 0]):
            pres = (G == k).any(1).reshape(nx, ny, nz)
            count += ndimage.binary_dilation(pres, st).ravel()
        ser = cart | (count >= 2)
        if rule in ("iface2", "iface3", "iface3c", "iface4"):   # cartilage cells: cartilage in series with bone, soft tissue in parallel
            E = np.where(cart, cart_par, np.where(ser, soft_series, voigt))
        else:
            E = np.where(ser, soft_series, voigt)
    if rule in ("iface3", "iface3c", "iface4"):
        # mixing on the constrained modulus M = lambda + 2 mu and the shear modulus G = mu: in series across a joint
        # (thin layers between stiff bones are laterally constrained, so a thin layer of nearly incompressible soft
        # tissue is as stiff in compression as cartilage), Voigt (lambda, mu averaged) elsewhere
        lam_i = Ev * nv / ((1 + nv) * (1 - 2 * nv)); mu_i = Ev / (2 * (1 + nv))
        H = [0, 1, 3]                                                          # hard phases
        with np.errstate(divide="ignore", invalid="ignore"):
            lamV = (f * lam_i).sum(1) / occ; muV = (f * mu_i).sum(1) / occ
            lam_h = (f[:, H] * lam_i[H]).sum(1) / hard; mu_h = (f[:, H] * mu_i[H]).sum(1) / hard
            M_h = lam_h + 2 * mu_h; G_h = mu_h
            Mc, Gc = lam_i[2] + 2 * mu_i[2], mu_i[2]; Ms, Gs = lam_i[4] + 2 * mu_i[4], mu_i[4]
            hM = np.where(hard > 0, hard / M_h, 0.0); hG = np.where(hard > 0, hard / G_h, 0.0)
            # cartilage cells: cartilage + hard in series, soft tissue alongside
            M_ch = fch / (f[:, 2] / Mc + hM); G_ch = fch / (f[:, 2] / Gc + hG)
            lam_c = (fch * (M_ch - 2 * G_ch) + f[:, 4] * lam_i[4]) / occ; mu_c = (fch * G_ch + f[:, 4] * mu_i[4]) / occ
            # joint cells without cartilage: soft tissue + hard in series
            M_s = occ / (f[:, 4] / Ms + hM); G_s = occ / (f[:, 4] / Gs + hG)
            lam_s = M_s - 2 * G_s; mu_s = G_s
        if rule == "iface4":
            # outside cartilage: majority rule between soft tissue and the hard phases (no partial-volume mixing of
            # soft tissue with bone, whose correct form depends on how the soft region is confined); fascia + soft
            # tissue mixed by volume (a tension band inside the soft tissue)
            fs, fh = f[:, 4], hard
            lam_hd = np.where(hard > 0, lam_h, 0.0); mu_hd = np.where(hard > 0, mu_h, 0.0)
            fasc_only = (f[:, 0] + f[:, 1] == 0) & (f[:, 3] > 0)
            lam_fv = np.where(fasc_only, (f[:, 3] * lam_i[3] + fs * lam_i[4]) / np.maximum(f[:, 3] + fs, 1e-9), 0.0)
            mu_fv = np.where(fasc_only, (f[:, 3] * mu_i[3] + fs * mu_i[4]) / np.maximum(f[:, 3] + fs, 1e-9), 0.0)
            soft_major = fs > fh
            lam_s = np.where(fasc_only, lam_fv, np.where(soft_major, lam_i[4], lam_hd))
            mu_s = np.where(fasc_only, mu_fv, np.where(soft_major, mu_i[4], mu_hd))
            lamV = lam_s; muV = mu_s
        if rule == "iface3c":
            # soft-tissue gaps without cartilage: Young's moduli in series, volume-averaged Poisson's ratio (iface2)
            nu_s = np.clip(nu, 0.0, 0.49)
            lam_s = soft_series * nu_s / ((1 + nu_s) * (1 - 2 * nu_s)); mu_s = soft_series / (2 * (1 + nu_s))
        lamX = np.where(cart, lam_c, np.where(ser, lam_s, lamV)); muX = np.where(cart, mu_c, np.where(ser, mu_s, muV))
        with np.errstate(divide="ignore", invalid="ignore"):
            E = muX * (3 * lamX + 2 * muX) / (lamX + muX); nu = lamX / (2 * (lamX + muX))
    active = np.where(ser, occ > 0.15, occ >= 0.5)
    E = np.where(active, E, 1.0); nu = np.where(active, np.clip(nu, 0.0, 0.4995 if rule in ("iface3", "iface3c", "iface4") else 0.49), 0.3)
    return E, nu, active, F, ser


def solve_with_springs(g, lam, mu, active, fixed_cells, loads, extra=None, tol=1e-6):
    """Like fe.solver.solve(method='gmg') with an optional extra stiffness callback extra(nmap, n) -> sparse."""
    from fe.gmg import pcg_gmg
    K, nmap, used, cells, dofs = assemble(g, lam, mu, active)
    n = K.shape[0]
    f = np.zeros(n); kd = np.zeros(n)
    nb_, ab_ = face_node_areas(g, fixed_cells, "z-")
    ln = nmap[nb_]; ok = ln >= 0
    for c in range(3):
        np.add.at(kd, 3 * ln[ok] + c, ab_[ok] * 1e6)
    for nodes, area, t in loads:
        ln = nmap[nodes]; ok = ln >= 0
        for c in range(3):
            np.add.at(f, 3 * ln[ok] + c, area[ok] * t[c])
    K = K + sp.diags(kd)
    if extra is not None:
        K = K + extra(nmap, n)
    u, info = pcg_gmg(K, f, used, (g.xe, g.ye, g.ze), tol=tol)
    U = np.full((g.nnode, 3), np.nan); U[used] = u.reshape(-1, 3)
    info["ndof"] = n
    # ground reaction per node (spring force on the structure = -k u)
    R = np.zeros((g.nnode, 3))
    np.add.at(R, nb_, -(ab_[:, None] * 1e6) * np.nan_to_num(U[nb_]))
    return U, info, R


def spring_matrix(g, springs, bone_nodes):
    """Rank-1 stiffness of each ligament spring: k (d . (mean u_A - mean u_B))^2; ends = bone nodes within 4 mm."""
    def build(nmap, n):
        rows, cols, vals = [], [], []
        X = g.node_xyz
        for s in springs:
            a, b = np.array(s["a"]), np.array(s["b"])
            d = (b - a) / np.linalg.norm(b - a)
            w = np.zeros(n)
            for p, sign, key in ((a, 1.0, "A"), (b, -1.0, "B")):
                cand = bone_nodes[s[key]]
                dist = np.linalg.norm(X[cand] - p, axis=1)
                sel = cand[dist <= max(4.0, np.sort(dist)[min(7, len(dist) - 1)])]
                ln = nmap[sel]; sel = sel[ln >= 0]; ln = ln[ln >= 0]
                for c in range(3):
                    np.add.at(w, 3 * ln + c, sign * d[c] / len(ln))
            nz_ = np.nonzero(w)[0]; wv = w[nz_]
            rows.append(np.repeat(nz_, len(nz_))); cols.append(np.tile(nz_, len(nz_))); vals.append(s["k"] * np.outer(wv, wv).ravel())
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return build


def element_forces(g, lam, mu, U, cells):
    """Internal nodal forces K_e u_e (n, 8, 3) of the given cells."""
    out = np.zeros((len(cells), 24))
    did = g.dim_id[cells]
    ue = np.nan_to_num(U[g.cell_nodes[cells]].reshape(len(cells), 24))
    for d in np.unique(did):
        m = did == d
        out[m] = lam[cells[m], None] * (ue[m] @ g.KL[d].T) + mu[cells[m], None] * (ue[m] @ g.KM[d].T)
    return out.reshape(len(cells), 8, 3)


def region_forces(g, lam, mu, U, active, target, label_of_cell, labels):
    """Force (N, engine frame) that each labelled group of outside cells exerts on the target cell set."""
    tn = np.zeros(g.nnode, bool); tn[g.cell_nodes[np.nonzero(target)[0]].ravel()] = True
    out = np.nonzero(active & ~target)[0]
    out = out[tn[g.cell_nodes[out]].any(1)]
    fe = element_forces(g, lam, mu, U, out)
    msk = tn[g.cell_nodes[out]]                                               # (n, 8)
    fsum = -(fe * msk[:, :, None]).sum(1)                                    # force on the target nodes
    res = {}
    lab = label_of_cell[out]
    for k, name in enumerate(labels):
        res[name] = fsum[lab == k].sum(0)
    return res


def run(pkl, out, h_fine=3.0, h_coarse=6.0, z_cut=150.0, ns=3, rule="iface", nu_soft=0.49, use_springs=True,
        roles=NORMAL_ROLES, cache=None, tol=1e-6, forces=VR.FORCES, tag=""):
    t0 = time.time()
    V = voxelize(pkl, h_fine, h_coarse, z_cut, ns, cache=cache)
    g = Grid(V["xe"], V["ye"], V["ze"])
    cls = {int(k): int(v) for k, v in V["cls"].items()}
    E, nu, active, F, ser = materials(V, roles, rule, nu_soft)
    lam, mu = lame(E, nu)
    L = V["L"]; nsub = L.shape[1]
    bf = lambda bodies: np.isin(L, list(bodies)).mean(1)
    cells = np.arange(g.ncell)
    fixed = cells[(g.cK == 0) & active]
    top = cells[(g.cK == g.nz - 1) & active]
    loads = []
    for name, Fz in forces.items():
        fb = bf(roles["bones"][name])
        c = top[fb[top] > 0.5]
        n_, a_ = face_node_areas(g, c, "z+")
        loads.append((n_, a_, np.array([0.0, 0.0, -Fz / a_.sum()])))
    extra = None
    springs = json.loads(str(V["springs_json"]))
    if use_springs and springs:
        # map spring ends to bones: names "Part 8" etc. -> bone of that cortical body
        bone_nodes = {}
        for s in springs:
            nm = s["name"].replace("Longitudinal - ", "")
            pa, pb_ = [x.strip() for x in nm.split(" To ")]
            for key, part in (("A", pa), ("B", pb_)):
                body = int(part.split()[-1]) - 4 if part.startswith("Part ") else None   # "Part k" <-> body k-4
                bone = [nm_ for nm_, bs in roles["bones"].items() if body in bs]
                s[key] = bone[0]
        for name, bodies in roles["bones"].items():
            fb = bf(bodies)
            bone_nodes[name] = np.unique(g.cell_nodes[cells[(fb > 0.5) & active]].ravel())
        extra = spring_matrix(g, springs, bone_nodes)
    U, info, R = solve_with_springs(g, lam, mu, active, fixed, loads, extra, tol)
    t1 = time.time()
    act = cells[active]
    sig, vm, eps = cell_stress(g, lam, mu, U, act)
    VM = np.full(g.ncell, np.nan); VM[act] = vm
    PR = np.full(g.ncell, np.nan)
    S = np.zeros((len(act), 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = sig[:, 0], sig[:, 1], sig[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sig[:, 3]; S[:, 1, 2] = S[:, 2, 1] = sig[:, 4]; S[:, 0, 2] = S[:, 2, 0] = sig[:, 5]
    PR[act] = -np.linalg.eigvalsh(S)[:, 0]                                    # compressive principal stress
    res = {"tag": tag, "rule": rule, "grid": list(g.shape), "h_fine": h_fine, "h_coarse": h_coarse, "z_cut": z_cut,
           "ns": ns, "nu_soft": nu_soft, "springs": bool(extra is not None), "ndof": int(info["ndof"]),
           "iters": info.get("iters"), "time_s": round(time.time() - t0, 1), "n_series_cells": int((ser & active).sum())}
    for key, b in roles["cart"].items():
        fb = bf((b,))
        for thr in (0.1, 0.25, 0.5):
            m = (fb >= thr) & active
            if m.sum():
                res[f"{key}_vm_mean@{thr}"] = float(np.mean(VM[m])); res[f"{key}_vm_max@{thr}"] = float(np.max(VM[m]))
                res[f"{key}_p_mean@{thr}"] = float(np.mean(PR[m])); res[f"{key}_n@{thr}"] = int(m.sum())
    # forces on talus and calcaneus by neighbouring region
    names = list(roles["cart"]) + ["soft", "fascia", "bone", "other"]
    lab = np.full(g.ncell, len(names) - 1, np.int16)
    dom = np.argmax(F[:, 1:6], 1) + 1
    lab[dom == CL.index("soft")] = names.index("soft")
    lab[dom == CL.index("fascia")] = names.index("fascia")
    lab[np.isin(dom, [1, 2])] = names.index("bone")
    for key, b in roles["cart"].items():                                     # any cartilage of the joint wins
        lab[bf((b,)) > 0] = names.index(key)
    for bone in ("talus", "calcaneus"):
        tgt = (bf(roles["bones"][bone]) >= 0.5) & active & ~(F[:, 3] > 0)
        rf = region_forces(g, lam, mu, U, active, tgt, lab, names)
        res[f"F_on_{bone}"] = {k: [round(float(x), 1) for x in v] for k, v in rf.items() if np.abs(v).sum() > 0.05}
    xc = g.node_xyz[:, 0]
    res["ground_Fz"] = {"heel(x<-40)": float(R[xc < -40, 2].sum()), "mid(-40..20)": float(R[(xc >= -40) & (xc < 20), 2].sum()),
                        "fore(x>=20)": float(R[xc >= 20, 2].sum())}
    nb_, ab_ = face_node_areas(g, fixed, "z-")
    A = np.bincount(nb_, weights=ab_, minlength=g.nnode)
    res["ground_peak_MPa"] = float((R[A > 0, 2] / A[A > 0]).max())
    res["ground_total_Fz"] = float(R[:, 2].sum())
    ft = bf(roles["bones"]["talus"]); fc = bf(roles["bones"]["calcaneus"])
    tal = np.unique(g.cell_nodes[cells[(ft > 0.5) & active]]); cal = np.unique(g.cell_nodes[cells[(fc > 0.5) & active]])
    res["talus_u_mean"] = np.nanmean(U[tal], 0).round(4).tolist(); res["calc_u_mean"] = np.nanmean(U[cal], 0).round(4).tolist()
    res["solve_s"] = round(t1 - t0, 1)
    np.savez_compressed(out, VM=VM.astype(np.float32), PR=PR.astype(np.float32), U=U.astype(np.float32),
                        xe=g.xe, ye=g.ye, ze=g.ze, active=active, E=E.astype(np.float32), res=json.dumps(res))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl"); ap.add_argument("out")
    ap.add_argument("--h", type=float, default=3.0); ap.add_argument("--hc", type=float, default=6.0)
    ap.add_argument("--zcut", type=float, default=150.0); ap.add_argument("--ns", type=int, default=3)
    ap.add_argument("--rule", default="iface"); ap.add_argument("--nu_soft", type=float, default=0.49)
    ap.add_argument("--no_springs", action="store_true"); ap.add_argument("--cache", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    r = run(a.pkl, a.out, a.h, a.hc, a.zcut, a.ns, a.rule, a.nu_soft, not a.no_springs, cache=a.cache, tag=a.tag)
    print(json.dumps(r, indent=1))
