"""Voxel replica of an ANSYS atlas model, version 3: split nodes between different bones (Claude, 2026-10-28).
See splitfe.py for the rule and replica2.py for voxelisation and material mixing."""
import os, sys, time, json, pickle, numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.environ.get("COMECH", os.path.dirname(HERE)))
import replica2 as R2
import splitfe as SF
import voxel_replica as VR
from fe.solver import Grid, lame

CL = R2.CL


def bone_share(V, roles):
    cls = {int(k): int(v) for k, v in V["cls"].items()}
    L = V["L"]; nb = max(max(cls), int(L.max()))
    grp, names = R2.bone_groups(cls, roles)
    lutg = np.zeros(nb + 1, np.int16)
    for b, k in grp.items():
        lutg[b] = k
    G = lutg[L]
    ng = max(names) + 1
    S = np.zeros((L.shape[0], ng), np.float32)
    for k in range(1, ng):
        S[:, k] = (G == k).mean(1)
    return S, names


def facet_tris(pkl, bodies, cache=None):
    """Boundary triangles (centroid, unit normal, area) of the given cartilage bodies of the atlas mesh."""
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    from frame import to_engine
    D = pickle.load(open(pkl, "rb"))
    ids = np.array(sorted(D["nodes"])); X = np.array([D["nodes"][k] for k in ids])
    pos = np.full(ids.max() + 1, -1, np.int64); pos[ids] = np.arange(len(ids))
    XE = to_engine(X)
    out = {}
    for b in D["bodies"]:
        if b["index"] not in bodies:
            continue
        E = np.array(b["elems"])[:, :4]
        faces = np.concatenate([E[:, [0, 1, 2]], E[:, [0, 1, 3]], E[:, [0, 2, 3]], E[:, [1, 2, 3]]])
        u, cnt = np.unique(np.sort(faces, 1), axis=0, return_counts=True)
        T = XE[pos[u[cnt == 1]]]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); a = 0.5 * np.linalg.norm(n, axis=1)
        out[b["index"]] = (T.mean(1), n / (2 * a[:, None] + 1e-12), a)
    if cache:
        pickle.dump(out, open(cache, "wb"))
    return out


def cart_vm(sig, nrm, nu_c=0.4):
    """Von Mises stress of a thin cartilage layer bonded between stiff bones, from the traction on its plane:
    sqrt((sigma_nn (1-2nu)/(1-nu))^2 + 3 tau^2) (in-plane strains of the layer ~ 0)."""
    S = np.zeros((len(sig), 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = sig[:, 0], sig[:, 1], sig[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sig[:, 3]; S[:, 1, 2] = S[:, 2, 1] = sig[:, 4]; S[:, 0, 2] = S[:, 2, 0] = sig[:, 5]
    t = np.einsum("nij,nj->ni", S, nrm)
    snn = (t * nrm).sum(1)
    tau = np.linalg.norm(t - snn[:, None] * nrm, axis=1)
    return np.sqrt((snn * (1 - 2 * nu_c) / (1 - nu_c)) ** 2 + 3 * tau ** 2), -snn, tau


def spring_matrix_cn(xyz, springs, bone_nodes):
    def build(nmap, n):
        rows, cols, vals = [], [], []
        for s in springs:
            a, b = np.array(s["a"]), np.array(s["b"])
            d = (b - a) / np.linalg.norm(b - a)
            w = np.zeros(n)
            for p, sign, key in ((a, 1.0, "A"), (b, -1.0, "B")):
                cand = bone_nodes[s[key]]
                cand = cand[nmap[cand] >= 0]
                dist = np.linalg.norm(xyz[cand] - p, axis=1)
                sel = cand[dist <= max(4.0, np.sort(dist)[min(7, len(dist) - 1)])]
                ln = nmap[sel]
                for c in range(3):
                    np.add.at(w, 3 * ln + c, sign * d[c] / len(ln))
            nz_ = np.nonzero(w)[0]; wv = w[nz_]
            rows.append(np.repeat(nz_, len(nz_))); cols.append(np.tile(nz_, len(nz_))); vals.append(s["k"] * np.outer(wv, wv).ravel())
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return build


def spring_ends(springs, roles):
    """Spring end names 'Part k' -> bone names, through roles['part_body'] (body index of each part name)."""
    pb_ = roles.get("part_body") or {f"Part {k + 4}": k for k in range(1, 70)}      # normal model: body k = Part k+4
    for s in springs:
        nm = s["name"].replace("Longitudinal - ", "")
        pa, pb = [x.strip() for x in nm.split(" To ")]
        for key, part in (("A", pa), ("B", pb)):
            body = pb_[part]
            s[key] = [n_ for n_, bs in roles["bones"].items() if body in bs][0]
    return springs


def run(pkl, out, h_fine=3.0, h_coarse=6.0, z_cut=150.0, ns=3, rule="iface", nu_soft=0.49, use_springs=True,
        split=True, roles=R2.NORMAL_ROLES, cache=None, tol=1e-6, forces=VR.FORCES, tag="", rigid=True, save=True,
        ground_occ=0.5, box=((-105, 125), (-50, 45), (0, 100)), cls_override=None):
    t0 = time.time()
    V = R2.voxelize(pkl, h_fine, h_coarse, z_cut, ns, box=box, cache=cache)
    if cls_override:                                   # e.g. give mis-assigned cartilage bodies their cartilage class
        V = dict(V); V["cls"] = dict(V["cls"])
        for b, c in cls_override.items():
            V["cls"][int(b)] = int(c)
    g = Grid(V["xe"], V["ye"], V["ze"])
    aniso_rule = rule.endswith("a") and rule.startswith("iface3")
    E, nu, active, F, ser = R2.materials(V, roles, "iface3" if aniso_rule else rule, nu_soft)
    lam, mu = lame(E, nu)
    L = V["L"]
    bf = lambda bodies: np.isin(L, list(bodies)).mean(1)
    S, bnames = bone_share(V, roles)
    cart = F[:, 3] > 0
    part = np.where((S.max(1) >= 0.5) & ~cart, S.argmax(1), 0)
    if split:
        cn, orig, nsplit = SF.split_connectivity(g, active, part, S)
    else:
        cn, orig, nsplit = g.cell_nodes, np.arange(g.nnode), 0
    t_split = time.time() - t0
    cells = np.arange(g.ncell)
    # ground: bonded only where the soft tissue reaches the plate (lowest sub-sample row of the bottom cells occupied),
    # as in the atlas, where the soft tissue is bonded to the plate only where it touches it (heel and forefoot)
    ns_ = int(round(L.shape[1] ** (1 / 3)))
    bot_row = (np.arange(L.shape[1]) % ns_) == 0
    occ_bot = (L[:, bot_row] > 0).mean(1)
    if "contact" in V:
        # bonded to the ground in proportion to the contact footprint of each bottom cell
        cf = V["contact"].ravel()[(g.cI * g.ny + g.cJ)]                       # per cell (i, j) -> footprint fraction
        fixed_cells = cells[(g.cK == 0) & active & (cf > 0)]
        cfw = cf[fixed_cells]
    else:
        fixed_cells = cells[(g.cK == 0) & active & (occ_bot >= ground_occ)]
        cfw = np.ones(len(fixed_cells))
    res_ground = {"n_fixed_cells": int(len(fixed_cells)), "contact_area_mm2": float((g.dx[g.cI[fixed_cells]] * g.dy[g.cJ[fixed_cells]] * cfw).sum())}
    fixed = SF.face_nodes_cn(g, cn, fixed_cells, "z-")
    fixed = (fixed[0], fixed[1] * np.repeat(cfw, 4))
    top = cells[(g.cK == g.nz - 1) & active]
    loads = []
    lines = roles.get("load_line", {})
    for name, Fz in forces.items():
        c = top[bf(roles["bones"][name])[top] > 0.5]
        n_, a_ = SF.face_nodes_cn(g, cn, c, "z+")
        if name in lines:
            # linearly varying traction on the cut section with the resultant on the line of action of the atlas load
            # (uniform pressure on the upper 70 mm of the bone surface, far above the cut): same force and moment
            X = g.node_xyz[orig[n_]][:, :2]; w = a_
            xb = (w[:, None] * X).sum(0) / w.sum(); dX = X - xb
            I = np.einsum("n,ni,nj->ij", w, dX, dX)
            e = np.array(lines[name]) - xb                       # offset of the line of action from the section centroid
            ab = np.linalg.solve(I, w.sum() * e)                 # t = -F/A (1 + ab . dX): sum(w t dX) = -F e
            tz = -Fz / w.sum() * (1 + dX @ ab)
            loads.append((n_, a_ * tz, np.array([0.0, 0.0, 1.0])))
            res_ground[f"load_offset_{name}"] = [round(float(x), 1) for x in e]
        else:
            loads.append((n_, a_, np.array([0.0, 0.0, -Fz / a_.sum()])))
    xyz = g.node_xyz[orig]
    bone_nodes = {}
    for name, bodies in roles["bones"].items():
        k = [i for i, nm in bnames.items() if nm == name][0]
        bone_nodes[name] = np.unique(cn[cells[(part == k) & active]].ravel())
    extra = None
    springs = json.loads(str(V["springs_json"]))
    springs = [s_ for s_ in springs if max(s_["a"][2], s_["b"][2]) < z_cut - 2 * h_coarse]
    if use_springs and springs:
        # each spring end goes to the nearest bone (names in the atlas files are not reliable)
        trees = {nm: cKDTree(xyz[nd]) for nm, nd in bone_nodes.items() if len(nd)}
        for s_ in springs:
            for key, p in (("A", s_["a"]), ("B", s_["b"])):
                s_[key] = min(trees, key=lambda nm: trees[nm].query(p)[0])
            s_["d_end"] = [round(float(trees[s_["A"]].query(s_["a"])[0]), 2), round(float(trees[s_["B"]].query(s_["b"])[0]), 2)]
        extra = spring_matrix_cn(xyz, springs, bone_nodes)
    groups = None
    if rigid:
        groups = [np.unique(cn[cells[(part == k) & active]].ravel()) for k in bnames if ((part == k) & active).sum() > 3]
    aniso = None
    if aniso_rule:
        # cartilage cells as laminates: bone / cartilage layers in series along the local joint normal (nearest
        # boundary triangle of the atlas cartilage), soft tissue alongside
        cart_bodies = [b for b, c in {int(k): int(v) for k, v in V["cls"].items()}.items() if c == CL.index("cartilage")]
        ftri_all = facet_tris(pkl, cart_bodies, cache=(cache.replace(".npz", "") + "_allcart.pkl") if cache else None)
        cen = np.concatenate([v[0] for v in ftri_all.values()]); nrm_all = np.concatenate([v[1] for v in ftri_all.values()])
        ac = cells[active & (F[:, 3] > 0)]
        # joint normal: smallest principal direction of the cartilage boundary points within 4 mm (thin layer)
        tree = cKDTree(cen)
        nrm_c = np.zeros((len(ac), 3))
        for ii, p in enumerate(g.cell_xyz[ac]):
            nb = tree.query_ball_point(p, 4.0)
            if len(nb) < 6:
                nb = tree.query(p, k=12)[1]
            Q = cen[nb] - cen[nb].mean(0)
            w_, V_ = np.linalg.eigh(Q.T @ Q)
            nrm_c[ii] = V_[:, 0]
        Ev = np.array([VR.MAT[c][0] for c in CL[1:6]]); nv = np.array([VR.MAT[c][1] for c in CL[1:6]]); nv[4] = nu_soft
        lam_i = Ev * nv / ((1 + nv) * (1 - 2 * nv)); mu_i = Ev / (2 * (1 + nv))
        fl = F[ac][:, [1, 2, 3, 4]]                                         # cortical, cancellous, cartilage, fascia
        occ_ = F[ac, 1:6].sum(1)
        f_par = F[ac, 5] / np.maximum(occ_, 1e-9)
        C = SF.laminate_C(fl, lam_i[[0, 1, 2, 3]], mu_i[[0, 1, 2, 3]], nrm_c, f_par, lam_i[4], mu_i[4])
        aniso = (ac, C)
        res_ground["n_aniso_cells"] = int(len(ac))
    U, info, R = SF.solve_cn(g, lam, mu, active, cn, orig, fixed, loads, extra, groups, tol, aniso=aniso)
    t1 = time.time()
    act = cells[active]
    sig, vm, eps = SF.cell_stress_cn(g, lam, mu, U, act, cn, aniso)
    VM = np.full(g.ncell, np.nan); VM[act] = vm
    Sg = np.zeros((len(act), 3, 3))
    Sg[:, 0, 0], Sg[:, 1, 1], Sg[:, 2, 2] = sig[:, 0], sig[:, 1], sig[:, 2]
    Sg[:, 0, 1] = Sg[:, 1, 0] = sig[:, 3]; Sg[:, 1, 2] = Sg[:, 2, 1] = sig[:, 4]; Sg[:, 0, 2] = Sg[:, 2, 0] = sig[:, 5]
    PR = np.full(g.ncell, np.nan); PR[act] = -np.linalg.eigvalsh(Sg)[:, 0]
    res = {"tag": tag, "rule": rule, "split": split, "n_split_nodes": int(nsplit), "grid": list(g.shape), "h_fine": h_fine,
           "h_coarse": h_coarse, "z_cut": z_cut, "ns": ns, "nu_soft": nu_soft, "springs": bool(extra is not None),
           "ndof": int(info["ndof"]), "iters": info.get("iters"), "flag": info.get("flag"), "t_prep_s": round(t_split, 1),
           "t_solve_s": round(t1 - t0 - t_split, 1), "sri": bool(SF.SRI), **res_ground}
    ftri = facet_tris(pkl, list(roles["cart"].values()), cache=(cache.replace(".npz", "") + "_facets.pkl") if cache else None)
    pos_act = -np.ones(g.ncell, np.int64); pos_act[act] = np.arange(len(act))
    for key, b in roles["cart"].items():
        fb = bf((b,))
        for thr in (0.25,):
            m = (fb >= thr) & active
            if m.sum():
                res[f"{key}_vm_mean@{thr}"] = float(np.mean(VM[m])); res[f"{key}_vm_max@{thr}"] = float(np.max(VM[m]))
                res[f"{key}_n@{thr}"] = int(m.sum())
        m = (fb > 0) & active
        if m.sum() and b in ftri:
            cen, nrm_t, _ = ftri[b]
            _, j = cKDTree(cen).query(g.cell_xyz[m])
            vmc, pn, tau = cart_vm(sig[pos_act[m]], nrm_t[j])
            w = fb[m]
            res[f"{key}_cvm_wmean"] = float((w * vmc).sum() / w.sum())          # cartilage-volume-weighted mean
            res[f"{key}_cvm_max@0.25"] = float(vmc[w >= 0.25].max()) if (w >= 0.25).any() else None
            res[f"{key}_p_wmean"] = float((w * pn).sum() / w.sum())
            res[f"{key}_tau_wmean"] = float((w * tau).sum() / w.sum())
    # von Mises statistics of every bone (cells of that bone's part; the atlas 'tc' value is the talus cortical maximum)
    for k, nm in bnames.items():
        m = (part == k) & active
        if m.sum() > 10:
            v = VM[m]
            res[f"bonevm_{nm}"] = [round(float(np.percentile(v, 50)), 4), round(float(np.percentile(v, 99)), 4), round(float(v.max()), 4)]
    # forces on the talus / calcaneus from each neighbouring region (cells of other parts touching their nodes)
    names = list(roles["cart"]) + ["soft", "fascia", "bone", "other"]
    lab = np.full(g.ncell, len(names) - 1, np.int16)
    dom = np.argmax(F[:, 1:6], 1) + 1
    lab[dom == CL.index("soft")] = names.index("soft")
    lab[dom == CL.index("fascia")] = names.index("fascia")
    lab[np.isin(dom, [1, 2])] = names.index("bone")
    for key, b in roles["cart"].items():
        lab[bf((b,)) > 0] = names.index(key)
    for bone in [b_ for b_ in ("talus", "calcaneus") if b_ in bnames.values()]:
        k = [i for i, nm in bnames.items() if nm == bone][0]
        tgt = (part == k) & active
        tn = np.zeros(len(orig), bool); tn[cn[cells[tgt]].ravel()] = True
        oc = cells[active & ~tgt]
        oc = oc[tn[cn[oc]].any(1)]
        fe = SF.element_forces_cn(g, lam, mu, U, oc, cn, aniso)
        fsum = -(fe * tn[cn[oc]][:, :, None]).sum(1)
        res[f"F_on_{bone}"] = {nm: [round(float(x), 1) for x in fsum[lab[oc] == i].sum(0)] for i, nm in enumerate(names)
                               if (lab[oc] == i).any()}
        if extra is not None:   # ligament springs acting on this bone
            fs = np.zeros(3)
            for s in springs:
                for key, other, sgn in (("A", "B", 1.0), ("B", "A", -1.0)):
                    if s[key] == bone:
                        a, b = np.array(s["a"]), np.array(s["b"]); d = (b - a) / np.linalg.norm(b - a)
                        ua = np.nanmean(U[bone_nodes[s["A"]][np.linalg.norm(xyz[bone_nodes[s["A"]]] - a, axis=1) <= 4.0]], 0)
                        ub = np.nanmean(U[bone_nodes[s["B"]][np.linalg.norm(xyz[bone_nodes[s["B"]]] - b, axis=1) <= 4.0]], 0)
                        ext = d @ (ub - ua)                       # spring elongation
                        fs += sgn * s["k"] * ext * d             # force on end A is +k*ext*d (pulls towards B)
            res[f"F_on_{bone}"]["springs"] = [round(float(x), 1) for x in fs]
    xc = xyz[:, 0]
    res["ground_Fz"] = {"heel(x<-40)": round(float(R[xc < -40, 2].sum()), 1),
                        "mid(-40..20)": round(float(R[(xc >= -40) & (xc < 20), 2].sum()), 1),
                        "fore(x>=20)": round(float(R[xc >= 20, 2].sum()), 1)}
    A = np.bincount(fixed[0], weights=fixed[1], minlength=len(orig))
    full = A >= 0.5 * h_fine ** 2
    p = R[full, 2] / A[full]
    res["ground_peak_MPa"] = float(p.max()) if full.any() else None
    res["ground_p99_MPa"] = float(np.percentile(p, 99)) if full.any() else None
    res["ground_total_Fz"] = float(R[:, 2].sum())
    for bone in [b_ for b_ in ("talus", "calcaneus", "talocalcaneal") if b_ in bone_nodes]:
        res[f"{bone}_u_mean"] = np.nanmean(U[bone_nodes[bone]], 0).round(4).tolist()
    res["time_s"] = round(time.time() - t0, 1)
    if save:
        np.savez_compressed(out, VM=VM.astype(np.float32), PR=PR.astype(np.float32), xe=g.xe, ye=g.ye, ze=g.ze,
                            active=active, E=E.astype(np.float32), nu=nu.astype(np.float32), part=part.astype(np.int16),
                            U=U.astype(np.float32), cn=cn.astype(np.int32), orig=orig.astype(np.int32),
                            bnames=json.dumps({int(k): v for k, v in bnames.items()}), res=json.dumps(res))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl"); ap.add_argument("out")
    ap.add_argument("--h", type=float, default=3.0); ap.add_argument("--hc", type=float, default=6.0)
    ap.add_argument("--zcut", type=float, default=150.0); ap.add_argument("--ns", type=int, default=3)
    ap.add_argument("--rule", default="iface"); ap.add_argument("--nu_soft", type=float, default=0.49)
    ap.add_argument("--no_springs", action="store_true"); ap.add_argument("--no_split", action="store_true")
    ap.add_argument("--no_rigid", action="store_true")
    ap.add_argument("--cache", default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    r = run(a.pkl, a.out, a.h, a.hc, a.zcut, a.ns, a.rule, a.nu_soft, not a.no_springs, not a.no_split,
            cache=a.cache, tag=a.tag, rigid=not a.no_rigid)
    keys = [k for k in r if not any(k.startswith(p) for p in ("ankle_", "syndesmosis_", "talofibular_"))]
    print(json.dumps({k: r[k] for k in keys}, indent=0))
