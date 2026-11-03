"""Voxel FE with split nodes between different bones (Claude, 2026-10-28).

On a voxel grid, two bones whose surfaces are closer than a cell end up in cells that share nodes, so they are welded
together and the joint between them (cartilage or soft tissue) is bypassed.  Here every active cell belongs to a part:
the bone that fills at least half of it (cells without cartilage), or the joint/soft part 0.  A node shared by cells of
two or more different bones is duplicated, one copy per bone, so different bones are connected only through part-0
cells (cartilage, soft tissue, fascia), as in the atlas where every bone is bonded to its cartilage and soft tissue but
never to another bone directly.  A part-0 cell at a split node is attached to the copy of the bone that fills most of
it (or, if it holds none of those bones, of the bone with most cells around the node).

The global stiffness is assembled on the new connectivity; the geometric multigrid of the engine is reused with each
copy interpolated like its original node."""
import numpy as np
import scipy.sparse as sp


def split_connectivity(g, active, part, bone_share):
    """part: (ncell,) 0 for joint/soft cells, k>0 bone id; bone_share: (ncell, nbone+1) fraction of each bone in the cell.
    Returns cn (ncell, 8) node ids (valid for active cells), orig (nnew,) original grid node of each node id."""
    cells = np.nonzero(active)[0]
    cn0 = g.cell_nodes[cells]                                   # (n, 8)
    pc = part[cells]
    # bone parts touching each node
    nb = bone_share.shape[1]
    touch = np.zeros((g.nnode, nb), np.int32)
    for l in range(8):
        np.add.at(touch, (cn0[:, l], pc), 1)
    touch[:, 0] = 0
    nparts = (touch > 0).sum(1)
    split = np.nonzero(nparts >= 2)[0]
    # new ids: originals keep their id; each extra (node, bone) pair gets a new id
    newid = {}
    nxt = g.nnode
    orig_extra = []
    for n in split:
        bones = np.nonzero(touch[n])[0]
        # the first bone keeps the original id
        for b in bones[1:]:
            newid[(n, b)] = nxt; orig_extra.append(n); nxt += 1
    orig = np.concatenate([np.arange(g.nnode), np.array(orig_extra, np.int64)])
    is_split = np.zeros(g.nnode, bool); is_split[split] = True
    first_bone = np.zeros(g.nnode, np.int64)
    first_bone[split] = [np.nonzero(touch[n])[0][0] for n in split]
    cn = g.cell_nodes.copy()
    rows, ls = np.nonzero(is_split[cn0])
    for r, l in zip(rows, ls):
        n = cn0[r, l]; c = cells[r]
        b = pc[r]
        if b == 0:
            cand = np.nonzero(touch[n])[0]
            sh = bone_share[c, cand]
            b = cand[np.argmax(sh)] if sh.max() > 0 else cand[np.argmax(touch[n, cand])]
        if b != first_bone[n]:
            cn[c, l] = newid[(n, b)]
    return cn, orig, len(split)


SRI = True          # selective reduced integration: volumetric (lambda) term at the cell centre (no volumetric locking)


def KL_family(g):
    """Lambda-part element matrices per family of cell dimensions (one-point if SRI, else 2x2x2 as in the engine)."""
    if not SRI:
        return g.KL
    if getattr(g, "_KL_sri", None) is None:
        DL = np.zeros((6, 6)); DL[:3, :3] = 1.0
        vol = g.udims.prod(1)
        g._KL_sri = np.stack([g.Bc[d].T @ DL @ g.Bc[d] * vol[d] for d in range(len(g.udims))])
    return g._KL_sri


def assemble_cn(g, lam, mu, cells, cn, aniso=None):
    nused = np.unique(cn[cells])
    nmap = -np.ones(cn.max() + 1, np.int64); nmap[nused] = np.arange(len(nused))
    ln = nmap[cn[cells]]
    dofs = (3 * ln[:, :, None] + np.arange(3)[None, None, :]).reshape(len(cells), 24)
    did = g.dim_id[cells]
    Ke = np.empty((len(cells), 576))
    KLf = KL_family(g)
    for d in np.unique(did):
        m = np.nonzero(did == d)[0]
        Ke[m] = lam[cells[m], None] * KLf[d].reshape(1, 576) + mu[cells[m], None] * g.KM[d].reshape(1, 576)
    if aniso is not None:
        ac, C = aniso
        pos = np.searchsorted(cells, ac)
        Ke[pos] = aniso_Ke(g, ac, C).reshape(len(ac), 576)
    d32 = dofs.astype(np.int64)
    rows = np.repeat(d32, 24, axis=1).ravel(); cols = np.tile(d32, (1, 24)).ravel()
    n = 3 * len(nused)
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    return K, nmap, nused


def face_nodes_cn(g, cn, cells, face):
    from fe.solver import _LOC
    ax = "xyz".index(face[0]); side = 1 if face[1] == "+" else 0
    sel = _LOC[:, ax] == side
    nodes = cn[cells][:, sel]
    d = [g.dx[g.cI[cells]], g.dy[g.cJ[cells]], g.dz[g.cK[cells]]]
    other = [d[i] for i in range(3) if i != ax]
    area = (other[0] * other[1] / 4)[:, None].repeat(4, 1)
    return nodes.ravel(), area.ravel()


def rigid_modes_cn(xyz, nmap, n, node_groups):
    rows, cols, vals = [], [], []
    col = 0
    for nodes in node_groups:
        ln = nmap[nodes]; ok = ln >= 0
        nodes, ln = nodes[ok], ln[ok]
        if len(nodes) < 4:
            continue
        X = xyz[nodes]; r = (X - X.mean(0)) / 20.0
        for d in range(3):
            rows.append(3 * ln + d); cols.append(np.full(len(ln), col)); vals.append(np.ones(len(ln))); col += 1
        for comps in [((1, -r[:, 2]), (2, r[:, 1])), ((0, r[:, 2]), (2, -r[:, 0])), ((0, -r[:, 1]), (1, r[:, 0]))]:
            for d, v in comps:
                rows.append(3 * ln + d); cols.append(np.full(len(ln), col)); vals.append(v)
            col += 1
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, col))


def solve_cn(g, lam, mu, active, cn, orig, fixed, loads, extra=None, node_groups=None, tol=1e-6, maxiter=800, aniso=None):
    """fixed: (node_ids, area) of the ground faces (k = 1e6 N/mm^3 per area); loads: list of (node_ids, area, t[3])."""
    from fe.gmg import pcg_gmg
    cells = np.nonzero(active)[0]
    K = None
    if aniso is None and getattr(pcg_gmg, "__name__", "") == "pcg_gmg_gpu":
        try:
            import gpusolve as _gs
            dev = _gs.pick_device()
            if dev is not None:
                K, nmap, nused, _ = _gs.assemble_gpu(g, lam, mu, cells, cn, KL_family(g), dev)
        except Exception:
            K = None
    if K is None:
        K, nmap, nused = assemble_cn(g, lam, mu, cells, cn, aniso)
    n = K.shape[0]
    f = np.zeros(n); kd = np.zeros(n)
    nb_, ab_ = fixed
    ln = nmap[nb_]
    for c in range(3):
        np.add.at(kd, 3 * ln + c, ab_ * 1e6)
    for nodes, area, t in loads:
        ln = nmap[nodes]
        for c in range(3):
            np.add.at(f, 3 * ln + c, area * t[c])
    K = K + sp.diags(kd)
    if extra is not None:
        K = K + extra(nmap, n)
    xyz = g.node_xyz[orig]
    Z = rigid_modes_cn(xyz, nmap, n, node_groups) if node_groups else None
    u, info = pcg_gmg(K, f, orig[nused], (g.xe, g.ye, g.ze), tol=tol, Z=Z, maxiter=maxiter)
    U = np.full((len(orig), 3), np.nan); U[nused] = u.reshape(-1, 3)
    R = np.zeros((len(orig), 3))
    np.add.at(R, nb_, -(ab_[:, None] * 1e6) * U[nb_])
    info["ndof"] = n
    return U, info, R


def cell_stress_cn(g, lam, mu, U, cells, cn, aniso=None):
    ue = np.nan_to_num(U[cn[cells]]).reshape(len(cells), 24)
    B = g.Bc[g.dim_id[cells]]
    eps = np.einsum("nij,nj->ni", B, ue)
    tr = eps[:, :3].sum(1)
    sig = np.empty_like(eps)
    sig[:, :3] = lam[cells, None] * tr[:, None] + 2 * mu[cells, None] * eps[:, :3]
    sig[:, 3:] = mu[cells, None] * eps[:, 3:]
    if aniso is not None:
        ac, C = aniso
        loc = np.searchsorted(cells, ac); ok = (loc < len(cells)) & (cells[np.minimum(loc, len(cells) - 1)] == ac)
        sig[loc[ok]] = np.einsum("nij,nj->ni", C[ok], eps[loc[ok]])
    sx, sy, sz, txy, tyz, txz = sig.T
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + txz ** 2))
    return sig, vm, eps


def element_forces_cn(g, lam, mu, U, cells, cn, aniso=None):
    out = np.zeros((len(cells), 24))
    did = g.dim_id[cells]
    ue = np.nan_to_num(U[cn[cells]].reshape(len(cells), 24))
    KLf = KL_family(g)
    for d in np.unique(did):
        m = did == d
        out[m] = lam[cells[m], None] * (ue[m] @ KLf[d].T) + mu[cells[m], None] * (ue[m] @ g.KM[d].T)
    if aniso is not None:
        ac, C = aniso
        o = np.argsort(cells); sc = cells[o]
        loc = np.searchsorted(sc, ac); ok = (loc < len(sc)) & (sc[np.minimum(loc, len(sc) - 1)] == ac)
        if ok.any():
            idx = o[loc[ok]]
            out[idx] = np.einsum("nij,nj->ni", aniso_Ke(g, ac[ok], C[ok]), ue[idx])
    return out.reshape(len(cells), 8, 3)


# ---------------- anisotropic (laminate) joint cells ----------------
_PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]          # Voigt order of the solver (xx yy zz xy yz xz)


def bond_matrix(R):
    """6x6 stress transformation (engineering-shear Voigt order of the solver) for rotations R (n,3,3), columns =
    local axes in global coordinates: sigma_g = M sigma_l, C_g = M C_l M^T."""
    n = len(R); M = np.zeros((n, 6, 6))
    for p, (i, j) in enumerate(_PAIRS):
        for q, (k, l) in enumerate(_PAIRS):
            M[:, p, q] = R[:, i, k] * R[:, j, l] + (R[:, i, l] * R[:, j, k] if k != l else 0.0)
    return M


def laminate_C(f, lam, mu, nrm, f_par=None, lam_par=None, mu_par=None):
    """Backus average of isotropic layers stacked along nrm (n,3): f (n,m) layer fractions (normalised inside),
    lam, mu (m,) layer constants; optional parallel phase (fraction f_par of the cell) added by volume averaging.
    Returns global-frame stiffness (n,6,6) in the solver's Voigt order."""
    w = f / np.maximum(f.sum(1, keepdims=True), 1e-12)
    P = lam + 2 * mu
    C33 = 1.0 / (w / P).sum(1)
    a = (w * lam / P).sum(1)
    C13 = a * C33
    C11 = (w * 4 * mu * (lam + mu) / P).sum(1) + a ** 2 * C33
    G_in = (w * mu).sum(1); G_tr = 1.0 / (w / mu).sum(1)
    C12 = C11 - 2 * G_in
    n = len(f); CL = np.zeros((n, 6, 6))
    CL[:, 0, 0] = CL[:, 1, 1] = C11; CL[:, 0, 1] = CL[:, 1, 0] = C12
    CL[:, 0, 2] = CL[:, 2, 0] = CL[:, 1, 2] = CL[:, 2, 1] = C13; CL[:, 2, 2] = C33
    CL[:, 3, 3] = G_in; CL[:, 4, 4] = CL[:, 5, 5] = G_tr
    if f_par is not None:
        Ciso = np.zeros((6, 6)); Ciso[:3, :3] = lam_par; Ciso[[0, 1, 2], [0, 1, 2]] += 2 * mu_par; Ciso[[3, 4, 5], [3, 4, 5]] = mu_par
        CL = (1 - f_par)[:, None, None] * CL + f_par[:, None, None] * Ciso[None]
    # local frame: e3 = nrm
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    t = np.where(np.abs(nrm[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(nrm, t); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(nrm, e1)
    R = np.stack([e1, e2, nrm], 2)                                 # columns = local axes
    M = bond_matrix(R)
    return np.einsum("nij,njk,nlk->nil", M, CL, M)


def gauss_B(g):
    """B matrices at the 2x2x2 Gauss points and weights per family of cell dimensions."""
    from fe.solver import _Bmat, _shape_grad
    if getattr(g, "_gB", None) is None:
        q = 1 / np.sqrt(3); out = []
        for d in g.udims:
            Bs = np.stack([_Bmat(_shape_grad(xi, et, ze, d)) for xi in (-q, q) for et in (-q, q) for ze in (-q, q)])
            out.append((Bs, d.prod() / 8))
        g._gB = out
    return g._gB


def aniso_Ke(g, cells, C):
    """Element stiffness (n,24,24) of cells with full anisotropic stiffness C (n,6,6), 2x2x2 Gauss."""
    gB = gauss_B(g); did = g.dim_id[cells]
    Ke = np.zeros((len(cells), 24, 24))
    for d in np.unique(did):
        m = np.nonzero(did == d)[0]
        Bs, w = gB[d]
        Ke[m] = w * np.einsum("gki,nkl,glj->nij", Bs, C[m], Bs)
    return Ke


def split_connectivity_fast(g, active, part, bone_share):
    """Vectorised split_connectivity (same result)."""
    cells = np.nonzero(active)[0]
    cn0 = g.cell_nodes[cells]                                   # (n, 8)
    pc = part[cells]
    nb = bone_share.shape[1]
    # (node, bone) pairs of bone cells, with the number of cells
    m = pc > 0
    nodes = cn0[m].ravel(); bones = np.repeat(pc[m], 8)
    key = nodes.astype(np.int64) * nb + bones
    ukey, cnt = np.unique(key, return_counts=True)
    unode = ukey // nb; ubone = ukey % nb
    first = np.ones(len(ukey), bool); first[1:] = unode[1:] != unode[:-1]
    npair = np.bincount(unode, minlength=g.nnode)
    is_split = npair >= 2
    # new ids for every non-first pair of a split node (in sorted order, as split_connectivity)
    extra = ~first & is_split[unode]
    newid = np.full(len(ukey), -1, np.int64)
    newid[extra] = g.nnode + np.arange(extra.sum())
    orig = np.concatenate([np.arange(g.nnode), unode[extra]])
    pos0 = np.searchsorted(unode, np.arange(g.nnode))          # first pair of every node
    cn = g.cell_nodes.copy()
    rows, ls = np.nonzero(is_split[cn0])
    if len(rows):
        n = cn0[rows, ls]; c = cells[rows]; b = pc[rows].astype(np.int64)
        z = np.nonzero(b == 0)[0]
        if len(z):
            # part-0 cell: bone of the node with the largest share in the cell, else the bone with most cells at the node
            nz_ = n[z]; k = npair[nz_]; kmax = int(k.max())
            best = np.full(len(z), -1, np.int64); bsh = np.full(len(z), -1.0); bcnt = np.full(len(z), -1)
            bcnt_b = np.full(len(z), -1, np.int64)
            for j in range(kmax):
                ok = j < k
                pidx = pos0[nz_[ok]] + j
                bb = ubone[pidx]; sh = bone_share[c[z][ok], bb]; ct = cnt[pidx]
                upd = sh > bsh[ok]
                ii = np.nonzero(ok)[0][upd]; best[ii] = bb[upd]; bsh[ii] = sh[upd]
                upd2 = ct > bcnt[ok]
                jj = np.nonzero(ok)[0][upd2]; bcnt_b[jj] = bb[upd2]; bcnt[jj] = ct[upd2]
            b[z] = np.where(bsh > 0, best, bcnt_b)
        pk = np.searchsorted(ukey, n.astype(np.int64) * nb + b)
        nid = newid[pk]
        sel = nid >= 0
        cn[c[sel], ls[sel]] = nid[sel]
    return cn, orig, int(is_split.sum())


def assemble_structured(g, lam, mu, cells, cn, aniso=None):
    """Same matrix as assemble_cn, assembled block-wise on the grid stencil: cells whose nodes are all grid nodes add
    their 3x3 node-pair blocks into a (node, 27 neighbours) block array (no sorting of 576 entries per cell); only cells
    that touch split-node copies go through the general COO assembly."""
    if aniso is not None:
        return assemble_cn(g, lam, mu, cells, cn, aniso)
    from fe.solver import _LOC
    nused = np.unique(cn[cells])
    nmap = -np.ones(cn.max() + 1, np.int64); nmap[nused] = np.arange(len(nused))
    n = 3 * len(nused)
    reg_m = (cn[cells] == g.cell_nodes[cells]).all(1)
    reg = cells[reg_m]; irr = cells[~reg_m]
    KLf = KL_family(g)
    did = g.dim_id[reg]
    gn = g.cell_nodes[reg]                                    # (nr, 8) grid node ids
    # compact index of the grid nodes used by regular cells
    ug, inv = np.unique(gn, return_inverse=True)
    inv = inv.reshape(gn.shape)
    B = np.zeros((len(ug), 27, 3, 3))
    M = np.zeros((len(ug), 27), bool)
    LOC = np.asarray(_LOC).astype(np.int64)
    off = ((LOC[None, :, :] - LOC[:, None, :]) + 1)          # (8, 8, 3) neighbour offset of corner b seen from a
    oidx = off[..., 0] * 9 + off[..., 1] * 3 + off[..., 2]   # (8, 8)
    for d in np.unique(did):
        m = did == d
        lc = lam[reg[m]][:, None, None]; mc = mu[reg[m]][:, None, None]
        KL4 = KLf[d].reshape(8, 3, 8, 3).transpose(0, 2, 1, 3); KM4 = g.KM[d].reshape(8, 3, 8, 3).transpose(0, 2, 1, 3)
        im = inv[m]
        for a in range(8):
            rows = im[:, a]
            for b in range(8):
                o = oidx[a, b]
                B[rows, o] += lc * KL4[a, b][None] + mc * KM4[a, b][None]
                M[rows, o] = True
    # block rows/cols -> global dofs
    nb_, ob_ = np.nonzero(M)
    gi = ug[nb_]
    # neighbour grid node id from the offset (grid node id = i + (nx+1) (j + (ny+1) k), fe/solver.Grid)
    nx1, ny1 = g.nx + 1, g.ny + 1
    i = gi % nx1; j = (gi // nx1) % ny1; k = gi // (nx1 * ny1)
    di, dj, dk = ob_ // 9 - 1, (ob_ // 3) % 3 - 1, ob_ % 3 - 1
    gj = (i + di) + nx1 * ((j + dj) + ny1 * (k + dk))
    R = nmap[gi]; Cc = nmap[gj]
    vals = B[nb_, ob_]                                        # (nblk, 3, 3)
    rows = (3 * R[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2).ravel()
    cols = (3 * Cc[:, None, None] + np.arange(3)[None, None, :]).repeat(3, 1).ravel()
    K = sp.coo_matrix((vals.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    if len(irr):
        ln = nmap[cn[irr]]
        dofs = (3 * ln[:, :, None] + np.arange(3)[None, None, :]).reshape(len(irr), 24)
        did2 = g.dim_id[irr]
        Ke = np.empty((len(irr), 576))
        for d in np.unique(did2):
            mm = np.nonzero(did2 == d)[0]
            Ke[mm] = lam[irr[mm], None] * KLf[d].reshape(1, 576) + mu[irr[mm], None] * g.KM[d].reshape(1, 576)
        K = K + sp.coo_matrix((Ke.ravel(), (np.repeat(dofs, 24, axis=1).ravel(), np.tile(dofs, (1, 24)).ravel())), shape=(n, n)).tocsr()
    return K, nmap, nused
