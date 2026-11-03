"""Linear-elastic voxel finite element solver on a rectilinear (tensor-product) grid.

8-node trilinear hexahedra, 2x2x2 Gauss quadrature, per-cell isotropic material (lambda, mu)
obtained from partial-volume mixing.  Elastic foundations (Winkler springs) on selected faces,
surface tractions on selected faces.  Solved with a sparse direct or preconditioned CG solver.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# local node order: l = a + 2b + 4c with (a,b,c) in {0,1}^3
_LOC = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)], dtype=float)


def _shape_grad(xi, eta, zeta, dims):
    """Gradients of the 8 trilinear shape functions w.r.t. physical coords at natural point."""
    a, b, c = dims
    s = 2 * _LOC - 1  # signs
    dN = np.zeros((8, 3))
    dN[:, 0] = s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta) / 8 * (2 / a)
    dN[:, 1] = s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * zeta) / 8 * (2 / b)
    dN[:, 2] = s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta) / 8 * (2 / c)
    return dN


def _Bmat(dN):
    B = np.zeros((6, 24))
    for n in range(8):
        dx, dy, dz = dN[n]
        B[0, 3 * n] = dx
        B[1, 3 * n + 1] = dy
        B[2, 3 * n + 2] = dz
        B[3, 3 * n] = dy; B[3, 3 * n + 1] = dx
        B[4, 3 * n + 1] = dz; B[4, 3 * n + 2] = dy
        B[5, 3 * n] = dz; B[5, 3 * n + 2] = dx
    return B


_DL = np.zeros((6, 6)); _DL[:3, :3] = 1.0           # lambda part
_DM = np.diag([2, 2, 2, 1, 1, 1]).astype(float)      # mu part (engineering shear strains)


# basis constitutive patterns for orthotropic (z-normal) materials:
# D = c0*P0 + c1*P1 + c2*P2 + c3*P3 + c4*P4 + c5*P5 with
# P0: D11=D22=1, P1: D12=D21=1, P2: D13=D31=D23=D32=1, P3: D33=1, P4: D44=1 (xy), P5: D55=D66=1 (yz,xz)
_P = np.zeros((6, 6, 6))
_P[0, 0, 0] = _P[0, 1, 1] = 1
_P[1, 0, 1] = _P[1, 1, 0] = 1
_P[2, 0, 2] = _P[2, 2, 0] = _P[2, 1, 2] = _P[2, 2, 1] = 1
_P[3, 2, 2] = 1
_P[4, 3, 3] = 1
_P[5, 4, 4] = _P[5, 5, 5] = 1


def ortho_coeffs(lam, mu, shear_factor=None):
    """Coefficients c (n,6) for isotropic (lam, mu), optionally with reduced out-of-plane shear."""
    c = np.stack([lam + 2 * mu, lam, lam, lam + 2 * mu, mu, mu], 1)
    if shear_factor is not None:
        c[:, 5] *= shear_factor
    return c


def element_basis(dims):
    g = 1 / np.sqrt(3)
    a, b, cc = dims
    detJ = a * b * cc / 8
    K = np.zeros((6, 24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = _Bmat(_shape_grad(xi, eta, zeta, dims))
                for k in range(6):
                    K[k] += B.T @ _P[k] @ B * detJ
    return K


def element_matrices(dims):
    """Return (K_lambda, K_mu, B_center) for a rectangular hex of size dims."""
    g = 1 / np.sqrt(3)
    a, b, c = dims
    detJ = a * b * c / 8
    KL = np.zeros((24, 24)); KM = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = _Bmat(_shape_grad(xi, eta, zeta, dims))
                KL += B.T @ _DL @ B * detJ
                KM += B.T @ _DM @ B * detJ
    Bc = _Bmat(_shape_grad(0, 0, 0, dims))
    return KL, KM, Bc


class Grid:
    def __init__(self, xe, ye, ze):
        self.xe, self.ye, self.ze = map(np.asarray, (xe, ye, ze))
        self.nx, self.ny, self.nz = len(xe) - 1, len(ye) - 1, len(ze) - 1
        self.shape = (self.nx, self.ny, self.nz)
        self.ncell = self.nx * self.ny * self.nz
        self.nnode = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        self.dx, self.dy, self.dz = np.diff(self.xe), np.diff(self.ye), np.diff(self.ze)
        self.xc = (self.xe[:-1] + self.xe[1:]) / 2
        self.yc = (self.ye[:-1] + self.ye[1:]) / 2
        self.zc = (self.ze[:-1] + self.ze[1:]) / 2
        I, J, K = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij")
        self.cI, self.cJ, self.cK = I.ravel(), J.ravel(), K.ravel()
        # cell -> 8 nodes (bit order)
        nid = lambda i, j, k: i + (self.nx + 1) * (j + (self.ny + 1) * k)
        self.cell_nodes = np.stack([nid(self.cI + a, self.cJ + b, self.cK + c)
                                    for (a, b, c) in _LOC.astype(int)], axis=1)
        # element matrix families by unique dims
        dims = np.stack([self.dx[self.cI], self.dy[self.cJ], self.dz[self.cK]], 1)
        ud, inv = np.unique(np.round(dims, 6), axis=0, return_inverse=True)
        self.dim_id = inv.ravel()
        self.udims = ud
        mats = [element_matrices(d) for d in ud]
        self.KL = np.stack([m[0] for m in mats]); self.KM = np.stack([m[1] for m in mats])
        self.Bc = np.stack([m[2] for m in mats])
        self.KB = np.stack([element_basis(d) for d in ud])      # (ndims, 6, 24, 24)
        self.cell_vol = dims.prod(1)
        Ni, Nj, Nk = np.meshgrid(np.arange(self.nx + 1), np.arange(self.ny + 1), np.arange(self.nz + 1), indexing="ij")
        # node coordinates in node-id order (id = i + (nx+1)(j + (ny+1)k))
        self.node_xyz = np.stack([self.xe[Ni.transpose(2, 1, 0).ravel()],
                                  self.ye[Nj.transpose(2, 1, 0).ravel()],
                                  self.ze[Nk.transpose(2, 1, 0).ravel()]], 1)
        self.cell_xyz = np.stack([self.xc[self.cI], self.yc[self.cJ], self.zc[self.cK]], 1)

    def cid(self, i, j, k):
        return (i * self.ny + j) * self.nz + k  # matches meshgrid ij ravel order


def lame(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def assemble(grid, lam, mu, active, coeffs=None):
    """Assemble global stiffness for active cells; returns K (compact dofs), node map."""
    cells = np.nonzero(active)[0]
    cn = grid.cell_nodes[cells]
    used = np.unique(cn)
    nmap = -np.ones(grid.nnode, dtype=np.int64)
    nmap[used] = np.arange(len(used))
    ln = nmap[cn]                                     # (ne, 8)
    dofs = (3 * ln[:, :, None] + np.arange(3)[None, None, :]).reshape(len(cells), 24)
    did = grid.dim_id[cells]
    # element matrices per family of cell dimensions (avoids materialising (ne, 6, 24, 24) temporaries)
    Ke = np.empty((len(cells), 576), np.float64)
    for d in np.unique(did):
        m = np.nonzero(did == d)[0]
        if coeffs is None:
            Ke[m] = lam[cells[m], None] * grid.KL[d].reshape(1, 576) + mu[cells[m], None] * grid.KM[d].reshape(1, 576)
        else:
            Ke[m] = coeffs[cells[m]] @ grid.KB[d].reshape(grid.KB.shape[1], 576)
    idx_t = np.int32 if 3 * len(used) < 2 ** 31 - 1 else np.int64
    d32 = dofs.astype(idx_t)
    rows = np.repeat(d32, 24, axis=1).ravel()
    cols = np.tile(d32, (1, 24)).ravel()
    n = 3 * len(used)
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    del Ke, rows, cols
    return K, nmap, used, cells, dofs


def face_node_areas(grid, cells, face):
    """Tributary node areas for the given face of each cell. face in {'x-','x+','y-','y+','z-','z+'}.
    Returns (node_ids, area) arrays (may contain repeats)."""
    ax = "xyz".index(face[0]); side = 1 if face[1] == "+" else 0
    sel = _LOC[:, ax] == side
    nodes = grid.cell_nodes[cells][:, sel]          # (n, 4)
    d = [grid.dx[grid.cI[cells]], grid.dy[grid.cJ[cells]], grid.dz[grid.cK[cells]]]
    other = [d[i] for i in range(3) if i != ax]
    area = (other[0] * other[1] / 4)[:, None].repeat(4, 1)
    return nodes.ravel(), area.ravel()


def rigid_modes(grid, nmap, n, node_groups):
    """Sparse matrix whose columns are rigid-body modes (3 translations, 3 rotations) of node groups."""
    rows, cols, vals = [], [], []
    col = 0
    for nodes in node_groups:
        ln = nmap[nodes]; ok = ln >= 0
        nodes, ln = nodes[ok], ln[ok]
        if len(nodes) < 4:
            continue
        X = grid.node_xyz[nodes]; c = X.mean(0); r = (X - c) / 20.0
        for d in range(3):
            rows.append(3 * ln + d); cols.append(np.full(len(ln), col)); vals.append(np.ones(len(ln)))
            col += 1
        # rotations: w x r
        for (a, b, sa, sb) in [(1, 2, -1, 1), (0, 2, 1, -1), (0, 1, -1, 1)]:
            # rotation about axis k: u_a = sa * r_b', u_b = sb * r_a'
            pass
        # about x: u_y = -r_z, u_z = r_y ; about y: u_x = r_z, u_z = -r_x ; about z: u_x = -r_y, u_y = r_x
        for comps in [((1, -r[:, 2]), (2, r[:, 1])), ((0, r[:, 2]), (2, -r[:, 0])), ((0, -r[:, 1]), (1, r[:, 0]))]:
            for d, v in comps:
                rows.append(3 * ln + d); cols.append(np.full(len(ln), col)); vals.append(v)
            col += 1
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, col))


def solve(grid, lam, mu, active, springs, loads, method="cg", tol=1e-8, x0=None, maxiter=20000, penalties=(),
          coeffs=None, node_groups=None, U0=None):
    """springs: list of (node_ids, area, kvec[3]) ; loads: list of (node_ids, area, traction[3]).
    Returns full nodal displacement array (nnode,3) (nan for unused nodes) and info."""
    K, nmap, used, cells, dofs = assemble(grid, lam, mu, active, coeffs)
    n = K.shape[0]
    if U0 is not None:
        x0 = np.nan_to_num(U0[used]).ravel()
    f = np.zeros(n)
    kd = np.zeros(n)
    for nodes, area, kvec in springs:
        ln = nmap[nodes]; ok = ln >= 0
        for c in range(3):
            np.add.at(kd, 3 * ln[ok] + c, area[ok] * kvec[c])
    for nodes, area, t in loads:
        ln = nmap[nodes]; ok = ln >= 0
        for c in range(3):
            np.add.at(f, 3 * ln[ok] + c, area[ok] * t[c])
    K = K + sp.diags(kd)
    info = {}
    # rank-1 penalties k * (w.u)^2 on weighted sums of dofs (e.g. mean translation of a surface);
    # w is sparse (a few hundred dofs), so k w w^T is added to K explicitly and seen by the preconditioner
    for nodes, weights, comp, k in penalties:
        ln = nmap[nodes]; ok = ln >= 0
        w = np.zeros(K.shape[0])
        np.add.at(w, 3 * ln[ok] + comp, weights[ok])
        w /= w.sum()
        nz = np.nonzero(w)[0]
        wv = w[nz]
        K = K + sp.csr_matrix((k * np.outer(wv, wv).ravel(), (np.repeat(nz, len(nz)), np.tile(nz, len(nz)))),
                              shape=K.shape)
    if method == "gmg":
        from fe.gmg import pcg_gmg
        Z = rigid_modes(grid, nmap, K.shape[0], node_groups) if node_groups else None
        u, inf = pcg_gmg(K, f, used, (grid.xe, grid.ye, grid.ze), tol=tol, x0=x0, Z=Z)
        info.update(inf)
    elif method == "direct":
        u = spla.spsolve(K.tocsc(), f, permc_spec="MMD_AT_PLUS_A")
        info["iters"] = 0
    else:
        d = K.diagonal()
        M = sp.diags(1 / d)
        it = [0]
        def cb(xk): it[0] += 1
        u, flag = spla.cg(K, f, x0=x0, rtol=tol, maxiter=maxiter, M=M, callback=cb)
        info["iters"] = it[0]; info["flag"] = flag
    U = np.full((grid.nnode, 3), np.nan)
    U[used] = u.reshape(-1, 3)
    info["ndof"] = n
    info["K"] = K
    info["f"] = f
    info["used"] = used
    return U, info


def cell_stress(grid, lam, mu, U, cells, coeffs=None):
    """Cauchy stress (6-vector, engineering order xx,yy,zz,xy,yz,xz) and von Mises at cell centres."""
    ue = U[grid.cell_nodes[cells]].reshape(len(cells), 24)
    B = grid.Bc[grid.dim_id[cells]]                      # (n,6,24)
    eps = np.einsum("nij,nj->ni", B, ue)
    if coeffs is None:
        tr = eps[:, :3].sum(1)
        sig = np.empty_like(eps)
        sig[:, :3] = lam[cells, None] * tr[:, None] + 2 * mu[cells, None] * eps[:, :3]
        sig[:, 3:] = mu[cells, None] * eps[:, 3:]
    else:
        D = np.einsum("nk,kij->nij", coeffs[cells], _P)
        sig = np.einsum("nij,nj->ni", D, eps)
    sx, sy, sz, txy, tyz, txz = sig.T
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + txz ** 2))
    return sig, vm, eps
