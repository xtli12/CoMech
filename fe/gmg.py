"""Geometric multigrid (Galerkin) preconditioned CG for voxel elasticity on a rectilinear grid."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _p1d(coords):
    """1D linear prolongation from every-other node (plus last) to all nodes."""
    n = len(coords)
    cidx = list(range(0, n, 2))
    if cidx[-1] != n - 1:
        cidx.append(n - 1)
    cidx = np.array(cidx)
    rows, cols, vals = [], [], []
    for ci, i in enumerate(cidx):
        rows.append(i); cols.append(ci); vals.append(1.0)
    for i in range(n):
        if i in set(cidx):
            continue
        b = np.searchsorted(cidx, i)
        a = b - 1
        xa, xb = coords[cidx[a]], coords[cidx[b]]
        w = (coords[i] - xa) / (xb - xa)
        rows += [i, i]; cols += [a, b]; vals += [1 - w, w]
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(cidx)))
    return P, coords[cidx]


class BlockJacobi:
    def __init__(self, A):
        n = A.shape[0] // 3
        A = A.tocsr()
        D = np.zeros((n, 3, 3))
        Ac = A.tocoo()
        m = (Ac.row // 3) == (Ac.col // 3)
        r, c, v = Ac.row[m], Ac.col[m], Ac.data[m]
        np.add.at(D, (r // 3, r % 3, c % 3), v)
        # regularise empty blocks
        bad = np.abs(np.linalg.det(D)) < 1e-30
        D[bad] = np.eye(3)
        self.Dinv = np.linalg.inv(D)

    def __call__(self, r):
        return np.einsum("nij,nj->ni", self.Dinv, r.reshape(-1, 3)).ravel()


class Level:
    def __init__(self, A):
        self.A = A.tocsr()
        self.M = BlockJacobi(self.A)
        # spectral radius of M A by power iteration
        x = np.random.default_rng(0).standard_normal(A.shape[0])
        lam = 1.0
        for _ in range(12):
            y = self.M(self.A @ x)
            lam = np.linalg.norm(y) / np.linalg.norm(x)
            x = y / np.linalg.norm(y)
        self.lmax = 1.1 * lam
        self.lmin = self.lmax / 30.0

    def smooth(self, x, b, deg=3):
        """Chebyshev smoothing with block-Jacobi preconditioning."""
        th = (self.lmax + self.lmin) / 2; de = (self.lmax - self.lmin) / 2
        r = b - self.A @ x
        sig = th / de
        rho = 1 / sig
        d = self.M(r) / th
        for k in range(deg):
            x = x + d
            if k == deg - 1:
                break
            r = r - self.A @ d
            rho_new = 1 / (2 * sig - rho)
            d = rho_new * rho * d + 2 * rho_new / de * self.M(r)
            rho = rho_new
        return x


class GMG:
    def __init__(self, A, used_nodes, xe, ye, ze, min_dofs=6000, max_levels=6):
        self.levels = []
        self.P = []
        nx, ny, nz = len(xe), len(ye), len(ze)
        act_dofs = (3 * used_nodes[:, None] + np.arange(3)).ravel()
        coords = (np.asarray(xe), np.asarray(ye), np.asarray(ze))
        Acur = A
        for lev in range(max_levels):
            self.levels.append(Level(Acur))
            if Acur.shape[0] <= min_dofs:
                break
            Px, cx = _p1d(coords[0]); Py, cy = _p1d(coords[1]); Pz, cz = _p1d(coords[2])
            Pn = sp.kron(Pz, sp.kron(Py, Px)).tocsr()
            Pd = sp.kron(Pn, sp.identity(3)).tocsr()
            Pd = Pd[act_dofs]
            keep = np.nonzero(np.asarray(abs(Pd).sum(0)).ravel() > 0)[0]
            Pd = Pd[:, keep].tocsr()
            Ac = (Pd.T @ Acur @ Pd).tocsr()
            self.P.append(Pd)
            Acur = Ac
            act_dofs = keep
            coords = (cx, cy, cz)
        self.coarse = spla.splu(Acur.tocsc(), permc_spec="MMD_AT_PLUS_A")

    def vcycle(self, b, l=0):
        if l == len(self.levels) - 1:
            return self.coarse.solve(b)
        L = self.levels[l]
        x = L.smooth(np.zeros_like(b), b)
        r = b - L.A @ x
        ec = self.vcycle(self.P[l].T @ r, l + 1)
        x = x + self.P[l] @ ec
        x = L.smooth(x, b)
        return x

    def as_linear_operator(self):
        n = self.levels[0].A.shape[0]
        return spla.LinearOperator((n, n), matvec=self.vcycle)


def pcg_gmg(A, b, used_nodes, grid_edges, tol=1e-7, maxiter=500, x0=None, A_op=None, Z=None):
    """PCG with a symmetric GMG V-cycle; optionally an additive coarse correction on the columns of Z
    (e.g. per-bone rigid-body modes)."""
    mg = GMG(A, used_nodes, *grid_edges)
    it = [0]
    def cb(xk): it[0] += 1
    M = mg.as_linear_operator()
    if Z is not None:
        Zd = Z.toarray() if sp.issparse(Z) else Z
        Aeff = A if A_op is None else A_op
        AZ = np.column_stack([Aeff @ Zd[:, i] for i in range(Zd.shape[1])])
        Ec = Zd.T @ AZ
        Ec = 0.5 * (Ec + Ec.T)
        Einv = np.linalg.pinv(Ec)
        n = A.shape[0]
        M = spla.LinearOperator((n, n), matvec=lambda r: mg.vcycle(r) + Zd @ (Einv @ (Zd.T @ r)))
    x, flag = spla.cg(A if A_op is None else A_op, b, x0=x0, rtol=tol, maxiter=maxiter, M=M, callback=cb)
    return x, dict(iters=it[0], flag=flag, levels=len(mg.levels))
