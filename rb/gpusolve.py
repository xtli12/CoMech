"""GPU version of the engine's solver (Claude, 2026-10-29): preconditioned conjugate gradients with a geometric
multigrid V-cycle (Galerkin coarse operators, Chebyshev smoothing with 3x3 block-Jacobi) and an additive coarse
correction on per-bone rigid-body modes -- the same algorithm as fe/gmg.pcg_gmg, with the iterations run in torch on a
GPU.  The multigrid hierarchy (prolongations and Galerkin products) is built on the CPU with scipy, as before.

pcg_gmg_gpu(A, b, used_nodes, grid_edges, tol, Z=None, maxiter=..., device="cuda:0") -> x (numpy), info
Falls back to fe.gmg.pcg_gmg when torch or a GPU is not available (device="cpu" runs the torch code on the CPU)."""
import os, gc, time, random, numpy as np
import scipy.sparse as sp

# CUDA availability through NVML: the default check initialises the CUDA driver, after which processes forked by
# multiprocessing (the experiment scripts' fork pools) cannot use the GPU
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
try:
    import torch
except Exception:                                    # pragma: no cover
    torch = None


def _p1d(coords):
    """1D linear prolongation from every other node (plus the last one) to all nodes (as fe.gmg._p1d, vectorised)."""
    n = len(coords)
    cidx = np.arange(0, n, 2)
    if cidx[-1] != n - 1:
        cidx = np.append(cidx, n - 1)
    rows = [cidx]; cols = [np.arange(len(cidx))]; vals = [np.ones(len(cidx))]
    fine = np.setdiff1d(np.arange(n), cidx)
    b = np.searchsorted(cidx, fine); a = b - 1
    xa, xb = coords[cidx[a]], coords[cidx[b]]
    w = (coords[fine] - xa) / (xb - xa)
    rows += [fine, fine]; cols += [a, b]; vals += [1 - w, w]
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, len(cidx)))
    return P, coords[cidx]


_PCACHE = {}


def hierarchy(A, used_nodes, xe, ye, ze, min_dofs=6000, max_levels=6):
    """Galerkin hierarchy as in fe.gmg.GMG: list of matrices A_l and prolongations P_l (CPU, scipy)."""
    As, Ps = [A.tocsr()], []
    act_dofs = (3 * np.asarray(used_nodes)[:, None] + np.arange(3)).ravel()
    coords = (np.asarray(xe, float), np.asarray(ye, float), np.asarray(ze, float))
    for lev in range(max_levels - 1):
        if As[-1].shape[0] <= min_dofs:
            break
        key = (lev,) + tuple(np.round((c - c[0]) / (c[-1] - c[0]), 9).tobytes() for c in coords)
        if key not in _PCACHE:
            Px, cx = _p1d(coords[0]); Py, cy = _p1d(coords[1]); Pz, cz = _p1d(coords[2])
            Pn = sp.kron(Pz, sp.kron(Py, Px)).tocsr()
            _PCACHE[key] = (sp.kron(Pn, sp.identity(3, format="csr")).tocsr(), (cx - coords[0][0]) / (coords[0][-1] - coords[0][0]),
                            (cy - coords[1][0]) / (coords[1][-1] - coords[1][0]), (cz - coords[2][0]) / (coords[2][-1] - coords[2][0]))
            if len(_PCACHE) > 24:
                _PCACHE.pop(next(iter(_PCACHE)))
        Pfull, ux, uy, uz = _PCACHE[key]
        cx = coords[0][0] + ux * (coords[0][-1] - coords[0][0]); cy = coords[1][0] + uy * (coords[1][-1] - coords[1][0])
        cz = coords[2][0] + uz * (coords[2][-1] - coords[2][0])
        Pd = Pfull[act_dofs]
        keep = np.nonzero(np.asarray(abs(Pd).sum(0)).ravel() > 0)[0]
        Pd = Pd[:, keep].tocsr()
        Ac = (Pd.T @ (As[-1] @ Pd)).tocsr()
        Ps.append(Pd); As.append(Ac)
        act_dofs = keep
        coords = (cx, cy, cz)
    return As, Ps


def block_diag_inv(A):
    """Inverses of the 3x3 diagonal blocks of a CSR matrix (n/3, 3, 3)."""
    A = A.tocsr()
    if not A.has_sorted_indices:
        A.sort_indices()
    n = A.shape[0] // 3
    base = 3 * np.arange(n)
    D = np.empty((n, 3, 3))
    for i in range(3):
        for j in range(3):
            D[:, i, j] = np.asarray(A[base + i, base + j]).ravel()
    bad = np.abs(np.linalg.det(D)) < 1e-30
    D[bad] = np.eye(3)
    return np.linalg.inv(D)


class _TLevel:
    def __init__(self, A, dev, dt):
        self.A = _csr(A, dev, dt)
        self.Dinv = torch.from_numpy(block_diag_inv(A)).to(dev, dt)
        self.n = A.shape[0]
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(self.n, generator=g, dtype=torch.float64).to(dev, dt)
        lam = 1.0
        for _ in range(12):
            y = self.M(self.A @ x)
            lam = float(torch.linalg.norm(y) / torch.linalg.norm(x))
            x = y / torch.linalg.norm(y)
        self.lmax = 1.1 * lam
        self.lmin = self.lmax / 30.0

    def M(self, r):
        return torch.bmm(self.Dinv, r.view(-1, 3, 1)).view(-1)

    def smooth(self, x, b, deg=3):
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
            d = rho_new * rho * d + (2 * rho_new / de) * self.M(r)
            rho = rho_new
        return x


def _csr(A, dev, dt):
    A = A.tocsr()
    if not A.has_sorted_indices:
        A.sort_indices()
    if A.nnz < 2 ** 31 - 1:
        ip, ix = A.indptr.astype(np.int32, copy=False), A.indices.astype(np.int32, copy=False)
    else:
        ip, ix = A.indptr.astype(np.int64, copy=False), A.indices.astype(np.int64, copy=False)
    npdt = np.float32 if dt == torch.float32 else np.float64
    return torch.sparse_csr_tensor(torch.from_numpy(ip).to(dev), torch.from_numpy(ix).to(dev),
                                   torch.from_numpy(A.data.astype(npdt, copy=False)).to(dev), size=A.shape)


class GPUMG:
    def __init__(self, As, Ps, dev, dt):
        self.dev, self.dt = dev, dt
        self.levels = [_TLevel(A, dev, dt) for A in As[:-1]]
        self.P = [_csr(P, dev, dt) for P in Ps]
        self.PT = [_csr(P.T.tocsr(), dev, dt) for P in Ps]
        Acn = As[-1].toarray(); Acn = 0.5 * (Acn + Acn.T)
        self.Lc = None; self.coarse_inv = None; self.mode = None
        if not os.environ.get("COMECH_TEST_NO_CUSOLVER"):
            try:
                self.Lc = torch.linalg.cholesky(torch.from_numpy(Acn).to(dev, torch.float64)); self.mode = "chol"
            except Exception:
                self.Lc = None
        if self.mode is None:
            # cuSOLVER unavailable (its handle needs free GPU memory): factorise on the CPU, triangular solves on the
            # GPU with cuBLAS
            try:
                self.Lc = torch.from_numpy(np.linalg.cholesky(Acn)).to(dev, torch.float64); self.mode = "trsm"
            except np.linalg.LinAlgError:
                self.coarse_inv = torch.from_numpy(np.linalg.pinv(Acn)).to(dev, torch.float64); self.mode = "pinv"

    def coarse(self, b):
        b64 = b.to(torch.float64).view(-1, 1)
        if self.mode == "chol":
            x = torch.cholesky_solve(b64, self.Lc)
        elif self.mode == "trsm":
            x = torch.linalg.solve_triangular(self.Lc.T, torch.linalg.solve_triangular(self.Lc, b64, upper=False), upper=True)
        else:
            x = self.coarse_inv @ b64
        return x.view(-1).to(self.dt)

    def vcycle(self, b, l=0):
        if l == len(self.levels):
            return self.coarse(b)
        L = self.levels[l]
        x = L.smooth(torch.zeros_like(b), b)
        r = b - L.A @ x
        ec = self.vcycle(self.PT[l] @ r, l + 1)
        x = x + self.P[l] @ ec
        return L.smooth(x, b)


def pick_device():
    if torch is None or not torch.cuda.is_available():
        return None
    dev = os.environ.get("COMECH_GPU")
    if dev is None:
        dev = str(os.getpid() % torch.cuda.device_count())
    return f"cuda:{dev}"


def _cpu_solver():
    """fe.gmg.pcg_gmg as defined in fe/gmg.py (rbfe replaces the module attribute by pcg_gmg_gpu)."""
    import fe.gmg as M
    f = getattr(M, "_pcg_gmg_cpu", None) or M.pcg_gmg
    if getattr(f, "__name__", "") == "pcg_gmg_gpu":
        raise RuntimeError("CPU solver not available")
    return f


def free_gpu(dev=None):
    """Return the memory cached by this process to the driver (several worker processes share each GPU)."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            if dev is not None and str(dev).startswith("cuda"):
                with torch.cuda.device(dev):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        except Exception:
            pass


def _pcg_gpu(As, Ps, b, Z, dev, dt, tol, maxiter, x0):
    if os.environ.get("COMECH_TEST_GPU_FAIL"):
        raise RuntimeError("test: simulated GPU failure")
    if os.environ.get("COMECH_TEST_GPU_FAIL_ONCE") == "1":
        os.environ["COMECH_TEST_GPU_FAIL_ONCE"] = "0"
        raise RuntimeError("test: simulated transient GPU failure")
    t1 = time.time()
    mg = GPUMG(As, Ps, dev, dt)
    A64 = _csr(As[0], dev, torch.float64)
    bt = torch.from_numpy(np.asarray(b, np.float64)).to(dev)
    Zt = ZTt = Einv = None
    if Z is not None:
        Zs = sp.csr_matrix(Z)
        AZ = (As[0] @ Zs)
        Ec = (Zs.T @ AZ).toarray(); Ec = 0.5 * (Ec + Ec.T)
        Einv = torch.from_numpy(np.linalg.pinv(Ec)).to(dev)
        Zt = _csr(Zs, dev, torch.float64); ZTt = _csr(Zs.T.tocsr(), dev, torch.float64)

    def prec(r):
        z = mg.vcycle(r.to(dt)).to(torch.float64)
        if Zt is not None:
            z = z + Zt @ (Einv @ (ZTt @ r))
        return z
    x = torch.zeros_like(bt) if x0 is None else torch.from_numpy(np.asarray(x0, np.float64)).to(dev)
    r = bt - A64 @ x
    bn = float(torch.linalg.norm(bt))
    thr = tol * (bn if bn > 0 else 1.0)
    z = prec(r); p = z.clone(); rz = torch.dot(r, z)
    it = 0; flag = 1
    rn = float(torch.linalg.norm(r))
    if rn <= thr:
        flag = 0
    while flag and it < maxiter:
        Ap = A64 @ p
        alpha = rz / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        it += 1
        if it % 5 == 0 or it < 5:
            rn = float(torch.linalg.norm(r))
            if rn <= thr:
                flag = 0; break
        z = prec(r)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    if flag:
        rn = float(torch.linalg.norm(r))
        flag = 0 if rn <= thr else maxiter
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)
    t2 = time.time()
    xn = x.cpu().numpy()
    info = dict(iters=it, flag=flag, t_gpu=round(t2 - t1, 2), relres=rn / (bn if bn > 0 else 1.0), device=dev,
                coarse=mg.mode)
    del mg, A64, bt, x, r, z, p, Zt, ZTt, Einv
    return xn, info


def pcg_gmg_gpu(A, b, used_nodes, grid_edges, tol=1e-7, maxiter=800, x0=None, Z=None, device=None, mg_dtype="float32"):
    """Same interface and result as fe.gmg.pcg_gmg (relative residual ||r|| <= tol ||b||).  A failed GPU attempt (out of
    memory while other workers hold the GPU, library handle errors) is retried after releasing the cached memory; after
    four failed attempts the system is solved on the CPU with fe.gmg.pcg_gmg."""
    dev = device or pick_device()
    if dev is None:
        return _cpu_solver()(A, b, used_nodes, grid_edges, tol=tol, maxiter=maxiter, x0=x0, Z=Z)
    t0 = time.time()
    As, Ps = hierarchy(A, used_nodes, *grid_edges)
    t1 = time.time()
    dt = torch.float32 if mg_dtype == "float32" else torch.float64
    err = None
    for attempt in range(4):
        xn = None
        try:
            xn, info = _pcg_gpu(As, Ps, b, Z, dev, dt, tol, maxiter, x0)
        except Exception as e:
            err = repr(e)[:300]
        free_gpu(dev)                    # outside the except block: the traceback no longer holds the GPU tensors
        if xn is not None:
            info.update(levels=len(As), t_hier=round(t1 - t0, 2), gpu_attempt=attempt)
            return xn, info
        if os.environ.get("COMECH_TEST_GPU_FAIL"):
            break
        time.sleep(5 + 10 * attempt + 10 * random.random())
    xn, info = _cpu_solver()(A, b, used_nodes, grid_edges, tol=tol, maxiter=maxiter, x0=x0, Z=Z)
    info = dict(info); info.update(fallback="cpu", gpu_error=err, t_hier=round(t1 - t0, 2))
    return xn, info


# ---------------------------------------------------------------- assembly on the GPU
def _gpu_lock(dev):
    """Exclusive lock per GPU for the memory-hungry assembly step (several workers share a GPU)."""
    import fcntl
    d = os.path.expanduser(os.environ.get("COMECH_LOCKDIR", "~/.comech_gpu_locks"))
    os.makedirs(d, exist_ok=True)
    f = open(os.path.join(d, f"asm_{dev.replace(':', '_')}.lock"), "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def assemble_gpu(g, lam, mu, cells, cn, KLf, dev, chunk=60000):
    """Global stiffness of the voxel elements (same matrix as splitfe.assemble_cn) assembled on the GPU.  Returns
    (K as scipy CSR with sorted indices, nmap, nused, K as torch CSR float64 on the GPU)."""
    nused = np.unique(cn[cells])
    nmap = -np.ones(cn.max() + 1, np.int64); nmap[nused] = np.arange(len(nused))
    n = 3 * len(nused)
    ln = nmap[cn[cells]]
    dofs = torch.from_numpy((3 * ln[:, :, None] + np.arange(3)[None, None, :]).reshape(len(cells), 24)).to(dev)
    did = torch.from_numpy(g.dim_id[cells].astype(np.int64)).to(dev)
    KLt = torch.from_numpy(np.asarray(KLf).reshape(len(KLf), 576)).to(dev, torch.float64)
    KMt = torch.from_numpy(np.asarray(g.KM).reshape(len(g.KM), 576)).to(dev, torch.float64)
    lt = torch.from_numpy(np.asarray(lam)[cells]).to(dev, torch.float64); mt = torch.from_numpy(np.asarray(mu)[cells]).to(dev, torch.float64)
    lock = _gpu_lock(dev)
    try:
        keys, vals = [], []
        for a in range(0, len(cells), chunk):
            sl = slice(a, a + chunk)
            v = lt[sl, None] * KLt[did[sl]] + mt[sl, None] * KMt[did[sl]]          # (c, 576)
            d = dofs[sl]
            k = (d[:, :, None] * n + d[:, None, :]).reshape(-1)
            uk, inv = torch.unique(k, return_inverse=True)
            vv = torch.zeros(len(uk), dtype=torch.float64, device=dev).index_add_(0, inv, v.reshape(-1))
            keys.append(uk); vals.append(vv)
            del v, k, inv
        k = torch.cat(keys); v = torch.cat(vals); del keys, vals
        uk, inv = torch.unique(k, return_inverse=True)
        vv = torch.zeros(len(uk), dtype=torch.float64, device=dev).index_add_(0, inv, v)
        del k, v, inv
        rows = uk // n; cols = uk % n
        del uk
        crow = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        crow[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
        del rows
        K = sp.csr_matrix((vv.cpu().numpy(), cols.cpu().numpy().astype(np.int32), crow.cpu().numpy()), shape=(n, n))
        del crow, cols, vv
    finally:
        del dofs, did, KLt, KMt, lt, mt
        free_gpu(dev)
        lock.close()
    K.has_sorted_indices = True
    return K, nmap, nused, None
