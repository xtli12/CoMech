"""Minimal reverse-mode automatic differentiation on numpy arrays (float32).

Only the operations needed by the CoMech models are implemented.  Every op records a closure that
propagates the output gradient to its parents; `backward()` runs them in reverse topological order.
"""
import numpy as np

DT = np.float32


class T:
    __slots__ = ("data", "grad", "_parents", "_back", "req")

    def __init__(self, data, req=False, parents=(), back=None):
        self.data = data if isinstance(data, np.ndarray) and data.dtype == DT else np.asarray(data, dtype=DT)
        self.grad = None
        self._parents = parents
        self._back = back
        self.req = req or any(p.req for p in parents)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"T{self.data.shape}"

    # ------------- graph -------------
    def backward(self, g=None):
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.req and id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data) if g is None else g.astype(DT)
        for node in reversed(topo):
            if node._back is not None and node.grad is not None:
                node._back(node.grad)
                if node._parents:      # free intermediate grads
                    node.grad = None if node._back is not None and not _is_leaf(node) else node.grad

    # ------------- operators -------------
    def __add__(self, o): return add(self, o)
    def __radd__(self, o): return add(self, o)
    def __sub__(self, o): return add(self, neg(_t(o)))
    def __rsub__(self, o): return add(_t(o), neg(self))
    def __mul__(self, o): return mul(self, o)
    def __rmul__(self, o): return mul(self, o)
    def __neg__(self): return neg(self)
    def __matmul__(self, o): return matmul(self, o)
    def __truediv__(self, o): return mul(self, 1.0 / o) if not isinstance(o, T) else mul(self, pow_(o, -1.0))
    def __pow__(self, p): return pow_(self, p)
    def sum(self, axis=None, keepdims=False): return sum_(self, axis, keepdims)
    def mean(self, axis=None, keepdims=False): return mean(self, axis, keepdims)
    def reshape(self, *s): return reshape(self, s[0] if len(s) == 1 else s)
    def transpose(self, *ax): return transpose(self, ax)


def _is_leaf(n):
    return not n._parents


def _t(x):
    return x if isinstance(x, T) else T(np.asarray(x, dtype=DT))


def _acc(p, g):
    if not p.req:
        return
    if p.grad is None:
        p.grad = g.astype(DT, copy=False)
    else:
        p.grad = p.grad + g


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


def param(arr):
    return T(np.asarray(arr, dtype=DT), req=True)


def add(a, b):
    a, b = _t(a), _t(b)
    out = T(a.data + b.data, parents=(a, b))
    def back(g):
        _acc(a, _unbroadcast(g, a.shape)); _acc(b, _unbroadcast(g, b.shape))
    out._back = back
    return out


def neg(a):
    out = T(-a.data, parents=(a,))
    out._back = lambda g: _acc(a, -g)
    return out


def mul(a, b):
    a, b = _t(a), _t(b)
    out = T(a.data * b.data, parents=(a, b))
    def back(g):
        if a.req: _acc(a, _unbroadcast(g * b.data, a.shape))
        if b.req: _acc(b, _unbroadcast(g * a.data, b.shape))
    out._back = back
    return out


def pow_(a, p):
    out = T(a.data ** p, parents=(a,))
    out._back = lambda g: _acc(a, g * p * a.data ** (p - 1))
    return out


def matmul(a, b):
    a, b = _t(a), _t(b)
    out = T(a.data @ b.data, parents=(a, b))
    def back(g):
        if a.req:
            ga = g @ np.swapaxes(b.data, -1, -2)
            _acc(a, _unbroadcast(ga, a.shape))
        if b.req:
            gb = np.swapaxes(a.data, -1, -2) @ g
            _acc(b, _unbroadcast(gb, b.shape))
    out._back = back
    return out


def sum_(a, axis=None, keepdims=False):
    out = T(a.data.sum(axis=axis, keepdims=keepdims), parents=(a,))
    def back(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        _acc(a, np.broadcast_to(g, a.shape))
    out._back = back
    return out


def mean(a, axis=None, keepdims=False):
    n = a.data.size if axis is None else np.prod([a.shape[i] for i in np.atleast_1d(axis)])
    return mul(sum_(a, axis, keepdims), 1.0 / n)


def reshape(a, shape):
    out = T(a.data.reshape(shape), parents=(a,))
    out._back = lambda g: _acc(a, g.reshape(a.shape))
    return out


def transpose(a, axes):
    axes = tuple(axes)
    inv = np.argsort(axes)
    out = T(np.transpose(a.data, axes), parents=(a,))
    out._back = lambda g: _acc(a, np.transpose(g, inv))
    return out


def concat(ts, axis=-1):
    ts = [_t(t) for t in ts]
    out = T(np.concatenate([t.data for t in ts], axis=axis), parents=tuple(ts))
    sizes = np.cumsum([0] + [t.shape[axis] for t in ts])
    def back(g):
        for t, s0, s1 in zip(ts, sizes[:-1], sizes[1:]):
            if t.req:
                sl = [slice(None)] * g.ndim
                sl[axis] = slice(s0, s1)
                _acc(t, g[tuple(sl)])
    out._back = back
    return out


def take(a, idx, axis=0):
    """Gather along axis with integer index array."""
    out = T(np.take(a.data, idx, axis=axis), parents=(a,))
    def back(g):
        z = np.zeros_like(a.data)
        if axis == 0:
            np.add.at(z, idx, g)
        else:
            zz = np.moveaxis(z, axis, 0)
            np.add.at(zz, idx, np.moveaxis(g, axis, 0))
        _acc(a, z)
    out._back = back
    return out


def spmm(S, a):
    """Constant scipy sparse matrix (n x m) times tensor (m, ...) -> (n, ...)."""
    sh = a.shape
    A2 = a.data.reshape(sh[0], -1)
    out = T(np.asarray(S @ A2, dtype=DT).reshape((S.shape[0],) + sh[1:]), parents=(a,))
    def back(g):
        _acc(a, np.asarray(S.T @ g.reshape(S.shape[0], -1), dtype=DT).reshape(sh))
    out._back = back
    return out


def relu(a):
    m = a.data > 0
    out = T(a.data * m, parents=(a,))
    out._back = lambda g: _acc(a, g * m)
    return out


def gelu(a):
    x = a.data
    c = DT(np.sqrt(2 / np.pi)); k = DT(0.044715)
    x2 = x * x
    th = np.tanh(c * (x + k * x2 * x))
    y = DT(0.5) * x * (1 + th)
    out = T(y, parents=(a,))
    def back(g):
        du = c * (1 + DT(3) * k * x2)
        d = DT(0.5) * (1 + th) + DT(0.5) * x * (1 - th * th) * du
        _acc(a, g * d)
    out._back = back
    return out


def tanh(a):
    y = np.tanh(a.data)
    out = T(y, parents=(a,))
    out._back = lambda g: _acc(a, g * (1 - y ** 2))
    return out


def sigmoid(a):
    y = 1 / (1 + np.exp(-a.data))
    out = T(y, parents=(a,))
    out._back = lambda g: _acc(a, g * y * (1 - y))
    return out


def exp(a):
    y = np.exp(a.data)
    out = T(y, parents=(a,))
    out._back = lambda g: _acc(a, g * y)
    return out


def log(a):
    out = T(np.log(a.data), parents=(a,))
    out._back = lambda g: _acc(a, g / a.data)
    return out


def sqrt(a):
    y = np.sqrt(a.data)
    out = T(y, parents=(a,))
    out._back = lambda g: _acc(a, g * 0.5 / np.maximum(y, 1e-12))
    return out


def layernorm(a, gamma, beta, eps=1e-5):
    x = a.data
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    rstd = 1 / np.sqrt(var + eps)
    xh = xc * rstd
    out = T(xh * gamma.data + beta.data, parents=(a, gamma, beta))
    def back(g):
        if gamma.req: _acc(gamma, _unbroadcast(g * xh, gamma.shape))
        if beta.req: _acc(beta, _unbroadcast(g, beta.shape))
        if a.req:
            gx = g * gamma.data
            n = x.shape[-1]
            dx = rstd * (gx - gx.mean(-1, keepdims=True) - xh * (gx * xh).mean(-1, keepdims=True))
            _acc(a, dx)
    out._back = back
    return out


def softmax(a, axis=-1):
    x = a.data - a.data.max(axis, keepdims=True)
    e = np.exp(x)
    y = e / e.sum(axis, keepdims=True)
    out = T(y, parents=(a,))
    def back(g):
        _acc(a, y * (g - (g * y).sum(axis, keepdims=True)))
    out._back = back
    return out


def hinge(a):
    return relu(a)


# ---------------- spectral convolution (FNO) ----------------
def spectral_conv3d(x, wr, wi, modes):
    """x: (B, Ci, X, Y, Z) real; weights (Ci, Co, m) real/imag on selected flat frequency indices `modes`
    (index array into the flattened fftn grid). y = Re(ifftn(sum_i W_io * fftn(x_i)))."""
    B, Ci, X, Y, Z = x.shape
    N = X * Y * Z
    Xf = np.fft.fftn(x.data, axes=(2, 3, 4)).reshape(B, Ci, N)[:, :, modes]       # (B,Ci,m) complex
    W = wr.data + 1j * wi.data                                                     # (Ci,Co,m)
    Yk = np.einsum("bim,iom->bom", Xf, W)
    full = np.zeros((B, W.shape[1], N), dtype=np.complex64 if DT == np.float32 else np.complex128)
    full[:, :, modes] = Yk
    y = np.fft.ifftn(full.reshape(B, W.shape[1], X, Y, Z), axes=(2, 3, 4)).real.astype(DT)
    out = T(y, parents=(x, wr, wi))
    def back(g):
        G = np.fft.fftn(g, axes=(2, 3, 4)).reshape(B, W.shape[1], N)[:, :, modes] / N   # (B,Co,m)
        if wr.req or wi.req:
            cg = np.einsum("bim,bom->iom", Xf, np.conj(G))      # sum_b X_i * conj(G_o)
            # d<g,y>/dWr = Re(conj(G) X), d/dWi = -Im(conj(G) X)
            _acc(wr, cg.real.astype(DT)); _acc(wi, (-cg.imag).astype(DT))
        if x.req:
            # adjoint: Re(ifftn(conj(W) * fftn(g))) restricted to modes, summed over outputs
            Gf = np.fft.fftn(g, axes=(2, 3, 4)).reshape(B, W.shape[1], N)[:, :, modes]
            Xk = np.einsum("bom,iom->bim", Gf, np.conj(W))
            full2 = np.zeros((B, Ci, N), dtype=np.complex64 if DT == np.float32 else np.complex128)
            full2[:, :, modes] = Xk
            gx = np.fft.ifftn(full2.reshape(B, Ci, X, Y, Z), axes=(2, 3, 4)).real
            _acc(x, gx.astype(DT))
    out._back = back
    return out


# ---------------- optimiser ----------------
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, clip=1.0):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.wd, self.clip = lr, b1, b2, eps, wd, clip
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def zero(self):
        for v in self.p.values():
            v.grad = None

    def step(self, lr=None):
        lr = lr or self.lr
        self.t += 1
        # global norm clipping
        tot = 0.0
        for v in self.p.values():
            if v.grad is not None:
                tot += float((v.grad.astype(np.float64) ** 2).sum())
        gn = np.sqrt(tot)
        sc = min(1.0, self.clip / (gn + 1e-12)) if self.clip else 1.0
        for k, v in self.p.items():
            if v.grad is None:
                continue
            g = v.grad * sc
            if self.wd:
                g = g + self.wd * v.data
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            v.data -= (lr * mh / (np.sqrt(vh) + self.eps)).astype(DT)
        return gn


# ---------------- truncated-DFT spectral convolution (identical to FFT + mode selection) ----------------
def dft_mats(n, m):
    """Forward (2m x n) and inverse (n x 2m) DFT matrices for frequencies -m..m-1 (mod n)."""
    k = np.array([i % n for i in range(-m, m)])
    idx = np.arange(n)
    F = np.exp(-2j * np.pi * np.outer(k, idx) / n)
    return F, np.conj(F.T)


def _tdft(x, Fs):
    """Apply per-axis matrices Fs=(Fx,Fy,Fz) to the last three axes of x."""
    y = np.tensordot(x, Fs[2], axes=([-1], [1]))                       # (..., X, Y, c)
    y = np.moveaxis(np.tensordot(y, Fs[1], axes=([-2], [1])), -1, -2)   # (..., X, b, c)
    y = np.moveaxis(np.tensordot(y, Fs[0], axes=([-3], [1])), -1, -3)   # (..., a, b, c)
    return y


def _fwd_real(x, F, Cz, Sz):
    """Truncated DFT of a real array over its last three axes; the z-axis is done with real matmuls."""
    y = (x @ Cz) - 1j * (x @ Sz)                                        # (..., X, Y, c)
    y = np.moveaxis(np.tensordot(y, F[1], axes=([-2], [1])), -1, -2)   # (..., X, b, c)
    y = np.moveaxis(np.tensordot(y, F[0], axes=([-3], [1])), -1, -3)   # (..., a, b, c)
    return y


def _inv_real(Yh, I, CzT, SzT):
    """Real part of the inverse truncated DFT; the z-axis is done last with real matmuls."""
    y = np.moveaxis(np.tensordot(Yh, I[0], axes=([-3], [1])), -1, -3)  # (..., X, b, c)
    y = np.moveaxis(np.tensordot(y, I[1], axes=([-2], [1])), -1, -2)   # (..., X, Y, c)
    return (y.real @ CzT) - (y.imag @ SzT)                              # (..., X, Y, Z)


def spectral_conv3d_dft(x, wr, wi, mats):
    """x: (B, Ci, X, Y, Z); w: (Ci, Co, a, b, c); mats = ((Fx,Ix),(Fy,Iy),(Fz,Iz)).
    Identical to FFT followed by selection of the retained modes and inverse FFT (real part)."""
    B, Ci, X, Y, Z = x.shape
    N = X * Y * Z
    CT = np.complex64 if DT == np.float32 else np.complex128
    F = [m[0].astype(CT) for m in mats]; I = [m[1].astype(CT) for m in mats]
    Fz = mats[2][0]
    Cz = np.ascontiguousarray(Fz.real.T).astype(DT); Sz = np.ascontiguousarray(-Fz.imag.T).astype(DT)   # (Z, c)
    CzT = np.ascontiguousarray(Fz.real).astype(DT); SzT = np.ascontiguousarray(-Fz.imag).astype(DT)     # (c, Z)
    Xh = _fwd_real(x.data, F, Cz, Sz)                                   # (B,Ci,a,b,c)
    W = (wr.data + 1j * wi.data).astype(CT)
    Yh = np.einsum("nipqr,iopqr->nopqr", Xh, W)
    y = (_inv_real(Yh, I, CzT, SzT) / N).astype(DT)
    out = T(y, parents=(x, wr, wi))
    def back(g):
        G = _fwd_real(g, F, Cz, Sz) / N                                 # (B,Co,a,b,c)
        if wr.req or wi.req:
            cg = np.einsum("nipqr,nopqr->iopqr", Xh, np.conj(G))
            _acc(wr, cg.real.astype(DT)); _acc(wi, (-cg.imag).astype(DT))
        if x.req:
            Xk = np.einsum("nopqr,iopqr->nipqr", G, np.conj(W))
            _acc(x, _inv_real(Xk, I, CzT, SzT).astype(DT))
    out._back = back
    return out
