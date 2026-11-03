"""CoMech and baselines, written on the numpy autodiff in ml/ad.py."""
import numpy as np, scipy.sparse as sp
from ml import ad
from ml.ad import T, param, layernorm, spmm, concat, softmax
from ml.ad import relu as gelu   # ReLU activations throughout (cheaper on CPU)
from ml.data import NTOK, TOK, NCH, PATCH, TOK_IDX, TOK_W, CELL_FOURIER

NEFF = 17
NACT = 7
NQF = CELL_FOURIER.shape[1]


class Module:
    def __init__(self):
        self.p = {}

    def add(self, name, arr):
        self.p[name] = param(arr)
        return self.p[name]

    def lin(self, name, i, o, rng, scale=1.0):
        self.add(name + ".W", rng.standard_normal((i, o)).astype(np.float32) * scale / np.sqrt(i))
        self.add(name + ".b", np.zeros(o, np.float32))

    def L(self, name, x):
        return x @ self.p[name + ".W"] + self.p[name + ".b"]

    def ln(self, name, d):
        self.add(name + ".g", np.ones(d, np.float32)); self.add(name + ".bb", np.zeros(d, np.float32))

    def LN(self, name, x):
        return layernorm(x, self.p[name + ".g"], self.p[name + ".bb"])

    def mlp(self, name, dims, rng, last_scale=1.0):
        for k in range(len(dims) - 1):
            self.lin(f"{name}.{k}", dims[k], dims[k + 1], rng, scale=last_scale if k == len(dims) - 2 else 1.0)
        return len(dims) - 1

    def MLP(self, name, x, n):
        for k in range(n):
            x = self.L(f"{name}.{k}", x)
            if k < n - 1:
                x = gelu(x)
        return x

    def state(self):
        return {k: v.data.copy() for k, v in self.p.items()}

    def load(self, st):
        for k, v in st.items():
            self.p[k].data[...] = v


# ---------------- shared blocks ----------------
def mixer_init(m, name, D, Ht, Hc, rng, film_dim=None):
    m.ln(name + ".ln1", D)
    m.lin(name + ".tok1", NTOK, Ht, rng); m.lin(name + ".tok2", Ht, NTOK, rng, scale=0.5)
    m.ln(name + ".ln2", D)
    m.lin(name + ".ch1", D, Hc, rng); m.lin(name + ".ch2", Hc, D, rng, scale=0.5)
    if film_dim:
        m.lin(name + ".film1", film_dim, 2 * D, rng, scale=0.1)
        m.lin(name + ".film2", film_dim, 2 * D, rng, scale=0.1)


def mixer(m, name, x, cond=None):
    """x: (B, NTOK, D); cond: (B, F) FiLM conditioning."""
    B, N, D = x.shape
    h = m.LN(name + ".ln1", x)
    if cond is not None:
        gb = m.L(name + ".film1", cond).reshape(B, 1, 2 * D)
        g, b = _split(gb, D)
        h = h * (g + 1.0) + b
    ht = h.transpose(0, 2, 1)                         # (B, D, N)
    ht = m.L(name + ".tok2", gelu(m.L(name + ".tok1", ht)))
    x = x + ht.transpose(0, 2, 1)
    h = m.LN(name + ".ln2", x)
    if cond is not None:
        g, b = _split(m.L(name + ".film2", cond).reshape(B, 1, 2 * D), D)
        h = h * (g + 1.0) + b
    x = x + m.L(name + ".ch2", gelu(m.L(name + ".ch1", h)))
    return x


def _split(gb, D):
    # slice along last axis with autodiff (via matmul with selection matrices)
    I = np.eye(2 * D, dtype=np.float32)
    return gb @ T(I[:, :D]), gb @ T(I[:, D:])


def interp_matrix(B, cells):
    """Sparse (B*Nq, B*NTOK) trilinear interpolation from token grid to query cells (B, Nq)."""
    Nq = cells.shape[1]
    rows = np.repeat(np.arange(B * Nq), 8)
    cols = (TOK_IDX[cells.ravel()] + (np.arange(B)[:, None, None] * NTOK).repeat(Nq, 1).reshape(-1, 1)).ravel()
    vals = TOK_W[cells.ravel()].ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(B * Nq, B * NTOK))


# ---------------- CoMech ----------------
class CoMech(Module):
    """Latent mechanical world model: E_s (patch-mixer) -> s; g(a) -> u; s' = D(s,u) (FiLM mixers);
    neural-field decoder Phi(x | s'); effect head Q(s')."""

    def __init__(self, D=64, Du=64, Ht=192, Hc=128, Hd=128, n_enc=2, n_dyn=2, seed=0, act_concat=False):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.D, self.Du, self.n_enc, self.n_dyn = D, Du, n_enc, n_dyn
        self.act_concat = act_concat
        pin = NCH * int(np.prod(PATCH)) + (NACT if act_concat else 0)
        self.lin("patch", pin, D, rng)
        self.add("pos", rng.standard_normal((NTOK, D)).astype(np.float32) * 0.02)
        for k in range(n_enc):
            mixer_init(self, f"enc{k}", D, Ht, Hc, rng, film_dim=Du if act_concat else None)
        self.ln("enc_out", D)
        if act_concat:
            self.nact = self.mlp("act", [NACT, Du, Du], rng)
        else:
            self.nact = self.mlp("act", [NACT, Du, Du], rng)
            for k in range(n_dyn):
                mixer_init(self, f"dyn{k}", D, Ht, Hc, rng, film_dim=Du)
        self.ndec = self.mlp("dec", [D + NQF + NCH, Hd, Hd, Hd, 4], rng)
        self.lin("pool_q", D, 1, rng)
        self.neff = self.mlp("eff", [2 * D, 128, NEFF], rng)

    # --- pieces ---
    def encode(self, patches, a=None):
        x = T(patches)
        if self.act_concat:
            B = patches.shape[0]
            x = concat([x, T(np.repeat(a[:, None, :], NTOK, 1))], -1)
        x = self.L("patch", x) + self.p["pos"]
        cond = self.MLP("act", T(a), self.nact) if self.act_concat else None
        for k in range(self.n_enc):
            x = mixer(self, f"enc{k}", x, cond)
        return self.LN("enc_out", x)

    def act_enc(self, a):
        return self.MLP("act", T(a), self.nact)

    def step(self, s, u):
        for k in range(self.n_dyn):
            s = mixer(self, f"dyn{k}", s, u)
        return s

    def decode(self, s, cells, mat):
        """s: (B, NTOK, D) tensor; cells: (B, Nq) int; mat: (B, Nq, NCH) material fractions at the query cell."""
        B, Nq = cells.shape
        W = interp_matrix(B, cells)
        f = spmm(W, s.reshape(B * NTOK, self.D)).reshape(B, Nq, self.D)
        x = concat([f, T(CELL_FOURIER[cells]), T(mat)], -1)
        return self.MLP("dec", x, self.ndec)

    def effect(self, s):
        B = s.shape[0]
        w = softmax(self.L("pool_q", s).reshape(B, NTOK), axis=1)          # (B, NTOK)
        att = (s * w.reshape(B, NTOK, 1)).sum(1)
        mean = s.mean(1)
        return self.MLP("eff", concat([mean, att], -1), self.neff)

    def rep(self, s):
        """Pooled latent representation (for clustering / equivalence in latent space)."""
        return s.mean(1)


# ---------------- geometry-to-field surrogate ----------------
class Geo2Field(CoMech):
    """Same encoder/decoder, but the input is the explicit post-operative geometry (+ load) and there is
    no latent transition: s' = E(post-op geometry, load)."""

    def __init__(self, **kw):
        kw.setdefault("act_concat", True)
        super().__init__(**kw)


# ---------------- DeepONet ----------------
class DeepONet(Module):
    def __init__(self, P=64, H=256, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.P = P
        bin_ = NTOK * NCH + NACT
        self.nb = self.mlp("branch", [bin_, H, H], rng)
        self.lin("bout", H, 4 * P, rng)
        self.lin("bbias", H, 4, rng)
        self.nt = self.mlp("trunk", [NQF + NCH, 128, 128, 4 * P], rng)
        self.neff = self.mlp("eff", [H, 128, NEFF], rng)

    def forward(self, geo_pooled, a, cells, mat):
        B, Nq = cells.shape
        h = gelu(self.MLP("branch", T(np.concatenate([geo_pooled, a], 1)), self.nb))
        bc = self.L("bout", h).reshape(B, 1, 4, self.P)
        tr = self.MLP("trunk", T(np.concatenate([CELL_FOURIER[cells], mat], -1)), self.nt).reshape(B, Nq, 4, self.P)
        y = (bc * tr).sum(-1) * (1.0 / np.sqrt(self.P)) + self.L("bbias", h).reshape(B, 1, 4)
        return y, self.MLP("eff", h, self.neff)


# ---------------- Fourier neural operator ----------------
def fno_modes(shape, m):
    X, Y, Z = shape
    kx = [i % X for i in range(-m[0], m[0])]
    ky = [i % Y for i in range(-m[1], m[1])]
    kz = [i % Z for i in range(-m[2], m[2])]
    idx = [(a * Y + b) * Z + c for a in kx for b in ky for c in kz]
    return np.unique(np.array(idx))


class FNO(Module):
    def __init__(self, width=16, modes=(6, 4, 4), layers=3, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        from ml.data import GSHAPE
        self.W, self.nl = width, layers
        self.mats = tuple(ad.dft_mats(n, m) for n, m in zip(GSHAPE, modes))     # truncated DFT = FFT + mode selection
        msh = tuple(2 * m for m in modes)
        self.lin("lift", NCH + 3 + NACT, width, rng)
        for l in range(layers):
            sc = 1.0 / (width * width)
            self.add(f"sw{l}r", (rng.standard_normal((width, width) + msh) * sc).astype(np.float32))
            self.add(f"sw{l}i", (rng.standard_normal((width, width) + msh) * sc).astype(np.float32))
            self.lin(f"pw{l}", width, width, rng)
        self.nproj = self.mlp("proj", [width, 64, 4], rng)
        self.neff = self.mlp("eff", [width * 2, 64, NEFF], rng)

    def forward(self, geo, a, cells):
        """geo: (B, NCH, X, Y, Z) post-operative geometry; a: (B, NACT); cells: (B, Nq)."""
        from ml.data import GSHAPE, CELL_UVW
        B = geo.shape[0]
        X, Y, Z = GSHAPE
        coords = CELL_UVW.T.reshape(1, 3, X, Y, Z).repeat(B, 0)
        av = np.broadcast_to(a[:, :, None, None, None], (B, NACT, X, Y, Z))
        inp = np.concatenate([geo, coords, av], 1).astype(np.float32)      # (B, Cin, X, Y, Z)
        h = T(inp.transpose(0, 2, 3, 4, 1))                                # channels last
        h = self.L("lift", h)                                              # (B,X,Y,Z,W)
        for l in range(self.nl):
            hc = h.transpose(0, 4, 1, 2, 3)                                # (B,W,X,Y,Z)
            sc = ad.spectral_conv3d_dft(hc, self.p[f"sw{l}r"], self.p[f"sw{l}i"], self.mats).transpose(0, 2, 3, 4, 1)
            h = sc + self.L(f"pw{l}", h)
            if l < self.nl - 1:
                h = gelu(h)
        flat = h.reshape(B, X * Y * Z, self.W)
        idx = (cells + (np.arange(B) * X * Y * Z)[:, None]).ravel()
        g = ad.take(flat.reshape(B * X * Y * Z, self.W), idx, 0).reshape(B, cells.shape[1], self.W)
        y = self.MLP("proj", g, self.nproj)
        pooled = concat([flat.mean(1), g.mean(1)], -1)
        return y, self.MLP("eff", pooled, self.neff)
