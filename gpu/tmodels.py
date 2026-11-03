"""PyTorch models for the GPU scaling experiment.

CoMechT      the CoMech world model of the paper (patch-mixer state encoder, FiLM-mixer latent transition,
             neural-field decoder, effect head), scaled in width and depth; act_concat=True gives the
             action-concatenated surrogate (no latent transition).
GINO         geometry-informed neural operator (Li et al., NeurIPS 2023): graph-kernel encoder from the cell
             point cloud to a regular latent grid, Fourier layers on the latent grid, graph-kernel decoder to
             arbitrary query cells.  Receives the explicit post-operative geometry and the load.
GeoFNO       geometry-aware Fourier neural operator (Li et al., JMLR 2023): learned deformation of the physical
             coordinates to a uniform computational domain, non-uniform Fourier transforms between the cells and
             the latent grid, Fourier layers in between.  On the engine's rectilinear grid the learned deformation is
             axis-aligned (one monotone map per axis).  Receives the explicit post-operative geometry and the load.

All models return normalised fields (log-stress, 3 displacement components) at query cells and the 17 normalised
effect descriptors, exactly as the numpy models in ml/models.py.
"""
import os, sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import NTOK, TOK, NCH, PATCH, TOK_IDX, TOK_W, CELL_FOURIER, CELL_UVW, GSHAPE, GRID

NEFF, NACT = 17, 7
NQF = CELL_FOURIER.shape[1]
PIN = NCH * int(np.prod(PATCH))


def patchify(g):
    """(B, C, X, Y, Z) -> (B, NTOK, C*48); identical ordering to ml.data.patchify."""
    B, C = g.shape[:2]
    px, py, pz = PATCH; tx, ty, tz = TOK
    g = g.reshape(B, C, tx, px, ty, py, tz, pz).permute(0, 2, 4, 6, 1, 3, 5, 7)
    return g.reshape(B, tx * ty * tz, C * px * py * pz)


def mlp(dims, last_scale=1.0):
    layers = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        if i == len(dims) - 2 and last_scale != 1.0:
            with torch.no_grad():
                lin.weight.mul_(last_scale)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class Film(nn.Module):
    def __init__(self, cdim, D):
        super().__init__()
        self.lin = nn.Linear(cdim, 2 * D)
        nn.init.normal_(self.lin.weight, std=0.1 / math.sqrt(cdim)); nn.init.zeros_(self.lin.bias)

    def forward(self, h, c):
        g, b = self.lin(c).unsqueeze(1).chunk(2, dim=-1)
        return h * (1.0 + g) + b


class Mixer(nn.Module):
    def __init__(self, D, Ht, Hc, film_dim=None):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.tok = mlp([NTOK, Ht, NTOK], last_scale=0.5)
        self.ch = mlp([D, Hc, D], last_scale=0.5)
        self.f1 = Film(film_dim, D) if film_dim else None
        self.f2 = Film(film_dim, D) if film_dim else None

    def forward(self, x, c=None):
        h = self.ln1(x)
        if c is not None:
            h = self.f1(h, c)
        x = x + self.tok(h.transpose(1, 2)).transpose(1, 2)
        h = self.ln2(x)
        if c is not None:
            h = self.f2(h, c)
        return x + self.ch(h)


class CoMechT(nn.Module):
    def __init__(self, D=192, Du=128, Ht=512, Hc=768, Hd=256, n_enc=4, n_dyn=4, n_dec=4, act_concat=False):
        super().__init__()
        self.D, self.act_concat = D, act_concat
        self.patch = nn.Linear(PIN + (NACT if act_concat else 0), D)
        self.pos = nn.Parameter(torch.randn(NTOK, D) * 0.02)
        self.act = mlp([NACT, Du, Du])
        self.enc = nn.ModuleList([Mixer(D, Ht, Hc, Du if act_concat else None) for _ in range(n_enc)])
        self.enc_out = nn.LayerNorm(D)
        self.dyn = None if act_concat else nn.ModuleList([Mixer(D, Ht, Hc, Du) for _ in range(n_dyn)])
        self.dec = mlp([D + NQF + NCH] + [Hd] * n_dec + [4])
        self.pool_q = nn.Linear(D, 1)
        self.eff = mlp([2 * D, 256, NEFF])
        self.register_buffer("tok_idx", torch.as_tensor(TOK_IDX, dtype=torch.long), persistent=False)
        self.register_buffer("tok_w", torch.as_tensor(TOK_W, dtype=torch.float32), persistent=False)
        self.register_buffer("cell_f", torch.as_tensor(CELL_FOURIER, dtype=torch.float32), persistent=False)

    def encode(self, patches, a=None):
        x, cond = patches, None
        if self.act_concat:
            x = torch.cat([x, a[:, None, :].expand(-1, NTOK, -1)], -1)
            cond = self.act(a)
        x = self.patch(x) + self.pos
        for blk in self.enc:
            x = blk(x, cond)
        return self.enc_out(x)

    def act_enc(self, a):
        return self.act(a)

    def step(self, s, u):
        for blk in self.dyn:
            s = blk(s, u)
        return s

    def decode(self, s, cells, mat):
        """s: (B, NTOK, D); cells: (B, Nq) long; mat: (B, Nq, NCH) material fractions at the query cells."""
        B, Nq = cells.shape
        gi = self.tok_idx[cells] + (torch.arange(B, device=s.device) * NTOK)[:, None, None]      # (B, Nq, 8)
        w = self.tok_w[cells].to(s.dtype)
        f = torch.einsum("bqkd,bqk->bqd", s.reshape(B * NTOK, -1)[gi], w)
        x = torch.cat([f, self.cell_f[cells].to(f.dtype), mat.to(f.dtype)], -1)
        return self.dec(x)

    def effect(self, s):
        w = torch.softmax(self.pool_q(s).squeeze(-1).float(), dim=1).to(s.dtype)
        att = (s * w[..., None]).sum(1)
        return self.eff(torch.cat([s.mean(1), att], -1))

    def rep(self, s):
        return s.mean(1)


# ---------------------------------------------------------------- Fourier layers
class SpectralConv3d(nn.Module):
    """Standard FNO spectral convolution on a uniform grid (rfftn, four corner blocks of retained modes)."""

    def __init__(self, cin, cout, modes):
        super().__init__()
        self.m = modes
        sc = 1.0 / (cin * cout)
        self.w = nn.ParameterList([nn.Parameter(sc * torch.randn(cin, cout, *modes, dtype=torch.cfloat)) for _ in range(4)])

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        m1, m2, m3 = self.m
        xf = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
        out = torch.zeros(B, self.w[0].shape[1], X, Y, Z // 2 + 1, dtype=torch.cfloat, device=x.device)
        mul = lambda a, w: torch.einsum("bixyz,ioxyz->boxyz", a, w)
        out[:, :, :m1, :m2, :m3] = mul(xf[:, :, :m1, :m2, :m3], self.w[0])
        out[:, :, -m1:, :m2, :m3] = mul(xf[:, :, -m1:, :m2, :m3], self.w[1])
        out[:, :, :m1, -m2:, :m3] = mul(xf[:, :, :m1, -m2:, :m3], self.w[2])
        out[:, :, -m1:, -m2:, :m3] = mul(xf[:, :, -m1:, -m2:, :m3], self.w[3])
        return torch.fft.irfftn(out, s=(X, Y, Z), dim=(-3, -2, -1), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, W, modes):
        super().__init__()
        self.spec = SpectralConv3d(W, W, modes)
        self.pw = nn.Conv3d(W, W, 1)

    def forward(self, x):
        return F.gelu(self.spec(x) + self.pw(x))


def _box():
    """Bounding box (mm) of the engine's analysis grid, read from fe.geometry.ML_BOX as in ml.data.cell_norm_coords;
    the parametric engine's [0, 96] x [-18, 30] x [0, 63] when ML_BOX is not defined.  These values were hard-coded
    here before 2026-10-31; with the real-bone engine they covered only ~6% of the analysis cells (Claude)."""
    import fe.geometry as _FG
    box = getattr(_FG, "ML_BOX", ((0, 96), (-18, 30), (0, 63)))
    return np.array([b[0] for b in box], float), np.array([b[1] for b in box], float)


def _latent_coords(shape):
    """Cell-centred coordinates of a uniform latent grid over the engine's bounding box, (N, 3) in mm."""
    lo, hi = _box()
    axes = [lo[d] + (np.arange(n) + 0.5) / n * (hi[d] - lo[d]) for d, n in enumerate(shape)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return G


def _neighbours(src, dst, r, cap):
    """For every dst point, indices of src points within radius r (padded with -1, at most cap)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(src)
    lists = tree.query_ball_point(dst, r)
    M = min(cap, max(len(l) for l in lists))
    idx = -np.ones((len(dst), M), np.int64)
    for i, l in enumerate(lists):
        if len(l) > M:     # keep the nearest
            d = np.linalg.norm(src[l] - dst[i], axis=1); l = list(np.asarray(l)[np.argsort(d)[:M]])
        idx[i, :len(l)] = l
    return idx


class GNOLayer(nn.Module):
    """Kernel integral v(x) = mean_{y in B_r(x)} k(x, y) * g(y), with k an MLP of (x - y, x)."""

    def __init__(self, W, kmlp, r):
        super().__init__()
        self.r = r
        self.k = mlp([6, kmlp, kmlp, W])

    def forward(self, g, nbr, rel, xq):
        """g: (B, Ns, W) source features; nbr: (Nd, M) source indices (-1 = pad); rel: (Nd, M, 3) x - y in mm;
        xq: (Nd, 3) normalised destination coordinates."""
        mask = (nbr >= 0).float()                                            # (Nd, M)
        kin = torch.cat([rel / self.r, xq[:, None, :].expand(-1, nbr.shape[1], -1)], -1)
        k = self.k(kin) * mask[..., None]                                    # (Nd, M, W)
        Nd, M = nbr.shape
        # same values as g[:, nbr.clamp(min=0)], but index_select back-propagates with index_add_ instead of the
        # sort-based index_put_ kernel, which took ~90% of a GINO training step on the A100 (profiled 2026-10-25)
        gg = g.index_select(1, nbr.clamp(min=0).reshape(-1)).view(g.shape[0], Nd, M, -1)   # (B, Nd, M, W)
        return (gg * k[None]).sum(2) / mask.sum(1).clamp(min=1.0)[None, :, None]

    def batched(self, g, nbr, rel, xq):
        """Per-sample destinations: g (B, Ns, W); nbr (B, Nd, M); rel (B, Nd, M, 3); xq (B, Nd, 3)."""
        B, Nd, M = nbr.shape
        mask = (nbr >= 0).float()
        kin = torch.cat([rel / self.r, xq[:, :, None, :].expand(-1, -1, M, -1)], -1)
        k = self.k(kin) * mask[..., None]                                    # (B, Nd, M, W)
        idx = nbr.clamp(min=0).reshape(B, Nd * M, 1).expand(-1, -1, g.shape[-1])
        gg = torch.gather(g, 1, idx).reshape(B, Nd, M, -1)
        return (gg * k).sum(2) / mask.sum(2).clamp(min=1.0)[..., None]


class GINO(nn.Module):
    def __init__(self, width=64, latent=(40, 24, 32), modes=(12, 8, 10), layers=4, r_in=3.2, r_out=3.2, kmlp=64):
        super().__init__()
        self.W, self.latent = width, latent
        cells = GRID.cell_xyz.astype(np.float64)
        lat = _latent_coords(latent)
        lo, hi = _box()
        enc = _neighbours(cells, lat, r_in, 64)              # latent node <- cells
        dec = _neighbours(lat, cells, r_out, 48)             # cell <- latent nodes
        rel_e = np.where(enc[..., None] >= 0, lat[:, None, :] - cells[enc.clip(0)], 0.0)
        rel_d = np.where(dec[..., None] >= 0, cells[:, None, :] - lat[dec.clip(0)], 0.0)
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
        self.register_buffer("enc_nbr", t(enc, torch.long), persistent=False)
        self.register_buffer("enc_rel", t(rel_e), persistent=False)
        self.register_buffer("lat_n", t((lat - lo) / (hi - lo)), persistent=False)
        self.register_buffer("dec_nbr", t(dec, torch.long), persistent=False)
        self.register_buffer("dec_rel", t(rel_d), persistent=False)
        self.register_buffer("cell_n", t((cells - lo) / (hi - lo)), persistent=False)
        self.register_buffer("cell_f", t(CELL_FOURIER), persistent=False)
        self.lift = mlp([NCH + 1 + 3, width, width])
        self.gno_in = GNOLayer(width, kmlp, r_in)
        self.lat_in = nn.Conv3d(width + 3 + 1, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(layers)])
        self.gno_out = GNOLayer(width, kmlp, r_out)
        self.head = mlp([width + NCH + NQF, 128, 128, 4])
        self.pool_q = nn.Linear(width, 1)
        self.eff = mlp([2 * width, 256, NEFF])

    def latent_field(self, geo, load):
        """geo: (B, NCH, X, Y, Z) post-operative geometry; load: (B,) load multiplier - 1."""
        B = geo.shape[0]
        g = geo.reshape(B, NCH, -1).transpose(1, 2)                                    # (B, Ncell, NCH)
        feats = torch.cat([g, load[:, None, None].expand(-1, g.shape[1], 1), self.cell_n[None].expand(B, -1, -1)], -1)
        h = self.lift(feats)                                                             # (B, Ncell, W)
        v = self.gno_in(h, self.enc_nbr, self.enc_rel, self.lat_n)                        # (B, Nl, W)
        Lx, Ly, Lz = self.latent
        v = v.transpose(1, 2).reshape(B, self.W, Lx, Ly, Lz)
        coords = self.lat_n.T.reshape(1, 3, Lx, Ly, Lz).expand(B, -1, -1, -1, -1)
        v = self.lat_in(torch.cat([v, coords, load[:, None, None, None, None].expand(-1, 1, Lx, Ly, Lz)], 1))
        for blk in self.blocks:
            v = blk(v)
        return v                                                                         # (B, W, Lx, Ly, Lz)

    def forward(self, geo, load, cells, mat):
        """Fields at query cells (B, Nq) and effects."""
        B = geo.shape[0]
        v = self.latent_field(geo, load)
        flat = v.reshape(B, self.W, -1).transpose(1, 2)                                  # (B, Nl, W)
        u = self.gno_out.batched(flat, self.dec_nbr[cells], self.dec_rel[cells], self.cell_n[cells])   # (B, Nq, W)
        y = self.head(torch.cat([u, mat, self.cell_f[cells]], -1))
        w = torch.softmax(self.pool_q(flat).squeeze(-1), 1)
        e = self.eff(torch.cat([flat.mean(1), (flat * w[..., None]).sum(1)], -1))
        return y, e


# ---------------------------------------------------------------- Geo-FNO
class AxisMap(nn.Module):
    """Learned monotone map from the cells along one axis to [0, 1): cumulative softplus of per-cell increments,
    initialised to the physical (non-uniform) cell widths."""

    def __init__(self, widths):
        super().__init__()
        w = np.asarray(widths, np.float64); w = w / w.sum() * len(w)
        self.theta = nn.Parameter(torch.as_tensor(np.log(np.expm1(w)), dtype=torch.float32))

    def forward(self):
        inc = F.softplus(self.theta)
        edges = torch.cumsum(inc, 0) / inc.sum()
        return edges - 0.5 * inc / inc.sum()                    # cell-centre coordinates in (0, 1)


def _freqs(m):
    return torch.cat([torch.arange(0, m), torch.arange(-m, 0)]).float()           # 2m integer frequencies


class NUSpectral(nn.Module):
    """Spectral convolution between two point sets given by separable coordinates (ax, ay, az) in [0,1):
    forward non-uniform DFT on the source coordinates, complex weights on 2m1 x 2m2 x 2m3 modes, inverse non-uniform
    DFT on the destination coordinates."""

    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes = modes
        sc = 1.0 / (cin * cout)
        self.w = nn.Parameter(sc * torch.randn(cin, cout, *(2 * m for m in modes), dtype=torch.cfloat))
        for d, m in enumerate(modes):
            self.register_buffer(f"k{d}", _freqs(m), persistent=False)

    @staticmethod
    def _basis(coord, k, sign):
        return torch.exp(sign * 2j * math.pi * coord[:, None].to(torch.float32) * k[None, :])       # (n, 2m)

    def forward(self, x, src, dst):
        """x: (B, C, nx, ny, nz) real on src; src/dst: tuples of 1-D coordinate tensors."""
        B, C, nx, ny, nz = x.shape
        ks = (self.k0, self.k1, self.k2)
        Ef = [self._basis(s, k, -1.0) for s, k in zip(src, ks)]
        Ei = [self._basis(d, k, 1.0) for d, k in zip(dst, ks)]
        a = x.float().to(torch.cfloat)
        a = torch.einsum("bcxyz,xk->bckyz", a, Ef[0])
        a = torch.einsum("bckyz,yl->bcklz", a, Ef[1])
        a = torch.einsum("bcklz,zm->bcklm", a, Ef[2]) / (nx * ny * nz)
        a = torch.einsum("bcklm,cdklm->bdklm", a, self.w)
        o = torch.einsum("bdklm,xk->bdxlm", a, Ei[0])                                    # Ei: (n_dst, 2m)
        o = torch.einsum("bdxlm,yl->bdxym", o, Ei[1])
        o = torch.einsum("bdxym,zm->bdxyz", o, Ei[2])
        return o.real


class GeoFNO(nn.Module):
    def __init__(self, width=64, latent=(48, 24, 32), modes=(12, 8, 10), layers=4):
        super().__init__()
        self.W, self.latent = width, latent
        self.maps = nn.ModuleList([AxisMap(GRID.dx), AxisMap(GRID.dy), AxisMap(GRID.dz)])
        for d, n in enumerate(latent):
            self.register_buffer(f"lat{d}", (torch.arange(n).float() + 0.5) / n, persistent=False)
        self.register_buffer("uvw", torch.as_tensor(CELL_UVW.T.reshape(3, *GSHAPE), dtype=torch.float32), persistent=False)
        self.register_buffer("cell_f", torch.as_tensor(CELL_FOURIER, dtype=torch.float32), persistent=False)
        self.lift = nn.Conv3d(NCH + 1 + 3, width, 1)
        self.sp_in = NUSpectral(width, width, modes)
        self.lat_bias = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(layers - 2)])
        self.sp_out = NUSpectral(width, width, modes)
        self.pw_out = nn.Conv3d(width, width, 1)
        self.head = mlp([width + NCH + NQF, 128, 128, 4])
        self.pool_q = nn.Linear(width, 1)
        self.eff = mlp([2 * width, 256, NEFF])

    def forward(self, geo, load, cells, mat):
        B = geo.shape[0]
        X, Y, Z = GSHAPE
        inp = torch.cat([geo, load[:, None, None, None, None].expand(-1, 1, X, Y, Z), self.uvw[None].expand(B, -1, -1, -1, -1)], 1)
        h0 = self.lift(inp)                                                              # (B, W, X, Y, Z) on cells
        src = tuple(m() for m in self.maps)
        lat = (self.lat0, self.lat1, self.lat2)
        v = F.gelu(self.sp_in(h0, src, lat) + self.lat_bias[None, :, None, None, None])
        for blk in self.blocks:
            v = blk(v)
        h = F.gelu(self.sp_out(v, lat, src) + self.pw_out(h0))                           # back on the cells
        flat = h.reshape(B, self.W, -1).transpose(1, 2)                                   # (B, Ncell, W)
        idx = cells
        u = torch.gather(flat, 1, idx[..., None].expand(-1, -1, self.W))
        y = self.head(torch.cat([u, mat, self.cell_f[cells]], -1))
        vl = v.reshape(B, self.W, -1).transpose(1, 2)
        w = torch.softmax(self.pool_q(vl).squeeze(-1), 1)
        e = self.eff(torch.cat([vl.mean(1), (vl * w[..., None]).sum(1)], -1))
        return y, e


def build(kind, cfg):
    if kind == "comech":
        return CoMechT(**cfg)
    if kind == "actcat":
        return CoMechT(act_concat=True, **cfg)
    if kind == "gino":
        return GINO(**cfg)
    if kind == "geofno":
        return GeoFNO(**cfg)
    raise ValueError(kind)


def count_params(m):
    return int(sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters()))
