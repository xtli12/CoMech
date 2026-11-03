"""Self-test before the long GPU runs: tensor layout, every model's forward/backward at the full batch size
(time and memory per step), and the predictor interface used by the experiment scripts.

    CUDA_VISIBLE_DEVICES=0 python3 gpu/selftest.py
Writes logs/selftest.json; exits with status 1 if anything fails.
"""
import os, sys, json, time, traceback, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ml.data import Norm, patchify as np_patchify, NCH, GRID
from gpu import config as CFG
from gpu.cache import Cache
from gpu.tdata import Pairs, Sampler
from gpu.tmodels import build, patchify, count_params
from gpu.ttrain import loss_comech, loss_actcat, loss_geo_micro, to_dev, model_cfg, paper_eps

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rep, ok = dict(device=str(dev), torch=torch.__version__), True


def check(label, fn):
    global ok
    t = time.time()
    try:
        r = fn(); rep[label] = dict(ok=True, s=round(time.time() - t, 2), **(r or {}))
        print("PASS", label, rep[label], flush=True)
    except Exception as e:
        ok = False; rep[label] = dict(ok=False, error=repr(e), trace=traceback.format_exc()[-2000:])
        print("FAIL", label, repr(e), flush=True); traceback.print_exc()


def t_patchify():
    x = np.random.default_rng(0).random((2, NCH) + tuple(GRID.shape)).astype(np.float32)
    d = np.abs(patchify(torch.as_tensor(x)).numpy() - np_patchify(x)).max()
    assert d == 0.0, d


norm = Norm.cached(); eps = paper_eps()
C = Cache(["test"], n=16)
pairs = Pairs(C, norm, eps, CFG.RHO)


def t_step(kind, Bf, Kb, Nq, n_steps=3):
    def run():
        torch.manual_seed(0)
        m = build(kind, model_cfg(kind)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        smp = Sampler(C, norm, pairs, seed=1, Bf=Bf, Kb=Kb, Nq=Nq, need_post=kind in CFG.GEO_KINDS)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times, losses = [], []
        for _ in range(n_steps):
            b = to_dev(smp.next(), dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.time(); opt.zero_grad()
            if kind == "comech":
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    L, logs = loss_comech(m, b, {}, eps)
                L.backward()
            elif kind == "actcat":
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    L, logs = loss_actcat(m, b)
                L.backward()
            else:
                n = len(b["tr"]); micro = getattr(CFG, "MICRO", {"gino": 20, "geofno": 10})[kind]; tot = 0.0
                for idx in np.array_split(np.arange(n), max(1, min(micro, n))):
                    Lf, Le = loss_geo_micro(m, b, torch.as_tensor(idx, device=dev))
                    ((Lf + Le) * len(idx) / n).backward(); tot += float(Lf + Le) * len(idx) / n
                L = torch.tensor(tot)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t); losses.append(float(L))
            assert np.isfinite(losses[-1]), losses
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
        est_h = float(np.median(times[1:] if len(times) > 1 else times)) * CFG.STEPS / 3600
        return dict(params=count_params(m), step_s=round(float(np.median(times)), 3), peak_gb=round(mem, 2),
                    est_hours_full_run=round(est_h, 2), losses=[round(x, 4) for x in losses])
    return run


def t_overfit():
    torch.manual_seed(0)
    m = build("comech", dict(D=64, Du=64, Ht=192, Hc=128, Hd=128, n_enc=2, n_dyn=2, n_dec=3)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    b = to_dev(Sampler(C, norm, pairs, seed=3, Bf=2, Kb=4, Nq=512).next(), dev)
    L0 = None
    for i in range(150):
        opt.zero_grad(); L, _ = loss_comech(m, b, {}, eps); L.backward(); opt.step()
        L0 = float(L) if L0 is None else L0
    assert float(L) < 0.5 * L0, (L0, float(L))
    return dict(first=round(L0, 4), last=round(float(L), 4))


def t_predictor(kind):
    def run():
        from gpu.tpredict import TorchPredictor
        from ml.data import Split
        from fe.geometry import VOID
        small = dict(comech=dict(D=32, Du=32, Ht=64, Hc=64, Hd=64, n_enc=1, n_dyn=1, n_dec=2),
                     gino=dict(width=16, latent=(16, 10, 12), modes=(4, 3, 3), layers=2, r_in=5.0, r_out=6.0, kmlp=16),
                     geofno=dict(width=16, latent=(24, 12, 16), modes=(4, 3, 3), layers=3))
        cfg = small.get(kind, small["comech"])
        m = build(kind, cfg)
        d = tempfile.mkdtemp()
        p = f"{d}/x.pt"; torch.save(dict(cfg=dict(kind=kind, model=cfg), state=m.state_dict()), p)
        P = TorchPredictor(p)
        S = Split("test", max_feet=1, load_post=True)
        pre = S.geo_pre[0].astype(np.float32); post = S.geo_post[0].astype(np.float32)
        cells = np.nonzero(S.lab[0, 1] != VOID)[0]
        e = P.effects(pre, S.act[0], post if P.kind == "geo" else None)
        assert e.shape == (8, 17), e.shape
        y, e1 = P.fields_and_effects(pre, S.act[0, 1:2], cells, post[1:2] if P.kind == "geo" else None)
        assert y.shape == (1, len(cells), 4) and e1.shape == (1, 17), (y.shape, e1.shape)
        assert np.allclose(e1[0], e[1], atol=2e-3), np.abs(e1[0] - e[1]).max()
        if kind == "comech":
            y2, e2 = P.rollout2(pre, S.act[0, 1:2], S.act[0, 2:3] * np.array([0, 0, 0, 1, 0, 0, 0], np.float32), cells[None])
            assert y2.shape == (1, len(cells), 4)
            s = P.m.encode(np_patchify(pre[None])); u = P.m.act_enc(np.zeros((1, 7), np.float32))
            assert P.m.effect(P.m.step(s, u)).data.shape == (1, 17)
        return dict(kind_seen=P.kind)
    return run


check("patchify_layout", t_patchify)
check("overfit_small_comech", t_overfit)
for k in ("comech", "actcat", "gino", "geofno"):
    check(f"predictor_{k}", t_predictor(k))
for k in ("comech", "actcat", "gino", "geofno"):
    check(f"full_batch_step_{k}", t_step(k, CFG.BF, CFG.KB, CFG.NQ, n_steps=4 if dev.type == "cuda" else 1))
rep["all_ok"] = ok
os.makedirs("logs", exist_ok=True)
json.dump(rep, open("logs/selftest.json", "w"), indent=1)
print("SELFTEST", "OK" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
