"""Train one run of the GPU scaling experiment (see gpu/config.py).

    CUDA_VISIBLE_DEVICES=0 python3 gpu/ttrain.py comechL_s0 [--steps N] [--smoke]

Writes results/models_gpu/<name>.pt (final weights) and logs/train_<name>.csv; checkpoints every 2,000 steps to
results/models_gpu/<name>.ckpt so that an interrupted run resumes where it stopped.
"""
import os, sys, time, json, math, argparse, pickle, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ml.data import Norm
from gpu import config as CFG
from gpu.cache import Cache
from gpu.tdata import Pairs, Sampler, Prefetch
from gpu.tmodels import build, patchify, count_params

OUT = "results/models_gpu"


def paper_eps():
    if CFG.EPS is not None:
        return CFG.EPS
    for p in ("results/models_long/comech_s0.pkl", "results/models/comech_s0.pkl"):
        if os.path.exists(p):
            return float(pickle.load(open(p, "rb"))["cfg"]["eps"])
    return 0.1326


def model_cfg(kind):
    return dict(comech=CFG.COMECH_L, actcat=CFG.COMECH_L, gino=CFG.GINO_CFG, geofno=CFG.GEOFNO_CFG)[kind]


def lr_at(it, steps):
    lr = CFG.LR * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / steps)))
    if it < CFG.WARMUP:
        lr *= (it + 1) / CFG.WARMUP
    return lr


def wmse(y, t, w=None):
    d = (y.float() - t) ** 2
    if w is not None:
        d = d * w[..., None]
    return d.mean()


def to_dev(b, dev):
    out = {}
    for k, v in b.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.as_tensor(v).to(dev, non_blocking=True)
        else:
            out[k] = v
    return out


def loss_comech(m, b, flags, eps):
    pre = patchify(b["pre"])
    s0 = m.encode(pre)
    s = s0[b["slot"]]
    s1 = m.step(s, m.act_enc(b["a"]))
    y = m.decode(s1, b["cells"], b["mat_pre"]); e = m.effect(s1)
    L = wmse(y, b["y"], b["w"]); logs = dict(field=float(L))
    Le = wmse(e, b["eff"]); logs["eff"] = float(Le); L = L + Le
    if flags.get("roll", 1.0) > 0 and len(b["roll"]):
        ia, ib = b["roll"][:, 0], b["roll"][:, 1]
        a2 = torch.zeros_like(b["a"][ib]); a2[:, 3] = b["a"][ib, 3]          # osteotomy-only second action
        s_ab = m.step(s1[ia], m.act_enc(a2))
        y2 = m.decode(s_ab, b["cells"][ib], b["mat_pre"][ib]); e2 = m.effect(s_ab)
        Lr = wmse(y2, b["y"][ib], b["w"][ib]) + wmse(e2, b["eff"][ib])
        logs["roll"] = float(Lr); L = L + flags.get("roll", 1.0) * Lr
    Z = b["eff"]
    if flags.get("swap", 1.0) > 0 and len(b["eq"]):
        ia, ib = b["eq"][:, 0], b["eq"][:, 1]
        ea, eb = e[ia].float(), e[ib].float()
        Ls = ((ea - eb) ** 2).mean() + ((ea - Z[ib]) ** 2).mean() + ((eb - Z[ia]) ** 2).mean()
        logs["swap"] = float(Ls); L = L + flags.get("swap", 1.0) * Ls
    if flags.get("sens", 1.0) > 0 and len(b["nb"]):
        ia, ib = b["nb"][:, 0], b["nb"][:, 1]
        ea, eb = e[ia].float(), e[ib].float()
        dist = lambda x, z: torch.sqrt(((x - z) ** 2).mean(1) + 1e-8)
        mu = CFG.MU_FRAC * eps
        h1 = torch.relu(dist(ea, Z[ia]) - dist(ea, Z[ib]) + mu)
        h2 = torch.relu(dist(eb, Z[ib]) - dist(eb, Z[ia]) + mu)
        Ln = (h1 + h2).mean()
        logs["sens"] = float(Ln); L = L + flags.get("sens", 1.0) * Ln
    return L, logs


def loss_actcat(m, b):
    pre = b["pre"][b["slot"]]
    s = m.encode(patchify(pre), b["a"])
    y = m.decode(s, b["cells"], b["mat_pre"]); e = m.effect(s)
    Lf, Le = wmse(y, b["y"], b["w"]), wmse(e, b["eff"])
    return Lf + Le, dict(field=float(Lf), eff=float(Le))


def loss_geo_micro(m, b, idx):
    y, e = m(b["post"][idx], b["a"][idx, 6], b["cells"][idx], b["mat_post"][idx])
    Lf, Le = wmse(y, b["y"][idx], b["w"][idx]), wmse(e, b["eff"][idx])
    return Lf, Le


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--steps", type=int, default=None); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    name = args.name
    kind, nfeet, seed, flags = CFG.RUNS[name]
    steps = args.steps or CFG.STEPS
    os.makedirs(OUT, exist_ok=True); os.makedirs("logs", exist_ok=True)
    final = f"{OUT}/{name}.pt"
    if os.path.exists(final) and not args.smoke:
        print("already trained", name); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed); np.random.seed(seed)
    norm = Norm.cached()
    eps = paper_eps()
    if args.smoke:
        C = Cache(["test"], n=6); nfeet = C.nf
    else:
        C = Cache(["train"] + (["train_x"] if nfeet > 180 else []), n=nfeet)
    pairs = Pairs(C, norm, eps, CFG.RHO)
    geo = kind in CFG.GEO_KINDS
    Bf, Kb, Nq = (2, 3, 256) if args.smoke else (CFG.BF, CFG.KB, CFG.NQ)
    mk = lambda w: Sampler(C, norm, pairs, seed=seed * 1000 + 17 * w + 1, Bf=Bf, Kb=Kb, Nq=Nq, need_post=geo)
    pf = Prefetch(mk, n_workers=1 if args.smoke else 4, depth=2 if args.smoke else 8)
    mcfg = model_cfg(kind)
    m = build(kind, mcfg).to(dev)
    nparams = count_params(m)
    opt = torch.optim.AdamW(m.parameters(), lr=CFG.LR, betas=(0.9, 0.999), weight_decay=CFG.WD)
    start, hist = 0, []
    ck = f"{OUT}/{name}.ckpt"
    if os.path.exists(ck) and not args.smoke:
        st = torch.load(ck, map_location=dev, weights_only=False)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["it"]; hist = st["hist"]
        print("resumed", name, "at", start, flush=True)
    use_amp = dev.type == "cuda" and kind in ("comech", "actcat")
    micro = getattr(CFG, "MICRO", {"gino": 20, "geofno": 10}).get(kind, 1)   # transitions per micro-batch = n / micro
    print(json.dumps(dict(name=name, kind=kind, feet=C.nf, steps=steps, params=nparams, eps=eps, dev=str(dev),
                          eq_pairs=sum(map(len, pairs.eq)), nb_pairs=sum(map(len, pairs.nb)))), flush=True)
    logf = open(f"logs/train_{name}.csv", "a")
    t0 = time.time(); m.train()
    for it in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, steps)
        b = to_dev(pf.next(), dev)
        opt.zero_grad(set_to_none=True)
        if kind == "comech":
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                L, logs = loss_comech(m, b, flags, eps)
            L.backward()
        elif kind == "actcat":
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                L, logs = loss_actcat(m, b)
            L.backward()
        else:
            n = len(b["tr"]); lf = le = 0.0
            for idx in np.array_split(np.arange(n), max(1, min(micro, n))):
                ii = torch.as_tensor(idx, device=dev)
                Lf, Le = loss_geo_micro(m, b, ii)
                ((Lf + Le) * (len(idx) / n)).backward()
                lf += float(Lf) * len(idx) / n; le += float(Le) * len(idx) / n
            logs = dict(field=lf, eff=le); L = torch.tensor(lf + le)
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CFG.CLIP)
        if not torch.isfinite(gn):
            print("non-finite gradient at", it, "- step skipped", flush=True); opt.zero_grad(set_to_none=True); continue
        opt.step()
        logs.update(loss=float(L), gn=float(gn), lr=lr_at(it, steps)); hist.append(logs)
        if (it + 1) % 100 == 0 or args.smoke:
            avg = {k: float(np.mean([h[k] for h in hist[-100:] if k in h])) for k in hist[-1]}
            el = time.time() - t0
            print(name, it + 1, f"{el:.0f}s", f"eta {el / (it + 1 - start) * (steps - it - 1) / 3600:.1f}h",
                  {k: round(v, 4) for k, v in avg.items()}, flush=True)
            logf.write(json.dumps(dict(it=it + 1, t=el, **avg)) + "\n"); logf.flush()
        if (it + 1) % 2000 == 0 and not args.smoke:
            torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), it=it + 1, hist=hist[-2000:]), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    pf.stop = True
    cfg = dict(kind=kind, name=name, model=mcfg, feet=C.nf, steps=steps, seed=seed, flags=flags, eps=eps, rho=CFG.RHO,
               Bf=Bf, Kb=Kb, Nq=Nq, lr=CFG.LR, wd=CFG.WD, warmup=CFG.WARMUP, params=nparams,
               train_time=time.time() - t0, gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
    out = final if not args.smoke else f"{OUT}/smoke_{name}.pt"
    torch.save(dict(cfg=cfg, state=m.state_dict(), hist_tail=hist[-500:]), out + ".tmp")
    os.replace(out + ".tmp", out)
    json.dump(cfg, open(out[:-3] + ".json", "w"), indent=1)
    if os.path.exists(ck) and not args.smoke:
        os.remove(ck)
    print("DONE", name, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
