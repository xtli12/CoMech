"""E5 equivalence precision and E6 planning regret / abstention."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.common import *
from exp.sweeps import test_feet, PLAN_T
from ml.data import Split, Norm
from ml.train import Pairs
from scipy.stats import spearmanr

# planning objective on standardised effects (z), lower is better
W = dict(overload=1.0, transfer=0.5, deformation=0.5, invasive=0.3)


def J_of(z, t, fusion, interp):
    over = z[..., EI["post_mean"]] + z[..., EI["mid_mean"]]
    trans = z[..., EI["tn_p99"]] + z[..., EI["ant_mean"]]
    deform = np.abs(z[..., EI["talus_evert"]]) + np.abs(z[..., EI["nav_drop"]])
    inv = np.abs(t) / 10.0 + 2.0 * fusion + 0.1 * interp
    return W["overload"] * over + W["transfer"] * trans + W["deformation"] * deform + W["invasive"] * inv


def equivalence(names, S, norm, eps):
    out = {}
    Zt = norm.eff(S.eff_raw)
    for name in names:
        pred = get_pred(name)
        dp, df = [], []
        for f in range(S.nf):
            post = S.geo_post[f].astype(np.float32) if pred.kind in ("geo", "fno") else None
            Zp = pred.effects(S.geo_pre[f].astype(np.float32), S.act[f], post)
            for i in range(S.K):
                for j in range(i + 1, S.K):
                    dp.append(np.sqrt(((Zp[i] - Zp[j]) ** 2).mean())); df.append(np.sqrt(((Zt[f, i] - Zt[f, j]) ** 2).mean()))
        dp, df = np.array(dp), np.array(df)
        P = dp <= eps; T = df <= eps
        prec = float((P & T).sum() / max(P.sum(), 1)); rec = float((P & T).sum() / max(T.sum(), 1))
        out[name] = dict(precision=prec, recall=rec, n_pred=int(P.sum()), n_true=int(T.sum()), n_pairs=int(len(dp)),
                         spearman=float(spearmanr(dp, df)[0]))
        print("equiv", name, out[name], flush=True)
    return out


def planning(names, norm):
    feet = test_feet()
    PL = {}
    for f in glob.glob("data/sweeps/plan_*.json"):
        r = json.load(open(f)); PL.setdefault(r["i"], {})[r["tag"]] = r
    # adjuncts to complete resection; fusion abandons the mobile joint (its facet stresses vanish by
    # construction) and is therefore not ranked by the same objective
    cands = [(t, ip, 0) for t in PLAN_T for ip in (0, 1)]
    tags = [f"t{t:+.0f}_i{ip}" for t, ip, _ in cands]
    out = {"cands": cands, "feet": {}, "models": {}}
    ids = [i for i in PL if all(t in PL[i] for t in tags)]
    for i in ids:
        z = norm.eff(np.array([[PL[i][t]["eff"][k] for k in EI] for t in tags]))
        Jfe = np.array([J_of(z[c], *[cands[c][0], cands[c][2], cands[c][1]]) for c in range(len(cands))])
        out["feet"][i] = dict(J_fe=Jfe.tolist(), best=int(Jfe.argmin()))
    cache = {}
    for i in ids:
        foot = feet[i]
        g, _ = geo_of(foot, Action())
        ref_fr = geo_of(replace_none(foot), Action(resect=1))[1]
        post_list, A = [], []
        for (t, ip, fu) in cands:
            gp, fr = geo_of(foot, Action(resect=1, osteo=t, interp=ip, fusion=fu))
            d = defect_fraction(fr, ref_fr)
            A.append(act_vec(1, d["mid"], d["post"], t, ip, fu)); post_list.append(gp.astype(np.float32))
        cache[i] = (g.astype(np.float32), np.stack(A), np.stack(post_list))
    out["actions"] = {i: cache[i][1].tolist() for i in ids}
    for name in names:
        pred = get_pred(name)
        res = []
        for i in ids:
            g, A, post = cache[i]
            zp = pred.effects(g, A, post if pred.kind in ("geo", "fno") else None)
            Jp = np.array([J_of(zp[c], cands[c][0], cands[c][2], cands[c][1]) for c in range(len(cands))])
            Jfe = np.array(out["feet"][i]["J_fe"])
            ch = int(Jp.argmin())
            rng_ = Jfe.max() - Jfe.min()
            res.append(dict(i=i, choice=ch, best=int(Jfe.argmin()), regret=float((Jfe[ch] - Jfe.min()) / rng_),
                            top1=bool(ch == Jfe.argmin()), rank_rho=float(spearmanr(Jp, Jfe)[0]), J_pred=Jp.tolist()))
        out["models"][name] = res
        print("plan", name, "median regret", round(float(np.median([r["regret"] for r in res])), 3),
              "top1", sum(r["top1"] for r in res), "/", len(res),
              "rho", round(float(np.median([r["rank_rho"] for r in res])), 3), flush=True)
    return out


def replace_none(foot):
    from dataclasses import replace
    return replace(foot, ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0)


def abstention(ens_names, norm):
    """Ensemble disagreement on in-range test feet vs out-of-range feet, and its relation to effect error."""
    preds = [get_pred(n) for n in ens_names]
    val = Split("val", load_post=False)
    def collect(S):
        dis, err = [], []
        Zt = norm.eff(S.eff_raw)
        for f in range(S.nf):
            Z = np.stack([p.effects(S.geo_pre[f].astype(np.float32), S.act[f]) for p in preds])   # (M, K, 17)
            dis += list(Z.std(0).mean(1)); err += list(np.sqrt(((Z.mean(0) - Zt[f]) ** 2).mean(1)))
        return np.array(dis), np.array(err)
    dv, ev = collect(val)
    thr = float(np.percentile(dv, 90))
    out = {"threshold": thr}
    for split in ["test", "ood"]:
        S = Split(split, load_post=False)
        d, e = collect(S)
        ab = d > thr
        out[split] = dict(abstain_rate=float(ab.mean()), err_accepted=float(np.median(e[~ab])) if (~ab).any() else None,
                          err_abstained=float(np.median(e[ab])) if ab.any() else None,
                          spearman_dis_err=float(spearmanr(d, e)[0]), dis=d.tolist(), err=e.tolist())
        print("abstain", split, {k: v for k, v in out[split].items() if not isinstance(v, list)}, flush=True)
    return out


def main(names, ens):
    norm = Norm.cached()
    import pickle
    src = model_path("comech_s0") if os.path.exists(model_path("comech_s0")) else "results/models_long/comech_s0.pkl"
    eps = pickle.load(open(src, "rb"))["cfg"]["eps"] if os.path.exists(src) else 0.13256   # same epsilon as in training
    S = Split("test", load_post=any(needs_post(n) for n in names))
    res = dict(eps=eps)
    res["equivalence"] = equivalence(names, S, norm, eps)
    res["planning"] = planning(names, norm)
    if ens != ["-"]:
        res["abstention"] = abstention(ens, norm)
    save_merged("results/eval_equiv_plan.json", res)


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2].split(","))
