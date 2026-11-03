"""E3b: label-free taxonomy test on 240 type-balanced synthetic coalitions.

Each coalition is compared with its own coalition-free twin (same foot), as the atlas fingerprints compare each
coalition with the intact joint.  Three representations are clustered without labels:
  geometry  - the bridge itself (coalition channel, patch-averaged);
  latent    - the change of the CoMech latent state caused by the coalition, s0(foot) - s0(twin);
  mechanics - the change of the predicted pre-operative effect vector, delta(foot) - delta(twin).
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.common import *
from ml.data import Norm
from ml.train import patchify
from dataclasses import replace
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI


def cluster_scores(X, labels, k, seeds=20, pcs=10):
    X = X[:, X.std(0) > 1e-8]
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=min(pcs, Xs.shape[1], Xs.shape[0] - 1), random_state=0).fit_transform(Xs)
    aris, nmis = [], []
    for s in range(seeds):
        c = KMeans(n_clusters=k, n_init=5, random_state=s).fit_predict(Z)
        aris.append(ARI(labels, c)); nmis.append(NMI(labels, c))
    return dict(ari=float(np.mean(aris)), ari_sd=float(np.std(aris)), nmi=float(np.mean(nmis)), emb=Z[:, :2].tolist())


def twins():
    fn = "data/cohort/taxonomy_twins.npz"
    if os.path.exists(fn):
        return np.load(fn)["twin"].astype(np.float32)
    feet = json.load(open("data/cohort/taxonomy.json"))
    out = []
    for f in feet:
        tw = replace(Foot(**f), ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0)
        out.append(geo_of(tw, Action())[0].astype(np.float16))
    out = np.stack(out); np.savez_compressed(fn, twin=out)
    return out.astype(np.float32)


def main(names):
    geo = np.load("data/cohort/taxonomy_geo.npz")["pre"].astype(np.float32)          # (240, C, X, Y, Z)
    tw = twins()
    feet = json.load(open("data/cohort/taxonomy.json"))
    atype = np.array([f["ctype"] for f in feet]); rtype = np.array([f["rtype"] for f in feet])
    t1 = atype == "I"
    out = {"labels": dict(atype=atype.tolist(), rtype=rtype.tolist())}
    bridge = patchify(geo[:, 2:3]).reshape(len(geo), -1)                             # coalition channel only
    out["geometry"] = dict(anatomy=cluster_scores(bridge, atype, 4), rozansky=cluster_scores(bridge[t1], rtype[t1], 4))
    print("geometry", out["geometry"]["anatomy"]["ari"], out["geometry"]["rozansky"]["ari"], flush=True)
    # FE mechanical fingerprints (engine ground truth), if the taxonomy sweeps are available
    fe_fp = None
    if all(os.path.exists(f"data/sweeps_tax/{i:03d}_{k}.json") for i in range(len(geo)) for k in ("coal", "twin")):
        norm = Norm.cached()
        E = lambda k: norm.eff(np.array([[json.load(open(f"data/sweeps_tax/{i:03d}_{k}.json"))["eff"][n] for n in EI]
                                         for i in range(len(geo))]))
        fe_fp = E("coal") - E("twin")
        out["fe_mechanics"] = dict(anatomy=cluster_scores(fe_fp, atype, 4), rozansky=cluster_scores(fe_fp[t1], rtype[t1], 4),
                                   fingerprints=fe_fp.tolist())
        print("fe_mechanics", out["fe_mechanics"]["anatomy"]["ari"], out["fe_mechanics"]["rozansky"]["ari"], flush=True)
    for name in names:
        pred = get_pred(name); m = pred.m
        dlat, deff = [], []
        for i in range(0, len(geo), 6):
            s_c = m.encode(patchify(geo[i:i + 6])); s_t = m.encode(patchify(tw[i:i + 6]))
            dlat.append((s_c.data - s_t.data).reshape(s_c.shape[0], -1))
            u = m.act_enc(np.zeros((s_c.shape[0], 7), np.float32))
            deff.append(m.effect(m.step(s_c, u)).data - m.effect(m.step(s_t, u)).data)
        dlat = np.concatenate(dlat); deff = np.concatenate(deff)
        out[name] = dict(latent=dict(anatomy=cluster_scores(dlat, atype, 4), rozansky=cluster_scores(dlat[t1], rtype[t1], 4)),
                         mechanics=dict(anatomy=cluster_scores(deff, atype, 4), rozansky=cluster_scores(deff[t1], rtype[t1], 4)),
                         fingerprints=deff.tolist())
        if fe_fp is not None:
            out[name]["fp_vs_fe_r"] = float(np.corrcoef(deff.ravel(), fe_fp.ravel())[0, 1])
        print(name, {k: (round(v["anatomy"]["ari"], 3), round(v["rozansky"]["ari"], 3)) for k, v in out[name].items()
                     if isinstance(v, dict) and "anatomy" in v}, flush=True)
    save_merged("results/eval_taxonomy.json", out)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
