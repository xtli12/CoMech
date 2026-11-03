"""Latent states of the adjuncts rehearsed for the virtual child of Fig. 7a, for the latent-rehearsal panels of
Figs. 1c and 4c.  Re-runs the planner of exp/e_maps.py exactly (same 42 candidates: osteotomy -10..10 mm in 1-mm
steps, with or without interposition, all after complete resection; same action vectors; the same three-model
ensemble and objective J) and checks that J and the chosen plan equal results/eval_maps.json.  It then stores, for
seed 0, the pre-operative latent state s and the 42 post-action latent states s' = D(s, u) (token means, as
Model.rep), projected on the first three principal components of these 43 states, together with J (ensemble mean),
the ensemble disagreement used for abstention (Methods) and the abstention threshold of
results/eval_equiv_plan.json.  Output: results/child_latent.json (about 30 s on one core).
Run from scripts/comech with COMECH_MODELS=results/models_long."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("COMECH_MODELS", "results/models_long")
from exp.common import *
from exp.e_equiv_plan import J_of
from dataclasses import replace

ENS = ["comech_s0", "comech_s1", "comech_s2"]


def main():
    ch = json.load(open("results/eval_maps.json"))["child"]
    foot = Foot(**ch["foot"])
    g_pre = geo_of(foot, Action())[0].astype(np.float32)
    twin = replace(foot, ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0)
    ref_fr = geo_of(twin, Action(resect=1))[1]
    cands = [(float(t), ip, 0) for t in range(-10, 11) for ip in (0, 1)]
    A = []
    for (t, ip, fu) in cands:
        d = defect_fraction(geo_of(foot, Action(resect=1, osteo=t, interp=ip, fusion=fu))[1], ref_fr)
        A.append(act_vec(1, d["mid"], d["post"], t, ip, fu))
    A = np.stack(A)
    preds = [get_pred(n) for n in ENS]
    Z = np.stack([p.effects(g_pre, A) for p in preds])
    J = np.array([[J_of(Z[m, c], cands[c][0], cands[c][2], cands[c][1]) for c in range(len(cands))]
                  for m in range(len(preds))]).mean(0)
    assert np.allclose(J, ch["J"], atol=1e-6), "J differs from results/eval_maps.json"
    choice = int(J.argmin())
    assert list(cands[choice]) == list(ch["choice"]), "chosen plan differs from results/eval_maps.json"
    dis = Z.std(0).mean(1)
    thr = json.load(open("results/eval_equiv_plan.json"))["abstention"]["threshold"]
    from ml.data import patchify
    m0 = preds[0].m
    s0 = m0.encode(patchify(g_pre[None])).data.mean(1)             # seed 0, pre-operative state (1, D)
    lat = preds[0].latent(g_pre, A).data.mean(1)                   # seed 0, post-action states (42, D)
    X = np.r_[s0, lat]
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = (X - mu) @ Vt[:3].T
    # orientation: PC1 grows with the osteotomy translation, PC2 is larger on the interposition branch, PC3 is
    # positive at the pre-operative state
    t = np.array([c[0] for c in cands]); ip = np.array([c[1] for c in cands])
    if np.corrcoef(P[1:, 0], t)[0, 1] < 0:
        P[:, 0] *= -1
    if P[1:][ip == 1, 1].mean() < P[1:][ip == 0, 1].mean():
        P[:, 1] *= -1
    if P[0, 2] < 0:
        P[:, 2] *= -1
    ev = (S ** 2 / (S ** 2).sum())[:3]
    out = dict(cands=cands, s=P[0].round(5).tolist(), pc=P[1:].round(5).tolist(), J=J.round(6).tolist(), choice=choice,
               disagreement=dis.round(6).tolist(), threshold=thr, n_abstain=int((dis > thr).sum()),
               explained=ev.round(4).tolist(), model="comech_s0", ensemble=ENS)
    json.dump(out, open("results/child_latent.json", "w"))
    print("choice", cands[choice], "explained", ev.round(3), "abstain", out["n_abstain"],
          "max disagreement", round(float(dis.max()), 4), "threshold", round(thr, 4))


if __name__ == "__main__":
    main()
