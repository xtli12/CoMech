"""Field maps: (i) FE vs CoMech on a held-out foot (Fig. 5a); (ii) deformation-guided surgical map for one
virtual child whose defect exceeds its tolerance, with the planner's adjunct and FE verification (Fig. 7a)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.common import *
from ml.data import Split, Norm
from dataclasses import replace
import fe.simulate as FS
from exp.e_equiv_plan import J_of


def main(name, ens):
    norm = Norm.cached()
    pred = get_pred(name)
    out = {}
    # ---------- (i) held-out foot with a posterior defect ----------
    S = Split("test", load_post=False)
    best = None
    for f in range(S.nf):
        for k in range(S.K):
            a = S.act[f, k]
            if a[0] == 1 and a[5] == 0 and a[2] >= 0.35 and abs(a[3]) < 0.05 and a[6] == 0:
                best = (f, k); break
        if best:
            break
    f, k = best
    lab = S.lab[f, k]; cells = np.nonzero(lab != VOID)[0]
    y, e = pred.fields_and_effects(S.geo_pre[f].astype(np.float32), S.act[f, k][None], cells)
    vm_p, u_p = norm.fields_inv(y[0])
    np.savez_compressed("results/map_heldout.npz", cells=cells, lab=lab, vm_fe=S.vm[f, k, cells].astype(np.float32),
                        u_fe=S.uc[f, k, cells].astype(np.float32), vm_p=vm_p, u_p=u_p, act=S.act[f, k],
                        foot=json.dumps(S.meta[f]["foot"]))
    out["heldout"] = dict(f=int(f), k=int(k), act=S.act[f, k].tolist(), ctype=S.ctype[f])
    # ---------- (ii) surgical map for one virtual child ----------
    C = json.load(open("results/eval_cohort.json"))
    rows = C["rows"]
    # child (v3e, 2026-11-01): a middle-facet (type I) coalition whose defect exceeds its middle-facet tolerance, the
    # median such shortfall.  Chosen after the real-bone results of the original rule (posterior-facet coalition,
    # types II/IV, whose best plan was resection alone) had been seen; CHILD_FACET=post restores the original rule.
    FAC = os.environ.get("CHILD_FACET", "mid")
    if FAC == "mid":
        cand = [r for r in rows if r["d_mid"] > r["tol_mid"] and r["foot"]["ctype"] == "I"]
    else:
        cand = [r for r in rows if r["d_post"] > r["tol_post"] and r["foot"]["ctype"] in ("II", "IV")]
    cand = cand or [r for r in rows if not r["within"]]
    child = sorted(cand, key=lambda r: r[f"reserve_{FAC}"])[len(cand) // 2]
    foot = Foot(**child["foot"])
    g_pre, _ = geo_of(foot, Action()); g_pre = g_pre.astype(np.float32)
    twin = replace(foot, ctype="none", rtype="none", cov_mid=0.0, cov_post=0.0, fib_frac=0.0)
    ref_fr = geo_of(twin, Action(resect=1))[1]
    preds = [get_pred(n) for n in ens]
    # exhaustive latent rehearsal over adjuncts to complete resection (osteotomy -10..10 mm in 1-mm steps,
    # with or without interposition)
    cands = [(float(t), ip, 0) for t in range(-10, 11) for ip in (0, 1)]
    A = []
    for (t, ip, fu) in cands:
        fr = geo_of(foot, Action(resect=1, osteo=t, interp=ip, fusion=fu))[1]
        d = defect_fraction(fr, ref_fr)
        A.append(act_vec(1, d["mid"], d["post"], t, ip, fu))
    A = np.stack(A)
    Z = np.stack([p.effects(g_pre, A) for p in preds])               # (M, n, 17)
    Jm = np.array([[J_of(Z[m, c], cands[c][0], cands[c][2], cands[c][1]) for c in range(len(cands))] for m in range(len(preds))])
    J = Jm.mean(0); ch = int(J.argmin()); i0 = cands.index((0.0, 0, 0))
    out["child"] = dict(i=child["i"], facet=FAC, n_candidates=len(cand), foot=child["foot"], d_post=child["d_post"], d_mid=child["d_mid"],
                        tol_post=child["tol_post"], tol_mid=child["tol_mid"], ratio=child["ratio"],
                        choice=cands[ch], J=J.tolist(), J_sd=Jm.std(0).tolist(), cands=cands)
    # fields for resection only and for the chosen plan (predicted) + FE verification of both
    for tag, c in [("resect", i0), ("plan", ch)]:
        t, ip, fu = cands[c]
        act = Action(resect=1, osteo=t, interp=ip, fusion=fu)
        r = FS.simulate(foot, act, tol=1e-6)
        lab = r["lab"]; cells = np.nonzero(lab != VOID)[0]
        y, e = pred.fields_and_effects(g_pre, A[c][None], cells)
        vm_p, u_p = norm.fields_inv(y[0])
        ucell = r["Urel"][GRID.cell_nodes].mean(1)
        np.savez_compressed(f"results/map_child_{tag}.npz", cells=cells, lab=lab, vm_fe=r["vm"][cells], u_fe=ucell[cells],
                            vm_p=vm_p, u_p=u_p, frac=np.round(r["frac"] * 255).astype(np.uint8))
        out["child"][f"eff_fe_{tag}"] = r["eff"]
        out["child"][f"eff_pred_{tag}"] = {n: float(v) for n, v in zip(EI, norm.eff_inv(Z.mean(0)[c][None])[0])}
    # tolerance contour: facet defect footprint at the child's posterior tolerance
    import fe.geometry as FG
    xs, ys = FG.map_axes(foot)
    tv_int = FootGeom(twin, Action()).top_view(xs, ys)
    tv_def = FootGeom(twin, Action(resect=1, d_mid=child["tol_mid"]) if FAC == "mid" else
                      Action(resect=1, d_post=child["tol_post"])).top_view(xs, ys)
    tv_pre = FootGeom(foot, Action()).top_view(xs, ys)
    post, mid, ant = tv_int["post"], tv_int["mid"], tv_int["ant"]
    dp = (mid & ~tv_def["mid"]) if FAC == "mid" else (post & ~tv_def["post"])   # defect at the child's tolerance
    br, fib = tv_pre["coal"], np.zeros_like(tv_pre["coal"])
    np.savez_compressed("results/map_child_masks.npz", xs=xs, ys=ys, post=post, mid=mid, ant=ant, tol_def=dp,
                        coal=(br | fib))
    json.dump(out, open("results/eval_maps.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out["child"].items() if k not in ("J", "J_sd", "cands")}, indent=1)[:2500])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
