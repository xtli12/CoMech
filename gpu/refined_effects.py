"""Effect-level discretisation check: the 17 effect descriptors of the standard-grid FE versus the twofold-refined FE
on the held-out test configurations (data/refined_test_f0.50, written by `gpu/refined.py solve`).

    python3 gpu/refined_effects.py          ->  results/eval_refined_effects.json

Reports, per descriptor, the relative difference and the correlation across configurations; the standardised effect
RMSE (same normalisation as the models' eff_rmse); the within-foot rank agreement of the facet stresses across the
eight actions of each foot; and the equivalence decisions (pairs of actions of one foot within eps of each other) of
the standard grid scored against those of the refined grid, i.e. the precision and recall that the engine itself
achieves on the equivalence task when the refined solution is taken as the reference.
"""
import os, sys, json, glob, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import spearmanr, pearsonr
from fe.simulate import EFFECT_NAMES
from ml.data import Norm

OUT = "data/refined_test_f0.50"


def load():
    feet = []
    for fn in sorted(glob.glob("data/bank/test/foot_*.json")):
        m = json.load(open(fn))
        recs = []
        for k, a in enumerate(m["actions"]):
            p = f"{OUT}/{m['fid']:04d}_{k}.npz"
            if not os.path.exists(p):
                recs = None
                break
            z = np.load(p)
            recs.append(dict(k=k, meta=a.get("meta"), act=a["act"], std=a["eff"], ref=json.loads(z["eff"].item()),
                             ndof_ref=int(z["ndof"])))
        if recs:
            feet.append(dict(fid=m["fid"], ctype=m["foot"].get("ctype"), actions=recs))
    return feet


def main():
    norm = Norm.cached()
    feet = load()
    rows = [r for f in feet for r in f["actions"]]
    print("feet", len(feet), "configurations", len(rows), flush=True)
    if not rows:
        return
    Es = np.array([[r["std"][n] for n in EFFECT_NAMES] for r in rows], float)
    Er = np.array([[r["ref"][n] for n in EFFECT_NAMES] for r in rows], float)
    per = {}
    for j, n in enumerate(EFFECT_NAMES):
        s, r = Es[:, j], Er[:, j]
        big = np.abs(s) > 1e-6 * max(float(np.abs(s).max()), 1e-12)
        rel = np.abs(r - s)[big] / np.abs(s)[big]
        sgn = (r - s)[big] / np.abs(s)[big]
        per[n] = dict(median_rel=float(np.median(rel)), q3_rel=float(np.percentile(rel, 75)),
                      median_signed_rel=float(np.median(sgn)),
                      nrmse=float(np.sqrt(np.mean((r - s) ** 2)) / (s.std() + 1e-12)),
                      pearson=float(pearsonr(s, r)[0]) if s.std() > 0 and r.std() > 0 else None,
                      spearman=float(spearmanr(s, r)[0]) if s.std() > 0 and r.std() > 0 else None)
    Zs, Zr = norm.eff(Es), norm.eff(Er)
    eff_rmse = np.sqrt(((Zs - Zr) ** 2).mean(1))
    # within-foot ordering of the facet stresses across the actions of each foot
    within = {}
    for n in ["mid_mean", "post_mean", "ant_mean", "tn_p99"]:
        j = EFFECT_NAMES.index(n)
        rh, i0 = [], 0
        for f in feet:
            K = len(f["actions"])
            s, r = Es[i0:i0 + K, j], Er[i0:i0 + K, j]
            i0 += K
            if np.ptp(s) > 0 and np.ptp(r) > 0:
                rh.append(float(spearmanr(s, r)[0]))
        within[n] = dict(median_rho=float(np.median(rh)), q1=float(np.percentile(rh, 25)), min=float(np.min(rh)), n=len(rh))
    # equivalence decisions of the standard grid scored against the refined grid (same eps as the models)
    src = "results/models_long/comech_s0.pkl"
    eps = pickle.load(open(src, "rb"))["cfg"]["eps"] if os.path.exists(src) else 0.13256
    ds, dr, i0 = [], [], 0
    for f in feet:
        K = len(f["actions"])
        zs, zr = Zs[i0:i0 + K], Zr[i0:i0 + K]
        i0 += K
        for a in range(K):
            for b in range(a + 1, K):
                ds.append(np.sqrt(((zs[a] - zs[b]) ** 2).mean()))
                dr.append(np.sqrt(((zr[a] - zr[b]) ** 2).mean()))
    ds, dr = np.array(ds), np.array(dr)
    P, T = ds <= eps, dr <= eps
    equiv = dict(eps=float(eps), precision=float((P & T).sum() / max(P.sum(), 1)), recall=float((P & T).sum() / max(T.sum(), 1)),
                 n_pred=int(P.sum()), n_true=int(T.sum()), n_pairs=int(len(ds)), spearman=float(spearmanr(ds, dr)[0]))
    summary = dict(n_feet=len(feet), n_config=len(rows),
                   eff_rmse=dict(median=float(np.median(eff_rmse)), q1=float(np.percentile(eff_rmse, 25)),
                                 q3=float(np.percentile(eff_rmse, 75))),
                   per_effect=per, within_foot=within, equivalence_standard_vs_refined=equiv,
                   ndof_refined_median=float(np.median([r["ndof_ref"] for r in rows])))
    os.makedirs("results", exist_ok=True)
    json.dump(dict(summary=summary, eff_rmse=eff_rmse.tolist(), feet=feet), open("results/eval_refined_effects.json", "w"))
    print("standardised effect RMSE, standard vs refined grid: median %.3f (IQR %.3f-%.3f)"
          % (summary["eff_rmse"]["median"], summary["eff_rmse"]["q1"], summary["eff_rmse"]["q3"]))
    print("%-12s %8s %8s %8s %7s %7s" % ("effect", "med|rel|", "med rel", "nRMSE", "r", "rho"))
    for n, v in per.items():
        print("%-12s %8.3f %+8.3f %8.3f %7s %7s" % (n, v["median_rel"], v["median_signed_rel"], v["nrmse"],
                                                  "-" if v["pearson"] is None else "%.3f" % v["pearson"],
                                                  "-" if v["spearman"] is None else "%.3f" % v["spearman"]))
    print("within-foot rank agreement:", {n: (round(v["median_rho"], 3), round(v["q1"], 3)) for n, v in within.items()})
    print("equivalence, standard grid vs refined grid:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in equiv.items()})
    print("REFINED_EFFECTS_DONE", flush=True)


if __name__ == "__main__":
    main()
