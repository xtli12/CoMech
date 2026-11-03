"""FE-verified envelopes for the virtual cohort: tolerance of each foot from FE defect sweeps on its coalition-free
twin, envelope status of the defect left by complete resection, the 50% rule, and agreement with CoMech."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.common import tolerance, TAU, boot_ci
from exp.sweeps_cohort import DEFS, FACETS

D = [0.0] + DEFS


def calibrated_tau(curve, target):
    """Threshold at which the reference-foot FE curve is crossed at the atlas tolerance (target, fraction)."""
    r = np.asarray(curve) / curve[0] - 1
    return float(np.interp(target, D, r))


def main():
    SW = {}
    for f in glob.glob("data/sweeps_cohort/*.json"):
        r = json.load(open(f)); SW[(r["i"], r["fac"], round(r["d"], 2))] = r["eff"]
    C = json.load(open("data/cohort/cohort.json"))
    P = {r["i"]: r for r in json.load(open("results/eval_cohort.json"))["rows"]}
    Z = json.load(open("results/eval_zeroshot.json"))["comech_s0"]["scores"]
    # atlas-calibrated thresholds: the engine's reference foot crosses them at the atlas tolerances (38% and 56%)
    tau_cal = {"mid": calibrated_tau(Z["curve_mid_fe"], 0.38), "post": calibrated_tau(Z["curve_post_fe"], 0.56)}
    rows = []
    for c in C:
        i, t = c["i"], c["foot"]["ctype"]
        rec = dict(i=i, ctype=t, ratio=c["ratio"], d_mid=c["d_mid"], d_post=c["d_post"], rule_ok=bool(c["ratio"] < 0.5))
        ok = {"pri": True, "cal": True}
        complete = True
        for fac in FACETS[t]:
            ys = [SW.get((i, "none", 0.0), {}).get(f"{fac}_mean")] + [SW.get((i, fac, d), {}).get(f"{fac}_mean") for d in DEFS]
            if any(y is None for y in ys):
                complete = False; break
            rec[f"fe_curve_{fac}"] = ys
            rec[f"fe_tol_{fac}"] = tolerance(D, ys)
            rec[f"fe_tol_{fac}_cal"] = tolerance(D, ys, tau=tau_cal[fac])
            rec[f"pred_tol_{fac}"] = P[i][f"tol_{fac}"]
            ok["pri"] &= c[f"d_{fac}"] <= rec[f"fe_tol_{fac}"]
            ok["cal"] &= c[f"d_{fac}"] <= rec[f"fe_tol_{fac}_cal"]
        if not complete:
            continue
        rec["fe_within"] = bool(ok["pri"]); rec["fe_within_cal"] = bool(ok["cal"])
        rec["pred_within"] = bool(P[i]["within"])
        rows.append(rec)
    n = len(rows)
    S = dict(n=n, tau_cal=tau_cal)
    for key in ["fe_within", "fe_within_cal"]:
        w = np.array([r[key] for r in rows]); rule = np.array([r["rule_ok"] for r in rows])
        S[key] = dict(within=int(w.sum()), rule_ok=int(rule.sum()), mis_total=int((w != rule).sum()),
                      mis_permissive=int((rule & ~w).sum()), mis_strict=int((~rule & w).sum()),
                      by_type={t: dict(n=int(sum(r["ctype"] == t for r in rows)),
                                       within=int(sum(r[key] for r in rows if r["ctype"] == t)),
                                       permissive=int(sum(r["rule_ok"] and not r[key] for r in rows if r["ctype"] == t)),
                                       strict=int(sum((not r["rule_ok"]) and r[key] for r in rows if r["ctype"] == t)))
                               for t in ["I", "II", "III", "IV"]})
    pw = np.array([r["pred_within"] for r in rows]); fw = np.array([r["fe_within"] for r in rows])
    S["pred_vs_fe"] = dict(agree=int((pw == fw).sum()), sens=float((pw & fw).sum() / max(fw.sum(), 1)),
                           beyond_detected=int((~pw & ~fw).sum()), beyond_fe=int((~fw).sum()),
                           false_within=int((pw & ~fw).sum()), false_beyond=int((~pw & fw).sum()))
    for fac in ["mid", "post"]:
        a = np.array([r[f"fe_tol_{fac}"] for r in rows if f"fe_tol_{fac}" in r]) * 100
        b = np.array([r[f"pred_tol_{fac}"] for r in rows if f"fe_tol_{fac}" in r]) * 100
        ac = np.array([r[f"fe_tol_{fac}_cal"] for r in rows if f"fe_tol_{fac}" in r]) * 100
        if len(a):
            from scipy.stats import spearmanr
            S[f"tol_{fac}"] = dict(n=len(a), fe_median=float(np.median(a)), fe_q=[float(np.percentile(a, 25)), float(np.percentile(a, 75))],
                                   fe_range=[float(a.min()), float(a.max())], fe_censored=int((a >= 69.99).sum()),
                                   fe_cal_median=float(np.median(ac)), fe_cal_range=[float(ac.min()), float(ac.max())],
                                   pred_mae=float(np.mean(np.abs(a - b))), pred_bias=float(np.mean(b - a)),
                                   pred_rho=float(spearmanr(a, b)[0]),
                                   const_mae=float(np.mean(np.abs(a - np.median(a)))))
    json.dump(dict(rows=rows, summary=S), open("results/eval_cohort_fe.json", "w"))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
