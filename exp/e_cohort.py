"""E4/E5 on the 135-foot virtual cohort: patient-specific tolerance, mechanical reserve, the 50% rule,
determinants of tolerance and the defect-correction exchange rate (CoMech ensemble predictions)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.common import *
from ml.data import Split, Norm
from scipy.stats import spearmanr

DD = np.round(np.arange(0, 0.701, 0.05), 3)
TT = np.array([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10.0])


def ens_effects(preds, g, A):
    return np.mean([p.effects(g, A) for p in preds], 0)


def main(names):
    norm = Norm.cached()
    preds = [get_pred(n) for n in names]
    C = json.load(open("data/cohort/cohort.json"))[: int(os.environ.get("COHORT_N", "100000"))]
    G = np.load("data/cohort/cohort_geo.npz")
    intact = G["intact"][: len(C)].astype(np.float32)
    rows = []
    for c, gi in zip(C, intact):
        rec = dict(i=c["i"], foot=c["foot"], ratio=c["ratio"], d_mid=c["d_mid"], d_post=c["d_post"])
        for fac in ["mid", "post"]:
            # tolerance curve and its dependence on osteotomy translation
            land = np.zeros((len(TT), len(DD)))
            for ti, t in enumerate(TT):
                A = np.stack([act_vec(1, d if fac == "mid" else 0, d if fac == "post" else 0,
                                      osteo=t) for d in DD])
                e = norm.eff_inv(ens_effects(preds, gi, A))
                land[ti] = e[:, EI[f"{fac}_mean"]]
            base = land[TT == 0][0]
            rec[f"curve_{fac}"] = base.tolist()
            rec[f"tol_{fac}"] = tolerance(DD, base)
            # landscape relative to the defect-free, uncorrected joint
            rec[f"land_{fac}"] = (land / base[0] - 1).tolist()
            # tolerance after an osteotomy of t mm, referenced to the defect-free, uncorrected joint
            rec[f"tol_{fac}_by_t"] = [tolerance(DD, land[ti], ref=base[0]) for ti in range(len(TT))]
        # mechanical reserve and envelope status
        for fac in ["mid", "post"]:
            tol = rec[f"tol_{fac}"]
            rec[f"reserve_{fac}"] = (tol - rec[f"d_{fac}"]) / max(tol, 1e-6)
        rec["within"] = bool(rec["d_mid"] <= rec["tol_mid"] and rec["d_post"] <= rec["tol_post"])
        rec["rule_ok"] = bool(rec["ratio"] < 0.5)
        rows.append(rec)
        if rec["i"] % 20 == 0:
            print(rec["i"], round(rec["tol_mid"], 3), round(rec["tol_post"], 3), rec["within"], rec["rule_ok"], flush=True)
    # summaries
    S = {}
    for fac in ["mid", "post"]:
        t = np.array([r[f"tol_{fac}"] for r in rows]) * 100
        S[f"tol_{fac}"] = dict(min=float(t.min()), max=float(t.max()), median=float(np.median(t)),
                               q1=float(np.percentile(t, 25)), q3=float(np.percentile(t, 75)),
                               censored=int((t >= 69.99).sum()))
    within = np.array([r["within"] for r in rows]); rule = np.array([r["rule_ok"] for r in rows])
    S["n"] = len(rows)
    S["within_envelope"] = int(within.sum())
    S["rule_ok"] = int(rule.sum())
    S["mis_total"] = int((within != rule).sum())
    S["mis_permissive"] = int((rule & ~within).sum())     # judged suitable although defect exceeds tolerance
    S["mis_strict"] = int((~rule & within).sum())         # judged unsuitable although within tolerance
    # determinants of the posterior tolerance
    names_p = Foot.vec_names()
    X = np.array([[r["foot"][k] for k in names_p] for r in rows], float)
    for fac in ["mid", "post"]:
        y = np.array([r[f"tol_{fac}"] for r in rows])
        rho = {k: float(spearmanr(X[:, j], y)[0]) for j, k in enumerate(names_p) if X[:, j].std() > 0}
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Xs = np.c_[np.ones(len(Xs)), Xs[:, X.std(0) > 0]]
        beta, *_ = np.linalg.lstsq(Xs, (y - y.mean()) / y.std(), rcond=None)
        r2 = 1 - ((Xs @ beta - (y - y.mean()) / y.std()) ** 2).sum() / (((y - y.mean()) / y.std()) ** 2).sum()
        keep = [k for j, k in enumerate(names_p) if X[:, j].std() > 0]
        S[f"det_{fac}"] = dict(spearman=rho, beta=dict(zip(keep, beta[1:].tolist())), r2=float(r2))
    # exchange rate (posterior facet): gain in tolerance per mm / per degree of correcting translation
    xr = []
    for r in rows:
        tb = np.array(r["tol_post_by_t"])
        al = r["foot"]["align"]; s = r["foot"]["scale"]
        sgn = 1.0 if al >= 0 else -1.0                      # valgus corrected by medialising (+), varus by lateralising (-)
        j0 = list(TT).index(0.0); j5 = list(TT).index(5.0 * sgn)
        per_mm = (tb[j5] - tb[j0]) * 100 / 5.0
        # alignment and osteotomy both act in the scaled (canonical) frame: 1 degree of hindfoot alignment moves the
        # heel by Z_REF*tan(1 deg) = 0.506 canonical mm, i.e. the same shift as a 0.506-mm osteotomy translation
        import fe.geometry as FG
        zr = FG.z_ref_align() if hasattr(FG, "z_ref_align") else 29.0     # real-bone engine: shear about the joint level
        per_deg = per_mm * zr * np.tan(np.deg2rad(1.0))
        xr.append(dict(i=r["i"], align=al, per_mm=float(per_mm), per_deg=float(per_deg)))
    pdg = np.array([x["per_deg"] for x in xr])
    S["xrate_post_per_deg"] = dict(median=float(np.median(pdg)), q1=float(np.percentile(pdg, 25)),
                                   q3=float(np.percentile(pdg, 75)), positive=int((pdg > 0).sum()))
    json.dump(dict(rows=rows, summary=S, xrate=xr, DD=DD.tolist(), TT=TT.tolist()), open("results/eval_cohort.json", "w"))
    print(json.dumps(S, indent=1)[:3000])


if __name__ == "__main__":
    main(sys.argv[1].split(","))
