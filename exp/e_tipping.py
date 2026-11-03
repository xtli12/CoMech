"""E4/E5 ground truth: tolerable defect (tipping point) on held-out feet and the osteotomy exchange rate,
FE versus model predictions."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.common import *
from exp.sweeps import DEFS, test_feet, intact
from ml.data import Split, Norm


def load_sweeps():
    S = {}
    for f in glob.glob("data/sweeps/*.json"):
        r = json.load(open(f))
        S[(r["kind"], r["i"], r["tag"], round(r["d"], 2))] = r["eff"]
    return S


def main(names):
    norm = Norm.cached()
    SW = load_sweeps()
    feet = test_feet()
    ids = sorted({k[1] for k in SW if k[0] == "sweep"})
    fe = {}
    for i in ids:
        base = SW.get(("sweep", i, "none", 0.0))
        if base is None:
            continue
        rec = {}
        for fac in ["mid", "post"]:
            ys = [base[f"{fac}_mean"]] + [SW.get(("sweep", i, fac, d), {}).get(f"{fac}_mean", np.nan) for d in DEFS[1:]]
            if np.isnan(ys).any():
                continue
            rec[fac] = dict(curve=ys, tol=tolerance(DEFS, ys))
            # load spill to the other facet / anterior facet / TN
            other = "post" if fac == "mid" else "mid"
            rec[fac]["spill_" + other] = [base[f"{other}_mean"]] + [SW[("sweep", i, fac, d)][f"{other}_mean"] for d in DEFS[1:]]
        # exchange rate: posterior tolerance after +/-5 mm osteotomy
        xr = {}
        for t in (5, -5):
            ys = [SW.get(("xrate", i, f"post_t{t:+d}", d), {}).get("post_mean", np.nan) for d in DEFS]
            if not np.isnan(ys).any() and "post" in rec:
                # referenced to the defect-free joint without correction (same reference as the cohort analysis)
                xr[t] = dict(curve=ys, tol=tolerance(DEFS, ys, ref=rec["post"]["curve"][0]))
        rec["xrate"] = xr
        fe[i] = rec
    out = {"fe": fe, "models": {}}
    # predictions
    for name in names:
        pred = get_pred(name)
        P = {}
        for i in fe:
            foot = intact(feet[i])
            g, _ = geo_of(foot, Action())
            g = g.astype(np.float32)
            rec = {}
            for fac in ["mid", "post"]:
                # all points are complete-resection actions on the coalition-free twin, as in the FE sweep
                # (d = 0: resection without facet defect, i.e. the defect-free joint)
                A = np.stack([act_vec(1, d if fac == "mid" else 0, d if fac == "post" else 0) for d in DEFS])
                post = None
                if pred.kind in ("geo", "fno"):
                    post = np.stack([geo_of(foot, Action(resect=1, d_mid=d if fac == "mid" else 0,
                                                         d_post=d if fac == "post" else 0))[0].astype(np.float32) for d in DEFS])
                e = norm.eff_inv(pred.effects(g, A, post))
                ys = e[:, EI[f"{fac}_mean"]].tolist()
                rec[fac] = dict(curve=ys, tol=tolerance(DEFS, ys))
            xr = {}
            for t in (5, -5):
                A = np.stack([act_vec(1, 0, d, osteo=t) for d in DEFS])
                post = None
                if pred.kind in ("geo", "fno"):
                    post = np.stack([geo_of(foot, Action(resect=1, d_post=d, osteo=t))[0].astype(np.float32) for d in DEFS])
                e = norm.eff_inv(pred.effects(g, A, post))
                ys = e[:, EI["post_mean"]].tolist()
                xr[t] = dict(curve=ys, tol=tolerance(DEFS, ys, ref=rec["post"]["curve"][0]))
            rec["xrate"] = xr
            P[i] = rec
        # scores
        err = {fac: [abs(P[i][fac]["tol"] - fe[i][fac]["tol"]) * 100 for i in fe if fac in fe[i]] for fac in ["mid", "post"]}
        cur = {fac: [float(np.mean(np.abs(np.array(P[i][fac]["curve"]) / P[i][fac]["curve"][0] -
                                           np.array(fe[i][fac]["curve"]) / fe[i][fac]["curve"][0])) * 100)
                     for i in fe if fac in fe[i]] for fac in ["mid", "post"]}
        out["models"][name] = dict(pred=P, tol_err_pp=err, curve_err_pct=cur)
        print(name, {f: (round(float(np.median(v)), 2), round(float(np.mean(v)), 2)) for f, v in err.items()}, flush=True)
    save_merged("results/eval_tipping.json", out)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
