"""FE verification of the CT-informed twins: for every foot read as a middle-facet (type I) coalition, a point
twin (reading coverage, engine-median shape, alignment +/-8 deg for a recorded valgus/varus, otherwise 0) is
solved by FE on its coalition-free twin over defects of 0-70%; its FE tolerance is compared with the defect left
by complete resection and with the CoMech ensemble's prediction for the same twin."""
import os, sys, json, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace, asdict
from fe.geometry import Foot, Action
import fe.simulate as S
from exp.e_ct_twins import read_csv

ROOT = "data/ct/fe"
DEFS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def point_twin(r):
    d = r["deformity"]
    al = 8.0 if d.startswith("valgus") else (-8.0 if d.startswith("varus") else 0.0)
    roz = r["roz"] if r["roz"] in ("R1", "R2", "R3", "R4", "unclass") else "R1"
    return Foot(scale=0.95, align=al, ctype="I", rtype=roz, cov_mid=float(r["cov_mid"]),
                fib_frac=0.5 if roz == "R3" else 0.0)


def job(args):
    r, d = args
    fn = f"{ROOT}/{r['foot']}_{d:.2f}.json"
    if os.path.exists(fn):
        return
    f = replace(point_twin(r), ctype="none", rtype="none", cov_mid=0.0, fib_frac=0.0)
    res = S.simulate(f, Action(resect=1, d_mid=d), tol=1e-5)
    json.dump(dict(foot=r["foot"], d=d, eff=res["eff"]), open(fn, "w"))
    print(r["foot"], d, flush=True)


def main(csv_path):
    os.makedirs(ROOT, exist_ok=True)
    R = [r for r in read_csv(csv_path) if r["type"] == "I"]
    J = [(r, d) for r in R for d in DEFS]
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NPROC", "1"))) as p:
        list(p.imap_unordered(job, J, chunksize=1))
    # analysis
    from exp.common import tolerance, get_pred, channels, act_vec, EI
    from exp.cohort import child_defect, coalition_ratio
    from ml.data import Norm
    norm = Norm.cached(); preds = [get_pred(n) for n in ["comech_s0", "comech_s1", "comech_s2"]]
    tau_cal = json.load(open("results/eval_cohort_fe.json"))["summary"]["tau_cal"]
    DD = np.round(np.arange(0, 0.701, 0.05), 3)
    rows = []
    for r in R:
        y = [json.load(open(f"{ROOT}/{r['foot']}_{d:.2f}.json"))["eff"]["mid_mean"] for d in DEFS]
        f = point_twin(r)
        dd, fr_int, _ = child_defect(f); cr = coalition_ratio(f)
        gi = channels(np.round(fr_int * 255).astype(np.uint8)).astype(np.float32)
        A = np.stack([act_vec(1, x, 0) for x in DD])
        yp = norm.eff_inv(np.mean([p.effects(gi, A) for p in preds], 0))[:, EI["mid_mean"]]
        rec = dict(foot=r["foot"], treatment=r["treatment"], deformity=r["deformity"], conf=r["conf"], cov_mid=r["cov_mid"],
                   rtype=f.rtype, align=f.align, ratio=cr["ratio"], d_mid=dd["d_mid"], fe_curve=y,
                   fe_tol=tolerance(DEFS, y), fe_tol_cal=tolerance(DEFS, y, tau=tau_cal["mid"]),
                   pred_tol=tolerance(DD, yp), pred_tol_cal=tolerance(DD, yp, tau=tau_cal["mid"]))
        rec.update(fe_within=bool(rec["d_mid"] <= rec["fe_tol"]), fe_within_cal=bool(rec["d_mid"] <= rec["fe_tol_cal"]),
                   pred_within=bool(rec["d_mid"] <= rec["pred_tol"]), rule_ok=bool(cr["ratio"] < 0.5),
                   fe_reserve=(rec["fe_tol"] - rec["d_mid"]) / max(rec["fe_tol"], 1e-6))
        rows.append(rec)
        print(r["foot"], round(rec["d_mid"], 2), round(rec["fe_tol"], 2), round(rec["pred_tol"], 2), rec["fe_within"], rec["rule_ok"], flush=True)
    json.dump(dict(rows=rows, defs=DEFS, tau_cal=tau_cal), open("results/eval_ct_twins_fe.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
