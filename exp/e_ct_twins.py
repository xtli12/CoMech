"""CT-informed twins of the real feet (preliminary readings of PACS exports, pending clinician adjudication).

Every foot is represented by K twins.  What the CT reading provides is fixed or sampled around the reading:
the anatomy-based coalition type and Rozansky type, the middle-facet coverage (uniform within +/-0.15 for
medium-confidence and +/-0.25 for low-confidence readings) and the side of a recorded deformity (valgus
+5..+12 deg, varus -12..-5 deg, otherwise -4..+4 deg).  What the exports cannot measure (size, facet axes,
sustentaculum position, dome, cartilage thickness; no pixel spacing or Hounsfield units) is drawn from the
engine's priors.  Feet whose coalition could not be characterised ("indeterminate") draw their type from the
composition of the three-centre cohort.  A type-misreading sensitivity analysis redraws the type from that
composition with probability 0.25 (low confidence) or 0.10 (medium confidence).

For each twin: coalition-to-posterior-facet ratio (50% rule), defect left by complete resection, CoMech
ensemble tolerance on the coalition-free twin, envelope status, mechanical reserve and ensemble disagreement
(abstention when it exceeds the 90th-percentile validation threshold)."""
import os, sys, csv, json, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace, asdict
from exp.common import *
from exp.cohort import coalition_ratio, child_defect
import exp.bank as B
from ml.data import Norm

DD = np.round(np.arange(0, 0.701, 0.05), 3)
FAC = {"I": ["mid"], "II": ["post"], "IV": ["mid", "post"], "III": []}
P_TYPE = {k: v / sum(B.COHORT_TYPES.values()) for k, v in B.COHORT_TYPES.items()}
P_ROZ = {k: v / sum(B.COHORT_ROZ_I.values()) for k, v in B.COHORT_ROZ_I.items()}
WIDTH = {"M": 0.15, "L": 0.25, "H": 0.08}


def pick(rng, P):
    k = list(P); return k[rng.choice(len(k), p=np.array([P[x] for x in k]))]


def read_csv(path):
    R = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        f = lambda k: float(r[k]) if r[k] not in ("", None) else None
        R.append(dict(foot=r["foot"], patient=r["patient"], side=r["side"], treatment=r["treatment"],
                      deformity=r["deformity"], postop_ct=r["postop_ct"], type=r["anatomic_type"],
                      roz=r["rozansky"].replace("?", ""), cov_mid=f("cov_mid"), cov_post=f("cov_post"),
                      conf=r["confidence"] or "L"))
    return R


def twin(rng, r, flip=False):
    base = B.sample_foot(rng, p_intact=0.0)
    d = r["deformity"]
    al = rng.uniform(5, 12) if d.startswith("valgus") else (rng.uniform(-12, -5) if d.startswith("varus") else rng.uniform(-4, 4))
    t = r["type"]; roz = r["roz"]
    p_mis = {"M": 0.10, "L": 0.25, "H": 0.05}[r["conf"]] if flip else 0.0
    source = "reading"
    if t == "indeterminate" or rng.random() < p_mis:
        t = pick(rng, P_TYPE); roz = ""; source = "prior"
    w = WIDTH[r["conf"]]
    kw = dict(align=al, ctype=t, cov_mid=0.0, cov_post=0.0, fib_frac=0.0)
    if t == "I":
        if roz not in ("R1", "R2", "R3", "R4", "unclass") or source == "prior":
            roz = pick(rng, P_ROZ)
        c = r["cov_mid"] if (source == "reading" and r["cov_mid"] is not None) else (rng.uniform(0.5, 1.0) if roz == "R4" else rng.uniform(0.2, 0.6))
        kw.update(rtype=roz, cov_mid=float(np.clip(rng.uniform(c - w, c + w) if source == "reading" else c, 0.1, 0.95)))
        if roz == "R3":
            kw["fib_frac"] = rng.uniform(0.3, 0.7)
    elif t == "II":
        kw.update(rtype="R5", cov_post=rng.uniform(0.15, 0.6))
    elif t == "III":
        kw.update(rtype="R5")
    else:
        kw.update(rtype="unclass", cov_mid=rng.uniform(0.3, 0.8), cov_post=rng.uniform(0.15, 0.5))
    return replace(base, **kw), source


def evaluate(foot, preds, norm, tau_cal):
    cr = coalition_ratio(foot)
    rec = dict(foot=asdict(foot), ratio=cr["ratio"], rule_ok=bool(cr["ratio"] < 0.5))
    fac = FAC[foot.ctype]
    if not fac:      # extra-articular: complete resection leaves no facet defect
        rec.update(d_mid=0.0, d_post=0.0, within=True, within_cal=True, dis=None)
        return rec
    d, fr_int, _ = child_defect(foot)
    rec.update(d)
    gi = channels(np.round(fr_int * 255).astype(np.uint8)).astype(np.float32)
    ok, okc, dis = True, True, []
    for fc in fac:
        A = np.stack([act_vec(1, x if fc == "mid" else 0, x if fc == "post" else 0) for x in DD])
        Z = np.stack([p.effects(gi, A) for p in preds])
        dis.append(float(Z.std(0).mean()))
        y = norm.eff_inv(Z.mean(0))[:, EI[f"{fc}_mean"]]
        tol = tolerance(DD, y); tolc = tolerance(DD, y, tau=tau_cal[fc])
        rec[f"tol_{fc}"] = tol; rec[f"tol_{fc}_cal"] = tolc
        rec[f"reserve_{fc}"] = (tol - rec[f"d_{fc}"]) / max(tol, 1e-6)
        ok &= rec[f"d_{fc}"] <= tol; okc &= rec[f"d_{fc}"] <= tolc
    rec.update(within=bool(ok), within_cal=bool(okc), dis=float(np.mean(dis)))
    return rec


def run_foot(args):
    r, K, seed = args
    norm = Norm.cached()
    preds = [get_pred(n) for n in ["comech_s0", "comech_s1", "comech_s2"]]
    tau_cal = json.load(open("results/eval_cohort_fe.json"))["summary"]["tau_cal"]
    out = dict(meta=r, twins={}, )
    for mode in ["main", "flip"]:
        rng = np.random.default_rng(seed)
        tw = []
        for k in range(K):
            f, src = twin(rng, r, flip=(mode == "flip"))
            e = evaluate(f, preds, norm, tau_cal); e["source"] = src
            tw.append(e)
        out["twins"][mode] = tw
    print(r["foot"], r["type"], flush=True)
    return out


def summarise(res, thr):
    rows = []
    for o in res:
        r = o["meta"]; T = o["twins"]["main"]; Tf = o["twins"]["flip"]
        w = np.array([t["within"] for t in T]); wc = np.array([t["within_cal"] for t in T]); ru = np.array([t["rule_ok"] for t in T])
        wf = np.array([t["within"] for t in Tf])
        dis = [t["dis"] for t in T if t["dis"] is not None]
        ab = float(np.mean([x > thr for x in dis])) if dis else 0.0
        def cls(p):
            return "within" if p >= 0.9 else ("beyond" if p <= 0.1 else "indeterminate")
        q = lambda key: [float(np.percentile([t[key] for t in T if key in t], p)) for p in (10, 50, 90)] if any(key in t for t in T) else None
        rows.append(dict(foot=r["foot"], type=r["type"], conf=r["conf"], treatment=r["treatment"], deformity=r["deformity"],
                         p_within=float(w.mean()), p_within_cal=float(wc.mean()), p_rule_ok=float(ru.mean()),
                         p_within_flip=float(wf.mean()), status=cls(w.mean()), status_cal=cls(wc.mean()),
                         status_flip=cls(wf.mean()), rule=cls(ru.mean()).replace("within", "resect alone").replace("beyond", "not alone"),
                         abstain_frac=ab, ratio=q("ratio"), d_mid=q("d_mid"), tol_mid=q("tol_mid"), reserve_mid=q("reserve_mid"),
                         d_post=q("d_post"), tol_post=q("tol_post")))
    return rows


def main(csv_path, K=32):
    R = read_csv(csv_path)
    thr = json.load(open("results/eval_equiv_plan.json"))["abstention"]["threshold"]
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NPROC", "1"))) as p:
        res = p.map(run_foot, [(r, K, 1000 + i) for i, r in enumerate(R)], chunksize=1)
    rows = summarise(res, thr)
    json.dump(dict(rows=rows, threshold=thr, K=K, raw=res), open("results/eval_ct_twins.json", "w"))
    for x in rows:
        print(x["foot"], x["type"], x["conf"], x["status"], x["status_cal"], x["rule"], round(x["p_within"], 2), round(x["p_rule_ok"], 2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32)
