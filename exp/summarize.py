"""Collect every number reported in the manuscript into results/numbers.json."""
import os, sys, json, glob, pickle, numpy as np
from scipy.stats import spearmanr, wilcoxon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = lambda n: json.load(open(f"results/{n}.json")) if os.path.exists(f"results/{n}.json") else None
N = {}
ALL = ["comech_s0", "comech_s1", "comech_s2", "comech_noswap", "comech_nosens", "comech_none", "actcat", "geo", "deeponet", "fno"]
SEEDS = ["comech_s0", "comech_s1", "comech_s2"]


def q(v):
    v = np.asarray([x for x in v if x == x], float)
    return dict(median=float(np.median(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                mean=float(v.mean()), n=int(len(v)))


# ---- bank and FE counts ----
B = R("bank_stats")
sw = len(glob.glob("data/sweeps/*.json"))
conv = sum(1 for l in open("results/convergence.log") if l.startswith("{"))
N["fe"] = dict(bank=B["total_solutions_bank"], atlas_analogue=33, sweeps=sw, convergence=conv,
               total=B["total_solutions_bank"] + 33 + sw + conv,
               feet={s: B[s]["feet"] for s in ["train", "val", "test", "ood"]},
               iters_median=B["iters_median"])
# ---- training ----
tr = {}
for m in ALL:
    p = f"{os.environ.get('COMECH_MODELS', 'results/models_long')}/{m}.pkl"
    if os.path.exists(p):
        d = pickle.load(open(p, "rb"))
        tr[m] = dict(steps=d["cfg"]["steps"], train_time_h=d["cfg"]["train_time"] / 3600,
                     params=int(sum(v.size for v in d["state"].values())), eps=d["cfg"]["eps"],
                     final_field=float(np.mean([h["field"] for h in d["hist"][-100:]])))
N["training"] = tr
# ---- fidelity ----
F = R("eval_fields")
if F:
    fid = {}
    for k, rows in F.items():
        m, sp = k.split("|")
        fid.setdefault(m, {})[sp] = {key: q([r[key] for r in rows if key in r]) for key in
                                     ["stress_l2", "stress_l2_art", "disp_l2", "hot_iou", "mid_peak_err", "post_peak_err",
                                      "mid_mean_err", "post_mean_err", "eff_rmse", "t"]}
    # seed spread of medians
    for sp in ["test", "ood"]:
        fid.setdefault("comech_seeds", {})[sp] = {key: dict(mean=float(np.mean([fid[s][sp][key]["median"] for s in SEEDS if s in fid])),
                                                           sd=float(np.std([fid[s][sp][key]["median"] for s in SEEDS if s in fid])))
                                                  for key in ["stress_l2", "stress_l2_art", "disp_l2", "hot_iou", "eff_rmse"]}
    # per action type for CoMech s0 (test)
    rows = F.get("comech_s0|test", [])
    fid["comech_s0_by_meta"] = {mt: q([r["stress_l2_art"] for r in rows if r["meta"] == mt]) for mt in set(r["meta"] for r in rows)}
    N["fidelity"] = fid
    tests = {}
    for sp in ["test", "ood"]:
        ref = {(r["f"], r["k"]): r for r in F.get(f"comech_s0|{sp}", [])}
        for m in ALL:
            if m == "comech_s0" or f"{m}|{sp}" not in F:
                continue
            rows = [(ref[(r["f"], r["k"])], r) for r in F[f"{m}|{sp}"] if (r["f"], r["k"]) in ref]
            for key in ["stress_l2", "stress_l2_art", "disp_l2", "eff_rmse"]:
                a = np.array([x[key] for x, _ in rows]); b = np.array([y[key] for _, y in rows])
                tests.setdefault(sp, {}).setdefault(m, {})[key] = dict(p=float(wilcoxon(a, b).pvalue), comech_better=int((a < b).sum()), n=len(a))
    N["fidelity_tests"] = tests
L = R("eval_latency")
if L:
    N["latency"] = dict(L, speedup_full=L["fe_solve_s_median"] / L["rollout_full_field_s"],
                        speedup_effects=L["fe_solve_s_median"] / (L["sweep200_effects_s"] / 200))
RO = R("eval_rollout")
if RO:
    N["rollout"] = {m: dict(two=q([r["two_step"]["stress_l2_art"] for r in v]), one=q([r["one_step"]["stress_l2_art"] for r in v]),
                            two_eff=q([r["eff2"] for r in v]), one_eff=q([r["eff1"] for r in v])) for m, v in RO.items()}
# ---- zero-shot ----
Z = R("eval_zeroshot")
if Z:
    N["zeroshot"] = {m: v["scores"] for m, v in Z.items()}
# ---- taxonomy ----
T = R("eval_taxonomy")
if T:
    tx = {"geometry": {t: T["geometry"][t]["ari"] for t in ["anatomy", "rozansky"]},
          "geometry_nmi": {t: T["geometry"][t]["nmi"] for t in ["anatomy", "rozansky"]}}
    if "fe_mechanics" in T:
        tx["fe_mechanics"] = {t: T["fe_mechanics"][t]["ari"] for t in ["anatomy", "rozansky"]}
        tx["fe_mechanics_nmi"] = {t: T["fe_mechanics"][t]["nmi"] for t in ["anatomy", "rozansky"]}
        v = [T[s_]["fp_vs_fe_r"] for s_ in SEEDS if s_ in T and "fp_vs_fe_r" in T[s_]]
        if v:
            tx["fp_vs_fe_r"] = dict(mean=float(np.mean(v)), sd=float(np.std(v)))
    for rep in ["latent", "mechanics"]:
        for t in ["anatomy", "rozansky"]:
            v = [T[s][rep][t]["ari"] for s in SEEDS if s in T]
            tx[f"{rep}_{t}"] = dict(mean=float(np.mean(v)), sd=float(np.std(v)))
            nm = [T[s][rep][t]["nmi"] for s in SEEDS if s in T]
            tx[f"{rep}_{t}_nmi"] = dict(mean=float(np.mean(nm)), sd=float(np.std(nm)))
    for m in ["comech_noswap", "comech_nosens", "comech_none"]:
        if m in T:
            tx[m] = {f"{rep}_{t}": T[m][rep][t]["ari"] for rep in ["latent", "mechanics"] for t in ["anatomy", "rozansky"]}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import silhouette_score
    at = np.array(T["labels"]["atype"]); rt = np.array(T["labels"]["rtype"]); t1 = at == "I"
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    def sup(X):
        X = np.asarray(X)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        return dict(acc_anatomy=float(cross_val_score(clf, X, at, cv=cv).mean()),
                    acc_rozansky=float(cross_val_score(clf, X[t1], rt[t1], cv=cv).mean()),
                    sil_anatomy=float(silhouette_score(StandardScaler().fit_transform(X), at)),
                    sil_rozansky=float(silhouette_score(StandardScaler().fit_transform(X[t1]), rt[t1])))
    if "fe_mechanics" in T:
        tx["fe_supervised"] = sup(T["fe_mechanics"]["fingerprints"])
    ss = [sup(T[s_]["fingerprints"]) for s_ in SEEDS if s_ in T]
    tx["comech_supervised"] = {k: dict(mean=float(np.mean([x[k] for x in ss])), sd=float(np.std([x[k] for x in ss]))) for k in ss[0]}
    N["taxonomy"] = tx
# ---- tipping ----
TP = R("eval_tipping")
if TP:
    fe = TP["fe"]
    N["tipping_fe"] = {fac: q([fe[i][fac]["tol"] * 100 for i in fe if fac in fe[i]]) for fac in ["mid", "post"]}
    tp = {}
    for m, v in TP["models"].items():
        tp[m] = {fac: q(v["tol_err_pp"][fac]) for fac in ["mid", "post"]}
        tp[m]["both_mean"] = float(np.mean(v["tol_err_pp"]["mid"] + v["tol_err_pp"]["post"]))
        tp[m]["curve"] = {fac: q(v["curve_err_pct"][fac]) for fac in ["mid", "post"]}
        for fac in ["mid", "post"]:
            ids = [i for i in fe if fac in fe[i]]
            a = np.array([fe[i][fac]["tol"] for i in ids]); b = np.array([v["pred"][i][fac]["tol"] for i in ids])
            tp[m][f"spearman_{fac}"] = float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else float("nan")
            tp[m][f"mae_{fac}"] = float(np.mean(np.abs(a - b)) * 100)
            tp[m][f"bias_{fac}"] = float(np.mean(b - a) * 100)
    for fac in ["mid", "post"]:
        ids = [i for i in fe if fac in fe[i]]
        a = np.array([fe[i][fac]["tol"] for i in ids])
        loo = np.array([(a.sum() - x) / (len(a) - 1) for x in a])
        N.setdefault("tipping_const", {})[fac] = dict(mae=float(np.mean(np.abs(a - loo)) * 100), sd=float(a.std() * 100),
                                                      range=[float(a.min() * 100), float(a.max() * 100)],
                                                      censored=int((a >= 0.6999).sum()), n=len(a))
    N["tipping"] = tp
    # exchange-rate verification on FE (posterior tolerance at +-5 mm)
    xr = []
    for i in fe:
        x = fe[i].get("xrate", {})
        if "5" in x and "-5" in x and "post" in fe[i]:
            p = TP["models"]["comech_s0"]["pred"][i]
            xr.append(dict(i=i, fe=[fe[i]["xrate"]["-5"]["tol"], fe[i]["post"]["tol"], fe[i]["xrate"]["5"]["tol"]],
                           pred=[p["xrate"]["-5"]["tol"], p["post"]["tol"], p["xrate"]["5"]["tol"]]))
    N["xrate_fe_check"] = xr
    if xr:
        fe_up = [100 * (x["fe"][2] - x["fe"][1]) for x in xr]; fe_dn = [100 * (x["fe"][0] - x["fe"][1]) for x in xr]
        pr_up = [100 * (x["pred"][2] - x["pred"][1]) for x in xr]; pr_dn = [100 * (x["pred"][0] - x["pred"][1]) for x in xr]
        N["xrate_fe_summary"] = dict(n=len(xr), fe_medial_pp=q(fe_up), fe_lateral_pp=q(fe_dn), pred_medial_pp=q(pr_up), pred_lateral_pp=q(pr_dn),
                                     fe_medial_up=int(sum(v > 0 for v in fe_up)), fe_lateral_down=int(sum(v < 0 for v in fe_dn)),
                                     sign_agree=int(sum(np.sign(a) == np.sign(b) for a, b in zip(fe_up + fe_dn, pr_up + pr_dn))),
                                     n_signs=2 * len(xr))
# ---- cohort ----
C = R("eval_cohort")
if C:
    S = C["summary"]
    det = {fac: sorted(S[f"det_{fac}"]["spearman"].items(), key=lambda kv: -abs(kv[1]))[:4] for fac in ["mid", "post"]}
    N["cohort"] = dict(S, top_determinants=det)
    TT = list(C["TT"]); j0 = TT.index(0.0)
    xd = {}
    for fac in ["mid", "post"]:
        for nm, sel, jt in [("valgus_medialize", lambda a: a > 0, TT.index(5.0)), ("varus_lateralize", lambda a: a < 0, TT.index(-5.0))]:
            v = [100 * (r[f"tol_{fac}_by_t"][jt] - r[f"tol_{fac}_by_t"][j0]) / 5.0 for r in C["rows"] if sel(r["foot"]["align"])]
            xd[f"{fac}_{nm}_pp_per_mm"] = q(v)
        v = [100 * (r[f"tol_{fac}_by_t"][TT.index(5.0)] - r[f"tol_{fac}_by_t"][j0]) / 5.0 for r in C["rows"]]
        xd[f"{fac}_medialize_all_pp_per_mm"] = q(v)
        v = [100 * (r[f"tol_{fac}_by_t"][TT.index(-5.0)] - r[f"tol_{fac}_by_t"][j0]) / 5.0 for r in C["rows"]]
        xd[f"{fac}_lateralize_all_pp_per_mm"] = q(v)
    N["cohort_xrate"] = xd
    rows = C["rows"]
    N["cohort"]["by_type"] = {}
    for t in ["I", "II", "III", "IV"]:
        rr = [r for r in rows if r["foot"]["ctype"] == t]
        N["cohort"]["by_type"][t] = dict(n=len(rr), within=sum(r["within"] for r in rr), rule_ok=sum(r["rule_ok"] for r in rr),
                                         ratio=q([r["ratio"] * 100 for r in rr]))
# ---- equivalence, planning, abstention ----
E = R("eval_equiv_plan")
if E:
    N["equivalence"] = E["equivalence"]; N["eps"] = E["eps"]
    pl = {}
    for m, rows in E["planning"]["models"].items():
        pl[m] = dict(regret=q([r["regret"] for r in rows]), top1=int(sum(r["top1"] for r in rows)), n=len(rows),
                     rho=q([r["rank_rho"] for r in rows]), regret_p90=float(np.percentile([r["regret"] for r in rows], 90)),
                     regret_max=float(np.max([r["regret"] for r in rows])))
    N["planning"] = pl
    # reference policies: a random adjunct, and the best single adjunct applied to every foot
    Jf = [np.array(v["J_fe"]) for v in E["planning"]["feet"].values()]
    rr = [float(np.mean((J - J.min()) / (J.max() - J.min()))) for J in Jf]
    fixed = [[float((J[c] - J.min()) / (J.max() - J.min())) for J in Jf] for c in range(len(Jf[0]))]
    cbest = int(np.argmin([np.median(f) for f in fixed]))
    N["planning_ref"] = dict(random=q(rr), best_fixed=q(fixed[cbest]), best_fixed_cand=E["planning"]["cands"][cbest],
                             fe_best_counts={str(E["planning"]["cands"][c]): int(sum(int(np.argmin(J)) == c for J in Jf)) for c in range(len(Jf[0]))})
    N["planning_fe_best"] = [E["planning"]["cands"][E["planning"]["feet"][str(i)]["best"] if isinstance(list(E["planning"]["feet"].keys())[0], str) else E["planning"]["feet"][i]["best"]] for i in E["planning"]["feet"]]
    A = E["abstention"]
    N["abstention"] = {k: ({kk: vv for kk, vv in v.items() if not isinstance(vv, list)} if isinstance(v, dict) else v) for k, v in A.items()}
CF = R("eval_cohort_fe")
if CF:
    N["cohort_fe"] = CF["summary"]
M = R("eval_maps")
if M:
    N["maps"] = {k: v for k, v in M["child"].items() if k not in ("J", "J_sd", "cands")}
    N["maps"]["heldout"] = M["heldout"]
json.dump(N, open("results/numbers.json", "w"), indent=1)
print(json.dumps(N, indent=1)[:6000])
