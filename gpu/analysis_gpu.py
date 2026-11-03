"""Every number of the GPU scaling experiment that is quoted in the manuscript, computed from the result files.

    (from the repository root)
    python gpu/analysis_gpu.py results_gpu results results_gpu/results/eval_cohort_fe.json
        ->  prints every number and writes results_gpu/numbers_gpu_text.json
"""
import os, sys, json, numpy as np
from scipy.stats import spearmanr

G, P = sys.argv[1], sys.argv[2]
CFE = sys.argv[3] if len(sys.argv) > 3 else None
RG = lambda n: json.load(open(os.path.join(G, "results", n + ".json")))
RP = lambda n: json.load(open(os.path.join(P, n + ".json")))
NG, NP = RG("numbers_gpu"), RP("numbers")
F = RG("eval_fields")
out = {}
rng = np.random.default_rng(0)
ENS = ["comechL_s0", "comechL_s1", "comechL_s2"]
KEYS = ["stress_l2", "stress_l2_art", "disp_l2", "hot_iou", "mid_peak_err", "post_peak_err", "eff_rmse"]


def med(rows, k):
    v = np.array([r[k] for r in rows if k in r and r[k] == r[k]])
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


# ---------------- 1. field fidelity ----------------
fid = {}
for key, rows in F.items():
    n, sp = key.split("|")
    fid.setdefault(n, {})[sp] = {k: med(rows, k) for k in KEYS}
out["fidelity"] = fid
print("== field fidelity, held-out (median cartilage | stress | displ | IoU | mid peak | post peak | effects) and out-of-range cartilage")
for n in sorted(fid, key=lambda n: (NG["training"][n]["feet"] if n in NG["training"] else 0, n)):
    t = fid[n]["test"]; o = fid[n].get("ood", {})
    print(f"{n:17s} {NG['training'][n]['feet']:5d}  " + "  ".join(f"{t[k][0]:.3f}" for k in KEYS)
          + (f"   ood art {o['stress_l2_art'][0]:.3f} iou {o['hot_iou'][0]:.3f}" if o else ""))
for sp in ("test", "ood"):
    for k in KEYS:
        v = [fid[n][sp][k][0] for n in ENS]
        out.setdefault("ens_seeds", {}).setdefault(sp, {})[k] = dict(mean=float(np.mean(v)), sd=float(np.std(v)), min=min(v), max=max(v))
e = out["ens_seeds"]
print(f"CoMech-L seeds (test): cartilage {e['test']['stress_l2_art']['mean']:.4f} +- {e['test']['stress_l2_art']['sd']:.4f} "
      f"[{e['test']['stress_l2_art']['min']:.3f}-{e['test']['stress_l2_art']['max']:.3f}], stress {e['test']['stress_l2']['mean']:.3f}, "
      f"displ {e['test']['disp_l2']['mean']:.3f}, IoU {e['test']['hot_iou']['mean']:.3f}, mid {e['test']['mid_peak_err']['mean']:.3f}, "
      f"post {e['test']['post_peak_err']['mean']:.3f}, effects {e['test']['eff_rmse']['mean']:.3f}; "
      f"ood cartilage {e['ood']['stress_l2_art']['mean']:.3f}, IoU {e['ood']['hot_iou']['mean']:.3f}")
pf = NP["fidelity"]
print("paper CPU (test cartilage): CoMech seeds %.4f; actcat %.4f; geo %.4f; fno %.4f; deeponet %.4f" % (
    pf["comech_seeds"]["test"]["stress_l2_art"]["mean"], pf["actcat"]["test"]["stress_l2_art"]["median"],
    pf["geo"]["test"]["stress_l2_art"]["median"], pf["fno"]["test"]["stress_l2_art"]["median"],
    pf["deeponet"]["test"]["stress_l2_art"]["median"]))
print("paper CPU (test displ): CoMech seeds %.4f; (ood cartilage %.4f)" % (pf["comech_seeds"]["test"]["disp_l2"]["mean"],
                                                                         pf["comech_seeds"]["ood"]["stress_l2_art"]["mean"]))
# by state (pre-operative coalition vs resection) for comechL_s0
for n in ["comechL_s0"]:
    rows = F[f"{n}|test"]
    by = {}
    for r in rows:
        by.setdefault(r["meta"], []).append(r["stress_l2_art"])
    out.setdefault("by_meta", {})[n] = {m: float(np.median(v)) for m, v in by.items()}
    print(n, "cartilage by action type:", {m: round(float(np.median(v)), 3) for m, v in by.items()})
print("Wilcoxon vs comechL_s0 (cartilage, test):", {k: ("%.1e" % v["p"], v["comech_better"]) for k, v in NG["tests"].items()})

# ---------------- 2. roll-out ----------------
print("== roll-out (two transitions vs one):", {n: (round(v["two"], 3), round(v["one"], 3)) for n, v in NG["rollout"].items()})
out["rollout"] = NG["rollout"]

# ---------------- 3. tolerance on held-out feet ----------------
T = RG("eval_tipping"); fe = T["fe"]; ids = sorted(fe, key=int)
T0 = RP("eval_tipping")


def boot_rho(p, a, n=4000):
    bs = []
    for _ in range(n):
        i = rng.integers(len(a), size=len(a))
        if np.ptp(a[i]) > 0 and np.ptp(p[i]) > 0:
            bs.append(spearmanr(p[i], a[i])[0])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


tol = {}
groups = [("paper_cpu_ens", None), ("comechL_ens", ENS)] + [(n, [n]) for n in NG["training"]]
for fac in ("post", "mid"):
    a = np.array([fe[i][fac]["tol"] for i in ids if fac in fe[i]]) * 100
    a0 = np.array([T0["fe"][i][fac]["tol"] for i in ids if fac in T0["fe"][i]]) * 100
    assert np.allclose(a, a0), "FE tolerances differ from the paper"
    loo = np.array([(a.sum() - x) / (len(a) - 1) for x in a])
    tol[fac] = dict(n=len(a), fe_min=float(a.min()), fe_max=float(a.max()), fe_median=float(np.median(a)),
                    censored=int((a >= 69.99).sum()), loo_mae=float(np.abs(a - loo).mean()), models={})
    for name, ms in groups:
        if ms is None:
            p = np.mean([[T0["models"][m]["pred"][i][fac]["tol"] for i in ids if fac in T0["fe"][i]]
                         for m in ["comech_s0", "comech_s1", "comech_s2"]], 0) * 100
        else:
            p = np.mean([[T["models"][m]["pred"][i][fac]["tol"] for i in ids if fac in fe[i]] for m in ms], 0) * 100
        lo, hi = boot_rho(p, a)
        tol[fac]["models"][name] = dict(mae=float(np.abs(p - a).mean()), bias=float((p - a).mean()),
                                        rho=float(spearmanr(p, a)[0]), rho_ci=[lo, hi])
out["tolerance"] = tol
print("== tolerance on 20 held-out feet (MAE pp, bias, Spearman rho [95% bootstrap CI])")
for fac in ("post", "mid"):
    t = tol[fac]
    print(f"{fac}: FE {t['fe_min']:.1f}-{t['fe_max']:.1f} (median {t['fe_median']:.1f}; censored {t['censored']}), LOO baseline {t['loo_mae']:.2f}")
    for name, v in t["models"].items():
        print(f"   {name:17s} MAE {v['mae']:5.2f} bias {v['bias']:+5.2f} rho {v['rho']:+.2f} [{v['rho_ci'][0]:+.2f}, {v['rho_ci'][1]:+.2f}]")
# osteotomy direction (exchange rate) on the 6 feet: direction of change predicted by each model
xr = {}
for name in ["comechL_s0", "actcatL", "gino", "geofno"]:
    ok = tot = 0
    for i in ids:
        f = fe[i]
        if "post" not in f or not f.get("xrate"):
            continue
        for t_ in ("5", "-5"):
            if t_ not in f["xrate"]:
                continue
            d_fe = f["xrate"][t_]["tol"] - f["post"]["tol"]
            pm = T["models"][name]["pred"][i]
            d_p = pm["xrate"][t_]["tol"] - pm["post"]["tol"]
            ok += int(np.sign(d_fe) == np.sign(d_p)); tot += 1
    xr[name] = [ok, tot]
out["xrate_direction"] = xr
print("osteotomy direction agreement (predicted vs FE change of posterior tolerance):", xr)

# ---------------- 4. virtual cohort, FE-verified ----------------
if CFE and os.path.exists(CFE):
    C = json.load(open(CFE))["summary"]; C0 = RP("eval_cohort_fe")["summary"]
    out["cohort_fe"] = dict(gpu=dict(pred_vs_fe=C["pred_vs_fe"], tol_mid={k: C["tol_mid"][k] for k in ("n", "pred_mae", "pred_bias", "pred_rho", "const_mae")},
                                     tol_post={k: C["tol_post"][k] for k in ("n", "pred_mae", "pred_bias", "pred_rho", "const_mae")}),
                            paper=dict(pred_vs_fe=C0["pred_vs_fe"], tol_mid={k: C0["tol_mid"][k] for k in ("n", "pred_mae", "pred_bias", "pred_rho", "const_mae")},
                                       tol_post={k: C0["tol_post"][k] for k in ("n", "pred_mae", "pred_bias", "pred_rho", "const_mae")}))
    print("== virtual cohort (CoMech-L ensemble vs FE; paper CoMech ensemble in brackets)")
    for k in ("pred_vs_fe", "tol_mid", "tol_post"):
        print(k, out["cohort_fe"]["gpu"][k], " [paper:", out["cohort_fe"]["paper"][k], "]")

# ---------------- 5. equivalence, planning, abstention ----------------
E = NG["equivalence"]; PL = NG["planning"]
pe = [E[n]["precision"] for n in ENS]; re_ = [E[n]["recall"] for n in ENS]
out["equivalence_ens"] = dict(precision=[float(np.mean(pe)), float(np.std(pe))], recall=[float(np.mean(re_)), float(np.std(re_))])
print(f"== equivalence: CoMech-L seeds P {np.mean(pe):.3f} +- {np.std(pe):.3f}, R {np.mean(re_):.3f} +- {np.std(re_):.3f}")
for n in NG["training"]:
    print(f"   {n:17s} P {E[n]['precision']:.2f} R {E[n]['recall']:.2f}  regret median {PL[n]['regret_median']:.4f} mean {PL[n]['regret_mean']:.4f} hits {PL[n]['top1']}/{PL[n]['n']}")
EQ = RG("eval_equiv_plan")
print("   n_true pairs:", {n: EQ["equivalence"][n]["n_true"] for n in ENS}, "n_pairs", EQ["equivalence"]["comechL_s0"]["n_pairs"])
print("== abstention (CoMech-L ensemble):", NG["abstention"])
out["abstention"] = NG["abstention"]

# ---------------- 6. zero-shot and taxonomy ----------------
Z = RG("eval_zeroshot")
zs = {}
for n, z in Z.items():
    s = z["scores"]
    zs[n] = dict(tau=s["tau_all_vs_fe"], align_fe=s["align_vs_fe"], align_atlas=s["align_vs_atlas"], fusion=s["fusion_below_all"],
                 tol_post=s["tol_post_effhead"], tol_mid=s["tol_mid_effhead"], tol_post_fe=s["tol_post_fe"], tol_mid_fe=s["tol_mid_fe"],
                 field=s["field_stress_l2_median"])
    # bridge stress relative to FE (99th percentile in the bridge) for the nine coalitions
    R = z["rows"]
    rat = [R[k]["pred"]["coal_p99"] / R[k]["fe"]["coal_p99"] for k in R if R[k].get("group") == "cls" and R[k]["fe"].get("coal_p99", 0) > 0]
    zs[n]["bridge_ratio_median"] = float(np.median(rat)) if rat else None
out["zeroshot"] = zs
print("== zero-shot atlas analogue")
for n in ENS + ["actcatL", "gino", "geofno"]:
    v = zs[n]
    print(f"   {n:12s} tau {v['tau']:+.2f} align {v['align_fe']} (atlas {v['align_atlas']}) fusion {v['fusion']} tol post {v['tol_post']:.3f}/{v['tol_post_fe']:.3f} "
          f"mid {v['tol_mid']:.3f}/{v['tol_mid_fe']:.3f} field {v['field']:.3f} bridge ratio {v['bridge_ratio_median']}")
TX = NG["taxonomy"]
print("== taxonomy (ARI): ", {n: (round(v["latent_anatomy"], 2), round(v["latent_rozansky"], 2), round(v["fp_vs_fe_r"], 3)) for n, v in TX.items()})
la = [TX[n]["latent_anatomy"] for n in ENS]; lr = [TX[n]["latent_rozansky"] for n in ENS]; fr = [TX[n]["fp_vs_fe_r"] for n in ENS]
out["taxonomy_ens"] = dict(latent_anatomy=[float(np.mean(la)), float(np.std(la))], latent_rozansky=[float(np.mean(lr)), float(np.std(lr))],
                           fp_vs_fe_r=[float(np.mean(fr)), float(np.std(fr))])
print("   seeds: latent anatomy %.2f +- %.2f, Rozansky %.2f +- %.2f, fingerprint r %.3f +- %.3f" % (np.mean(la), np.std(la), np.mean(lr), np.std(lr), np.mean(fr), np.std(fr)))
ET = RG("eval_taxonomy")
for n in ["comechL_s0"]:
    v = ET.get(n, {})
    for k in ("classify", "classification", "acc"):
        if k in v:
            print("   classification:", json.dumps(v[k])[:300])

# ---------------- 7. latency ----------------
L = NG["latency_gpu"]; out["latency"] = L
print(f"== latency on {L['device']}: full field {1000 * L['full_field_s']:.1f} ms, 200 effects {1000 * L['sweep200_effects_s']:.1f} ms "
      f"(paper CPU {1000 * NP['latency']['rollout_full_field_s']:.0f} ms and {NP['latency']['sweep200_effects_s']:.2f} s; FE {NP['latency']['fe_solve_s_median']:.2f} s)")

# ---------------- 8. refined grid ----------------
if os.path.exists(os.path.join(G, "results", "eval_refined.json")):
    RF = RG("eval_refined")
    rf = {}
    if "fe_base" in RF:
        rf["fe_base"] = {k: med(RF["fe_base"], k) for k in KEYS[:4]}
    for n, rows in RF.items():
        if n == "fe_base":
            continue
        rf[n] = dict(vs_refined={k: med([r["vs_refined"] for r in rows], k) for k in KEYS[:4]},
                     vs_standard={k: med([r["vs_standard"] for r in rows], k) for k in KEYS[:4]}, n=len(rows))
    out["refined"] = rf
    print("== against the twofold-refined FE (median cartilage, displacement)")
    if "fe_base" in rf:
        print(f"   standard-grid FE: cartilage {rf['fe_base']['stress_l2_art'][0]:.3f} [{rf['fe_base']['stress_l2_art'][1]:.3f}-{rf['fe_base']['stress_l2_art'][2]:.3f}], "
              f"stress {rf['fe_base']['stress_l2'][0]:.3f}, displ {rf['fe_base']['disp_l2'][0]:.3f}, IoU {rf['fe_base']['hot_iou'][0]:.3f}")
    for n, v in rf.items():
        if n != "fe_base":
            print(f"   {n:17s} vs refined {v['vs_refined']['stress_l2_art'][0]:.3f} (displ {v['vs_refined']['disp_l2'][0]:.3f})"
                  f"   vs standard {v['vs_standard']['stress_l2_art'][0]:.3f} (displ {v['vs_standard']['disp_l2'][0]:.3f})  n={v['n']}")
RE = RG("eval_refined_effects")["summary"]
out["refined_effects"] = RE
print("== effects, refined vs standard grid:", {k: RE[k] for k in ("n_config", "eff_rmse")})
print("   within-foot:", RE["within_foot"]); print("   equivalence:", RE["equivalence_standard_vs_refined"])
for k in ("post_mean", "mid_mean", "ant_mean", "tn_p99", "nav_drop", "talus_evert"):
    v = RE["per_effect"][k]
    print(f"   {k:11s} median |rel| {v['median_rel']:.3f}, signed {v['median_signed_rel']:+.3f}, r {v['pearson']:.3f}, rho {v['spearman']:.3f}")
json.dump(out, open(os.path.join(G, "numbers_gpu_text.json"), "w"), indent=1, default=float)
print("wrote", os.path.join(G, "numbers_gpu_text.json"))
