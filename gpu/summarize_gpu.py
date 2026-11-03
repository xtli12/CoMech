"""Compact summary of the GPU scaling experiment: results/numbers_gpu.json and logs/summary_gpu.txt."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import spearmanr, wilcoxon
from gpu import config as CFG

R = lambda n: json.load(open(f"results/{n}.json")) if os.path.exists(f"results/{n}.json") else None
med = lambda rows, k: float(np.median([r[k] for r in rows if k in r])) if rows else None
KEYS = ["stress_l2", "stress_l2_art", "disp_l2", "hot_iou", "mid_peak_err", "post_peak_err", "eff_rmse"]
N, lines = {}, []
say = lambda s: (lines.append(s), print(s))

F = R("eval_fields")
if F:
    N["fields"] = {}
    for key in sorted(F):
        name, split = key.split("|")
        N["fields"].setdefault(name, {})[split] = {k: med(F[key], k) for k in KEYS}
    ens = [n for n in CFG.ENSEMBLE if f"{n}|test" in F]
    if ens:
        N["fields"]["comechL_ens_mean"] = {sp: {k: float(np.mean([N["fields"][n][sp][k] for n in ens])) for k in KEYS}
                                           for sp in ("test", "ood") if all(f"{n}|{sp}" in F for n in ens)}
    say("== field fidelity (medians; cartilage stress, displacement, IoU, effects) ==")
    for n, v in N["fields"].items():
        t = v.get("test", {})
        if t:
            say(f"{n:20s} test  art {t['stress_l2_art']:.3f}  disp {t['disp_l2']:.3f}  iou {t['hot_iou']:.3f}  eff {t['eff_rmse']:.3f}"
                + (f"   | ood art {v['ood']['stress_l2_art']:.3f}" if v.get("ood") else ""))
    if "comechL_s0|test" in F:
        ref = {(r["f"], r["k"]): r for r in F["comechL_s0|test"]}
        N["tests"] = {}
        for key in F:
            n, sp = key.split("|")
            if sp != "test" or n == "comechL_s0":
                continue
            pr = [(ref[(r["f"], r["k"])]["stress_l2_art"], r["stress_l2_art"]) for r in F[key] if (r["f"], r["k"]) in ref]
            a, b = np.array(pr).T
            N["tests"][n] = dict(p=float(wilcoxon(a, b).pvalue), comech_better=int((a < b).sum()), n=len(a))
RO = R("eval_rollout")
if RO:
    N["rollout"] = {n: dict(two=float(np.median([r["two_step"]["stress_l2_art"] for r in rows])),
                            one=float(np.median([r["one_step"]["stress_l2_art"] for r in rows]))) for n, rows in RO.items()}
    say("== roll-out (cartilage error, two transitions vs one) ==")
    for n, v in N["rollout"].items():
        say(f"{n:20s} {v['two']:.3f} vs {v['one']:.3f}")
Z = R("eval_zeroshot")
if Z:
    N["zeroshot"] = {n: {k: v for k, v in z["scores"].items() if not isinstance(v, list)} for n, z in Z.items() if "scores" in z}
    say("== zero-shot atlas analogue ==")
    for n, s in N["zeroshot"].items():
        say(f"{n:20s} tau {s.get('tau_all_vs_fe', float('nan')):+.2f}  align {s.get('align_vs_fe')}  "
            f"tol post {s.get('tol_post_effhead', float('nan')):.2f}/{s.get('tol_post_fe', float('nan')):.2f}  "
            f"mid {s.get('tol_mid_effhead', float('nan')):.2f}/{s.get('tol_mid_fe', float('nan')):.2f}")
T = R("eval_taxonomy")
if T:
    N["taxonomy"] = {n: dict(latent_anatomy=v["latent"]["anatomy"]["ari"], latent_rozansky=v["latent"]["rozansky"]["ari"],
                             mech_anatomy=v["mechanics"]["anatomy"]["ari"], mech_rozansky=v["mechanics"]["rozansky"]["ari"],
                             fp_vs_fe_r=v.get("fp_vs_fe_r")) for n, v in T.items() if isinstance(v, dict) and "latent" in v}
TP = R("eval_tipping")
if TP:
    fe = TP["fe"]; ids = sorted(fe, key=int)
    N["tipping"] = {}
    for fac in ("mid", "post"):
        a = np.array([fe[i][fac]["tol"] for i in ids if fac in fe[i]]) * 100
        loo = np.array([(a.sum() - x) / (len(a) - 1) for x in a])
        N["tipping"][f"loo_mae_{fac}"] = float(np.abs(a - loo).mean())
        preds = {n: np.array([TP["models"][n]["pred"][i][fac]["tol"] for i in ids if fac in fe[i]]) * 100 for n in TP["models"]}
        ens = [n for n in CFG.ENSEMBLE if n in preds]
        if ens:
            preds["comechL_ens"] = np.mean([preds[n] for n in ens], 0)
        for n, p in preds.items():
            N["tipping"].setdefault(n, {})[fac] = dict(mae=float(np.abs(p - a).mean()), bias=float((p - a).mean()),
                                                      rho=float(spearmanr(p, a)[0]))
    say("== tolerance (MAE, pp; Spearman rho with FE; leave-one-out mean baseline) ==")
    say(f"baseline LOO: mid {N['tipping']['loo_mae_mid']:.1f}  post {N['tipping']['loo_mae_post']:.1f}")
    for n, v in N["tipping"].items():
        if isinstance(v, dict) and "post" in v:
            say(f"{n:20s} post {v['post']['mae']:.1f} (rho {v['post']['rho']:+.2f})  mid {v['mid']['mae']:.1f} (rho {v['mid']['rho']:+.2f})")
E = R("eval_equiv_plan")
if E:
    eq = E.get("equivalence", {}); pl = E.get("planning", {}).get("models", {})
    N["equivalence"] = {n: dict(precision=v["precision"], recall=v["recall"]) for n, v in eq.items() if isinstance(v, dict) and "precision" in v}
    N["planning"] = {n: dict(regret_median=float(np.median([r["regret"] for r in rows])), regret_mean=float(np.mean([r["regret"] for r in rows])),
                             top1=int(sum(r["top1"] for r in rows)), n=len(rows)) for n, rows in pl.items()}
    if "abstention" in E:
        N["abstention"] = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not isinstance(vv, list)})
                           for k, v in E["abstention"].items()}
    say("== equivalence (precision, recall) and planning (median regret, FE-optimal hits) ==")
    for n in N["equivalence"]:
        p = N["planning"].get(n, {})
        say(f"{n:20s} P {N['equivalence'][n]['precision']:.2f} R {N['equivalence'][n]['recall']:.2f}   regret {p.get('regret_median', float('nan')):.3f} hits {p.get('top1')}")
L = R("eval_latency_gpu")
if L:
    N["latency_gpu"] = L; say(f"== GPU latency: full field {1000 * L['full_field_s']:.1f} ms, 200 effects {1000 * L['sweep200_effects_s']:.1f} ms ({L['device']})")
RF = R("eval_refined")
if RF:
    N["refined"] = {"fe_base": {k: med(RF["fe_base"], k) for k in KEYS[:4]}} if "fe_base" in RF else {}
    for n, rows in RF.items():
        if n != "fe_base":
            N["refined"][n] = dict(vs_refined={k: float(np.median([r["vs_refined"][k] for r in rows])) for k in KEYS[:4]},
                                   vs_standard={k: float(np.median([r["vs_standard"][k] for r in rows])) for k in KEYS[:4]})
    say("== against the twofold-refined FE (cartilage stress error) ==")
    if "fe_base" in N["refined"]:
        say(f"{'standard-grid FE':20s} {N['refined']['fe_base']['stress_l2_art']:.3f}")
    for n, v in N["refined"].items():
        if n != "fe_base":
            say(f"{n:20s} vs refined {v['vs_refined']['stress_l2_art']:.3f}   vs standard {v['vs_standard']['stress_l2_art']:.3f}")
cfgs = {}
for n in CFG.RUNS:
    p = f"results/models_gpu/{n}.json"
    if os.path.exists(p):
        c = json.load(open(p)); cfgs[n] = dict(params=c["params"], feet=c["feet"], steps=c["steps"], hours=c["train_time"] / 3600, gpu=c.get("gpu"))
N["training"] = cfgs
json.dump(N, open("results/numbers_gpu.json", "w"), indent=1)
os.makedirs("logs", exist_ok=True)
open("logs/summary_gpu.txt", "w").write("\n".join(lines) + "\n")
