"""Compare the open-source engine's atlas-analog series with the Ansys atlas (fe_data.py)."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.fe_atlas import CLS_MODELS, CLS, MID, POST, ALIGN, ALIGN_TYPES, DEFECTS
from scipy.stats import kendalltau, spearmanr

recs = {r["name"]: r for r in (json.load(open(f)) for f in glob.glob("results/atlas_analog/rec_*.json"))}
E = {k: v["eff"] for k, v in recs.items()}
res = {}
# map atlas metric names -> engine metric
M = {"ant_max": "ant_p99", "mid_max": "mid_p99", "post_max": "post_p99", "ant_mean": "ant_mean", "mid_mean": "mid_mean",
     "post_mean": "post_mean", "tn_max": "tn_p99", "cc_max": "cc_p99"}
coal_names = ["Roz I", "Roz II", "Roz III", "Roz IV", "Roz V", "Type I", "Type II", "Type III", "Type IV"]

# 1. stress concentration at the bridge (engine: bridge P99; atlas: talocalcaneal complex peak)
tc_int = E["Normal"]["tc_p99"]
ratios = {n: E[n]["coal_p99"] / tc_int for n in coal_names}
at_int = CLS["tc_max"][0]
at_ratios = {n: CLS["tc_max"][CLS_MODELS.index(n)] / at_int for n in coal_names}
res["bridge_ratio_engine"] = [min(ratios.values()), max(ratios.values())]
res["bridge_ratio_atlas"] = [min(at_ratios.values()), max(at_ratios.values())]
# 2. fusion below every partial coalition (talocalcaneal complex)
res["fusion_tc"] = E["Fusion"]["tc_p99"]
res["fusion_below_all_engine"] = bool(all(E["Fusion"]["tc_p99"] < E[n]["tc_p99"] for n in coal_names))
res["fusion_tc_vs_intact_engine"] = E["Fusion"]["tc_p99"] / tc_int
# 3. orderings
types = ["Type I", "Type II", "Type III", "Type IV"]
roz = ["Roz I", "Roz II", "Roz III", "Roz IV", "Roz V"]
def tau(names):
    a = [CLS["tc_max"][CLS_MODELS.index(n)] for n in names]; b = [E[n]["coal_p99"] for n in names]
    t, p = kendalltau(a, b)
    conc = sum(1 for i in range(len(names)) for j in range(i + 1, len(names)) if (a[i] - a[j]) * (b[i] - b[j]) > 0)
    return float(t), conc, len(names) * (len(names) - 1) // 2
res["tau_types"] = tau(types); res["tau_roz"] = tau(roz); res["tau_all9"] = tau(coal_names)
res["roz1_highest_engine"] = max(roz, key=lambda n: E[n]["coal_p99"]) == "Roz I"
res["engine_coal_p99"] = {n: E[n]["coal_p99"] for n in coal_names}
# 4. coefficient of variation of facet peaks within taxonomies (ant, mid, post, TN)
def cv(names, src):
    out = []
    for k in ["ant_max", "mid_max", "post_max", "tn_max"]:
        if src == "atlas":
            v = np.array([CLS[k][CLS_MODELS.index(n)] for n in names])
        else:
            v = np.array([E[n][M[k]] for n in names])
        out.append(float(v.std(ddof=1) / v.mean()))
    return out
res["cv_types_engine"] = cv(types, "engine"); res["cv_roz14_engine"] = cv(roz[:4], "engine")
res["cv_types_atlas"] = cv(types, "atlas"); res["cv_roz14_atlas"] = cv(roz[:4], "atlas")
# 5. alignment: 48 signs (4 types x 2 directions x 6 regions)
regions = [("ant_max", "ant_p99"), ("mid_max", "mid_p99"), ("post_max", "post_p99"), ("tn_max", "tn_p99"),
           ("cc_max", "cc_p99"), ("tc_max", "coal_p99")]
agree, total, rows = 0, 0, []
for t in types:
    for j, pos in [(1, "varus10"), (2, "valgus10")]:
        for ka, ke in regions:
            a = ALIGN[t][ka][j] - ALIGN[t][ka][0]
            e = E[f"{t} {pos}"][ke] - E[t][ke]
            ok = np.sign(a) == np.sign(e)
            agree += int(ok); total += 1
            rows.append(dict(type=t, pos=pos, region=ka, atlas_pct=100 * a / ALIGN[t][ka][0],
                             engine_pct=100 * e / max(E[t][ke], 1e-9), agree=bool(ok)))
res["align_sign_agree"] = [agree, total]
res["align_rows"] = rows
# region-wise agreement
for ka, _ in regions:
    rr = [r for r in rows if r["region"] == ka]
    res[f"align_agree_{ka}"] = [sum(r["agree"] for r in rr), len(rr)]
# 6. defect curves
def curve(prefix, key):
    base = E["Normal"][key]
    return [0.0] + [100 * (E[f"{prefix} {d}"][key] - base) / base for d in DEFECTS[1:]]
res["mid_defect_engine"] = {k: curve("mid", k) for k in ["mid_p99", "mid_mean", "ant_p99", "ant_mean", "post_mean", "tn_p99"]}
res["post_defect_engine"] = {k: curve("post", k) for k in ["post_p99", "post_mean", "mid_mean", "mid_p99", "ant_mean", "tn_p99"]}
def at_curve(D, k):
    return [100 * (v - D[k][0]) / D[k][0] for v in D[k]]
res["mid_defect_atlas"] = {k: at_curve(MID, k) for k in ["mid_max", "mid_mean", "ant_max", "ant_mean", "post_mean", "tn_max"]}
res["post_defect_atlas"] = {k: at_curve(POST, k) for k in ["post_max", "post_mean", "mid_mean", "ant_mean", "tn_max"]}
res["rho_post_mean"] = float(spearmanr(res["post_defect_engine"]["post_mean"], res["post_defect_atlas"]["post_mean"])[0])
res["rho_mid_ant"] = float(spearmanr(res["mid_defect_engine"]["ant_p99"], res["mid_defect_atlas"]["ant_max"])[0])
res["rho_mid_mid"] = float(spearmanr(res["mid_defect_engine"]["mid_p99"], res["mid_defect_atlas"]["mid_max"])[0])
res["intact_engine"] = E["Normal"]
json.dump(res, open("results/atlas_compare.json", "w"), indent=1)
for k, v in res.items():
    if k not in ("align_rows", "intact_engine", "mid_defect_engine", "post_defect_engine", "mid_defect_atlas", "post_defect_atlas"):
        print(k, v)
print("post mean engine", np.round(res["post_defect_engine"]["post_mean"], 1))
print("post mean atlas ", np.round(res["post_defect_atlas"]["post_mean"], 1))
print("mid p99 engine", np.round(res["mid_defect_engine"]["mid_p99"], 1))
print("mid max atlas ", np.round(res["mid_defect_atlas"]["mid_max"], 1))
print("ant (mid def) engine", np.round(res["mid_defect_engine"]["ant_p99"], 1), "atlas", np.round(res["mid_defect_atlas"]["ant_max"], 1))
for r in rows:
    print(f"{r['type']:9s} {r['pos']:8s} {r['region']:8s} atlas {r['atlas_pct']:+7.1f}% engine {r['engine_pct']:+7.1f}% {'OK' if r['agree'] else '--'}")
