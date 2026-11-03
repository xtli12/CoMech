"""Counts for the CT-informed twin analysis (text, Fig. 8d and Supplementary Table 18)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T = json.load(open("results/eval_ct_twins.json")); F = json.load(open("results/eval_ct_twins_fe.json"))
rows = T["rows"]; fe = {r["foot"]: r for r in F["rows"]}
oper = lambda r: r["treatment"].startswith("resection")
S = {}
S["n_feet"] = len(rows); S["n_patients"] = len({r["foot"].split("-")[0] for r in rows})
for t in ["I", "III", "indeterminate"]:
    R = [r for r in rows if r["type"] == t]
    S[t] = dict(n=len(R), conf_M=sum(r["conf"] == "M" for r in R), conf_L=sum(r["conf"] == "L" for r in R),
                operated=sum(oper(r) for r in R), osteotomy=sum("osteotomy" in r["treatment"] for r in R),
                within=sum(r["status"] == "within" for r in R), beyond=sum(r["status"] == "beyond" for r in R),
                indet=sum(r["status"] == "indeterminate" for r in R),
                within_cal=sum(r["status_cal"] == "within" for r in R), beyond_cal=sum(r["status_cal"] == "beyond" for r in R),
                rule_alone=sum(r["rule"] == "resect alone" for r in R), rule_not=sum(r["rule"] == "not alone" for r in R),
                permissive=sum(r["rule"] == "resect alone" and r["status"] == "beyond" for r in R),
                permissive_cal=sum(r["rule"] == "resect alone" and r["status_cal"] == "beyond" for r in R),
                flip_changed=sum(r["status_flip"] != r["status"] for r in R),
                p_within_median=float(np.median([r["p_within"] for r in R])) if R else None,
                ratio_median=float(np.median([r["ratio"][1] for r in R])) if R else None,
                ratio_max=float(max(r["ratio"][2] for r in R)) if R else None,
                abstain_any=sum(r["abstain_frac"] > 0 for r in R),
                d_mid_median=float(np.median([r["d_mid"][1] for r in R if r["d_mid"]])) if t == "I" else None,
                tol_mid_median=float(np.median([r["tol_mid"][1] for r in R if r["tol_mid"]])) if t == "I" and R else None)
# operated type I feet
I = [r for r in rows if r["type"] == "I"]
S["I_operated_beyond"] = sum(oper(r) and r["status"] == "beyond" for r in I)
S["I_conservative_beyond"] = sum((not oper(r)) and r["status"] == "beyond" for r in I)
S["I_osteotomy_beyond"] = sum("osteotomy" in r["treatment"] and r["status"] == "beyond" for r in I)
S["rule_alone_all"] = sum(r["rule"] == "resect alone" for r in rows)
S["abstain_frac_all"] = float(np.mean([r["abstain_frac"] for r in rows if r["type"] != "III"]))
# FE point twins
FR = F["rows"]
a = np.array([r["fe_tol"] for r in FR]); b = np.array([r["pred_tol"] for r in FR])
S["fe"] = dict(n=len(FR), within=sum(r["fe_within"] for r in FR), within_cal=sum(r["fe_within_cal"] for r in FR),
               pred_within=sum(r["pred_within"] for r in FR), agree=sum(r["pred_within"] == r["fe_within"] for r in FR),
               rule_ok=sum(r["rule_ok"] for r in FR), permissive=sum(r["rule_ok"] and not r["fe_within"] for r in FR),
               tol_median=float(np.median(a)), tol_range=[float(a.min()), float(a.max())], d_median=float(np.median([r["d_mid"] for r in FR])),
               d_range=[float(min(r["d_mid"] for r in FR)), float(max(r["d_mid"] for r in FR))],
               ratio_range=[float(min(r["ratio"] for r in FR)), float(max(r["ratio"] for r in FR))],
               mae_pp=float(np.mean(np.abs(a - b)) * 100), bias_pp=float(np.mean(b - a) * 100),
               censored=int((a >= 0.6999).sum()))
pc = lambda x: f"{x}"
S["table_lines"] = [
    f"{S['n_feet']} feet, {S['n_patients']} patients",
    f"Type I: {S['I']['n']}",
    f"  beyond envelope: {S['I']['beyond']}",
    f"  within: {S['I']['within']}; unresolved: {S['I']['indet']}",
    f"  50% rule: resect alone {S['I']['rule_alone']}",
    f"  rule permissive: {S['I']['permissive']}",
    f"Type III: {S['III']['n']}",
    f"  within (no facet defect): {S['III']['within']}",
    f"Indeterminate: {S['indeterminate']['n']}",
    f"  within: {S['indeterminate']['within']}; beyond: {S['indeterminate']['beyond']}",
    f"  unresolved: {S['indeterminate']['indet']}",
    f"FE point twins (type I): {S['fe']['n']}",
    f"  beyond (FE): {S['fe']['n'] - S['fe']['within']}",
    f"  CoMech agrees: {S['fe']['agree']}/{S['fe']['n']}",
]
T["summary"] = S
json.dump(T, open("results/eval_ct_twins.json", "w"))
print(json.dumps({k: v for k, v in S.items() if k != "table_lines"}, indent=1))
print("\n".join(S["table_lines"]))
