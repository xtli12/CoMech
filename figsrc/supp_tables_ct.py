"""Supplementary Table 18: preliminary CT readings and twin results per foot."""
import os, sys, json, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = json.load(open(f"{ROOT}/results/eval_ct_twins.json")); F = json.load(open(f"{ROOT}/results/eval_ct_twins_fe.json"))
fe = {r["foot"]: r for r in F["rows"]}
R = {r["foot"]: r for r in csv.DictReader(open(f"{ROOT}/data/ct/ct_readings.csv", encoding="utf-8-sig"))}
TX = {"resection": "R", "resection + medialising calcaneal osteotomy": "R+MO", "resection + lateralising calcaneal osteotomy": "R+LO",
      "conservative": "C", "conservative (awaiting surgery)": "C", "awaiting surgery": "C"}
PL = lambda p: "coronal" if p.startswith("coronal") else ("transverse" if p.startswith("native transverse") else "pseudo-cor.")
pc = lambda x: "--" if x is None else f"{100 * x:.0f}"


def esc(s):
    return s.replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")


def main(out):
    L = []
    L.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{CT-informed twins of 39 feet: preliminary readings and envelope status.} Readings were made on PACS JPEG exports with an AI-assisted protocol and are pending clinician adjudication (Methods and Supplementary Note~5). Plane: native coronal, native transverse or pseudo-coronal reformat. Type: anatomy-based (I middle facet, III extra-articular posteromedial; ? indeterminate); Roz., Rozansky morphology among type I. Cov., middle-facet coverage read from CT. Conf., reading confidence (M medium, L low). Tx, treatment received (R resection; MO/LO medializing/lateralizing calcaneal osteotomy; C conservative or awaiting surgery). Ratio, coalition-to-posterior-facet area ratio of the twins (median). P(within), share of """ + str(T.get("K", 24)) + r""" twins whose complete resection stays within the envelope (primary threshold; calibrated threshold in parentheses); P(rule), share for which the 50\% rule recommends resection alone. $d$ and $d^\star_{\mathrm{FE}}$, middle-facet defect and FE tolerance of the point twin (type I only).}
\label{stab:ct}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llllrrlllrrrr}
\toprule
Foot & Plane & Type & Roz. & Cov. & Ratio & Conf. & Tx & Deformity & P(within) & P(rule) & $d$ & $d^\star_{\mathrm{FE}}$\\
 & & & & (\%) & (\%) & & & & (\%) & (\%) & (\%) & (\%)\\
\midrule""")
    order = sorted(T["rows"], key=lambda r: r["foot"])
    for r in order:
        c = R[r["foot"]]; f = fe.get(r["foot"])
        typ = {"I": "I", "III": "III", "indeterminate": "?"}[r["type"]]
        L.append(" & ".join([r["foot"], PL(c["plane"]), typ, (esc(c["rozansky"]) or "--") if r["type"] == "I" else "--", pc(float(c["cov_mid"])) if c["cov_mid"] and r["type"] == "I" else "--",
                             f"{100 * r['ratio'][1]:.0f}", r["conf"], TX.get(r["treatment"], r["treatment"]),
                             esc(r["deformity"].replace(" (side in record unclear)", "$^a$")) or "--",
                             f"{100 * r['p_within']:.0f} ({100 * r['p_within_cal']:.0f})", f"{100 * r['p_rule_ok']:.0f}",
                             pc(f["d_mid"]) if f else "--", pc(f["fe_tol"]) if f else "--"]) + r"\\")
    L.append(r"""\bottomrule
\end{tabular}

\raggedright\scriptsize $^a$Side of the recorded varus unclear in the clinical record. Indeterminate feet draw their coalition type from the composition of the three-center cohort, so their P(within) reflects that composition rather than their CT.
\end{table}""")
    open(out, "w").write("\n".join(L) + "\n")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/paper/supp_tables_ct.tex")
