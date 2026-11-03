"""LaTeX tables for the open-source engine: convergence, sampling ranges, agreement with the atlas."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper")

conv = [json.loads(l) for l in open(os.path.join(ROOT, "results/convergence.log")) if l.startswith("{")]
cmp_ = json.load(open(os.path.join(ROOT, "results/atlas_compare.json")))
out = []
RB = os.path.isdir(os.path.join(ROOT, "rb"))     # real-bone engine (Claude, 2026-11-01): cell sizes instead of grid factors
ENG = "the data engine" if RB else "the open-source engine"

# ---------------- S9: convergence ----------------
keys = [("mid_mean", "Middle facet, mean"), ("post_mean", "Posterior facet, mean"), ("mid_p99", "Middle facet, peak"),
        ("post_p99", "Posterior facet, peak"), ("ant_mean", "Anterior facet, mean"), ("tn_p99", "Talonavicular, peak"),
        ("cc_p99", "Calcaneocuboid, peak"), ("tc_p99", "Talocalcaneal complex, peak")]
cases = [c for c in ["Normal", "Post 50", "Mid 30", "Type I"]]
avail = {(r["fine"], r["name"]): r for r in conv}
fines = sorted({r["fine"] for r in conv}, reverse=True)
out.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{Grid refinement of """ + ENG + r""".} Effect descriptors (MPa) of the reference foot """ + (r"""on cells of 1.5~mm (the engine's), 1.125~mm and 1.0~mm""" if RB else r"""on the base grid (spacing $\times$1) and on uniformly refined grids ($\times$0.75, $\times$0.5)""") + r"""; degrees of freedom in the header. Peaks are 99th percentiles. The """ + (r"""1.0-mm cells were""" if RB else r"""$\times$0.5 grid was""") + r""" solved for the intact joint and the 50\% posterior defect.}
\label{stab:conv}
\footnotesize
\begin{tabular}{l""" + "r" * (len(fines) * 2) + r"""}
\toprule
 & \multicolumn{""" + str(len(fines)) + r"""}{c}{Intact joint} & \multicolumn{""" + str(len(fines)) + r"""}{c}{50\% posterior defect}\\
Descriptor""" + ("".join(f" & {max(1.0, 1.5 * f):g}~mm" for f in fines) if RB else "".join(f" & $\\times${f:g}" for f in fines)) * 2 + r"""\\
DOF""" + "".join(f" & {avail[(f, 'Normal')]['ndof']:,}" if (f, "Normal") in avail else " & --" for f in fines) +
           "".join(f" & {avail[(f, 'Post 50')]['ndof']:,}" if (f, "Post 50") in avail else " & --" for f in fines) + r"""\\
\midrule
""")
for k, lab in keys:
    row = lab
    for case in ["Normal", "Post 50"]:
        for f in fines:
            row += f" & {avail[(f, case)]['eff'][k]:.3f}" if (f, case) in avail else " & --"
    out.append(row + r"\\" + "\n")
# relative effect of the defect
row = r"\midrule Posterior mean, defect/intact"
for f in fines:
    row += " & " if True else ""
row = r"\midrule \multicolumn{" + str(1 + len(fines)) + r"}{l}{Posterior-facet mean, 50\% defect relative to intact}"
for f in fines:
    if (f, "Normal") in avail and (f, "Post 50") in avail:
        row += f" & {avail[(f, 'Post 50')]['eff']['post_mean'] / avail[(f, 'Normal')]['eff']['post_mean']:.2f}"
    else:
        row += " & --"
out.append(row + r"\\" + "\n")
out.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- S10: sampling ranges ----------------
out.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{Virtual-foot parameters of TCC-MechBank.} Ranges are uniform; reference values are those of the atlas-analogue reference foot. Out-of-range feet differ only in scale and alignment.}
\label{stab:ranges}
\footnotesize
\begin{tabular}{llll}
\toprule
Parameter & Reference & Training, validation and test & Out-of-range\\
\midrule
Isotropic scale & 1.00 & 0.86--1.04 & 0.80--0.86 or 1.04--1.08\\
Posterior-facet semi-axes (mm) & 12.0, 9.5 & $\times$0.85--1.15 each & as training\\
Middle-facet semi-axes (mm) & 6.5, 4.5 & $\times$0.8--1.2 each & as training\\
Anterior-facet semi-axes (mm) & 4.5, 4.0 & $\times$0.8--1.2 each & as training\\
Anterior facet & separate & separate 60\%, confluent 25\%, absent 15\% & as training\\
Sustentaculum position $y$ (mm) & 11.0 & 9.5--12.5 & as training\\
Posterior-facet convexity (mm) & 1.5 & 0.5--2.5 & as training\\
Cartilage thickness (mm) & 2.0 & 1.6--2.4 & as training\\
Hindfoot alignment ($^\circ$) & 0 & $-12$ to 12 & $\pm$12--16\\
Coalition growth direction ($^\circ$) & 0 & $-20$ to 20 & as training\\
\midrule
Coalition type & -- & none 15\%; types I--IV: equal mixture of & as training\\
 & & cohort proportions (44/15/69/7) and uniform & \\
Rozansky morphology (type I) & -- & equal mixture of 20/18/7/3 and uniform & as training\\
Middle-facet coverage & -- & I--III: 0.2--0.6; IV: 0.5--1.0; type IV: 0.3--0.8 & as training\\
Posterior-facet coverage & -- & type II: 0.15--0.6; type IV: 0.15--0.5 & as training\\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- S11: agreement with the atlas ----------------
c = cmp_
def pairs(t):
    return f"{t[1]}/{t[2]}"
# alignment signs without the atlas's type III and IV valgus models, whose solver input contained material errors
# (found 2026-10-31 when the atlas was re-derived from its files; Methods); same rows as exp/atlas_compare.py
BAD = {("Type III", "valgus10"), ("Type IV", "valgus10")}
AR = [r for r in c["align_rows"] if (r["type"], r["pos"]) not in BAD]
def agree(region=None):
    rr = [r for r in AR if region is None or r["region"] == region]
    return f"{sum(r['agree'] for r in rr)}/{len(rr)}", len(rr)
post_s, n_post = agree("post_max"); all_s, n_all = agree(); tc_s, n_tc = agree("tc_max")
rows = [
    ("Stress concentrated at every partial bridge (bridge/intact)",
     f"{c['bridge_ratio_atlas'][0]:.1f}--{c['bridge_ratio_atlas'][1]:.1f}$\\times$",
     f"{c['bridge_ratio_engine'][0]:.1f}--{c['bridge_ratio_engine'][1]:.1f}$\\times$", "yes"),
    ("Complete fusion below every partial coalition (talocalcaneal complex)", "yes", "yes" if c["fusion_below_all_engine"] else "no",
     "yes" if c["fusion_below_all_engine"] else "no"),
    ("Rozansky type I carries the highest bridge stress", "yes", "yes" if c["roz1_highest_engine"] else "no",
     "yes" if c["roz1_highest_engine"] else "no"),
    ("Ordering of Rozansky types I--V (concordant pairs)", "--", pairs(c["tau_roz"]), "partly"),
    ("Ordering of anatomy-based types I--IV (concordant pairs)", "--", pairs(c["tau_types"]), "no"),
    ("Anatomy-based types more variable than Rozansky I--IV (anterior, middle, posterior, TN)", "4/4",
     f"{sum(a > b for a, b in zip(c['cv_types_engine'], c['cv_roz14_engine']))}/4", "partly"),
    ("Valgus loads, varus unloads the posterior facet (signs)", f"{n_post}/{n_post}", post_s, "yes"),
    (f"All {n_all} alignment effects (signs)", f"{n_all}/{n_all}", all_s, "partly"),
    ("Alignment effects on bridge stress (signs)", f"{n_tc}/{n_tc}", tc_s, "no"),
    ("Posterior defect: residual-facet mean rises monotonically and accelerates",
     "+15\\% at 50\\%, +33\\% at 70\\%",
     f"+{c['post_defect_engine']['post_mean'][5]:.0f}\\% at 50\\%, +{c['post_defect_engine']['post_mean'][7]:.0f}\\% at 70\\%",
     "yes (larger)"),
    ("Middle defect: load spills to the anterior facet",
     f"+{c['mid_defect_atlas']['ant_max'][7]:.0f}\\% at 70\\%", f"+{c['mid_defect_engine']['ant_p99'][7]:.0f}\\% at 70\\%",
     "yes (larger)"),
    ("Middle defect: residual-facet peak maximal at 30\\%, then falls", "yes", "no (monotonic)", "no"),
]
if RB:      # real-bone engine: every "reproduced" judgment from the numbers (the parametric table above is unchanged)
    def frac_word(n, d):
        f = n / d
        return "yes" if f >= 0.999 else "mostly" if f >= 0.67 else "partly" if f >= 0.4 else "no"
    ip, ia = c["post_defect_engine"]["post_mean"], c["post_defect_atlas"]["post_mean"]
    mono = all(b > a for a, b in zip(ip[2:-1], ip[3:])); accel = (ip[7] - ip[5]) > (ip[5] - ip[3])
    r50, r70 = ip[5] / max(ia[5], 1e-9), ip[7] / max(ia[7], 1e-9)
    size = " (larger)" if min(r50, r70) > 1.5 else " (smaller)" if max(r50, r70) < 0.67 else ""
    sp_e, sp_a = c["mid_defect_engine"]["ant_p99"][7], c["mid_defect_atlas"]["ant_max"][7]
    mp = c["mid_defect_engine"]["mid_p99"]
    peak30 = int(np.argmax(mp)) == 3 and mp[-1] < mp[3]
    mono_mp = all(b >= a - 0.5 for a, b in zip(mp[:-1], mp[1:]))
    J = {
        "Ordering of Rozansky types I--V (concordant pairs)": frac_word(c["tau_roz"][1], c["tau_roz"][2]),
        "Ordering of anatomy-based types I--IV (concordant pairs)": frac_word(c["tau_types"][1], c["tau_types"][2]),
        "Anatomy-based types more variable than Rozansky I--IV (anterior, middle, posterior, TN)":
            frac_word(sum(a > b for a, b in zip(c["cv_types_engine"], c["cv_roz14_engine"])), 4),
        "Valgus loads, varus unloads the posterior facet (signs)": frac_word(*map(int, post_s.split("/"))),
        f"All {n_all} alignment effects (signs)": frac_word(*map(int, all_s.split("/"))),
        "Alignment effects on bridge stress (signs)": frac_word(*map(int, tc_s.split("/"))),
        "Posterior defect: residual-facet mean rises monotonically and accelerates": ("yes" + size) if (mono and accel) else "no",
        "Middle defect: load spills to the anterior facet":
            ("yes" + (" (larger)" if sp_e > 1.5 * sp_a else "")) if sp_e >= 0.25 * sp_a else "no (smaller)",
    }
    rows = [(a, b, ("yes" if peak30 else "no (monotonic)" if mono_mp else "no") if a.startswith("Middle defect: residual-facet peak")
             else e, J.get(a, d) if not a.startswith("Middle defect: residual-facet peak") else ("yes" if peak30 else "no"))
            for a, b, e, d in rows]
out.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{Qualitative laws of the atlas reproduced by """ + ENG + r""".} Atlas-analogue series of the reference foot (11 coalition, 8 alignment and 14 defect configurations) compared with the Ansys atlas. Engine peaks are 99th percentiles; atlas peaks are maxima. Alignment signs exclude the type III and IV valgus models of the atlas, whose solver input contained material errors (Methods).}
\label{stab:agree}
\footnotesize
\begin{tabular}{p{7.2cm}p{2.5cm}p{2.7cm}l}
\toprule
Law & Atlas & Engine & Reproduced\\
\midrule
""")
for r in rows:
    out.append(" & ".join(r) + r"\\" + "\n")
out.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
open(os.path.join(OUTDIR, "supp_tables_engine.tex"), "w").write("".join(out))
print("".join(out)[:3000])
