"""Supplementary Tables 13-17: models and training, field fidelity, zero-shot benchmark, task performance by model,
FE-verified virtual cohort.  Reads results/numbers.json and results/eval_*.json."""
import os, sys, json, pickle, numpy as np


def fmt_p(p):
    """Exact P value, two significant figures; scientific notation below 0.001 (as in the main text)."""
    import math
    if p >= 0.001:
        return f"{p:.{max(1 - math.floor(math.log10(p)), 0)}f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper")
R = lambda n: json.load(open(os.path.join(ROOT, "results", n + ".json"))) if os.path.exists(os.path.join(ROOT, "results", n + ".json")) else None
N = R("numbers"); F = R("eval_fields"); ZS = R("eval_zeroshot"); TP = R("eval_tipping"); E = R("eval_equiv_plan"); CF = R("eval_cohort_fe")
MODELS = os.environ.get("COMECH_MODELS", os.path.join(ROOT, "results", "models_long"))
ORDER = ["comech_s0", "comech_s1", "comech_s2", "comech_noswap", "comech_nosens", "comech_none", "actcat", "deeponet", "geo", "fno"]
NAME = {"comech_s0": r"\CoMech, seed 0", "comech_s1": r"\CoMech, seed 1", "comech_s2": r"\CoMech, seed 2",
        "comech_noswap": r"\quad $-$ effect-swap", "comech_nosens": r"\quad $-$ action sensitivity", "comech_none": r"\quad $-$ both",
        "actcat": "Action-concatenated", "deeponet": "DeepONet", "geo": r"Geometry-to-field$^*$", "fno": r"FNO$^*$"}
sys.path.insert(0, ROOT)
try:                                    # number of latent tokens of the analysis grid (912 parametric, 1,320 real-bone; Claude, 2026-11-03)
    from ml.data import TOK as _TOK
    NTOK = int(np.prod(_TOK))
except Exception:
    NTOK = 912
ARCH = {"comech": r"Mixer state encoder (2 blocks, " + f"{NTOK:,}" + r" tokens $\times$ 64); action MLP; FiLM-modulated mixer transition (2 blocks); neural-field decoder; effect head",
        "actcat": r"As \CoMech, but the action is concatenated to every patch and modulates every encoder block; no transition",
        "deeponet": r"Branch MLP on patch-averaged pre-operative geometry and action; trunk MLP on Fourier features and local material",
        "geo": r"As the action-concatenated model, but given the post-operative voxel geometry and load instead of the action",
        "fno": r"Fourier neural operator on the full grid: post-operative geometry, coordinates and load; 3 spectral layers (width 16; $\pm$6/4/4 modes)"}
avail = [m for m in ORDER if os.path.exists(os.path.join(MODELS, m + ".pkl"))]
RB = os.path.isdir(os.path.join(ROOT, "rb"))
# real-bone packs carry only the CoMech seeds' weights; the other models' size and training time come from
# results/models_meta.json (server task s31_models_meta) when their pickles are absent (Claude, 2026-11-01)
META = (R("models_meta") or {}).get("models", {})
avail_meta = [m for m in ORDER if m not in avail and m in META and "npar" in META[m]]
o = []
# ---------------- S13 models and training ----------------
o.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{Models and training.} All models were trained on the same batches (four feet $\times$ five actions; 1,024 query cells per transition, half from the articular and surgical region) with Adam (gradient clipping at 1) """ + (r"""on the CPUs of the server, one process per model (two threads; DeepONet one, the geometry-to-field model three, FNO twenty)""" if RB else r"""on one CPU core""") + r""": 1,600 steps with a 100-step warm-up and cosine decay from $10^{-3}$, then 2,400 steps with a warm restart of the same schedule from $5\times10^{-4}$. """ + (r"""All ten models were trained at the same time, so the training times are indicative.""" if RB else r"""Training times were measured on a shared two-core machine and are indicative.""") + r""" $^*$Given the explicit post-operative geometry instead of the action.}
\label{stab:mlmodels}
\footnotesize
\begin{tabular}{p{3.3cm}p{8.2cm}rrr}
\toprule
Model & Architecture & Parameters & Steps & Hours\\
\midrule
""")
for m in [x for x in ORDER if x in avail or x in avail_meta]:
    if m in avail:
        d = pickle.load(open(os.path.join(MODELS, m + ".pkl"), "rb"))
        kind = d["kind"]; npar = int(sum(v.size for v in d["state"].values()))
    else:
        d = dict(kind=META[m]["kind"], cfg=META[m]["cfg"]); kind = d["kind"]; npar = META[m]["npar"]
    arch = ARCH[kind] if m in ("comech_s0", "actcat", "deeponet", "geo", "fno") else (r"as seed 0" if m.startswith("comech_s") else r"as \CoMech, without the term")
    o.append(f"{NAME[m]} & {arch} & {npar:,} & {d['cfg']['steps']:,} & {d['cfg']['train_time'] / 3600:.1f}\\\\\n")
eps = N["eps"]
o.append(r"""\midrule
\multicolumn{5}{p{15.5cm}}{Losses: \CoMech, $\mathcal{L}_{\mathrm{field}}+\mathcal{L}_{\mathrm{eff}}+\mathcal{L}_{\mathrm{roll}}+\mathcal{L}_{\mathrm{swap}}+\mathcal{L}_{\mathrm{sens}}$ ($\epsilon$ = """ + f"{eps:.3f}" + r""", the 12th percentile of within-foot effect distances; $\rho=0.25$; $\mu=\epsilon/2$); ablations drop $\mathcal{L}_{\mathrm{swap}}$, $\mathcal{L}_{\mathrm{sens}}$ or both; baselines, $\mathcal{L}_{\mathrm{field}}+\mathcal{L}_{\mathrm{eff}}$.}\\
\bottomrule
\end{tabular}
\end{table}

""")
# ---------------- S14 fidelity ----------------
KEYS = [("stress_l2", "Stress"), ("stress_l2_art", "Cartilage"), ("disp_l2", "Displ."), ("hot_iou", "IoU"),
        ("mid_peak_err", "Mid peak"), ("post_peak_err", "Post peak"), ("eff_rmse", "Effects")]
o.append(r"""\begin{landscape}
\begin{table}[htbp]
\centering
\caption{\textbf{Field and effect fidelity.} Medians [interquartile ranges] over all solutions of 40 held-out feet (320) and 20 out-of-range feet (160). Stress, cartilage and displacement: relative $L_2$ error of von Mises stress over all occupied cells and over articular cartilage, and of displacement (calcaneus-fixed frame); IoU, intersection-over-union of the top 5\% of cartilage stress; mid and post peak: relative error of the 99th-percentile stress of the residual middle and posterior facets; effects: root-mean-square error of the 17 standardized effect descriptors. $P$, two-sided Wilcoxon signed-rank test of cartilage error against \CoMech seed 0 on paired solutions. $^*$Given the explicit post-operative geometry instead of the action.}
\label{stab:fidelity}
\scriptsize
\begin{tabular}{ll""" + "r" * len(KEYS) + r"""r}
\toprule
Split & Model & """ + " & ".join(k[1] for k in KEYS) + r""" & $P$\\
\midrule
""")
RES_MODELS = [m for m in ORDER if m in avail or m in avail_meta or not avail_meta and m in avail]   # models with results
for sp, spn in [("test", "Held-out"), ("ood", "Out-of-range")]:
    first = True
    for m in RES_MODELS:
        if f"{m}|{sp}" not in F:
            continue
        rows = F[f"{m}|{sp}"]
        cells = []
        for k, _ in KEYS:
            v = np.array([r[k] for r in rows if k in r and r[k] == r[k]])
            cells.append(f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}--{np.percentile(v, 75):.2f}]")
        t = N["fidelity_tests"].get(sp, {}).get(m, {}).get("stress_l2_art")
        ptxt = "--" if t is None else fmt_p(t["p"])
        o.append(f"{spn if first else ''} & {NAME[m]} & " + " & ".join(cells) + f" & {ptxt}\\\\\n")
        first = False
    o.append(r"\midrule" + "\n" if sp == "test" else "")
o.append(r"""\bottomrule
\end{tabular}
\end{table}
\end{landscape}

""")
# ---------------- S15 zero-shot ----------------
o.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{Zero-shot atlas analogue on the engine's reference foot} (never used in training; Methods). $\tau$, Kendall's rank correlation of bridge stress (99th percentile) among the nine coalitions with the engine's FE; pairs, concordant pairwise orderings among anatomy-based types (6 pairs) and Rozansky types (10 pairs) against the engine (FE) and the atlas; ratio, median predicted/FE bridge stress; signs, agreement of the 48 alignment effects with the engine and of 36 with the atlas (its type III and IV valgus models, whose solver input contained material errors, are excluded; Methods); fusion, complete fusion below every partial coalition; tolerance, tolerable defect from the effect head (engine FE: middle """ +
         f"{100 * ZS['comech_s0']['scores']['tol_mid_fe']:.0f}\\%, posterior {100 * ZS['comech_s0']['scores']['tol_post_fe']:.0f}\\%" + r"""). $^*$Given the post-operative geometry.}
\label{stab:zeroshot}
\scriptsize
\begin{tabular}{lrcccrcccrrr}
\toprule
 & & \multicolumn{2}{c}{Pairs vs FE} & \multicolumn{2}{c}{Pairs vs atlas} & & \multicolumn{2}{c}{Signs} & & \multicolumn{2}{c}{Tolerance (\%)} \\
Model & $\tau$ & types & Roz. & types & Roz. & ratio & FE & atlas & fusion & mid & post \\
\midrule
""")
COALS = ["Roz I", "Roz II", "Roz III", "Roz IV", "Roz V", "Type I", "Type II", "Type III", "Type IV"]
sys.path.insert(0, ROOT)
from exp.fe_atlas import ALIGN
BAD = {("Type III", "valgus10"), ("Type IV", "valgus10")}     # atlas models with material errors (2026-10-31; Methods)
REGS = [("ant_p99", "ant_max"), ("mid_p99", "mid_max"), ("post_p99", "post_max"), ("tn_p99", "tn_max"), ("cc_p99", "cc_max"),
        ("coal_p99", "tc_max")]


def signs_vs_atlas(Rr):
    """Sign agreement of the predicted alignment effects with the atlas, as in exp/e_zeroshot.py, without the BAD models."""
    ok = tot = 0
    for t in ["Type I", "Type II", "Type III", "Type IV"]:
        for j, pos in [(1, "varus10"), (2, "valgus10")]:
            if (t, pos) in BAD:
                continue
            for ke, ka in REGS:
                dp = Rr[f"{t} {pos}"]["pred"][ke] - Rr[t]["pred"][ke]
                da = ALIGN[t][ka][j] - ALIGN[t][ka][0]
                ok += int(np.sign(dp) == np.sign(da)); tot += 1
    return ok, tot


for m in RES_MODELS:
    if m not in ZS:
        continue
    s = ZS[m]["scores"]; Rr = ZS[m]["rows"]
    # coalitions whose FE bridge descriptor is zero (no cell with >= 25% coalition; Rozansky III in the real-bone
    # engine) have no ratio (Claude, 2026-11-01)
    ratio = np.median([Rr[c]["pred"]["coal_p99"] / Rr[c]["fe"]["coal_p99"] for c in COALS if Rr[c]["fe"]["coal_p99"] > 0])
    o.append(f"{NAME[m]} & {s['tau_all_vs_fe']:.2f} & {s['types_vs_fe'][0]}/6 & {s['roz_vs_fe'][0]}/10 & {s['types_vs_atlas'][0]}/6 & "
             f"{s['roz_vs_atlas'][0]}/10 & {ratio:.2f} & {s['align_vs_fe'][0]}/48 & {'%d/%d' % signs_vs_atlas(Rr)} & "
             f"{'yes' if s['fusion_below_all'] else 'no'} & {100 * s['tol_mid_effhead']:.0f} & {100 * s['tol_post_effhead']:.0f}\\\\\n")
o.append(r"""\bottomrule
\end{tabular}
\end{table}

""")
# ---------------- S16 tasks by model ----------------
o.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{Tolerance, equivalence and planning by model.} Tolerance: mean absolute error (MAE) and bias in percentage points and Spearman $\rho$ against FE sweeps on 20 held-out feet (assigning each foot the mean of the others gives """ +
         f"{N['tipping_const']['post']['mae']:.1f} points for the posterior and {N['tipping_const']['mid']['mae']:.1f} for the middle facet" + r"""). Equivalence: precision and recall of predicted mechanically equivalent action pairs on held-out feet (""" +
         f"{E['equivalence']['comech_s0']['n_true']} of {E['equivalence']['comech_s0']['n_pairs']}" + r""" pairs equivalent by FE). Planning: regret of the chosen adjunct as a fraction of the FE range over ten candidates on 16 held-out coalition feet (median, mean) and the number of feet in which the FE-optimal adjunct was chosen. $^*$Given the post-operative geometry.}
\label{stab:tasks}
\scriptsize
\begin{tabular}{lrrrrrrrrrrr}
\toprule
 & \multicolumn{3}{c}{Posterior tolerance} & \multicolumn{3}{c}{Middle tolerance} & \multicolumn{2}{c}{Equivalence} & \multicolumn{3}{c}{Planning}\\
Model & MAE & bias & $\rho$ & MAE & bias & $\rho$ & prec. & recall & median & mean & FE-optimal\\
\midrule
""")
for m in RES_MODELS:
    if m not in N["tipping"]:
        continue
    t = N["tipping"][m]; eq = E["equivalence"].get(m); pl = N["planning"].get(m)
    f = lambda x: "--" if x != x else f"{x:.2f}"
    o.append(f"{NAME[m]} & {t['mae_post']:.1f} & {t['bias_post']:+.1f} & {f(t['spearman_post'])} & {t['mae_mid']:.1f} & {t['bias_mid']:+.1f} & "
             f"{f(t['spearman_mid'])} & " + (f"{eq['precision']:.2f} & {eq['recall']:.2f}" if eq else "-- & --") + " & " +
             (f"{pl['regret']['median']:.3f} & {pl['regret']['mean']:.3f} & {pl['top1']}/{pl['n']}" if pl else "-- & -- & --") + "\\\\\n")
ref = N["planning_ref"]
o.append(r"\midrule" + "\n" + f"Random adjunct & & & & & & & & & {ref['random']['median']:.3f} & {ref['random']['mean']:.3f} & --\\\\\n")
o.append(f"Best fixed adjunct & & & & & & & & & {ref['best_fixed']['median']:.3f} & {ref['best_fixed']['mean']:.3f} & --\\\\\n")
o.append(r"""\bottomrule
\end{tabular}
\end{table}

""")
# ---------------- S17 virtual cohort ----------------
if CF:
    S = CF["summary"]
    o.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{FE-verified envelopes of the virtual cohort and the 50\% rule.} For each foot the coalition was resected completely and the effective defect of each facet measured; tolerances were solved by FE on the coalition-free twin (defects 0--70\% in steps of 10\%) for the facets that a coalition of that type involves. Primary threshold: residual-facet mean stress +20\% over the defect-free joint. Calibrated thresholds: chosen so that the engine's reference foot crosses them at the atlas tolerances (38\% middle, 56\% posterior): """ +
             f"middle {100 * S['tau_cal']['mid']:.0f}\\%, posterior {100 * S['tau_cal']['post']:.0f}\\%" + r""". The 50\% rule recommends resection alone when the coalition is smaller than half of the posterior facet; permissive, rule recommends resection alone although the defect exceeds the tolerance; strict, the reverse.}
\label{stab:cohort}
\footnotesize
\begin{tabular}{lrrrrrrr}
\toprule
 & & \multicolumn{3}{c}{Primary threshold} & \multicolumn{3}{c}{Calibrated thresholds}\\
Type & Feet & within & permissive & strict & within & permissive & strict\\
\midrule
""")
    for t in ["I", "II", "III", "IV"]:
        a = S["fe_within"]["by_type"][t]; b = S["fe_within_cal"]["by_type"][t]
        o.append(f"{t} & {a['n']} & {a['within']} & {a['permissive']} & {a['strict']} & {b['within']} & {b['permissive']} & {b['strict']}\\\\\n")
    a = S["fe_within"]; b = S["fe_within_cal"]
    o.append(r"\midrule" + "\n" + f"All & {S['n']} & {a['within']} & {a['mis_permissive']} & {a['mis_strict']} & {b['within']} & {b['mis_permissive']} & {b['mis_strict']}\\\\\n")
    tp_, tm_ = S.get("tol_post"), S.get("tol_mid")
    o.append(r"""\midrule
\multicolumn{8}{p{12cm}}{FE tolerance: posterior facet """ + (f"{tp_['fe_range'][0]:.0f}--{tp_['fe_range'][1]:.0f}\\% (median {tp_['fe_median']:.0f}\\%; calibrated {tp_['fe_cal_range'][0]:.0f}--{tp_['fe_cal_range'][1]:.0f}\\%)" if tp_ else "") +
             r"""; middle facet """ + (f"{tm_['fe_range'][0]:.0f}--{tm_['fe_range'][1]:.0f}\\% (median {tm_['fe_median']:.0f}\\%, {tm_['fe_censored']} censored at 70\\%)" if tm_ else "") +
             r""". \CoMech ensemble against FE (primary threshold): """ + f"{S['pred_vs_fe']['agree']} of {S['n']} feet classified alike; {S['pred_vs_fe']['beyond_detected']} of {S['pred_vs_fe']['beyond_fe']} feet beyond the envelope identified; {S['pred_vs_fe']['false_beyond']} placed beyond wrongly" +
             (f"; posterior tolerance MAE {tp_['pred_mae']:.1f} points (cohort median {tp_['const_mae']:.1f})" if tp_ else "") + r""".}\\
\bottomrule
\end{tabular}
\end{table}
""")
open(os.path.join(OUTDIR, "supp_tables_models.tex"), "w").write("".join(o))
print("ok", avail)
