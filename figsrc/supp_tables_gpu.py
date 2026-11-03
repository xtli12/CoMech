"""Supplementary Tables 19-22: the GPU scaling experiment (models and training; field fidelity; task performance;
comparison with the twofold-refined FE).  Reads the unpacked results_gpu_<date>.zip and the paper's results/.

    python figsrc/supp_tables_gpu.py [results_gpu_dir] [out_dir]      ->  supp_tables_gpu.tex
"""
import os, sys, json, numpy as np


def fmt_p(p):
    """Exact P value, two significant figures; scientific notation below 0.001 (as in the main text)."""
    import math
    if p >= 0.001:
        return f"{p:.{max(1 - math.floor(math.log10(p)), 0)}f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_gpu")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "paper")
RG = lambda n: json.load(open(os.path.join(G, "results", n + ".json"))) if os.path.exists(os.path.join(G, "results", n + ".json")) else None
RP = lambda n: json.load(open(os.path.join(ROOT, "results", n + ".json")))
NG = RG("numbers_gpu"); FG = RG("eval_fields") or {}; RE = RG("eval_refined_effects"); NP = RP("numbers")
ST = None
for p in [os.path.join(G, "logs", "selftest.json")]:
    if os.path.exists(p):
        ST = json.load(open(p))

ORDER = ["comechL_s0", "comechL_s1", "comechL_s2", "comechL_noswap", "comechL_nosens", "comechL_none", "actcatL", "gino",
         "geofno", "comechL_s0_n900", "comechL_s0_n450", "comechL_s0_n180", "actcatL_n900", "actcatL_n450", "actcatL_n180",
         "gino_n180", "geofno_n180"]
NAME = {"comechL_s0": r"\CoMech-L, seed 0", "comechL_s1": r"\CoMech-L, seed 1", "comechL_s2": r"\CoMech-L, seed 2",
        "comechL_noswap": r"\quad $-$ effect-swap", "comechL_nosens": r"\quad $-$ action sensitivity",
        "comechL_none": r"\quad $-$ both", "actcatL": "Action-concatenated-L", "gino": r"GINO$^*$", "geofno": r"Geo-FNO$^*$"}
for base, lab in [("comechL_s0", r"\CoMech-L, seed 0"), ("actcatL", "Action-concatenated-L"), ("gino", r"GINO$^*$"),
                  ("geofno", r"Geo-FNO$^*$")]:
    for n in (180, 450, 900):
        NAME[f"{base}_n{n}"] = lab
CPU = {"comech_s0": r"\CoMech, seed 0", "comech_s1": r"\CoMech, seed 1", "comech_s2": r"\CoMech, seed 2",
       "actcat": "Action-concatenated", "deeponet": "DeepONet", "geo": r"Geometry-to-field$^*$", "fno": r"FNO$^*$"}
CFG = {}
for n in ORDER:
    p = os.path.join(G, "results", "models_gpu", n + ".json")
    if os.path.exists(p):
        CFG[n] = json.load(open(p))
avail = [n for n in ORDER if n in CFG]
RESUMED = {}                                   # runs that were interrupted once and resumed from a checkpoint
for n in avail:
    lp = os.path.join(G, "logs", f"train_{n}.log")
    if os.path.exists(lp):
        for line in open(lp, errors="replace"):
            if line.startswith("resumed "):
                RESUMED[n] = int(line.split()[-1])
feet = lambda n: CFG[n].get("feet")
kind = lambda n: CFG[n].get("kind")
o = []

# ---------------- models and training ----------------
step_s = {}
if ST:
    for k in ("comech", "actcat", "gino", "geofno"):
        r = ST.get(f"full_batch_step_{k}") or {}
        if isinstance(r, dict) and "step_s" in r:
            step_s[k] = (r["step_s"], r.get("peak_gb"))
# hardware and interruptions of this run (Claude, 2026-11-03): the paper's A100 runs shared one GPU three at a time;
# the real-bone runs had one RTX 4090 each (four at a time) and none was interrupted
_lat = RG("eval_latency_gpu") or {}
_dev = _lat.get("device", "")
if os.path.isdir(os.path.join(ROOT, "rb")) and _dev:
    _mem = " (24 GB)" if "4090" in _dev else ""
    HW = (f"Step time was measured by the self-test before training, for one model running alone on one {_dev}{_mem}; "
          "the runs themselves ran four at a time, one per GPU, sharing the processors and disk of one server, so their "
          "wall-clock hours are somewhat longer than step time multiplied by the number of steps.")
else:
    HW = ("Step time was measured for one model running alone on one NVIDIA A100 (80 GB); the runs themselves shared the "
          "GPU three at a time, so their wall-clock hours overstate the cost of a single run.")
RES = ((r" $^\dagger$Interrupted once when the pipeline was restarted and resumed from its last checkpoint (model and "
        "optimizer state; steps " + " and ".join(f"{v:,}" for v in RESUMED.values())
        + ", respectively); wall-clock time after resumption.") if RESUMED else "")
o.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{GPU scaling experiment: models and training.} All runs used the same pre-specified settings, fixed before any result was seen: 40,000 steps of AdamW (learning rate $5\times10^{-4}$ with a 1,000-step linear warm-up and cosine decay to 2\% of that value; weight decay 0.01; gradient clipping at 1), batches of 16 feet $\times$ 5 actions with 4,096 query cells per transition, and the losses of the paper (\CoMech-L: all five terms; ablations drop effect-swap, action sensitivity or both; baselines: field and effect losses). \CoMech-L and the action-concatenated model were trained with bfloat16 autocast, GINO and Geo-FNO in single precision with gradients accumulated over five micro-batches. The weights after the last step were evaluated; there was no early stopping, checkpoint selection or tuning on held-out data. Training sets are nested: the paper's 180 training feet, extended by the first 270, 720 or 1,620 of 1,620 additional feet drawn from the same distribution with a new seed (450, 900 and 1,800 feet). """ + HW + r""" $^*$Given the explicit post-operative geometry instead of the action.""" + RES + r"""}
\label{stab:gpumodels}
\footnotesize
\begin{tabular}{lrrrrr}
\toprule
Model & Training feet & Parameters & Steps & s per step (alone) & Wall-clock h (shared)\\
\midrule
""")
for n in avail:
    c = CFG[n]; ss = step_s.get(kind(n))
    dag = "$^\\dagger$" if n in RESUMED else ""
    o.append(f"{NAME[n]}{dag} & {feet(n):,} & {c['params']:,} & {c['steps']:,} & "
             + (f"{ss[0]:.3f}" if ss else "--") + f" & {c['train_time'] / 3600:.1f}\\\\\n")
o.append(r"""\bottomrule
\end{tabular}
\end{table}

""")

# ---------------- field fidelity ----------------
KEYS = [("stress_l2", "Stress"), ("stress_l2_art", "Cartilage"), ("disp_l2", "Displ."), ("hot_iou", "IoU"),
        ("mid_peak_err", "Mid peak"), ("post_peak_err", "Post peak"), ("eff_rmse", "Effects")]
TESTS = NG.get("tests", {})
o.append(r"""\begin{landscape}
\begin{table}[htbp]
\centering
\caption{\textbf{GPU scaling experiment: field and effect fidelity.} Medians [interquartile ranges] over all solutions of the 40 held-out feet (320) and the 20 out-of-range feet (160), with the metrics of Supplementary Table~\ref{stab:fidelity}. $P$, two-sided Wilcoxon signed-rank test of cartilage error against \CoMech-L seed 0 on paired solutions. $^*$Given the explicit post-operative geometry instead of the action.}
\label{stab:gpufidelity}
\scriptsize
\begin{tabular}{llr""" + "r" * len(KEYS) + r"""r}
\toprule
Split & Model & Feet & """ + " & ".join(k[1] for k in KEYS) + r""" & $P$\\
\midrule
""")
for sp, spn in [("test", "Held-out"), ("ood", "Out-of-range")]:
    first = True
    for n in avail:
        rows = FG.get(f"{n}|{sp}")
        if not rows:
            continue
        cells = []
        for k, _ in KEYS:
            v = np.array([r[k] for r in rows if k in r and r[k] == r[k]])
            cells.append(f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}--{np.percentile(v, 75):.2f}]" if len(v) else "--")
        t = TESTS.get(n) if sp == "test" else None
        ptxt = "--" if not t else fmt_p(t["p"])
        o.append(f"{spn if first else ''} & {NAME[n]} & {feet(n):,} & " + " & ".join(cells) + f" & {ptxt}\\\\\n")
        first = False
    o.append(r"\midrule" + "\n" if sp == "test" else "")
o.append(r"""\bottomrule
\end{tabular}
\end{table}
\end{landscape}

""")

# ---------------- task performance ----------------
TP = NG.get("tipping", {}); EQ = NG.get("equivalence", {}); PL = NG.get("planning", {})
T0 = RP("eval_tipping"); ids = sorted(T0["fe"], key=int)
cpu_tol = {}
for fac in ("post", "mid"):
    a = np.array([T0["fe"][i][fac]["tol"] for i in ids if fac in T0["fe"][i]]) * 100
    p = np.mean([[T0["models"][m]["pred"][i][fac]["tol"] for i in ids if fac in T0["fe"][i]]
                 for m in ["comech_s0", "comech_s1", "comech_s2"]], 0) * 100
    cpu_tol[fac] = dict(mae=float(np.abs(p - a).mean()), rho=float(spearmanr(p, a)[0]))
sg = lambda x: f"{x:.2f}".replace("-", "$-$")
tol = lambda d, fac: f"{d[fac]['mae']:.1f} ({sg(d[fac]['rho'])})" if d and fac in d else "--"
o.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{GPU scaling experiment: tolerance, equivalence and planning.} Tolerance: mean absolute error (percentage points) of the tolerable defect of the posterior and middle facets on 20 held-out feet against FE sweeps, with Spearman's $\rho$ in parentheses; the cohort-mean baseline assigns every foot the mean FE tolerance of the other feet. Equivalence: precision and recall of mechanically equivalent pairs among the 1,120 action pairs of the held-out feet. Planning: median regret of the chosen adjunct on 16 held-out coalition feet (fraction of the FE range) and the number of feet in which the FE-optimal adjunct was chosen. Ensembles average the tolerances of the three seeds. $^*$Given the explicit post-operative geometry instead of the action.}
\label{stab:gputasks}
\footnotesize
\begin{tabular}{lrrrrrr}
\toprule
Model & Feet & Posterior tolerance & Middle tolerance & Precision & Recall & Regret (hits)\\
\midrule
""")
o.append(f"Cohort-mean baseline & -- & {TP.get('loo_mae_post', float('nan')):.1f} & {TP.get('loo_mae_mid', float('nan')):.1f} & & & \\\\\n")
o.append(f"\\CoMech ensemble (CPU, paper) & 180 & {tol(cpu_tol, 'post')} & {tol(cpu_tol, 'mid')} & & & \\\\\n")
if "comechL_ens" in TP:
    o.append(f"\\CoMech-L ensemble & 1,800 & {tol(TP['comechL_ens'], 'post')} & {tol(TP['comechL_ens'], 'mid')} & & & \\\\\n")
o.append(r"\midrule" + "\n")
for n in avail:
    e = EQ.get(n, {}); pl = PL.get(n, {})
    o.append(f"{NAME[n]} & {feet(n):,} & {tol(TP.get(n), 'post')} & {tol(TP.get(n), 'mid')} & "
             + (f"{e['precision']:.2f} & {e['recall']:.2f}" if e else "-- & --") + " & "
             + (f"{100 * pl['regret_median']:.1f}\\% ({pl['top1']})" if pl else "--") + "\\\\\n")
o.append(r"""\bottomrule
\end{tabular}
\end{table}

""")

# ---------------- refined-grid comparison ----------------
RF = NG.get("refined", {})
RK = [("stress_l2", "Stress"), ("stress_l2_art", "Cartilage"), ("disp_l2", "Displ."), ("hot_iou", "IoU")]
o.append(r"""\begin{table}[htbp]
\centering
\caption{\textbf{Discretization error of the engine.} The 320 held-out configurations were re-solved """ + (r"""on 1.0-mm cells instead of 1.5~mm (about 3.4 times as many cells), sampled on an analysis grid refined twofold""" if os.path.isdir(os.path.join(ROOT, "rb")) else r"""on a grid refined twofold in every direction (about eight times as many cells)""") + r""", and the refined solution was restricted to the standard grid by averaging the eight child cells of every standard cell. Top, medians of the field metrics of the standard-grid FE and of each model against the refined solution and, for the models, against the standard-grid FE on which they were trained. Bottom, the 17 effect descriptors of the refined against the standard grid: median and interquartile range of the relative difference (refined minus standard, relative to standard) and Pearson's $r$ across configurations. $^*$Given the explicit post-operative geometry instead of the action.}
\label{stab:refined}
\footnotesize
\begin{tabular}{lr""" + "r" * (2 * len(RK)) + r"""}
\toprule
 & & \multicolumn{""" + str(len(RK)) + r"""}{c}{Against refined FE} & \multicolumn{""" + str(len(RK)) + r"""}{c}{Against standard-grid FE}\\
\cmidrule(lr){3-""" + str(2 + len(RK)) + r"""}\cmidrule(lr){""" + str(3 + len(RK)) + "-" + str(2 + 2 * len(RK)) + r"""}
Model & Feet & """ + " & ".join(k[1] for k in RK) + " & " + " & ".join(k[1] for k in RK) + r"""\\
\midrule
""")
if "fe_base" in RF:
    o.append("Standard-grid FE & & " + " & ".join(f"{RF['fe_base'][k]:.2f}" for k, _ in RK) + " & " + " & ".join("--" for _ in RK) + "\\\\\n")
for m, lab in list(CPU.items()) + [(n, NAME[n]) for n in avail]:
    if m not in RF:
        continue
    tag = r" (CPU)" if m in CPU else ""
    nf = "180" if m in CPU else f"{feet(m):,}"
    o.append(f"{lab}{tag} & {nf} & " + " & ".join(f"{RF[m]['vs_refined'][k]:.2f}" for k, _ in RK) + " & "
             + " & ".join(f"{RF[m]['vs_standard'][k]:.2f}" for k, _ in RK) + "\\\\\n")
o.append(r"""\bottomrule
\end{tabular}

\vspace{8pt}
\begin{tabular}{lrrr@{\hspace{2.2em}}lrrr}
\toprule
Descriptor & Median & IQR & $r$ & Descriptor & Median & IQR & $r$\\
\midrule
""")
if RE:
    rows = [r for f in RE["feet"] for r in f["actions"]]
    DESC = [("post_mean", "Posterior facet, mean"), ("mid_mean", "Middle facet, mean"), ("ant_mean", "Anterior facet, mean"),
            ("post_p99", "Posterior facet, peak"), ("mid_p99", "Middle facet, peak"), ("ant_p99", "Anterior facet, peak"),
            ("tn_p99", "Talonavicular, peak"), ("cc_p99", "Calcaneocuboid, peak"), ("tc_p99", "Talocalcaneal, peak"),
            ("share_post", "Posterior load share"), ("share_mid", "Middle load share"), ("share_ant", "Anterior load share"),
            ("heel_share", "Heel share"), ("talus_dz", "Talar vertical translation"), ("talus_dy", "Talar medial translation"),
            ("talus_evert", "Talar eversion"), ("nav_drop", "Navicular drop")]
    cells = []
    for e, lab in DESC:
        s = np.array([r["std"][e] for r in rows]); rf = np.array([r["ref"][e] for r in rows])
        ok = np.abs(s) > 1e-6 * np.abs(s).max()
        d = (rf - s)[ok] / np.abs(s)[ok] * 100
        q1, md, q3 = np.percentile(d, [25, 50, 75])
        r_ = np.corrcoef(s, rf)[0, 1] if s.std() > 0 and rf.std() > 0 else float("nan")
        pm = lambda x: f"{x:+.0f}".replace("-", "$-$")
        cells.append(f"{lab} & {pm(md)}\\% & {pm(q1)} to {pm(q3)}\\% & {r_:.2f}")
    half = (len(cells) + 1) // 2
    for i in range(half):
        right = cells[half + i] if half + i < len(cells) else " & & & "
        o.append(cells[i] + " & " + right + "\\\\\n")
o.append(r"""\bottomrule
\end{tabular}
\end{table}

""")
os.makedirs(OUTDIR, exist_ok=True)
open(os.path.join(OUTDIR, "supp_tables_gpu.tex"), "w").write("".join(o))
print("wrote", os.path.join(OUTDIR, "supp_tables_gpu.tex"))
