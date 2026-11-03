"""E3a: zero-shot reproduction of the atlas-analog series on the reference foot (never used in training)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.common import *
from exp.scenarios import atlas_scenarios
from exp.fe_atlas import CLS_MODELS, CLS, ALIGN
from ml.data import Split, Norm
from ml.evaluate import field_metrics
from scipy.stats import kendalltau

COALS = ["Roz I", "Roz II", "Roz III", "Roz IV", "Roz V", "Type I", "Type II", "Type III", "Type IV"]
TYPES = ["Type I", "Type II", "Type III", "Type IV"]
ROZ = ["Roz I", "Roz II", "Roz III", "Roz IV", "Roz V"]


def region_stats(vm, lab, cells, frac):
    """Same definitions as the engine's effect descriptors: P99 over cells of the region and mean weighted by
    cell volume x material fraction."""
    import fe.geometry as FG
    band = FG.subtalar_band(GRID.cell_xyz[cells])
    vol = GRID.cell_vol[cells]
    out = {}
    for n, L in [("ant", C_ANT), ("mid", C_MID), ("post", C_POST), ("tn", C_TN), ("cc", C_CC), ("coal", COAL)]:
        m = lab == L
        w = vol[m] * frac[cells][m, L]
        out[n + "_p99"] = float(np.percentile(vm[m], 99)) if m.sum() > 3 else 0.0
        out[n + "_mean"] = float((vm[m] * w).sum() / w.sum()) if m.sum() > 3 else 0.0
    out["tc_p99"] = float(np.percentile(vm[band], 99))
    return out


def pairs_agree(a, b, names):
    c = t = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t += 1; c += int((a[names[i]] - a[names[j]]) * (b[names[i]] - b[names[j]]) > 0)
    return c, t


def main(names):
    norm = Norm.cached()
    sc = atlas_scenarios()
    ref_fr = geo_of(Foot(), Action(resect=1))[1]
    geos = []
    for i, (grp, name, foot, act) in enumerate(sc):
        g_pre, _ = geo_of(foot, Action())
        g_post, fr_post = geo_of(foot, act)
        # the atlas defect series is built atlas-style (defect on the intact joint, resect flag unset); its geometry is
        # identical to complete resection with that defect, which is how the models receive it
        rs = 1 if (act.resect or act.d_mid > 0 or act.d_post > 0) else 0
        d = defect_fraction(fr_post, ref_fr) if rs else {"mid": 0.0, "post": 0.0}
        a = act_vec(rs, d["mid"], d["post"], act.osteo, act.interp, act.fusion, act.load)
        fe = np.load(f"results/atlas_analog/fields_{i:02d}.npz")
        geos.append((grp, name, g_pre.astype(np.float32), g_post.astype(np.float32), a, fe["lab"], fe["vm"], fe["U"], fr_post))
        if name == "Normal":
            # reference of the defect series, queried like the defects: complete resection without facet defect
            # (identical geometry and FE solution to the intact joint)
            geos.append(("ref", "Normal ref", g_pre.astype(np.float32), g_post.astype(np.float32), act_vec(1, 0.0, 0.0),
                         fe["lab"], fe["vm"], fe["U"], fr_post))
    grid = GRID
    out = {}
    for mname in names:
        pred = get_pred(mname)
        R = {}
        for grp, name, gpre, gpost, a, lab, vm_fe, U_fe, frp in geos:
            cells = np.nonzero(lab != VOID)[0]
            y, e = pred.fields_and_effects(gpre, a[None], cells, gpost[None])
            vm_p, u_p = norm.fields_inv(y[0])
            ucell_fe = U_fe[grid.cell_nodes[cells]].mean(1)
            fm = field_metrics(vm_p, u_p, vm_fe[cells], ucell_fe, lab[cells])
            st_p = region_stats(vm_p, lab[cells], cells, frp); st_f = region_stats(vm_fe[cells], lab[cells], cells, frp)
            eff_p = norm.eff_inv(e)[0]
            R[name] = dict(group=grp, fm=fm, pred=st_p, fe=st_f, eff_pred={n: float(eff_p[EI[n]]) for n in EI})
        # ---- scores ----
        P = {n: R[n]["pred"]["coal_p99"] for n in COALS}; F = {n: R[n]["fe"]["coal_p99"] for n in COALS}
        A = {n: CLS["tc_max"][CLS_MODELS.index(n)] for n in COALS}
        s = {}
        s["tau_all_vs_fe"] = float(kendalltau([P[n] for n in COALS], [F[n] for n in COALS])[0])
        s["types_vs_fe"] = pairs_agree(P, F, TYPES); s["roz_vs_fe"] = pairs_agree(P, F, ROZ)
        s["types_vs_atlas"] = pairs_agree(P, A, TYPES); s["roz_vs_atlas"] = pairs_agree(P, A, ROZ)
        s["fusion_below_all"] = bool(all(R["Fusion"]["pred"]["tc_p99"] < R[n]["pred"]["tc_p99"] for n in COALS))
        # defect curves and tolerances
        dd = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for fac in ["mid", "post"]:
            yp = [R["Normal ref"]["pred"][f"{fac}_mean"]] + [R[f"{fac} {int(d * 100)}"]["pred"][f"{fac}_mean"] for d in dd[1:]]
            yf = [R["Normal ref"]["fe"][f"{fac}_mean"]] + [R[f"{fac} {int(d * 100)}"]["fe"][f"{fac}_mean"] for d in dd[1:]]
            ye = [R["Normal ref"]["eff_pred"][f"{fac}_mean"]] + [R[f"{fac} {int(d * 100)}"]["eff_pred"][f"{fac}_mean"] for d in dd[1:]]
            s[f"curve_{fac}_pred"] = yp; s[f"curve_{fac}_fe"] = yf; s[f"curve_{fac}_effhead"] = ye
            s[f"tol_{fac}_pred"] = tolerance(dd, yp); s[f"tol_{fac}_fe"] = tolerance(dd, yf); s[f"tol_{fac}_effhead"] = tolerance(dd, ye)
        # alignment signs (48): prediction vs engine FE, prediction vs atlas
        regs = [("ant_p99", "ant_max"), ("mid_p99", "mid_max"), ("post_p99", "post_max"), ("tn_p99", "tn_max"),
                ("cc_p99", "cc_max"), ("coal_p99", "tc_max")]
        a_fe = a_at = tot = 0
        for t in TYPES:
            for j, pos in [(1, "varus10"), (2, "valgus10")]:
                for ke, ka in regs:
                    dp = R[f"{t} {pos}"]["pred"][ke] - R[t]["pred"][ke]
                    df = R[f"{t} {pos}"]["fe"][ke] - R[t]["fe"][ke]
                    da = ALIGN[t][ka][j] - ALIGN[t][ka][0]
                    a_fe += int(np.sign(dp) == np.sign(df)); a_at += int(np.sign(dp) == np.sign(da)); tot += 1
        s["align_vs_fe"] = [a_fe, tot]; s["align_vs_atlas"] = [a_at, tot]
        s["field_stress_l2_median"] = float(np.median([R[n]["fm"]["stress_l2"] for n in R if R[n]["group"] != "ref"]))
        s["field_disp_l2_median"] = float(np.median([R[n]["fm"]["disp_l2"] for n in R if R[n]["group"] != "ref"]))
        out[mname] = dict(scores=s, rows=R)
        print(mname, {k: (v if not isinstance(v, list) or len(v) <= 2 else "...") for k, v in s.items()}, flush=True)
    save_merged("results/eval_zeroshot.json", out)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
