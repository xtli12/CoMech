"""E2: field fidelity on held-out and out-of-range feet, roll-out composition, and latency."""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.common import *
from ml.data import Split, Norm
from ml.evaluate import evaluate_split, field_metrics
from ml.train import ROLL_OST_ONLY


def main(names):
    norm = Norm.cached()
    fn = "results/eval_fields.json"
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for split in ["test", "ood"]:
        todo = [n for n in names if f"{n}|{split}" not in out]
        if not todo:
            continue
        S = Split(split, load_post=any(needs_post(n) for n in todo))
        for name in todo:
            pred = get_pred(name)
            t = time.time()
            rows = evaluate_split(pred, S, norm)
            out[f"{name}|{split}"] = rows
            print(name, split, len(rows), f"{time.time() - t:.0f}s",
                  {k: round(float(np.median([r[k] for r in rows if k in r])), 4)
                   for k in ["stress_l2", "stress_l2_art", "disp_l2", "hot_iou", "mid_peak_err", "post_peak_err", "eff_rmse"]},
                  flush=True)
        json.dump(out, open(fn, "w"))
    # ---- roll-out composition (CoMech): resection then osteotomy vs one-step vs FE ----
    S = Split("test", load_post=False)
    fn = "results/eval_rollout.json"
    comp = json.load(open(fn)) if os.path.exists(fn) else {}
    for name in [n for n in names if n.startswith("comech") and n not in comp]:
        pred = get_pred(name)
        rows = []
        for f in range(S.nf):
            m = S.metas[f]
            if "roll_a" not in m:
                continue
            ka, kb = m.index("roll_a"), m.index("roll_b")
            lab = S.lab[f, kb]; cells = np.nonzero(lab != VOID)[0]
            pre = S.geo_pre[f].astype(np.float32)
            a1 = S.act[f, ka][None]; a2 = ROLL_OST_ONLY(S.act[f, kb])[None]
            y2, e2 = pred.rollout2(pre, a1, a2, cells[None])
            y1, e1 = pred.fields_and_effects(pre, S.act[f, kb][None], cells)
            vm_t = S.vm[f, kb, cells].astype(np.float32); u_t = S.uc[f, kb, cells].astype(np.float32)
            et = norm.eff(S.eff_raw[f, kb][None])[0]
            r2 = field_metrics(*norm.fields_inv(y2[0]), vm_t, u_t, lab[cells])
            r1 = field_metrics(*norm.fields_inv(y1[0]), vm_t, u_t, lab[cells])
            rows.append(dict(f=f, two_step=r2, one_step=r1, eff2=float(np.sqrt(((e2[0] - et) ** 2).mean())),
                             eff1=float(np.sqrt(((e1[0] - et) ** 2).mean()))))
        comp[name] = rows
    json.dump(comp, open("results/eval_rollout.json", "w"))
    # ---- latency ----
    lat = {}
    if os.environ.get("SKIP_LATENCY") or (os.path.exists("results/eval_latency.json") and os.environ.get("REDO_LATENCY") is None):
        return
    S = Split("test", max_feet=4, load_post=False)
    pred = get_pred("comech_s0")
    pre = S.geo_pre[0].astype(np.float32)
    cells = np.nonzero(S.lab[0, 1] != VOID)[0]
    ts = []
    for rep in range(5):
        t = time.time(); pred.fields_and_effects(pre, S.act[0, 1][None], cells); ts.append(time.time() - t)
    lat["rollout_full_field_s"] = float(np.median(ts)); lat["ncells"] = int(len(cells))
    # sweep: 200 candidate operations, effects only (encode once)
    A = np.repeat(S.act[0, 1][None], 200, 0); A[:, 3] = np.linspace(-1, 1, 200)
    ts = []
    for rep in range(3):
        t = time.time(); pred.effects(pre, A); ts.append(time.time() - t)
    lat["sweep200_effects_s"] = float(np.median(ts))
    # FE timing: the same held-out configurations re-solved now, on the same core, to the bank tolerance
    import fe.simulate as FS
    from fe.geometry import Foot, Action
    fe_t = []
    for f in range(3):
        for k in (1, 2):
            a = S.acts_raw[f][k]
            t = time.time(); FS.simulate(Foot(**S.meta[f]["foot"]), Action(**a), tol=1e-5); fe_t.append(time.time() - t)
    lat["fe_solve_s_median"] = float(np.median(fe_t)); lat["fe_solve_s_all"] = fe_t
    bank_t = []
    for fjs in glob.glob("data/bank/test/foot_*.json"):
        bank_t += [a["time"] for a in json.load(open(fjs))["actions"]]
    lat["fe_bank_log_s_median"] = float(np.median(bank_t))
    json.dump(lat, open("results/eval_latency.json", "w"), indent=1)
    print(lat)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
