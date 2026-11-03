"""Evaluate all trained GPU models with the paper's experiment scripts (unchanged), plus GPU latency.

    COMECH_MODELS=results/models_gpu python3 gpu/run_eval.py [names]
"""
import os, sys, json, time, traceback, numpy as np
os.environ.setdefault("COMECH_MODELS", "results/models_gpu")
os.environ["SKIP_LATENCY"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu import config as CFG

MD = os.environ["COMECH_MODELS"]


def trained(names=None):
    names = names or list(CFG.RUNS)
    return [n for n in names if os.path.exists(f"{MD}/{n}.pt")]


def step(label, fn, *a):
    t = time.time()
    print(f"==== {label} ({len(a[0]) if a and isinstance(a[0], list) else ''})", flush=True)
    try:
        fn(*a)
        print(f"==== {label} done in {time.time() - t:.0f}s", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"==== {label} FAILED (continuing)", flush=True)


def latency(name):
    import torch
    from ml.data import Split
    from exp.common import get_pred
    from fe.geometry import VOID
    S = Split("test", max_feet=2, load_post=False)
    pred = get_pred(name)
    pre = S.geo_pre[0].astype(np.float32)
    cells = np.nonzero(S.lab[0, 1] != VOID)[0]
    A = np.repeat(S.act[0, 1][None], 200, 0)
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)
    for _ in range(3):
        pred.fields_and_effects(pre, S.act[0, 1][None], cells); pred.effects(pre, A)
    t1, t2 = [], []
    for _ in range(20):
        sync(); t = time.time(); pred.fields_and_effects(pre, S.act[0, 1][None], cells); sync(); t1.append(time.time() - t)
        sync(); t = time.time(); pred.effects(pre, A); sync(); t2.append(time.time() - t)
    out = dict(model=name, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", ncells=int(len(cells)),
               full_field_s=float(np.median(t1)), sweep200_effects_s=float(np.median(t2)))
    json.dump(out, open("results/eval_latency_gpu.json", "w"), indent=1)
    print("latency", out, flush=True)


if __name__ == "__main__":
    names = trained(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
    print("models", names, flush=True)
    if not names:
        sys.exit("no trained models in " + MD)
    kinds = {n: CFG.RUNS[n][0] for n in names}
    cm = [n for n in names if kinds[n] == "comech"]
    ens = [n for n in CFG.ENSEMBLE if n in names]
    import exp.e_fields as EF, exp.e_zeroshot as EZ, exp.e_taxonomy as ET, exp.e_tipping as ETP
    import exp.e_equiv_plan as EE, exp.e_cohort as EC
    step("fields + roll-out", EF.main, names)
    step("zero-shot atlas analogue", EZ.main, names)
    if cm:
        step("taxonomy", ET.main, cm)
    step("tolerance", ETP.main, names)
    step("equivalence / planning / abstention", EE.main, names, ens if len(ens) == 3 else ["-"])
    if len(ens) == 3:
        step("virtual cohort (ensemble)", EC.main, ens)
    if "comechL_s0" in names:
        step("latency", latency, "comechL_s0")
    print("EVAL_DONE", flush=True)
