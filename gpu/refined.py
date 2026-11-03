"""Grid-refined FE reference for the 320 held-out test configurations, and model errors against it.

    python3 gpu/refined.py solve NPROC          # CPU: re-solve every test configuration on a grid refined twofold
    python3 gpu/refined.py compare [names]      # errors of the standard-grid FE and of every model vs the refined FE

The refined solution is restricted to the standard grid by averaging the eight child cells of every standard cell
(von Mises stress over active children; displacement in the calcaneus-fixed frame).  This quantifies how much of a
model's error against the standard-grid FE is discretization error of the engine itself.
"""
import os, sys, json, glob, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fe.geometry import Foot, Action, VOID
import fe.simulate as SIM

OUT = "data/refined_test_f0.50"
FINE = 0.5


def children_index():
    g, f = SIM.get_grid(1.0), SIM.get_grid(FINE)
    assert f.shape == tuple(2 * n for n in g.shape), (g.shape, f.shape)
    I, J, K = g.cI, g.cJ, g.cK
    ch = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                ch.append(((2 * I + a) * f.ny + (2 * J + b)) * f.nz + (2 * K + c))
    return np.stack(ch, 1)                                         # (ncell, 8) refined cell ids


def job(args):
    fn, k = args
    m = json.load(open(fn))
    dst = f"{OUT}/{m['fid']:04d}_{k}.npz"
    if os.path.exists(dst):
        return
    t = time.time()
    r = SIM.simulate(Foot(**m["foot"]), Action(**m["actions"][k]["act"]), fine=FINE, tol=1e-6)
    gf = SIM.get_grid(FINE)
    ch = children_index()
    act = r["active"][ch]                                           # (ncell, 8)
    vm = r["vm"][ch]
    n_act = act.sum(1)
    vm_c = np.where(n_act > 0, (vm * act).sum(1) / np.maximum(n_act, 1), np.nan)
    ucell_f = r["Urel"][gf.cell_nodes].mean(1)                     # (ncell_f, 3)
    uc = ucell_f[ch]
    uc_c = np.where(n_act[:, None] > 0, (uc * act[..., None]).sum(1) / np.maximum(n_act, 1)[:, None], np.nan)
    np.savez_compressed(dst + ".tmp.npz", vm=vm_c.astype(np.float32), uc=uc_c.astype(np.float32), n_act=n_act.astype(np.int8),
                        eff=json.dumps(r["eff"]), ndof=r["ndof"])
    os.replace(dst + ".tmp.npz", dst)
    print(os.path.basename(dst), round(time.time() - t, 1), "s", r["ndof"], "dof", flush=True)


def solve(nproc):
    os.makedirs(OUT, exist_ok=True)
    J = []
    for fn in sorted(glob.glob("data/bank/test/foot_*.json")):
        J += [(fn, k) for k in range(len(json.load(open(fn))["actions"]))]
    print("refined jobs", len(J), flush=True)
    from multiprocessing import Pool
    with Pool(nproc) as p:
        list(p.imap_unordered(job, J, chunksize=1))
    print("REFINED_SOLVE_DONE", flush=True)


def compare(names):
    from ml.data import Split, Norm
    from ml.evaluate import field_metrics
    from exp.common import get_pred, needs_post
    norm = Norm.cached()
    S = Split("test", load_post=any(needs_post(n) for n in names))
    fn_out = "results/eval_refined.json"
    out = json.load(open(fn_out)) if os.path.exists(fn_out) else {}
    ref = {}
    for f in range(S.nf):
        for k in range(S.K):
            p = f"{OUT}/{f:04d}_{k}.npz"
            if os.path.exists(p):
                ref[(f, k)] = np.load(p)
    print("refined references", len(ref), flush=True)
    if not ref:
        return

    def cells_of(f, k):
        lab = S.lab[f, k]; r = ref[(f, k)]
        return np.nonzero((lab != VOID) & np.isfinite(r["vm"]) & (r["n_act"] > 0))[0]

    if "fe_base" not in out:
        rows = []
        for (f, k), r in ref.items():
            c = cells_of(f, k); lab = S.lab[f, k][c]
            m = field_metrics(S.vm[f, k, c].astype(np.float32), S.uc[f, k, c].astype(np.float32), r["vm"][c], r["uc"][c], lab)
            m.update(f=f, k=k, meta=S.metas[f][k]); rows.append(m)
        out["fe_base"] = rows
    for name in names:
        if name in out:
            continue
        pred = get_pred(name)
        rows = []
        t = time.time()
        for f in range(S.nf):
            pre = S.geo_pre[f].astype(np.float32)
            post = S.geo_post[f].astype(np.float32) if (S.geo_post is not None and pred.kind in ("geo", "fno")) else None
            for k in range(S.K):
                if (f, k) not in ref:
                    continue
                c = cells_of(f, k); lab = S.lab[f, k][c]; r = ref[(f, k)]
                y, _ = pred.fields_and_effects(pre, S.act[f, k:k + 1], c, None if post is None else post[k:k + 1])
                vm_p, u_p = norm.fields_inv(y[0])
                m_ref = field_metrics(vm_p, u_p, r["vm"][c], r["uc"][c], lab)
                m_std = field_metrics(vm_p, u_p, S.vm[f, k, c].astype(np.float32), S.uc[f, k, c].astype(np.float32), lab)
                rows.append(dict(f=f, k=k, meta=S.metas[f][k], vs_refined=m_ref, vs_standard=m_std))
        out[name] = rows
        print(name, len(rows), f"{time.time() - t:.0f}s",
              "cartilage vs refined", round(float(np.median([r["vs_refined"]["stress_l2_art"] for r in rows])), 4),
              "vs standard", round(float(np.median([r["vs_standard"]["stress_l2_art"] for r in rows])), 4), flush=True)
        json.dump(out, open(fn_out, "w"))
    json.dump(out, open(fn_out, "w"))
    print("REFINED_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "solve":
        solve(int(sys.argv[2]))
    else:
        from gpu import config as CFG
        md = os.environ.get("COMECH_MODELS", "results/models_gpu")
        gpu_names = [n for n in CFG.RUNS if os.path.exists(f"{md}/{n}.pt")]
        paper = [n for n in ["comech_s0", "comech_s1", "comech_s2", "actcat", "geo", "deeponet", "fno"]
                 if os.path.exists(f"results/models_long/{n}.pkl")]
        names = sys.argv[2].split(",") if len(sys.argv) > 2 else gpu_names
        compare(names)
        if paper and len(sys.argv) <= 2:              # the paper's CPU models, for comparison
            os.environ["COMECH_MODELS"] = "results/models_long"
            import importlib, exp.common
            importlib.reload(exp.common)
            compare(paper)
