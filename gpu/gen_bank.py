"""Regenerate the voxel fields of TCC-MechBank and generate additional training feet (CPU, many processes).

The paper's bank JSON files (actions + effects) ship in results/fe_ground_truth.zip, but the voxel fields (npz) do
not.  This script re-solves every foot with the paper's random seeds and writes the npz next to the JSON.  Because
the engine is deterministic, the 17 effect descriptors of every re-solved configuration must agree with the stored
JSON; the agreement is checked and written to logs/bank_qc_<split>.json (a mismatch means the server's numerical
libraries differ and the run stops before any model is trained).

    python3 gpu/gen_bank.py paper  NPROC          # train/val/test/ood of the paper (252 feet), with QC
    python3 gpu/gen_bank.py extra  NPROC N SEED   # N additional training feet in data/bank/train_x (new seed)
"""
import os, sys, json, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import asdict
from fe.geometry import Foot, Action
import fe.simulate as SIM
from exp.bank import make_jobs, ROOT

PAPER = [("train", 180, 11), ("test", 40, 13), ("val", 12, 12), ("ood", 20, 14)]


def solve(job):
    fid, split, foot, acts, meta = job
    d = f"{ROOT}/{split}"
    npz, js = f"{d}/foot_{fid:04d}.npz", f"{d}/foot_{fid:04d}.json"
    if os.path.exists(npz) and os.path.exists(js):
        return split, fid, None
    old = json.load(open(js)) if os.path.exists(js) else None
    recs, vm_l, uc_l, lab_l, fr_l = [], [], [], [], []
    U0 = None
    t0 = time.time()
    for k, act in enumerate(acts):
        t = time.time()
        r = SIM.simulate(foot, act, tol=1e-5, U0=U0)
        if k == 0:
            U0 = r["U"]
        grid = SIM.get_grid()
        ucell = r["Urel"][grid.cell_nodes].mean(1)
        vm_l.append(r["vm"].astype(np.float16)); uc_l.append(ucell.astype(np.float16))
        lab_l.append(r["lab"].astype(np.int8)); fr_l.append(np.round(r["frac"] * 255).astype(np.uint8))
        recs.append(dict(k=k, meta=meta[k], act=asdict(act), eff=r["eff"], iters=r["iters"], time=time.time() - t))
    qc = None
    if old is not None:      # determinism check against the stored effects of the paper
        diffs = []
        for a, b in zip(old["actions"], recs):
            for n, v in a["eff"].items():
                w = b["eff"][n]
                diffs.append(abs(w - v) / max(abs(v), 1e-3))
        same_act = all(a["act"] == b["act"] for a, b in zip(old["actions"], recs))
        qc = dict(max_rel=float(max(diffs)), median_rel=float(np.median(diffs)), same_actions=bool(same_act))
    tmp = npz + ".tmp.npz"
    np.savez_compressed(tmp, vm=np.stack(vm_l), ucell=np.stack(uc_l), lab=np.stack(lab_l), frac=np.stack(fr_l))
    os.replace(tmp, npz)
    if old is None:
        with open(js, "w") as fh:
            json.dump(dict(fid=fid, split=split, foot=asdict(foot), actions=recs), fh)
    print(split, fid, round(time.time() - t0, 1), "s", "" if qc is None else f"qc max_rel={qc['max_rel']:.2e}", flush=True)
    return split, fid, qc


def run(jobs, nproc):
    from multiprocessing import Pool
    out = []
    with Pool(nproc) as p:
        for r in p.imap_unordered(solve, jobs, chunksize=1):
            out.append(r)
    return out


if __name__ == "__main__":
    mode, nproc = sys.argv[1], int(sys.argv[2])
    os.makedirs("logs", exist_ok=True)
    if mode == "paper":
        splits = sys.argv[3].split(",") if len(sys.argv) > 3 else [s for s, _, _ in PAPER]
        jobs = []
        for split, n, seed in PAPER:
            if split in splits:
                os.makedirs(f"{ROOT}/{split}", exist_ok=True)
                jobs += make_jobs(split, n, seed, ood=(split == "ood"))
        res = run(jobs, nproc)
        qcs = [q for _, _, q in res if q is not None]
        bad = [(s, f, q) for s, f, q in res if q is not None and (q["max_rel"] > 1e-3 or not q["same_actions"])]
        summary = dict(n_checked=len(qcs), n_bad=len(bad), worst=max([q["max_rel"] for q in qcs], default=0.0),
                       bad=[(s, f) for s, f, _ in bad][:20])
        if not qcs and os.path.exists("logs/bank_qc_paper.json"):
            # every foot was already solved and checked by an earlier run: keep that QC record instead of overwriting
            # it with an empty one (happens when the pipeline is restarted)
            summary = json.load(open("logs/bank_qc_paper.json"))
            print("QC (recorded by the earlier run)", summary, flush=True)
        else:
            json.dump(summary, open("logs/bank_qc_paper.json", "w"), indent=1)
            print("QC", summary, flush=True)
        if bad:
            print("QC FAILED: re-solved effects differ from the paper by >0.1%; stop and report logs/bank_qc_paper.json", flush=True)
            sys.exit(3)
    elif mode == "extra":
        n, seed = int(sys.argv[3]), int(sys.argv[4])
        os.makedirs(f"{ROOT}/train_x", exist_ok=True)
        run(make_jobs("train_x", n, seed), nproc)
    print("BANK_DONE", mode, flush=True)
