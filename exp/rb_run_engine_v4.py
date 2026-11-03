"""Run every FE solve of the real-bone engine with a pool of workers sharing the GPUs (Claude, 2026-10-29).

Order: atlas-analog scenarios (33), TCC-MechBank (train 180 / test 40 / val 12 / ood 20 feet, same seeds and sampling
as exp/run_bank.sh), held-out sweeps (exp/sweeps.py), then -- once exp/cohort.py has written the virtual cohort and
the taxonomy set -- the taxonomy fingerprints (exp/sweeps_tax.py) and the cohort envelopes (exp/sweeps_cohort.py).
Every task is the scripts' own function, so the outputs have exactly their format; finished outputs are skipped.
Environment: NWORK (workers, default 12), NGPU (default 4), PROGRESS (json file written every minute)."""
import os, sys, json, time, traceback, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
NW = int(os.environ.get("NWORK", "12")); NGPU = int(os.environ.get("NGPU", "4"))
PROGRESS = os.environ.get("PROGRESS", os.path.join("logs", "rb-progress.json"))
ERRDIR = os.environ.get("ERRDIR", "logs")
STOP = os.path.join(ROOT, "STOP")


def _init(counter):
    with counter.get_lock():
        i = counter.value; counter.value += 1
    os.environ["COMECH_GPU"] = str(i % NGPU)


def task(t):
    kind, key, payload = t
    t0 = time.time()
    try:
        if kind == "atlas":
            import exp.run_atlas as M
            os.makedirs(M.OUT, exist_ok=True)
            if not os.path.exists(f"{M.OUT}/rec_{payload:02d}.json"):
                M.run(payload)
        elif kind == "bank":
            import exp.bank as M
            M.solve_foot(payload)
        elif kind == "sweep":
            import exp.sweeps as M
            M.run(payload)
        elif kind == "tax":
            import exp.sweeps_tax as M
            M.run(payload)
        elif kind == "cohort":
            import exp.sweeps_cohort as M
            M.run(payload)
        return kind, key, "ok", time.time() - t0
    except Exception:
        tb = traceback.format_exc()
        try:
            open(os.path.join(ERRDIR, f"rbe-err-{kind}-{key}.txt"), "w").write(tb)
        except OSError:
            pass
        return kind, key, "fail", time.time() - t0


def phase_a():
    import exp.bank as B
    import exp.sweeps as SW
    from exp.scenarios import atlas_scenarios
    T = [("atlas", i, i) for i in range(len(atlas_scenarios()))]
    for split, n, seed in (("train", 180, 11), ("test", 40, 13), ("val", 12, 12), ("ood", 20, 14)):
        os.makedirs(f"{B.ROOT}/{split}", exist_ok=True)
        for j in B.make_jobs(split, n, seed, ood=(split == "ood")):
            if not os.path.exists(f"{B.ROOT}/{split}/foot_{j[0]:04d}.json"):
                T.append(("bank", f"{split}{j[0]}", j))
    os.makedirs(SW.ROOT, exist_ok=True)
    for k, j in enumerate(SW.jobs()):
        kind, i, tag, d, foot, act = j
        if not os.path.exists(f"{SW.ROOT}/{kind}_{i:03d}_{tag}_{d:.2f}.json"):
            T.append(("sweep", k, j))
    return T


def phase_b():
    import exp.sweeps_tax as ST
    os.makedirs(ST.ROOT, exist_ok=True)
    T = [("tax", f"{i}{k}", (i, k)) for i in range(240) for k in ("coal", "twin")
         if not os.path.exists(f"{ST.ROOT}/{i:03d}_{k}.json")]
    import exp.sweeps_cohort as SC
    os.makedirs(SC.ROOT, exist_ok=True)
    T += [("cohort", f"{j[0]}{j[1]}{j[2]}", j) for j in SC.J if not os.path.exists(f"{SC.ROOT}/{j[0]:03d}_{j[1]}_{j[2]:.2f}.json")]
    return T


def cohort_running():
    r = subprocess.run(["pgrep", "-u", os.environ.get("USER", ""), "-f", "exp/cohort.py"], capture_output=True)
    return r.returncode == 0


def phase_c():
    """Additional training feet of the GPU scaling experiment (gpu/config.py: 1,620 feet, seed 21) in data/bank/train_x."""
    import exp.bank as B
    sys.path.insert(0, os.path.join(ROOT, "gpu"))
    from gpu.config import EXTRA_FEET, EXTRA_SEED
    os.makedirs(f"{B.ROOT}/train_x", exist_ok=True)
    return [("bank", f"train_x{j[0]}", j) for j in B.make_jobs("train_x", EXTRA_FEET, EXTRA_SEED)
            if not os.path.exists(f"{B.ROOT}/train_x/foot_{j[0]:04d}.json")]


def write_progress(st):
    st["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        st["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        st["mem"] = subprocess.run(["free", "-g"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[:2]
    except Exception:
        pass
    tmp = PROGRESS + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, PROGRESS)


def main():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    # geometry of the virtual cohort and of the taxonomy set (no FE) in a separate process
    coh = None
    if not os.path.exists("data/cohort/taxonomy.json") and not cohort_running():
        coh = subprocess.Popen([sys.executable, "-u", "exp/cohort.py"], stdout=open("logs_cohort.txt", "w"), stderr=subprocess.STDOUT)
    st = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "nwork": NW, "done": {}, "fail": {}, "total": {}, "sec": {}}
    t0 = time.time()
    with ctx.Pool(NW, initializer=_init, initargs=(counter,), maxtasksperchild=8) as pool:
        for phase in os.environ.get("RB_PHASES", "A,B").split(","):
            if phase == "B":
                while (coh is not None and coh.poll() is None) or cohort_running():
                    write_progress(st); time.sleep(30)
                if not os.path.exists("data/cohort/taxonomy.json"):
                    st["note"] = "cohort.py failed: phase B skipped"; break
            T = {"A": phase_a, "B": phase_b, "C": phase_c}[phase]()
            for k, _, _ in T:
                st["total"][k] = st["total"].get(k, 0) + 1
            write_progress(st)
            last = time.time()
            for kind, key, res, sec in pool.imap_unordered(task, T, chunksize=1):
                d = st["done"] if res == "ok" else st["fail"]
                d[kind] = d.get(kind, 0) + 1
                st["sec"][kind] = st["sec"].get(kind, 0.0) + sec
                st["elapsed_s"] = round(time.time() - t0)
                if time.time() - last > 60:
                    write_progress(st); last = time.time()
                if os.path.exists(STOP):
                    st["note"] = "stopped"; write_progress(st); pool.terminate(); return
            st["phase_" + phase + "_done"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_progress(st)
    st["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_progress(st)


if __name__ == "__main__":
    main()
