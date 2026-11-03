"""Train a list of models sequentially (one CPU core per process)."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import Split, Norm
from ml.train import train, Pairs

SPECS = {
    "comech_s0": ("comech", dict(seed=0)),
    "comech_s1": ("comech", dict(seed=1)),
    "comech_s2": ("comech", dict(seed=2)),
    "comech_noswap": ("comech", dict(seed=0, swap=0.0)),
    "comech_nosens": ("comech", dict(seed=0, sens=0.0)),
    "comech_none": ("comech", dict(seed=0, swap=0.0, sens=0.0)),
    "actcat": ("actcat", dict(seed=0)),
    "geo": ("geo", dict(seed=0)),
    "deeponet": ("deeponet", dict(seed=0)),
    "fno": ("fno", dict(seed=0, micro=4)),
}

if __name__ == "__main__":
    names = sys.argv[1].split(","); steps = int(sys.argv[2])
    os.makedirs("results/models", exist_ok=True)
    need_post = any(SPECS[n][0] in ("geo", "fno") for n in names)
    S = Split("train", load_post=need_post); norm = Norm.cached(); pairs = Pairs(S, norm)
    print("train feet", S.nf, "eps", pairs.eps, "eq pairs", sum(len(e) for e in pairs.eq), "nb pairs",
          sum(len(n) for n in pairs.nb), flush=True)
    for n in names:
        out = f"results/models/{n}.pkl"
        if os.path.exists(out):
            continue
        try:                                   # lock so that parallel queues never train the same model twice
            os.close(os.open(out + ".lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        kind, kw = SPECS[n]
        t = time.time()
        train(kind, S, norm, pairs, steps=steps, log_every=100, out=out, **kw)
        print("DONE", n, round(time.time() - t), flush=True)
