"""Second training phase: continue every model from its 1,600-step checkpoint with a warm restart of the cosine
schedule (peak learning rate 5e-4) for a further STEPS steps.  Identical for all models."""
import os, sys, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import Split, Norm
from ml.train import train, Pairs
from exp.train_models import SPECS

if __name__ == "__main__":
    names = sys.argv[1].split(","); steps = int(sys.argv[2])
    src, dst = "results/models", "results/models_long"
    os.makedirs(dst, exist_ok=True)
    need_post = any(SPECS[n][0] in ("geo", "fno") for n in names)
    S = Split("train", load_post=need_post); norm = Norm.cached(); pairs = Pairs(S, norm)
    for n in names:
        out = f"{dst}/{n}.pkl"
        if os.path.exists(out):
            continue
        try:                                   # lock so that parallel queues never train the same model twice
            os.close(os.open(out + ".lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        while not os.path.exists(f"{src}/{n}.pkl"):
            time.sleep(30)
        time.sleep(5)
        init = pickle.load(open(f"{src}/{n}.pkl", "rb"))
        kind, kw = SPECS[n]
        t = time.time()
        train(kind, S, norm, pairs, steps=steps, lr=5e-4, log_every=100, out=out, init=init, **kw)
        print("DONE", n, round(time.time() - t), flush=True)
