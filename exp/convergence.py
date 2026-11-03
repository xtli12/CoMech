"""Mesh convergence of the effect descriptors on three uniformly refined grids."""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fe.simulate as S
from fe.geometry import Foot, Action
from exp.scenarios import COAL_MORPH
from dataclasses import asdict
cases = [("Normal", Foot(), Action()), ("Post 50", Foot(), Action(d_post=0.5)), ("Mid 30", Foot(), Action(d_mid=0.3)),
         ("Type I", Foot(**COAL_MORPH["Type I"]), Action())]
out = []
for f in [1.0, 0.75, 0.5]:
    for name, foot, act in cases:
        t = time.time()
        r = S.simulate(foot, act, fine=f, tol=1e-6, return_fields=False)
        rec = dict(fine=f, name=name, eff=r["eff"], ndof=r["ndof"], iters=r["iters"], time=time.time() - t)
        out.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump(out, open("results/convergence.json", "w"))
