"""Remove unreadable bank records (a worker killed while writing): every foot_XXXX.json must load and its npz must
open with the four arrays; broken pairs are deleted so that the engine recomputes them (Claude, 2026-10-29).
Usage: python exp/rb_check_bank.py data/bank/train_x [more dirs]"""
import os, sys, glob, json
import numpy as np
bad = 0
for d in sys.argv[1:]:
    for js in sorted(glob.glob(f"{d}/foot_*.json")):
        npz = js.replace(".json", ".npz")
        ok = os.path.exists(npz)
        try:
            m = json.load(open(js)); ok = ok and len(m["actions"]) == 8
            if ok:
                z = np.load(npz)
                ok = all(k in z.files for k in ("vm", "ucell", "lab", "frac")) and z["vm"].shape[0] == 8
        except Exception:
            ok = False
        if not ok:
            bad += 1
            for p in (js, npz):
                if os.path.exists(p):
                    os.remove(p)
    for npz in glob.glob(f"{d}/foot_*.npz"):          # npz without json: an interrupted foot
        if not os.path.exists(npz.replace(".npz", ".json")):
            os.remove(npz)
print("removed", bad, "broken records")
