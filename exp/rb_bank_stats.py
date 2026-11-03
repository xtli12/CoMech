"""Summary of TCC-MechBank (results/bank_stats.json, read by summarize.py and fill_text.py): feet, solutions,
coalition types, Rozansky subtypes of type I feet and action kinds per split; median and IQR of solver iterations and
median solve time (Claude, 2026-10-29; same fields as the file of the paper)."""
import os, sys, json, glob, collections, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out, its, ts = {}, [], []
for split in ("train", "test", "val", "ood"):
    F = [json.load(open(f)) for f in sorted(glob.glob(f"data/bank/{split}/foot_*.json"))]
    types = collections.Counter(f["foot"]["ctype"] for f in F)
    roz = collections.Counter(f["foot"]["rtype"] for f in F if f["foot"]["ctype"] == "I")
    acts = collections.Counter(a["meta"] for f in F for a in f["actions"])
    out[split] = dict(feet=len(F), solutions=sum(len(f["actions"]) for f in F), types=dict(types), roz_typeI=dict(roz),
                      actions=dict(acts))
    its += [a["iters"] for f in F for a in f["actions"] if a.get("iters") is not None]
    ts += [a["time"] for f in F for a in f["actions"]]
out["total_solutions_bank"] = sum(out[s]["solutions"] for s in ("train", "test", "val", "ood"))
out["iters_median"] = float(np.median(its)); out["iters_iqr"] = [float(np.percentile(its, 25)), float(np.percentile(its, 75))]
out["time_median_contended"] = float(np.median(ts))
json.dump(out, open("results/bank_stats.json", "w"))
print(json.dumps(out))
