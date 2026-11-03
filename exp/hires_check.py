"""Check the refined-grid numbers quoted for the deformation-guided surgical map (Fig. 7a, Rehearsal section).
Needs data/hires/child_{resect,plan}_f{1.00,0.50}.npz from exp/hires_showcase.py (CPU only, ~1.5 min in total):
    python3 exp/hires_showcase.py 1.0 child_resect,child_plan
    python3 exp/hires_showcase.py 0.5 child_resect,child_plan
    python3 exp/hires_check.py            # writes results/hires_check.json
"""
import json, numpy as np

KEYS = ["post_mean", "mid_mean", "tn_p99", "talus_evert", "nav_drop"]
out = {}
for fine in ["1.00", "0.50"]:
    R = np.load(f"data/hires/child_resect_f{fine}.npz"); P = np.load(f"data/hires/child_plan_f{fine}.npz")
    er, ep = json.loads(str(R["eff"])), json.loads(str(P["eff"]))
    out[f"f{fine}"] = dict(
        ndof=[int(R["ndof"]), int(P["ndof"])],
        change_pct={k: 100 * (abs(ep[k]) - abs(er[k])) / abs(er[k]) for k in KEYS},   # change in magnitude, plan vs resection alone
        resect={k: er[k] for k in KEYS}, plan={k: ep[k] for k in KEYS})
json.dump(out, open("results/hires_check.json", "w"), indent=1)
for g, v in out.items():
    print(g, v["ndof"], {k: round(x, 1) for k, x in v["change_pct"].items()})
