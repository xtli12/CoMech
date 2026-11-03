"""Atlas-analog scenarios on the reference foot (open-source engine counterpart of the 33-model atlas)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fe.geometry import Foot, Action

# coalition morphologies on the reference foot (modelling choices, documented in the Supplement)
COAL_MORPH = {
    "Normal": dict(ctype="none", rtype="none"),
    "Roz I": dict(ctype="I", rtype="R1", cov_mid=0.35),
    "Roz II": dict(ctype="I", rtype="R2", cov_mid=0.35),
    "Roz III": dict(ctype="I", rtype="R3", cov_mid=0.5, fib_frac=0.5),
    "Roz IV": dict(ctype="I", rtype="R4", cov_mid=1.0),
    "Roz V": dict(ctype="II", rtype="R5", cov_post=0.3),
    "Type I": dict(ctype="I", rtype="R4", cov_mid=0.5),
    "Type II": dict(ctype="II", rtype="R5", cov_post=0.4),
    "Type III": dict(ctype="III", rtype="R5"),
    "Type IV": dict(ctype="IV", rtype="unclass", cov_mid=0.5, cov_post=0.3),
}


def atlas_scenarios():
    sc = []
    for name, kw in COAL_MORPH.items():
        sc.append(("cls", name, Foot(**kw), Action()))
    sc.append(("cls", "Fusion", Foot(), Action(fusion=1)))
    for t in ["Type I", "Type II", "Type III", "Type IV"]:
        for pos, ang in [("varus10", -10.0), ("valgus10", 10.0)]:
            sc.append(("align", f"{t} {pos}", Foot(align=ang, **COAL_MORPH[t]), Action()))
    for facet in ["mid", "post"]:
        for d in [10, 20, 30, 40, 50, 60, 70]:
            a = Action(d_mid=d / 100) if facet == "mid" else Action(d_post=d / 100)
            sc.append(("defect", f"{facet} {d}", Foot(), a))
    return sc
