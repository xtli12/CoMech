"""Real-bone CoMech engine: one simulation (Claude, 2026-10-29).

Drop-in replacement of the parametric fe/simulate.py (kept as fe/simulate_parametric.py).  The FE model is the voxel
replica of the atlas foot (rb/rbfe.py: 1.5-mm cells, split nodes between bones, ground contact footprint, leg loads,
ligament springs; GPU solver when available); descriptors and fields are returned in the format of the parametric
engine on the 2-mm analysis grid (rb/rbsim.py): eff (17 descriptors + extras), vm, lab, frac (per analysis cell) and
Urel (relative displacement at the analysis grid nodes)."""
import fe.geometry  # noqa: F401  (puts rb/ on the path)
import rbsim as _RS
from rbsim import EFFECT_NAMES, get_grid

BODY_LOAD = 360.0


def simulate(foot, act, fine=1.0, mat_override=None, return_fields=True, method="gmg", tol=1e-6, frac_hook=None,
             U0=None):
    """mat_override, frac_hook and U0 (warm start) of the parametric engine are not used."""
    r = _RS.simulate(foot, act, fine=fine, return_fields=return_fields, tol=tol)
    r["U"] = r.get("Urel")            # nodal displacement on the analysis grid, relative to the calcaneus
    return r
