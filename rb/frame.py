"""Engine-like frame for the atlas model (Claude, 2026-10-28): x anterior, y medial, z height above the ground plate.
Built from the local coordinate system in which ANSYS applied the leg forces (CS 14 of ds.dat): its x axis is the
plate normal, the plate is 12 mm thick (local x in [-6, 6]) and the foot stands on its top face (local x = 6)."""
import numpy as np
ORIGIN = np.array([-35.8959885737977, -59.0915876019794, 429.706394275391])
ANG = np.deg2rad([-0.407865969789822, -12.0171517608226, -91.9582383830423])      # THXY, THYZ, THZX


def _R():
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return Rz(ANG[0]) @ Rx(ANG[1]) @ Ry(ANG[2])        # columns: local axes in global coordinates


R_LOCAL = _R()


def to_engine(X):
    """Global (CT) coordinates -> engine frame: x = -local y (anterior), y = -local z (medial), z = local x - 6."""
    L = (np.asarray(X, float) - ORIGIN) @ R_LOCAL
    return np.stack([-L[..., 1], -L[..., 2], L[..., 0] - 6.0], -1)


def dir_to_engine(v):
    L = np.asarray(v, float) @ R_LOCAL
    return np.stack([-L[..., 1], -L[..., 2], L[..., 0]], -1)


def set_frame(origin, ang_deg):
    """Use another load coordinate system (MAPDL 'local' origin and THXY, THYZ, THZX in degrees)."""
    global ORIGIN, ANG, R_LOCAL
    ORIGIN = np.asarray(origin, float); ANG = np.deg2rad(ang_deg); R_LOCAL = _R()
