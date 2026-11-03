"""View constants of the rendered figures for the engine in use (Claude, 2026-11-01).  The parametric engine keeps the
values the figure scripts had; the real-bone engine (atlas foot, frame x anterior, y medial, z up from the top of the
ground plate) uses views centred on its subtalar facets and stronger smoothing of the bone surfaces (its 0.5-mm label
raster would otherwise show terraces)."""
import numpy as np
from fe.geometry import ML_BOX
PARAMETRIC = tuple(tuple(float(v) for v in b) for b in ML_BOX) == ((0.0, 96.0), (-18.0, 30.0), (0.0, 63.0))
if PARAMETRIC:
    FACET_C = (45.0, 3.0)                 # superior views of the calcaneal articular surfaces (talus removed)
    FACET_SCALE = 1.0                     # multiplies the orthographic half-widths of those views
    ACT_TARGET = np.array([40.5, 0.5, 28.0]); ACT_ORTHO = 44.0     # action panels (render_actions.py)
    OBLIQUE = dict(eye=(-40.0, 150.0, 120.0), target=(45.0, 3.0, 28.0))
    TALUS_C = FACET_C; TALUS_SCALE = 1.0  # inferior views of the talus (render_heldout.py)
    SIGMA = {"bone": 1.6, "cart": 1.1, "coal": 1.3, "interp": 1.2, "fuse": 1.3}
    ISO = {}                              # iso level of the smoothed occupancy (default 0.5)
    NSIGMA = {}                           # smoothing of the shading normals (default: as the surface)
else:
    # facet cartilage of the reference foot: posterior centroid (-52.8, -1.1, 63.5), middle (-39.1, 16.4, 57.9),
    # anterior (-24.8, 6.0, 52.9); facets span x -65.5..-20.5, y -12.5..21 mm; calcaneus x -96.5..-22, y -21..24 mm
    FACET_C = (-43.0, 2.5)
    FACET_SCALE = 0.92
    ACT_TARGET = np.array([-59.0, 1.5, 50.0]); ACT_ORTHO = 44.0
    OBLIQUE = dict(eye=(-150.0, 150.0, 150.0), target=(-55.0, 2.0, 50.0))
    TALUS_C = (-39.0, 8.5); TALUS_SCALE = 1.0   # talus of the reference foot: x -67..-11.5, y -13.5..31 mm
    # cartilage: about 1 mm thick on a 0.5-mm raster; strong surface smoothing opens holes in thin parts and closes
    # the notches of the defects (and moves the surface off the cells that colour it), so the surface keeps a light
    # smoothing and only the shading normals are smoothed strongly (removes the staircase ripples)
    SIGMA = {"bone": 4.5, "cart": 1.5, "coal": 2.0, "interp": 2.0, "fuse": 2.4}
    ISO = {}
    NSIGMA = {"cart": 3.5}
