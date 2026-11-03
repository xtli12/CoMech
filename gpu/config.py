"""Pre-specified configuration of the GPU scaling experiment (fixed before any result is seen).

All runs use the same optimizer, schedule, number of steps, batch composition and query-point budget.  Hyper-
parameters were set once a priori and are not tuned on held-out data; the final checkpoint of every run is
evaluated (no early stopping, no checkpoint selection).
"""
STEPS = 40_000            # 10x the 4,000 CPU steps of the paper
WARMUP = 1_000
LR = 5e-4
WD = 0.01
CLIP = 1.0
BF, KB, NQ = 16, 5, 4096  # feet per batch, actions per foot, query cells per transition (paper: 4, 5, 1024)
EPS = None                # effect-equivalence threshold; None = read from the paper's comech_s0 checkpoint (0.1326)
RHO = 0.25
MU_FRAC = 0.5
EXTRA_FEET = 1_620        # additional training feet (180 + 1,620 = 1,800 = 10x the paper)
EXTRA_SEED = 21

# model sizes
COMECH_L = dict(D=192, Du=128, Ht=512, Hc=768, Hd=256, n_enc=4, n_dyn=4, n_dec=4)
GINO_CFG = dict(width=64, latent=(32, 20, 24), modes=(12, 8, 10), layers=4, r_in=3.5, r_out=3.5, kmlp=64)
GEOFNO_CFG = dict(width=64, latent=(48, 24, 32), modes=(12, 8, 10), layers=4)

# run name -> (kind, number of training feet, seed, loss flags); priority order = launch order
RUNS = {
    # --- main scaled comparison, 1,800 training feet ---
    "comechL_s0":      ("comech", 1800, 0, dict()),
    "actcatL":         ("actcat", 1800, 0, dict()),
    "gino":            ("gino",   1800, 0, dict()),
    "geofno":          ("geofno", 1800, 0, dict()),
    "comechL_s1":      ("comech", 1800, 1, dict()),
    "comechL_s2":      ("comech", 1800, 2, dict()),
    "comechL_noswap":  ("comech", 1800, 0, dict(swap=0.0)),
    "comechL_nosens":  ("comech", 1800, 0, dict(sens=0.0)),
    "comechL_none":    ("comech", 1800, 0, dict(swap=0.0, sens=0.0)),
    # --- same models on the paper's 180 training feet (separates model scale from data scale) ---
    "comechL_s0_n180": ("comech", 180, 0, dict()),
    "actcatL_n180":    ("actcat", 180, 0, dict()),
    "gino_n180":       ("gino",   180, 0, dict()),
    "geofno_n180":     ("geofno", 180, 0, dict()),
    # --- data-scaling curve ---
    "comechL_s0_n450": ("comech", 450, 0, dict()),
    "comechL_s0_n900": ("comech", 900, 0, dict()),
    "actcatL_n450":    ("actcat", 450, 0, dict()),
    "actcatL_n900":    ("actcat", 900, 0, dict()),
}

ENSEMBLE = ["comechL_s0", "comechL_s1", "comechL_s2"]
MAIN = ["comechL_s0", "comechL_s1", "comechL_s2", "comechL_noswap", "comechL_nosens", "comechL_none",
        "actcatL", "gino", "geofno"]
GEO_KINDS = ("gino", "geofno")          # models that receive the explicit post-operative geometry (marked *)
# number of micro-batches each 80-transition step of the geometry models is split into (gradient accumulation;
# an implementation detail that does not change the loss, only memory use and speed)
MICRO = {"gino": 5, "geofno": 5}
