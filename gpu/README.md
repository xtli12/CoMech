# GPU scaling experiment

PyTorch port of CoMech, scaled in width and depth (CoMech-L), trained for 40,000 steps on ten times the training feet (180 + 1,620 = 1,800), together with an action-concatenated surrogate of the same size and two geometry-aware neural operators given the post-operative geometry (GINO and Geo-FNO). A refined-grid FE reference of the held-out test configurations measures how much of every model's error is due to the discretization of the engine itself. The configuration is fixed in `config.py` before any result is seen: the same optimizer, schedule, number of steps, batch composition and query budget for every run, no tuning on held-out data and no checkpoint selection.

## Running

```bash
export RB_TEMPLATE=/path/to/rb_template.npz        # label raster of the atlas foot (see the main README)
bash gpu/start.sh                                  # whole experiment in the background
bash gpu/status.sh                                 # progress
bash gpu/stop.sh                                   # stop; start.sh resumes where it stopped
```

Options: `GPUS=0,1,2,3` (GPUs to use), `NPROC` (CPU processes for data generation; default from cores and memory), `PY` (Python interpreter), `STAGES` (default `env,data,refined,train,eval,pack`).

| Stage | Contents | Device |
|---|---|---|
| `env` | environment check (`env_check.py`) | — |
| `data` | voxel fields of TCC-MechBank re-solved with the paper's seeds and checked against the stored effect descriptors (determinism check), 1,620 additional training feet (seed 21), training cache (`gen_bank.py`, `cache.py`) | CPU |
| `refined` | the 320 held-out test configurations on a grid refined twofold (`refined.py solve`) | CPU |
| `train` | self-test, then every run of `config.RUNS` (`selftest.py`, `ttrain.py`) | GPU |
| `eval` | predictions and metrics on the standard grid and against the refined-grid reference (`run_eval.py`, `refined.py compare`) | GPU |
| `pack` | summary (`summarize_gpu.py`) and a results archive without model weights (`pack.py`) | CPU |

The numbers of the paper are then computed from the packed results by `analysis_gpu.py` (and `refined_effects.py` for the effect descriptors on the refined grid).

Requirements: PyTorch with CUDA in addition to the packages of the main `requirements.txt`.
