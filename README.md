# CoMech

Code for **"A biomechanical world model for in silico rehearsal of pediatric hindfoot surgery"** (manuscript under review; the citation will be added on publication).

CoMech is a co-constructed biomechanical world model of the pediatric hindfoot with talocalcaneal coalition. The foot is the state, a surgical operation is the action (complete resection of the coalition, calcaneal osteotomy, interposition, subtalar fusion, load), and the post-operative von Mises stress field, displacement field and 17 effect descriptors are the outcome. CoMech is trained on finite element (FE) solutions produced by a voxel FE data engine built on the foot of an FE atlas. It is used to rehearse adjuncts to complete resection in latent space; the chosen plan is then verified by FE.

> **Scope.** Every result in the paper is in silico: CoMech is trained and tested on solutions of the data engine, not on patient outcomes. The CT-informed twins of real feet are exploratory, rest on preliminary readings and are not validated against clinical outcomes.

## Repository layout

| Path | Contents |
|---|---|
| `rb/` | Data engine on the atlas foot: real-bone geometry and the virtual-foot and coalition parameters (`rbgeom.py`), voxel replica of the atlas with split nodes between bones, ground contact and ligament springs (`rbfe.py`, `splitfe.py`, `replica2.py`, `replica3.py`, `voxel_replica.py`, `frame.py`), GPU solver (`gpusolve.py`), analysis grid and effect descriptors (`rbsim.py`) |
| `fe/` | Engine interface used by all experiments (`geometry.py`, `simulate.py`); voxel FE solver with geometric-multigrid preconditioned conjugate gradients (`solver.py`, `gmg.py`); the parametric hindfoot of the first version of the work (`geometry_parametric.py`, `simulate_parametric.py`); patch test |
| `ml/` | Reverse-mode automatic differentiation in NumPy with gradient checks (`ad.py`, `test_ad.py`); data pipeline, CoMech and the baselines (action-concatenated surrogate, geometry-to-field model, DeepONet, Fourier neural operator), training and evaluation |
| `exp/` | Atlas analogue, TCC-MechBank, held-out sweeps, virtual cohort and taxonomy set; parallel engine runner (`rb_run_engine_v4.py`); training; experiments (`e_*.py`); CT-informed twins; high-resolution checks; summaries (`summarize.py`) |
| `gpu/` | GPU scaling experiment: PyTorch port of CoMech (CoMech-L), GINO and Geo-FNO baselines, extended training set, refined-grid FE reference (see `gpu/README.md`) |
| `figsrc/`, `analysis/` | Figures and supplementary tables of the paper (house style in `figsrc/figstyle.py`) |

## Requirements

Python 3.10 or later with NumPy, SciPy, scikit-learn, scikit-image, Matplotlib and Pillow:

```bash
python -m pip install -r requirements.txt
```

PyTorch with CUDA is needed for the GPU solver of the engine (the CPU solver is used otherwise) and for the GPU scaling experiment.

## Inputs that are not in this repository

| Input | Status |
|---|---|
| Label raster of the atlas foot (`rb_template.npz`) | Derived from the de-identified CT of one adult foot; available from the corresponding author on reasonable request. Point `RB_TEMPLATE` to it. |
| Generated datasets and result files | Regenerated deterministically by this code from the label raster (random seeds are fixed in the code); also available from the corresponding author on reasonable request under a data-use agreement. |
| Clinical data of the three-center cohort (CT exports, CT reading table) | Not shared; de-identified data are available from the corresponding author on reasonable request, subject to approval by the participating institutions. |
| Screenshots of the commercial FE atlas | Not distributed; the atlas tables used by the figure scripts are in `figsrc/fe_data.py` and the Supplementary Information. |

## Reproducing the experiments

All scripts are run from the repository root; each step skips outputs that already exist, so runs can be interrupted and resumed.

```bash
export RB_TEMPLATE=/path/to/rb_template.npz

python fe/test_patch.py && python ml/test_ad.py       # checks of the FE element and of the autodiff gradients

python exp/cohort.py                                   # virtual cohort and taxonomy geometries
python exp/rb_run_engine_v4.py                         # every FE solve: atlas analogue (33), TCC-MechBank
                                                       # (252 feet x 8 actions), held-out sweeps (572),
                                                       # taxonomy fingerprints (480), cohort envelopes (577)
python exp/train_models.py comech_s0 1600              # first stage, then a warm restart:
python exp/train_continue.py comech_s0 2400            #   the same for every model (see exp/run_all.sh)

export COMECH_MODELS=results/models_long
python exp/e_fields.py comech_s0,comech_s1,comech_s2   # experiments: e_*.py (see exp/run_all.sh for the full list)
python exp/summarize.py                                # results/numbers.json
```

`bash exp/run_all.sh` runs the whole pipeline in the order used for the paper (ten models; the paper's runs used a workstation with 48 CPU threads and four GPUs). `bash figsrc/make_figures.sh` renders the three-dimensional images and builds the figures and supplementary tables. Some figure panels also need inputs that are not distributed (screenshots of the FE atlas, CT tiles); those scripts stop with a missing-file error without them.

Environment variables: `RB_TEMPLATE` (label raster), `COMECH_MODELS` (model directory of the experiments, `results/models_long`), `NWORK` and `NGPU` (engine workers and GPUs), `RENDER_OUT` (rendered images), `CT_READ_DIR` (CT data of the readings, if available).

## License

MIT License; see `LICENSE`.

## Citation

The citation will be added when the paper is published.
