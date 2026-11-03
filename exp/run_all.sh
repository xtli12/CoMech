#!/bin/bash
# Full CoMech pipeline on the data engine of the atlas foot, in the order used for the paper.
#   RB_TEMPLATE=/path/to/rb_template.npz bash exp/run_all.sh
# RB_TEMPLATE: label raster of the atlas foot (available from the corresponding author on request).
# Optional: PYTHON (interpreter), NWORK / NGPU (engine workers / GPUs, see exp/rb_run_engine_v4.py).
# Every step skips outputs that already exist, so the script can be interrupted and restarted.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
: "${RB_TEMPLATE:?set RB_TEMPLATE to the label raster of the atlas foot (rb_template.npz)}"
export RB_TEMPLATE COMECH="$(pwd)"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
mkdir -p logs results

# ---- checks
$PY fe/test_patch.py
$PY ml/test_ad.py

# ---- data engine: virtual cohort and taxonomy geometries, then every FE solve (atlas analogue, TCC-MechBank,
#      held-out sweeps, taxonomy fingerprints, cohort envelopes) with a pool of workers sharing the GPUs
$PY exp/cohort.py
$PY -u exp/rb_run_engine_v4.py
$PY -c "from ml.data import Norm; Norm.cached()"

# ---- training: 1,600 steps, then a warm restart for 2,400 steps (identical for every model)
ALL=comech_s0,comech_s1,comech_s2,comech_noswap,comech_nosens,comech_none,actcat,deeponet,geo,fno
for m in ${ALL//,/ }; do
  $PY -u exp/train_models.py $m 1600
  $PY -u exp/train_continue.py $m 2400
done

# ---- experiments
export COMECH_MODELS=results/models_long
CM=comech_s0,comech_s1,comech_s2,comech_noswap,comech_nosens,comech_none
ENS=comech_s0,comech_s1,comech_s2
$PY -u exp/atlas_compare.py
SKIP_LATENCY=1 $PY -u exp/e_fields.py $ALL
$PY -u exp/e_zeroshot.py $ALL
$PY -u exp/e_taxonomy.py $CM
$PY -u exp/e_tipping.py $ALL
$PY -u exp/e_cohort.py $ENS
$PY -u exp/e_equiv_plan.py $ALL $ENS
$PY -u exp/convergence.py > results/convergence.log
$PY -u exp/e_cohort_fe.py
$PY -u exp/e_maps.py comech_s0 $ENS
# CT-informed twins: need the CT reading table (data/ct/ct_readings.csv), which is not distributed
if [ -f data/ct/ct_readings.csv ]; then
  NPROC=6 $PY -u exp/e_ct_twins.py data/ct/ct_readings.csv
  NPROC=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -u exp/e_ct_twins_fe.py data/ct/ct_readings.csv
  $PY -u exp/summarize_ct.py
fi
# high-resolution FE of the rendered showcase cases and the grid check of the rehearsed plan
TAGS=child_pre,child_resect,child_plan,P09L_pre,P09L_resect,ref_intact
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -u exp/hires_showcase.py 1.0 $TAGS
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -u exp/hires_showcase.py 0.5 $TAGS
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -u exp/hires_check.py
# latency: run on an otherwise idle machine
REDO_LATENCY=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -u exp/e_fields.py comech_s0
$PY -u exp/rb_bank_stats.py
$PY -u exp/summarize.py                     # results/numbers.json

# ---- figures and supplementary tables
bash figsrc/make_figures.sh
