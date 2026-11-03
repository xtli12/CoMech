#!/bin/bash
# Render the three-dimensional images and build the figures and supplementary tables of the paper.
#   RB_TEMPLATE=/path/to/rb_template.npz bash figsrc/make_figures.sh [outdir] [gpu_results_dir]
# Needs results/ and data/ from the pipeline (exp/run_all.sh).  Panels built from inputs that are not distributed
# (screenshots of the FE atlas: ATLAS_CROPS / FIG2_CROPS; CT data of the readings: data/ct, CT_READ_DIR) are skipped
# with a message when those inputs are missing.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
O=${1:-out/figures}; OT=${O}_tables; G=${2:-results_gpu}
: "${RB_TEMPLATE:?set RB_TEMPLATE to the label raster of the atlas foot (rb_template.npz)}"
export RB_TEMPLATE COMECH="$(pwd)" COMECH_MODELS=${COMECH_MODELS:-results/models_long}
export RENDER_OUT=${RENDER_OUT:-results/renders} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p "$O" "$OT" "$RENDER_OUT" logs
run() { "$@" > "logs/fig_$(basename "${2:-$1}" .py).log" 2>&1 || echo "skipped or failed: $* (see logs/)"; }

# renders (sphere tracing of the engine feet) and the latent states of the virtual child
for f in render_maps render_heldout render_actions render_icons render_icons_roz child_latent; do run $PY figsrc/$f.py; done
[ -f data/ct/ct_readings.csv ] && run $PY figsrc/render_twin.py

# main figures (Fig. 3 and Supplementary Figs 1-5 are built from the FE atlas: analysis/)
run $PY figsrc/fig1_concept_v3.py "$O"
export FIG2_RB=1 FIG2_RENDERS="$RENDER_OUT"
run $PY analysis/fig2.py "$O/fig2"
run $PY analysis/fig3.py "$O/fig3"
run $PY figsrc/fig4_concept.py "$O"
run $PY figsrc/fig567_v3.py "$O"
run $PY figsrc/fig8_v3.py "$O/fig8"

# supplementary figures and tables
run $PY analysis/supp_figs.py "$O"
for f in supp_engine supp_fig8 supp_fig9; do run $PY figsrc/$f.py "$O"; done
run $PY figsrc/supp_fig11.py "$G" "$O"
run $PY analysis/supp_tables.py "$OT/supp_tables.tex"
for f in supp_tables_engine supp_tables_models; do run $PY figsrc/$f.py "$OT"; done
run $PY figsrc/supp_tables_ct.py "$OT/supp_tables_ct.tex"
run $PY figsrc/supp_tables_gpu.py "$G" "$OT"
echo "figures in $O, tables in $OT"
