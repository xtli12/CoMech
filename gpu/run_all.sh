#!/bin/bash
# GPU scaling experiment for CoMech: data (CPU) -> refined FE reference (CPU) -> self-test, training, evaluation (GPU)
# -> summary and packed results.  Every stage skips work that is already done, so the script can simply be re-run.
#
#   bash gpu/start.sh                      # runs this script in the background (recommended)
#   GPUS=0,1,2,3 NPROC=48 bash gpu/run_all.sh
#
# GPUs: the GPUs in $GPUS are released with ~/gpu_hold.py only for the self-test, training and evaluation, and
# are re-occupied with ~/gpu_hold.py when this script ends for any reason (normal end, error or kill).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
GPUS=${GPUS:-0,1,2,3}
STAGES=${STAGES:-env,data,refined,train,eval,pack}
HOLD=${HOLD:-$HOME/gpu_hold.py}
mkdir -p logs results/models_gpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*"; }
has() { [[ ",$STAGES," == *",$1,"* ]]; }

# ---- process counts from cores and memory ----
CORES=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)   # nproc honours OMP_NUM_THREADS, which is exported as 1 above
MEMGB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "${NPROC:-auto}" = auto ]; then NPROC=$(( CORES > 8 ? CORES - 4 : CORES )); M=$(( MEMGB / 2 )); [ $M -lt $NPROC ] && NPROC=$M; fi
NPROC_HI=$(( MEMGB / 4 )); [ $NPROC_HI -gt $NPROC ] && NPROC_HI=$NPROC; [ $NPROC_HI -lt 1 ] && NPROC_HI=1
log "cores=$CORES mem_avail=${MEMGB}GB NPROC=$NPROC NPROC_HI=$NPROC_HI GPUS=$GPUS STAGES=$STAGES"

# ---- GPU hold handling ----
RELEASED=0
release_gpus() {
  if [ -f "$HOLD" ]; then
    log "releasing GPUs $GPUS (gpu_hold.py stop)"; python3 "$HOLD" stop --gpus "$GPUS"; RELEASED=1
    for i in $(seq 1 24); do            # wait until the placeholder jobs have freed the memory (max 2 min)
      busy=0
      for g in ${GPUS//,/ }; do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
        [ -n "$u" ] && [ "$u" -gt 2000 ] && busy=1
      done
      [ $busy = 0 ] && break; sleep 5
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  else
    log "WARNING: $HOLD not found; GPUs are used without releasing a hold"
  fi
}
rehold_gpus() {
  if [ "$RELEASED" = 1 ] && [ -f "$HOLD" ]; then
    log "re-occupying GPUs $GPUS (gpu_hold.py start)"; python3 "$HOLD" start --gpus "$GPUS"; python3 "$HOLD" status; RELEASED=0
  fi
}
trap rehold_gpus EXIT
trap 'log "interrupted"; kill $(jobs -p) 2>/dev/null; pkill -TERM -f "gpu/ttrain.py" 2>/dev/null; sleep 3; exit 130' INT TERM

# ---- 0. environment ----
if has env; then
  log "== environment"; $PY gpu/env_check.py; rc=$?
  if [ $rc = 2 ]; then log "missing Python packages (see above); install them and re-run"; exit 2; fi
  if [ $rc = 3 ]; then log "PyTorch cannot see a GPU; check the CUDA build of torch and re-run"; exit 3; fi
fi

# ---- 1. data (CPU only; GPUs stay held) ----
if has data; then
  log "== data"
  [ -d data/sweeps ] || $PY -m zipfile -e results/fe_ground_truth.zip .
  [ -d results/atlas_analog ] || $PY -m zipfile -e results/atlas_analog_fields.zip .
  [ -f data/cohort/taxonomy_geo.npz ] || $PY exp/cohort.py
  $PY gpu/gen_bank.py paper "$NPROC" || { log "bank regeneration / QC failed - see logs/bank_qc_paper.json"; exit 4; }
  $PY - <<'EOF' || exit 5
import json; q = json.load(open("logs/bank_qc_paper.json")); print("bank QC:", q)
assert q["n_bad"] == 0, "QC failed"
EOF
  $PY gpu/gen_bank.py extra "$NPROC" 1620 21
  $PY gpu/cache.py train train_x val test ood
fi

# ---- 2. refined FE reference of the test split (CPU) ----
if has refined; then
  log "== refined FE reference (320 configurations, grid refined twofold)"
  $PY gpu/refined.py solve "$NPROC_HI"
fi

# ---- 3. self-test and training (GPU) ----
if has train; then
  release_gpus
  G0=${GPUS%%,*}
  log "== self-test on GPU $G0"
  CUDA_VISIBLE_DEVICES=$G0 $PY gpu/selftest.py > logs/selftest.log 2>&1; rc=$?
  tail -25 logs/selftest.log
  if [ $rc != 0 ]; then log "self-test FAILED - send logs/selftest.log; nothing was trained"; exit 6; fi
  log "== training on GPUs $GPUS"
  rm -rf results/models_gpu/locks        # stale locks of runs killed by an earlier stop/restart
  mkdir -p results/models_gpu/locks
  for g in ${GPUS//,/ }; do
    (
      for name in $($PY gpu/list_runs.py); do
        [ -f "results/models_gpu/$name.pt" ] && continue
        mkdir "results/models_gpu/locks/$name" 2>/dev/null || continue
        log "GPU $g: start $name"
        CUDA_VISIBLE_DEVICES=$g $PY gpu/ttrain.py "$name" >> "logs/train_$name.log" 2>&1
        if [ -f "results/models_gpu/$name.pt" ]; then log "GPU $g: done $name"; else log "GPU $g: FAILED $name (see logs/train_$name.log)"; fi
        rmdir "results/models_gpu/locks/$name" 2>/dev/null
      done
    ) &
  done
  wait
  log "training finished"; ls -la results/models_gpu/*.pt 2>/dev/null
fi

# ---- 4. evaluation (GPU) ----
if has eval; then
  [ "$RELEASED" = 1 ] || release_gpus
  G0=${GPUS%%,*}
  log "== evaluation on GPU $G0"
  CUDA_VISIBLE_DEVICES=$G0 COMECH_MODELS=results/models_gpu $PY gpu/run_eval.py > logs/eval.log 2>&1
  tail -5 logs/eval.log
  [ -d data/refined_test_f0.50 ] && CUDA_VISIBLE_DEVICES=$G0 COMECH_MODELS=results/models_gpu $PY gpu/refined.py compare > logs/refined_compare.log 2>&1
  tail -3 logs/refined_compare.log 2>/dev/null
fi
rehold_gpus

# ---- 5. summary and packed results ----
if has pack; then
  $PY gpu/summarize_gpu.py
  $PY gpu/pack.py
fi
log "ALL_DONE"
