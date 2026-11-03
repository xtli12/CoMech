#!/bin/bash
# One-screen progress report.
cd "$(dirname "$0")/.."
echo "== $(date '+%F %T')"
if [ -f logs/run_all.pid ] && kill -0 "$(cat logs/run_all.pid)" 2>/dev/null; then echo "running (PID $(cat logs/run_all.pid))"; else echo "not running"; fi
echo "-- data"
for s in train:180 test:40 val:12 ood:20 train_x:1620; do d=${s%%:*}; n=${s##*:}
  c=$(ls data/bank/$d/foot_*.npz 2>/dev/null | wc -l); echo "  bank $d: $c / $n"; done
[ -f logs/bank_qc_paper.json ] && echo "  QC: $(cat logs/bank_qc_paper.json | tr -d '\n ')"
for s in train train_x val test ood; do [ -f data/cache/$s/meta.json ] && echo "  cache $s: ok"; done
echo "  refined test FE: $(ls data/refined_test_f0.50/*.npz 2>/dev/null | wc -l) / 320"
echo "-- training"
[ -f logs/selftest.log ] && grep -E "SELFTEST|FAIL|full_batch" logs/selftest.log | cut -c1-200
for f in logs/train_*.log; do [ -f "$f" ] || continue; n=$(basename "$f" .log); n=${n#train_}
  if [ -f "results/models_gpu/$n.pt" ]; then echo "  $n: DONE"; else echo "  $n: $(grep -E '^[a-zA-Z0-9_]+ [0-9]+ ' "$f" | tail -1 | cut -c1-120)"; fi; done
echo "-- evaluation"
[ -f logs/eval.log ] && grep "^====" logs/eval.log | tail -8
[ -f logs/summary_gpu.txt ] && echo "-- summary" && cat logs/summary_gpu.txt
ls results_gpu_*.zip 2>/dev/null && echo "^ results to send back"
echo "-- GPUs"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
[ -f "$HOME/gpu_hold.py" ] && python3 "$HOME/gpu_hold.py" status 2>/dev/null | tail -6
tail -3 logs/run_all.log 2>/dev/null
