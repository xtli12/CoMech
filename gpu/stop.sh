#!/bin/bash
# Stop the experiment and re-occupy the GPUs.  Re-running gpu/start.sh later resumes where it stopped.
cd "$(dirname "$0")/.."
GPUS=${GPUS:-0,1,2,3}
if [ -f logs/run_all.pid ]; then
  PID=$(cat logs/run_all.pid)
  kill -TERM -- -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
  sleep 8
  pkill -TERM -f "gpu/ttrain.py" 2>/dev/null; pkill -TERM -f "gpu/run_eval.py" 2>/dev/null; sleep 3
fi
[ -f "$HOME/gpu_hold.py" ] && python3 "$HOME/gpu_hold.py" start --gpus "$GPUS"
[ -f "$HOME/gpu_hold.py" ] && python3 "$HOME/gpu_hold.py" status
