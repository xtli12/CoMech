#!/bin/bash
# Start the whole GPU experiment in the background (survives closing the terminal / SSH session).
#   bash gpu/start.sh                 (optional: GPUS=0,1,2,3 NPROC=48 PY=python3 bash gpu/start.sh)
cd "$(dirname "$0")/.."
mkdir -p logs
if [ -f logs/run_all.pid ] && kill -0 "$(cat logs/run_all.pid)" 2>/dev/null; then
  echo "already running (PID $(cat logs/run_all.pid)); progress: bash gpu/status.sh"; exit 1
fi
setsid nohup bash gpu/run_all.sh >> logs/run_all.log 2>&1 < /dev/null &
echo $! > logs/run_all.pid
sleep 2
echo "started (PID $(cat logs/run_all.pid))"
echo "progress:  bash gpu/status.sh      full log:  tail -f logs/run_all.log      stop:  bash gpu/stop.sh"
