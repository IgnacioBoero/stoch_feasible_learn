#!/usr/bin/env bash
set -euo pipefail

SWEEP_ID="${1:-}"
COUNT_PER_GPU="${2:-8}"

if [ -z "$SWEEP_ID" ]; then
  echo "Usage: bash launch_two_gpu_sweep.sh <entity/project/sweep_id> [count_per_gpu]"
  echo "Example: bash launch_two_gpu_sweep.sh iboero/maxent-feasible-fmnist/abc123 8"
  exit 1
fi

echo "Launching two W&B agents:"
echo "  GPU 0: $COUNT_PER_GPU runs"
echo "  GPU 1: $COUNT_PER_GPU runs"
echo "  Sweep: $SWEEP_ID"

CUDA_VISIBLE_DEVICES=0 wandb agent "$SWEEP_ID" --count "$COUNT_PER_GPU" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 wandb agent "$SWEEP_ID" --count "$COUNT_PER_GPU" &
PID1=$!

wait "$PID0"
wait "$PID1"

echo "Both sweep agents finished."

