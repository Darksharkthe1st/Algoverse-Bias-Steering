#!/bin/bash
cd /home/ubuntu/Algoverse-Bias-Steering
set -a; source .env; set +a
source .venv/bin/activate
for c in c1 c8 c16 c20 c30; do
  echo "=== $c starting at $(date) ===" >> logs/linear_add_queue.log
  python -m src.bias_steer run configs/exp/linear_add_${c}_qwen3_8b.py > logs/linear_add_${c}.log 2>&1
  status=$?
  echo "=== $c finished at $(date) with exit $status ===" >> logs/linear_add_queue.log
  if [ $status -ne 0 ]; then
    echo "=== $c FAILED, halting queue ===" >> logs/linear_add_queue.log
    break
  fi
done
echo "=== queue done at $(date) ===" >> logs/linear_add_queue.log
