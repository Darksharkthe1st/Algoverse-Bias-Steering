#!/bin/bash
cd /home/ubuntu/Algoverse-Bias-Steering
set -a; source .env; set +a
source .venv/bin/activate

# wait for adaptive_ablation (pid 5388) to finish
while kill -0 5388 2>/dev/null; do sleep 20; done
echo "=== adaptive_ablation finished at $(date), starting fixed_add ===" >> logs/queue.log

python -m src.bias_steer run configs/exp/fixed_add_qwen3_8b.py > logs/fixed_add_qwen3_8b.log 2>&1
echo "=== fixed_add finished at $(date). STOPPING QUEUE for adaptive_add calibration. ===" >> logs/queue.log
