#!/bin/bash
cd /home/ubuntu/Algoverse-Bias-Steering
set -a; source .env; set +a
source .venv/bin/activate
while kill -0 41484 2>/dev/null; do sleep 20; done
echo "=== c16 finished at $(date), starting c20 ===" >> logs/queue2.log
python -m src.bias_steer run configs/exp/adaptive_add_linear_c20_qwen3_8b.py > logs/adaptive_add_linear_c20.log 2>&1
echo "=== c20 finished at $(date) ===" >> logs/queue2.log
