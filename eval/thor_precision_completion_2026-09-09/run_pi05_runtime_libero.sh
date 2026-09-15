#!/usr/bin/env bash
set -euo pipefail
study_root=/home/guanming/ifl_eval/thor_precision_completion_20260909
export PATH=/home/guanming/.local/bin:/home/guanming/frt_env/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$study_root/pi05-public-sim-source"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
exec /home/guanming/frt_env/bin/python "$study_root/serve_pi05_runtime_libero.py" --precision "$1" --identity "$study_root/pi05-public-sim-$1-identity.json" > "$study_root/pi05-public-sim-$1.log" 2>&1
