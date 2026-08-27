#!/usr/bin/env bash
# Rerun the README pi05 H100 pair (207 -> 73 ms, 2.84x, BITEXACT) with the exact protocol.
#
#   torch baseline arm : lerobot pi05 eager, full chunk = prefill + 10 denoise steps
#   T1 arm             : pi05_iwm static-KV CUDA graph + step tables (torch only, no engine)
#   checkpoint         : lerobot/pi05_libero_finetuned_v044 (bf16-stored)
#   sampling           : median of 15 chunks after 2 warm chunks, one idle H100
#
# Needs an interpreter with lerobot >= 0.6.1 and CUDA torch (this box: /home/ubuntu/tools/pi05env313).
#
#   IFL_PI05_PY=/path/to/python CUDA_VISIBLE_DEVICES=7 examples/pi05_vla/reproduce_h100.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${IFL_PI05_PY:-python}"

# The pair is a single-GPU measurement; refuse to run without an explicit device so a shared
# box never times against a GPU somebody else is training on.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "set CUDA_VISIBLE_DEVICES to one idle GPU (the published pair is a solo-H100 protocol)" >&2
  exit 2
fi

exec "$PY" "$HERE/reproduce_h100.py"
