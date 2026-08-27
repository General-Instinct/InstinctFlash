#!/usr/bin/env bash
# Rerun the README LingBot-VLA-4B H100 pair (671 -> 185 ms, 3.62x, BITEXACT) with the exact
# protocol.
#
#   torch baseline arm : official LingbotVLAServer.infer, in-process, 12 calls p50 / 3 warmup
#   T1 arm             : static-KV CUDA graph on the 10-step denoise loop (torch only, no engine)
#   checkpoint         : robbyant/lingbot-vla-4b-posttrain-robotwin (chunk 50, 10 steps, 3 cams)
#
# This drives verify_static_capture.py, THE protocol artifact behind the row: it measures the
# stock arm first, installs the capture on the same instance with identical torch seeds, gates
# bitexactness on six cases (incl. a re-prefilled new prompt), and prints both p50s. The README
# table's stock number is the official websocket server (670.9 ms); the in-process stock this
# script measures agreed within 0.3% (672.7 ms) — same computation, minus the ws hop.
#
# Needs the upstream environment (this box: /home/ubuntu/lingbot-vla-repo/.venv) and the
# checkpoint in the local HF cache.
#
#   IFL_VLA4B_PY=/home/ubuntu/lingbot-vla-repo/.venv/bin/python CUDA_VISIBLE_DEVICES=7 \
#     examples/lingbot_vla/reproduce_h100.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${IFL_VLA4B_PY:-python}"
ROOT="${LINGBOT_VLA_ROOT:-$HOME/lingbot-vla-repo}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "set CUDA_VISIBLE_DEVICES to one idle GPU (the published pair is a solo-H100 protocol)" >&2
  exit 2
fi
if [ ! -f "$ROOT/deploy/lingbot_vla_policy.py" ]; then
  echo "LINGBOT_VLA_ROOT ($ROOT) is not the upstream checkout" >&2
  exit 2
fi

# Upstream's reset() resolves configs/robot_configs/<robot>.yaml relative to its project root.
cd "$ROOT"
exec "$PY" "$HERE/verify_static_capture.py"
