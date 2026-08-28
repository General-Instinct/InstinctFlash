#!/usr/bin/env bash
# Rerun the README LingBot-VLA-V2 H100 pair with the exact protocol.
#
#   torch baseline arm : upstream LingbotVLAv2Server.infer, eager, in-process,
#                        12 calls p50 / 3 warmup
#   DEFAULT arm        : Runtime.from_pretrained with no flags — static-KV denoise graph +
#                        vision/prefill graphs + GPU preprocessing, self-check gated
#   checkpoint         : robbyant/lingbot-vla-v2-6b-robotwin (use_length 50, 10 steps, 3 cams)
#   quality gate       : 6 fixed-seed cases, ours-vs-stock inside the family's recorded
#                        stock-vs-stock envelope (fused-MoE nondeterminism; tier NUMERIC)
#
# This drives reproduce_h100.py, which measures what the Runtime DEFAULT serves — not a module
# in isolation. The module-level 6-case gate for the denoise graph alone remains
# verify_static_capture.py.
#
# Needs the upstream environment (this box: /home/ubuntu/lingbot-vla-v2-repo/.venv), the
# checkpoint in the local HF cache, and the Qwen3-VL-4B-Instruct processor cached.
#
#   IFL_VLA2_PY=/home/ubuntu/lingbot-vla-v2-repo/.venv/bin/python CUDA_VISIBLE_DEVICES=7 \
#     examples/lingbot_vla_v2/reproduce_h100.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${IFL_VLA2_PY:-python}"
ROOT="${LINGBOT_VLA_V2_ROOT:-$HOME/lingbot-vla-v2-repo}"
export QWEN3VL_PATH="${QWEN3VL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "set CUDA_VISIBLE_DEVICES to one idle GPU (the published pair is a solo-H100 protocol)" >&2
  exit 2
fi
if [ ! -f "$ROOT/deploy/lingbot_vla_v2_policy.py" ]; then
  echo "LINGBOT_VLA_V2_ROOT ($ROOT) is not the upstream checkout" >&2
  exit 2
fi

# Upstream's reset() resolves configs/robot_configs/<robot>.yaml relative to its project root.
cd "$ROOT"
exec "$PY" "$HERE/reproduce_h100.py"
