#!/usr/bin/env bash
# Rerun the README GR00T-N1.7-3B H100 pair with the exact protocol.
#
#   torch baseline arm : NVIDIA Gr00tPolicy.get_action eager, DROID embodiment,
#                        p50 of 15 calls / 3 warmup on the fixed obs0 case
#   DEFAULT arm        : Runtime.from_pretrained on this package with no flags — fast decode +
#                        backbone fastpath + DiT CUDA graphs, self-check gated
#   checkpoint         : nvidia/GR00T-N1.7-3B
#   quality gate       : 6 fixed-seed cases (two prompt switches), ours vs stock exact
#                        equality — the family tier is BITEXACT
#
# This drives reproduce_h100.py, which measures what the Runtime DEFAULT serves — not a module
# in isolation. The module-level decomposition (fastpaths and capture, arm by arm) remains
# verify_fastpaths.py.
#
# Needs the upstream environment (this box: /home/ubuntu/Isaac-GR00T/.venv), the checkpoint in
# the local HF cache, and a warm cache for the Qwen3 tokenizer (HF_HUB_OFFLINE=1 is fine).
#
#   IFL_GROOT_PY=/home/ubuntu/Isaac-GR00T/.venv/bin/python CUDA_VISIBLE_DEVICES=7 \
#     examples/groot_n17/reproduce_h100.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${IFL_GROOT_PY:-python}"
export GR00T_ROOT="${GR00T_ROOT:-$HOME/Isaac-GR00T}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "set CUDA_VISIBLE_DEVICES to one idle GPU (the published pair is a solo-H100 protocol)" >&2
  exit 2
fi
if [ ! -d "$GR00T_ROOT/gr00t" ]; then
  echo "GR00T_ROOT ($GR00T_ROOT) is not the Isaac-GR00T checkout" >&2
  exit 2
fi

exec "$PY" "$HERE/reproduce_h100.py"
