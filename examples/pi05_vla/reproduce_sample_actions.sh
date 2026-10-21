#!/usr/bin/env bash
# Compare pi05 eager vs InstinctFlash sample_actions latency in live LIBERO episodes.
# Uses the Python script's defaults unless arguments are supplied.
# Needs an interpreter with LeRobot, LIBERO, InstinctFlash, and CUDA torch installed.
#
#   IFL_PI05_PY=/path/to/python CUDA_VISIBLE_DEVICES=0 examples/pi05_vla/reproduce_sample_actions.sh --output-dir /path/to/new/results
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${IFL_PI05_PY:-python}"

# Select an idle GPU explicitly for this single-GPU measurement.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "set CUDA_VISIBLE_DEVICES to one idle GPU" >&2
  exit 2
fi

exec "$PY" "$HERE/reproduce_sample_actions.py" "$@"
