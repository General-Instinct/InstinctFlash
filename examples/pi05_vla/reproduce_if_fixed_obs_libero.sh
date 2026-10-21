#!/usr/bin/env bash
# Compare pi05 eager vs InstinctFlash using one fixed real LIBERO observation.
# Median of 15 full chunks after 2 warm chunks; observation preparation is excluded.
# Needs an interpreter with LeRobot, LIBERO, InstinctFlash, and CUDA torch installed.
#
#   IFL_PI05_PY=/path/to/python CUDA_VISIBLE_DEVICES=0 examples/pi05_vla/reproduce_if_fixed_obs_libero.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${IFL_PI05_PY:-python}"

# Select an idle GPU explicitly for this single-GPU measurement.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "set CUDA_VISIBLE_DEVICES to one idle GPU" >&2
  exit 2
fi

exec "$PY" "$HERE/reproduce_if_fixed_obs_libero.py"
