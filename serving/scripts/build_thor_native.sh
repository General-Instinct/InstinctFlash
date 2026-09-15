#!/usr/bin/env bash
# Fresh Thor kernel, FMHA, BF16 and flash-rt wheel build. No environment installation.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -B "${script_dir}/build_thor_native.py" "$@"
