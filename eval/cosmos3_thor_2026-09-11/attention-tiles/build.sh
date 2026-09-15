#!/usr/bin/env bash
set -euo pipefail
probe_dir=$(cd -- "$(dirname -- "$0")" && pwd)
: "${CUTLASS_ROOT:?Set CUTLASS_ROOT to the pinned CUTLASS checkout}"
/usr/local/cuda/bin/nvcc -O3 --expt-relaxed-constexpr -arch=sm_110a -shared -Xcompiler -fPIC \
  -I"$CUTLASS_ROOT/include" -I"$CUTLASS_ROOT/tools/util/include" \
  -I"$CUTLASS_ROOT/examples/77_blackwell_fmha" \
  "$probe_dir/bf16_fmha.cu" -o "$probe_dir/bf16_fmha.so"
