#!/usr/bin/env bash
set -euo pipefail
: "${GEMM_SOURCE:?Path to unchanged serving/csrc/gemm}"
probe_dir=$(cd -- "$(dirname -- "$0")" && pwd)
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_110a -shared -Xcompiler -fPIC \
  -I"$GEMM_SOURCE" "$probe_dir/bridge.cu" "$GEMM_SOURCE/gemm_runner.cu" \
  -L/usr/local/cuda/lib64 -lcublasLt -lcublas -lcudart -o "$probe_dir/bf16_gemm.so"
