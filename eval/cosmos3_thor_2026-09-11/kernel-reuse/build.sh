#!/usr/bin/env bash
set -euo pipefail
probe_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(cd -- "$probe_dir/../../.." && pwd)
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -O3 --fmad=false -arch=sm_110a -shared -Xcompiler -fPIC \
  -I"$repo_dir/serving/csrc/kernels" "$probe_dir/bridge.cu" \
  "$repo_dir/serving/csrc/kernels/norm.cu" \
  "$repo_dir/serving/csrc/kernels/activation.cu" \
  "$repo_dir/serving/csrc/kernels/elementwise.cu" \
  "$repo_dir/serving/csrc/kernels/silu_mul_qwen36.cu" \
  -o "$probe_dir/kernels.so"
