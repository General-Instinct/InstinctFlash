#!/usr/bin/env bash
set -euo pipefail
cutlass_root=${1:?CUTLASS checkout required}
output_dir=${2:?output directory required}
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$output_dir"
/usr/local/cuda/bin/nvcc -shared -Xcompiler -fPIC -std=c++17 \
  --expt-relaxed-constexpr -O3 -arch=sm_110a \
  -I"$cutlass_root/include" -I"$cutlass_root/tools/util/include" \
  "$source_dir/linear_relu2_sm110.cu" -o "$output_dir/libinstinctflash_bf16.so"
