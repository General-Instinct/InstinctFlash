#!/usr/bin/env bash
# Build only the vendored FA2 extension for Thor; never install into an environment.
# Example (Python needs pybind11; CUDA 13 must support SM110):
#   ./serving/scripts/build_thor_fa2.sh --python /path/to/python3.12 \
#     --cmake /path/to/cmake --nvcc /usr/local/cuda/bin/nvcc \
#     --build-root /dev/shm/ifl-fa2-build --output-dir /path/to/new/fa2 --jobs 2
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -B "${script_dir}/build_thor_fa2.py" "$@"
