FlashRT inference backend
=========================

FlashRT contains the CUDA kernels and model frontends used by InstinctFlash.
Use the main repository's ``Runtime`` interface to load a checkpoint and choose
native or FP8 execution. Installing this backend does not enable FP8 implicitly.
The frontend source, CUDA kernels and build tools are available in this directory.

Install the Thor build
----------------------

The ``thor-2026-09-15`` release provides the same libraries built from the
source below. For the Python 3.12 / aarch64 vendor environments using
Torch 2.11.0+cu130, replace the bootstrap's pure Python backend wheel::

    python -m pip install --force-reinstall --no-deps \
      'https://github.com/General-Instinct/InstinctFlash/releases/download/thor-2026-09-15/flash_rt-0.1.0-cp312-cp312-linux_aarch64.whl#sha256=3262c54f7094dd3b77ef801e7208b157d891d8f50240374ed4f363c7f13e3c7f'
    python -m pip check

Cosmos uses Python 3.13 and the separate BF16 C library. Download and verify it
in a directory you will keep, then pass its absolute path to the reproduction
command's ``--library IFL_BF16_KERNEL_LIBRARY=...`` option::

    curl --fail --location --output libinstinctflash_bf16.so \
      https://github.com/General-Instinct/InstinctFlash/releases/download/thor-2026-09-15/libinstinctflash_bf16.so
    echo '9578bb119e00bca6b74b92fddcef060da6e426211c76d73fb11e9cfaf276cd6e  libinstinctflash_bf16.so' | sha256sum --check

The library was built with CUDA 13.2.78 and GCC 13.3 for SM110. Keep the
documented vendor stack; other CUDA/Python/GPU combinations need their own
build and qualification. The Python 3.12 wheel cannot be installed in Cosmos's
Python 3.13 environment.

Jetson Thor build
-----------------

The Thor build targets SM110 and CPython 3.12 on Linux aarch64. It requires a CUDA
toolkit with SM110 support, a C++ compiler, CMake, and Python development headers.
Install the Python build dependencies in the intended Python environment::

    python -m pip install 'setuptools<82' wheel pybind11

The standalone FA2 build supplies the attention extension needed by the Thor
frontends. Build directories must be new; each build retains its source hashes,
compiler commands and results. Run from the main repository root::

    python serving/scripts/build_thor_fa2.py \
      --python "$(command -v python)" \
      --build-root /tmp/ifl-fa2-build --output-dir /tmp/ifl-fa2-output

Fetch the exact CUTLASS revision used by the remaining native kernels::

    git clone https://github.com/NVIDIA/cutlass.git /tmp/ifl-cutlass
    git -C /tmp/ifl-cutlass checkout da5e086dab31d63815acafdac9a9c5893b1c69e2

Then build the engine, FMHA and shared BF16 library. Supply the FA2 library path
and SHA-256 from its completed build report::

    python serving/scripts/build_thor_native.py \
      --python "$(command -v python)" --cutlass-root /tmp/ifl-cutlass \
      --fa2-artifact /path/to/flash_rt_fa2.cpython-312-aarch64-linux-gnu.so \
      --fa2-sha256 FA2_SHA256 \
      --build-root /tmp/ifl-native-build --output-dir /tmp/ifl-native-output
    python -m pip install --force-reinstall --no-deps /tmp/ifl-native-output/wheels/flash_rt-*.whl
    python -m pip check

The backend wheel contains its Python frontends, configuration files and native
libraries. The separate ``native/libinstinctflash_bf16.so`` output can be selected
with ``IFL_BF16_KERNEL_LIBRARY`` for the Edge NUMERIC recipe. Build/import checks
do not certify model actions or task success; run the main repository's paired
reproduction and serving smoke commands in the model's vendor environment.
The explicit reinstall replaces the source-only wheel of the same version
while preserving the vendor environment's checked dependencies.

RTX 5090 build
--------------

For checkpoint-native pi0.5 LIBERO, use a separate Python 3.10 environment with
``examples/pi05_vla/requirements-sm120.lock`` and the hash-checked official
dependency installer ``scripts/install_pi05_transformers.py``. The
``pi05-sm120`` package extra declares its direct model/simulator dependencies;
the lock file pins the complete tested environment. Do not merge its LeRobot
dependency constraints into a different model's environment.

The public SM120 pi0.5 frontend requires the real 8-D robot state (end-effector
position, axis-angle orientation, and two gripper joint positions). It computes
the checkpoint's 50-action chunk and returns the qualified 10-action horizon.
Use ``model.calibrate(real_samples, prompt=task, percentile=...)`` before
``model.predict(images, state=robot_state)`` for fixed-data calibration. The
checkpoint's own processors supply state tokens and MEAN_STD action decoding.
Full commands and paired qualification evidence are in
``examples/pi05_vla/README.md``.

RTX 5090 uses Linux x86_64, CPython 3.10, CUDA 12.8 and ``GPU_ARCH=120``.
It cannot reuse the Thor CPython 3.12/aarch64 wheel. Start from the
LingBot-VA environment pins and install the complete verification extras::

    python -m pip install -r requirements-serving.txt \
      --extra-index-url https://download.pytorch.org/whl/cu126
    python -m pip install -e '.[test,diffusion,serve,viz]'
    python -m pip install -e './serving[all]'

The legacy/GROOT attention import requires a local FlashAttention build on
this ABI. FlashAttention 2.8.3 supports a native SM120-only build::

    CUDA_HOME=/path/to/cuda-12.8 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=8 \
      python -m pip wheel --no-build-isolation --no-deps \
      --wheel-dir /path/to/wheels flash-attn==2.8.3
    python -m pip install /path/to/wheels/flash_attn-2.8.3-*.whl

Fetch the same CUTLASS revision used by the Thor release, then configure the
full backend for SM120. ``FA2_ARCH_NATIVE_ONLY`` deliberately excludes other
GPU architectures from this host-specific wheel::

    git clone --depth 1 --branch v4.4.2 \
      https://github.com/NVIDIA/cutlass.git serving/third_party/cutlass
    cmake -S serving -B /path/to/flashrt-sm120-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/path/to/cuda-12.8/bin/nvcc \
      -DPython3_EXECUTABLE="$(command -v python)" \
      -DGPU_ARCH=120 -DFA2_ARCH_NATIVE_ONLY=ON \
      -DFLASH_RT_BUILD_JAX_FFI=ON
    cmake --build /path/to/flashrt-sm120-build --parallel 8
    python -m pip wheel --no-build-isolation --no-deps \
      --wheel-dir /path/to/wheels ./serving

The resulting wheel contains ``flash_rt_kernels`` (SM120a),
``flash_rt_fa2`` (SM120), and, when JAX is installed at configure time,
``flash_rt_jax_ffi`` (SM120a). The separate ``flash_rt_fp4`` module remains
SM100/SM110-only: compiling its SM100 multicast kernels for SM120 produces a
loadable binary but its first GEMM is rejected at runtime. SM120 NVFP4/W4A4
routes live in ``flash_rt_kernels`` and are selected independently.

Build the LingBot-VA A1--A7 bit-exact native chain from the repository root::

    cmake -S instinctflash/native -B /path/to/ifl-sm120-build \
      -DCMAKE_CUDA_COMPILER=/path/to/cuda-12.8/bin/nvcc \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build /path/to/ifl-sm120-build --parallel 8

Export the seven library paths listed in ``instinctflash/native/README.md``.
The planner will apply only the prefix whose ABI and exact Torch/CUDA/cuBLASLt
requirements load successfully.

Other platforms
---------------

The main ``CMakeLists.txt`` supports additional GPU targets. Set ``GPU_ARCH``
explicitly and build against the Python ABI used for inference. A Thor wheel
cannot be installed on x86_64 or a different CPython ABI. JAX FFI is optional:
Torch-only builds pass ``-DFLASH_RT_BUILD_JAX_FFI=OFF``.

Licenses
--------

FlashRT's license is in ``LICENSE``. Vendored FlashAttention and CUTLASS sources
retain their original notices; the binary wheel carries copies under
``flash_rt/notices``. The main InstinctFlash package has its own license.
