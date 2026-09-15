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
