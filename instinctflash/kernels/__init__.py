"""GPU kernels for InstinctFlash's own inference engines.

Design rules shared by the engine kernels:

* Activations are quantized to FP8 E4M3 *per row* at the moment they are
  produced (norms, activations, attention outputs). A row's scale travels next
  to it, so no calibration pass is needed and one outlier token cannot degrade
  the resolution of every other token.
* Weights are quantized per output channel. FP8 GEMMs (``torch._scaled_mm``,
  :mod:`.fp8`) therefore run with unit scales and write their raw accumulator in
  BF16; the consuming kernel applies ``row_scale[m] * col_scale[n]`` (plus bias)
  while it reads the tile, so the rescale costs no extra launch or memory pass.
* Residual streams stay in FP32, except in the LingBot-VA engine, which keeps
  upstream's BF16 residual stream.
* Shapes that change between requests (prompt length) are read from device
  memory, so one captured graph serves every request that fits its buffers.
* On SM110 (Jetson Thor) matrix-product kernels can be compiled for sm100 so
  ``tl.dot`` uses tcgen05; kernels opt in per launch (:mod:`._triton`).

:mod:`.gemm_bf16` is the BF16 GEMM + ReLU² behind
:mod:`instinctflash.backends.bf16_linear` (Cosmos Edge), not an FP8 engine
kernel. :mod:`.cublaslt` is an experimental direct cuBLASLt FP8 GEMM used only
by ``benchmarks/native/bench_cublaslt.py`` and its unit test; the engines do
not use it.

Importing this package does not import Triton or require a GPU; the Triton
kernel modules (activation, attention, gemm_bf16, norm, rope, wan) import
Triton when they are themselves imported.
"""
