Native PyTorch LingBot-VA at 2V/4A
=================================

Jetson Thor prediction p50, September 15, 2026:

.. list-table::
   :header-rows: 1

   * - Native PyTorch 2V/4A
     - InstinctFlash FP8 2V/4A
   * - 2071.29 ms
     - 459.10 ms (4.51×)

The native cell is a new 21-call capture. The Flash cell retains the
`published same-day capture <../public_release_2026-09-15/qualification/va/run/cells/va-2v4a-fp8/receipt.json>`_;
all request, seed and executed-action feedback hashes match. The speedup compares
the same 2V/4A schedule. See `raw native timings <run_native_v1/cells/va-eager_native-2v4a/receipt.json>`_,
`saved actions <run_native_v1/cells/va-eager_native-2v4a/receipt.npz>`_ and
`independent audit <independent_audit_v1.json>`_.

This additional cell runs the original upstream ``VA_Server`` at two video
steps and four action steps, using the released Robotwin checkpoint. It retains
the checkpoint's BF16 precision, video CFG5 and action guidance1. The default
checkpoint declaration remains 25V/50A; the receipt records the explicit
sampling override separately. No InstinctFlash optimization passes are installed.

Install from the current source checkout and prepare the ``va`` vendor and
checkpoint environment using `the reproduction guide <../../REPRODUCE.rst>`_.
The initial ``thor-2026-09-15`` release wheel predates this additional benchmark
option; the updated source-installed benchmark is required. Restore both vendor
and asset activations. Run on an otherwise idle Thor from the checkout root::

    python -I -B -m benchmarks.regression.user_e2e \
      --matrix eval/va_native_2v4a_2026-09-15/matrix_v1.json \
      --cell va-eager_native-2v4a \
      --output-root va-native-2v4a-results \
      --fixture benchmarks/regression/fixtures/recorded_inputs_v1.npz

    python -I -B eval/va_native_2v4a_2026-09-15/validate_native_v1.py \
      --run va-native-2v4a-results \
      --output va-native-2v4a-results/validation.json

Use a new output directory. The capture rejects inherited ``IFL_*`` optimizer
options, competing GPU compute processes, wrong loaded schedules and nonfinite
actions. Its sampling and request loop are the existing public benchmark:
seven episodes of three cycles each, with the first episode discarded.
The table uses the median of the twelve measured continuation calls. First-cycle
and warmup timings remain in the raw receipt; episode resets occur outside timed
prediction calls.

The fixture, prompts, seeds 1300–1320 and recorded action feedback match the
published full-schedule Native PyTorch and InstinctFlash 2V/4A captures exactly.
``matrix_v1.json`` is a separate single-cell measurement, not a replacement
for the original three-arm checkpoint-default reproduction matrix. Its
validation checks the protocol, finite outputs and latency; it does not
certify task success or equivalence to the original 25V/50A policy.
