RTX 4090 clean-install qualification
====================================

Recorded 2026-09-27 (UTC) on one NVIDIA GeForce RTX 4090 (24564 MiB, driver 610.43.02; container memory limit 143 GB) from source commit ``218c193`` (the internal commit the runs cloned; this release carries the same package, adapter, benchmark and script code). Prediction p50 in milliseconds; each run compares the checkpoint's eager native reference, the default Runtime and the selected execution mode with the same inputs, warmups and measurement counts. Task quality is not certified.

.. list-table::
   :header-rows: 1

   * - Model
     - Cell
     - Precision
     - p50 (ms)
   * - pi05
     - pi05-eager_native (fp8 run)
     - native
     - 325.2
   * - pi05
     - pi05-runtime_default (fp8 run)
     - native
     - 108.2
   * - pi05
     - pi05-runtime_selected (fp8 run)
     - fp8
     - 113.7
   * - vla4
     - vla4-eager_native (fp8 run)
     - native
     - 867.8
   * - vla4
     - vla4-runtime_default (fp8 run)
     - native
     - 206.0
   * - vla4
     - vla4-runtime_selected (fp8 run)
     - fp8
     - 248.9
   * - vla2
     - vla2-eager_native (fp8 run)
     - native
     - 1070.8
   * - vla2
     - vla2-runtime_default (fp8 run)
     - native
     - 1037.2
   * - vla2
     - vla2-runtime_selected (fp8 run)
     - fp8
     - 172.7
   * - vla2
     - vla2-eager_native (numeric run)
     - native
     - 1053.7
   * - vla2
     - vla2-runtime_default (numeric run)
     - native
     - 1041.4
   * - vla2
     - vla2-runtime_selected (numeric run)
     - native
     - 170.1
   * - groot
     - groot-eager_native (fp8 run)
     - native
     - 190.8
   * - groot
     - groot-runtime_default (fp8 run)
     - native
     - 84.7
   * - groot
     - groot-runtime_selected (fp8 run)
     - fp8
     - 96.0
   * - va
     - va-eager_native (fp8 run)
     - native
     - 9346.3
   * - va
     - va-runtime_default (fp8 run)
     - native
     - 3607.9
   * - va
     - va-runtime_selected (fp8 run)
     - fp8
     - 5835.2
   * - va
     - va-2v4a-fp8 (fp8 run)
     - fp8
     - 708.3
   * - edge
     - edge-eager_native (fp8 run)
     - native
     - 1236.4
   * - edge
     - edge-runtime_default (fp8 run)
     - native
     - 1303.0
   * - edge
     - edge-runtime_selected (fp8 run)
     - fp8
     - 1208.7
   * - edge
     - edge-eager_native (numeric run)
     - native
     - 1176.9
   * - edge
     - edge-runtime_default (numeric run)
     - native
     - 1238.2
   * - edge
     - edge-runtime_selected (numeric run)
     - native
     - 1196.8
   * - nano
     - nano-eager_native (numeric run)
     - native
     - 12801.6
   * - nano
     - nano-runtime_default (numeric run)
     - native
     - 13577.6
   * - nano
     - nano-runtime_selected (numeric run)
     - native
     - 13733.3

Not tested on this node:

* dreamzero: The pinned model files (81.45 GB) exceed this node's 71.0 GB /dev/shm, and its 32.2 GB root filesystem also holds the environment; DreamZero was not installed or run on this node. This is a capacity limit of this node, not an observed failure.
