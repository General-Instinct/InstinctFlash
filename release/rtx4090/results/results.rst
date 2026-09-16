RTX 4090 measured results
=========================

Complete requested scope; full-eight qualification remains incomplete.

Each model run used one NVIDIA GeForce RTX 4090. Latencies are median public Runtime prediction time in milliseconds; the main table selects the fastest audited runtime at the original checkpoint schedule. History models use steady history predictions; setup and WebSocket round trips are outside these medians.

Measurements span separate host/environment cohorts. Each speedup compares paired runs within its own cohort; this table is not a comparison between hosts. The cohort column and environment records below identify the driver, allocation and measured Torch builds.

.. csv-table::
   :header: "Model", "Cohort", "PyTorch p50 (ms)", "InstinctFlash p50 (ms)"

   LingBot-VA,old_4x4090_cu130,8139.07,3411.92 (2.39×; native)
   LingBot-VLA-4B,old_4x4090_cu130,808.11,203.10 (3.98×; native)
   LingBot-VLA-V2-6B,old_4x4090_cu130,959.83,174.32 (5.51×; fp8)
   pi05,old_4x4090_cu130,308.72,109.77 (2.81×; native)
   GR00T N1.7,old_4x4090_cu130,162.69,83.86 (1.94×; native)
   Cosmos3 Edge DROID,replacement_1x4090_cu130,1276.81,1305.00 (0.98×; native)
   Cosmos3 Nano DROID,—,—,Not measured: capacity excluded
   DreamZero DROID,old_4x4090_cu130,113018.28,96925.01 (1.17×; fp8)

Cosmos3 Nano was not tested on the capacity-assessed node (driver 580.159.04): its cgroup host-memory limit was 31.00 GB. The current FP8 loading path requires at least 41.79 GB of original and packed CPU weights. Native CPU weights require 29.10 GB; the remaining margin was not validated. The family was excluded without an OOM trial under the requested scope. This is not an observed OOM or an unsupported-GPU result; no excluded mode counts as tested or passed. The exact hardware, original checkpoint, static tensor evidence and scope instruction are bound in results.json.

Changed computation
-------------------

These modes do not replace the main checkpoint-schedule measurements. VA 2V/4A uses its measured PyTorch 2V/4A baseline. Dynamic-cache comparisons are labeled when their PyTorch baseline uses the checkpoint cache schedule. Equal declared adaptive policy can produce different actual compute/reuse decisions; these are whole-runtime comparisons. Original per-call traces remain in the raw receipts. Native precision alone does not establish byte equality.

.. csv-table::
   :header: "Mode", "Cohort", "PyTorch p50 (ms)", "InstinctFlash p50 (ms)", "Comparison"

   LingBot-VA / 2v4a-fp8,old_4x4090_cu130,1147.51,627.33 (1.83×),same declared sampling policy
   LingBot-VA / 2v4a-native,old_4x4090_cu130,1147.51,452.64 (2.54×),same declared sampling policy
   DreamZero DROID / dynamic-fp8,old_4x4090_cu130,113018.28,53556.88 (2.11×),checkpoint baseline; changed computation
   DreamZero DROID / dynamic-native,old_4x4090_cu130,113018.28,66026.97 (1.71×),checkpoint baseline; changed computation

Cohort ``old_4x4090_cu130``: dreamzero, groot, pi05, va, vla2, vla4. Tested PyTorch builds: 2.11.0+cu130. Host inventory: 4 GPUs; each model used one. GPU memory from the host inventory: 24564 MiB; driver 580.82.09; PCIe maximum Gen4 x16; power limit 425 W. Effective cgroup RAM allocation: 367123234816 bytes (341.91 GiB); physical host RAM: 503.68 GiB. CPU quota: 92.16; logical CPUs: 192. This records this cohort's tested allocation; minimum and peak host RAM were not measured.

Cohort ``replacement_1x4090_cu130``: edge. Tested PyTorch builds: 2.10.0+cu130. Host inventory: 1 GPU; each model used one. GPU memory from the host inventory: 24564 MiB; driver 580.159.04; PCIe maximum Gen4 x16; power limit 450 W. Effective cgroup RAM allocation: 30999998464 bytes (28.87 GiB); physical host RAM: 251.53 GiB. CPU quota: 10.2; logical CPUs: 96. This records this cohort's tested allocation; minimum and peak host RAM were not measured.

Action differences are empirical comparisons on the recorded inputs. These results do not validate robot task quality. ``results.json`` records every admitted cell, exact schedule, loaded-source hashes, raw receipt/action member hashes and archive hashes. The adjacent ``evidence/`` directory contains the original, unmodified bytes.
