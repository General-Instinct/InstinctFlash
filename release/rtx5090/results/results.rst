RTX 5090 measured results
=========================

Complete requested scope; full-eight qualification remains incomplete.

Each model run used one NVIDIA GeForce RTX 5090. Latencies are median public Runtime prediction time in milliseconds; the main table selects the fastest audited runtime at the original checkpoint schedule. History models use steady history predictions; setup and WebSocket round trips are outside these medians.

.. csv-table::
   :header: "Model", "PyTorch p50 (ms)", "InstinctFlash p50 (ms)"

   LingBot-VA,8143.99,3358.39 (2.42×; native)
   LingBot-VLA-4B,803.49,102.84 (7.81×; native)
   LingBot-VLA-V2-6B,980.52,145.18 (6.75×; numeric)
   pi05,303.81,100.67 (3.02×; native)
   GR00T N1.7,159.07,86.89 (1.83×; native)
   Cosmos3 Edge DROID,951.24,958.22 (0.99×; native)
   Cosmos3 Nano DROID,—,—
   DreamZero DROID,—,—

This is recorded pipeline coverage, separate from no-offload headline eligibility. Historical VA timings and full action evidence remain recorded; their CPU staging outside prediction timing is not a GPU-resident end-to-end result.

DreamZero DROID was not tested on the bound host with a 54.00 GB cgroup allocation. Original plus packed weight payload totals 54.66 GB. Mapped or reclaimable pages mean this arithmetic is not a proved nonreclaimable-RAM lower bound. Reliable native cold-load headroom was not established. All four modes were excluded under the requested skip policy without an OOM trial; none counts as tested or passed. This is not an unsupported-GPU finding or a minimum-RAM measurement. The unchanged exclusion and exact catalog mapping are included.

Cosmos3 Nano was not timed: its original BF16 comparison arm requires decoder-layer streaming under the unchanged 8 GiB reserve on this host. All three paired modes are excluded by the no-offload benchmark policy. This is not an observed OOM or a claim that explicit FP8 cannot fit. The static arithmetic and original source references are included.

Additional declared schedules
-----------------------------

These modes do not replace the main checkpoint-schedule measurements. VA 2V/4A uses its measured PyTorch 2V/4A baseline. Dynamic-cache comparisons are labeled when their PyTorch baseline uses the checkpoint cache schedule. Equal declared adaptive policy can produce different actual compute/reuse decisions; these are whole-runtime comparisons. Original per-call traces remain in the raw receipts. Native precision alone does not establish byte equality.

.. csv-table::
   :header: "Mode", "PyTorch p50 (ms)", "InstinctFlash p50 (ms)", "Comparison"

   LingBot-VA / 2v4a-fp8,1104.14,621.48 (1.78×),same declared sampling policy
   LingBot-VA / 2v4a-native,1104.14,419.30 (2.63×),same declared sampling policy

Cohort ``node5090``: edge, groot, pi05, va, vla2, vla4. Actual GPU UUID: 004614f4-173a-187b-4e67-a2e1799e4d42; device-reported GPU memory: 33670758400 bytes. Measured Torch builds: 2.10.0+cu130, 2.11.0+cu130. The original saved device receipt is included. This receipt establishes neither a host-RAM minimum nor a whole-machine memory peak.

Action differences are empirical comparisons on the recorded inputs. These results do not validate robot task quality. ``results.json`` records every admitted cell, exact schedule, loaded-source hashes, raw receipt/action member hashes and archive hashes. The adjacent ``evidence/`` directory contains the original, unmodified bytes.

All observed modes
------------------

.. csv-table::
   :header: "Model", "Cell", "Mode", "Precision", "p50 (ms)", "Declared NFE"

   LingBot-VA,va-2v4a-fp8,2v4a-fp8,fp8,621.477720,"{""action"": 4, ""video"": 2}"
   LingBot-VA,va-2v4a-native,2v4a-native,native,419.303255,"{""action"": 4, ""video"": 2}"
   LingBot-VA,va-runtime_selected,fp8,fp8,5190.021708,"{""action"": 50, ""video"": 25}"
   LingBot-VA,va-runtime_default,native,native,3358.389085,"{""action"": 50, ""video"": 25}"
   LingBot-VA,va-eager_native,pytorch,native,8143.990757,"{""action"": 50, ""video"": 25}"
   LingBot-VA,va-eager_native-2v4a,pytorch,native,1104.138507,"{""action"": 4, ""video"": 2}"
   LingBot-VLA-4B,vla4-runtime_selected,fp8,fp8,104.452771,"{""action"": 10, ""prefix"": 1}"
   LingBot-VLA-4B,vla4-runtime_default,native,native,102.844143,"{""action"": 10, ""prefix"": 1}"
   LingBot-VLA-4B,vla4-eager_native,pytorch,native,803.487758,"{""action"": 10, ""prefix"": 1}"
   LingBot-VLA-V2-6B,vla2-runtime_selected,fp8,fp8,148.381351,"{""action"": 10, ""prefix"": 1}"
   LingBot-VLA-V2-6B,vla2-runtime_default,native,native,988.531967,"{""action"": 10, ""prefix"": 1}"
   LingBot-VLA-V2-6B,vla2-runtime_selected,numeric,native,145.181685,"{""action"": 10, ""prefix"": 1}"
   LingBot-VLA-V2-6B,vla2-eager_native,pytorch,native,980.523107,"{""action"": 10, ""prefix"": 1}"
   pi05,pi05-runtime_selected,fp8,fp8,106.460926,"{""action"": 10, ""prefix"": 1}"
   pi05,pi05-runtime_default,native,native,100.669625,"{""action"": 10, ""prefix"": 1}"
   pi05,pi05-eager_native,pytorch,native,303.809207,"{""action"": 10, ""prefix"": 1}"
   GR00T N1.7,groot-runtime_selected,fp8,fp8,96.685597,"{""action"": 4, ""backbone"": 1}"
   GR00T N1.7,groot-runtime_default,native,native,86.891397,"{""action"": 4, ""backbone"": 1}"
   GR00T N1.7,groot-eager_native,pytorch,native,159.065904,"{""action"": 4, ""backbone"": 1}"
   Cosmos3 Edge DROID,edge-runtime_selected,fp8,fp8,1168.971040,"{""action"": 4, ""prefix"": 1}"
   Cosmos3 Edge DROID,edge-runtime_default,native,native,958.223257,"{""action"": 4, ""prefix"": 1}"
   Cosmos3 Edge DROID,edge-runtime_selected,numeric,native,959.493183,"{""action"": 4, ""prefix"": 1}"
   Cosmos3 Edge DROID,edge-eager_native,pytorch,native,951.239992,"{""action"": 4, ""prefix"": 1}"

The complete Runtime ratios and action differences are in results.json. Selecting a faster mode does not isolate an FP8 arithmetic benefit.
