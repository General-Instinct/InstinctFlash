Current public pipeline results
===============================

The completed public pipeline passed paired inference and WebSocket checks for eight model families, plus six separately qualified foreign-framework captures. The table contains nine operating points, including LingBot-VA at 2V/4A.

These are recorded-input inference timings on Jetson Thor. They do not establish task success, transfer the prior Cosmos task SCREEN, or certify complete historical reproduction. Current source qualification and historical numerical comparison are separate.

The complete machine-readable evidence is `summary.json <current_public_pipeline_v1/snapshot_v5/summary.json>`__. Its SHA256 is ``7cba2fff8629c9ac7eb8365929e3ded30f1d42ee2b2237c3fea30e73a3be7d57``.

Prediction p50 (ms)
-------------------

.. csv-table:: Declared prediction operating points
   :header-rows: 1

   "Model","Native PyTorch","LeRobot","vLLM-Omni","InstinctFlash"
   "LingBot-VA","15506.32; 25V/50A","Unmeasured","Unsupported","2891.74; FP8, 25V/50A; 5.36x versus native"
   "↳ LingBot-VA @2V/4A","Unmeasured","1171.08; native, 2V/4A","Unsupported","459.10; FP8, 2V/4A; 33.78x versus full 25V/50A native"
   "LingBot-VLA-4B","624.22; NFE10","Unsupported","Unsupported","221.53; FP8, NFE10; 2.82x versus native"
   "LingBot-VLA-V2-6B","734.56; NFE10","Unsupported","Unsupported","394.11; FP8, NFE10; 1.86x versus native"
   "Cosmos3 Edge DROID","3393.78; UniPC4 CFG3","Unsupported","1074.11; compiled, UniPC4 CFG3","1048.01; native, NUMERIC, UniPC4 CFG3; 3.24x versus native"
   "Cosmos3 Nano DROID","10184.68; UniPC4 CFG3","Unsupported","4417.63; compiled, UniPC4 CFG3","4772.38; native, NUMERIC, UniPC4 CFG3; 2.13x versus native"
   "pi05","408.58; NFE10","93.92; compiled, NFE1","Unsupported","51.85; FP8, NFE10; 7.88x versus native"
   "GR00T N1.7","139.50; NFE4","247.01; native, NFE4","Not qualified","117.30; native, NFE4; 1.19x versus native"
   "DreamZero DROID","23563.08; fixed 8/16 DiT","Unsupported","8729.64; compiled, upstream step cache","11899.42; FP8, 16 solver updates, dynamic cache; 1.98x versus native"

.. csv-table:: DreamZero: all four current cells
   :header-rows: 1

   "Cell","Role","Precision","Schedule/cache","p50 (ms)"
   "dreamzero-eager_native","Native reference; vendor encoder compilation","native","fixed 8/16 DiT","23563.08"
   "dreamzero-runtime_default","Runtime default control (BITEXACT)","native","fixed 8/16 DiT","23876.20"
   "dreamzero-runtime_selected","Native runtime-selected control (BITEXACT)","native","fixed 8/16 DiT","23285.73"
   "dreamzero-dynamic-fp8","Published and WebSocket route (BEHAVIORAL)","FP8","16 solver updates, dynamic cache","11899.42"

Speedups divide unrounded p50 values. Framework inputs, warmups, prompt/history handling, precision, compilation and schedules differ; foreign columns are not matched-compute comparisons. LeRobot pi05 uses compiled NFE1 while the paired pi05 routes retain NFE10.

VA timings select early continuations. Its 2V/4A native cell is unmeasured; that row's speedup explicitly uses the full 25V/50A native reference. DreamZero's native reference keeps vendor encoder compilation and an eager DiT with the checkpoint's fixed 8/16 DiT mask. The published DreamZero route is FP8 with dynamic cache and 16 solver updates; its separate native runtime-selected control is still required by the paired validator.

Unsupported means no matching policy in the pinned support registry. Not qualified means an available route has no validated measurement here. WebSocket checks cover six calls across two reset episodes and successful server close; their round-trip times are not used as benchmark p50 values.

Historical numerical comparisons
--------------------------------

Each current arm is compared to its own frozen historical arm where precision and request contracts are comparable. Exact bytes are reported without changing thresholds. A valid mismatch remains visible and does not become a task-quality or causal claim. Earlier failed attempts and the negative Nano diagnostic remain preserved even when a later ordinary run matches historical bytes.

.. csv-table:: Own-arm action archives
   :header-rows: 1

   "Family","Exact / comparable","Different","Non-comparable","Assessment"
   "pi05","6/6","0","0","passed"
   "GR00T N1.7","3/3","0","0","passed"
   "LingBot-VLA-4B","3/3","0","0","passed"
   "LingBot-VLA-V2-6B","3/3","0","0","passed"
   "LingBot-VA","4/4","0","0","passed"
   "Cosmos3 Edge DROID","3/3","0","0","passed"
   "Cosmos3 Nano DROID","3/3","0","0","passed"
   "DreamZero DROID","0/3","3","1","failed_historical_equivalence"

The new DreamZero native runtime-selected control is not comparable to the older same-ID FP8 control; its exactness and action difference are null. The other three declared DreamZero routes are compared separately. This table counts action archives, not successful tasks or episodes.

Raw evidence and reproduction
-----------------------------

* pi05: `plan <qualification/pi05/run/plan.json>`__; `run <qualification/pi05/run/run.json>`__; `paired CPU replay <current_public_pipeline_v1/snapshot_v5/pi05_paired_replay.json>`__; `WS pi05-runtime_selected <qualification/pi05/serving_fp8_v2/receipt.json>`__.
* GR00T N1.7: `plan <qualification/groot/run/plan.json>`__; `run <qualification/groot/run/run.json>`__; `paired CPU replay <current_public_pipeline_v1/snapshot_v5/groot_paired_replay.json>`__; `WS groot-runtime_selected <qualification/groot/serving/receipt.json>`__.
* LingBot-VLA-4B: `plan <qualification/vla4/run/plan.json>`__; `run <qualification/vla4/run/run.json>`__; `paired CPU replay <current_public_pipeline_v1/snapshot_v5/vla4_paired_replay.json>`__; `WS vla4-runtime_selected <qualification/vla4/serving/receipt.json>`__.
* LingBot-VLA-V2-6B: `plan <qualification/vla2/run/plan.json>`__; `run <qualification/vla2/run/run.json>`__; `paired CPU replay <current_public_pipeline_v1/snapshot_v5/vla2_paired_replay.json>`__; `WS vla2-runtime_selected <qualification/vla2/serving/receipt.json>`__.
* LingBot-VA: `plan <qualification/va/run/plan.json>`__; `run <qualification/va/run/run.json>`__; `paired CPU replay <current_public_pipeline_v1/snapshot_v5/va_paired_replay.json>`__; `WS va-runtime_selected <qualification/va/serving_full/receipt.json>`__; `WS va-2v4a-fp8 <qualification/va/serving_2v4a/receipt.json>`__.
* Cosmos3 Edge DROID: `plan <qualification/edge/pytorch_triton_v1/run/plan.json>`__; `run <qualification/edge/pytorch_triton_v1/run/run.json>`__; `paired CPU replay <current_public_pipeline_v1/snapshot_v5/edge_paired_replay.json>`__; `WS edge-runtime_selected <qualification/edge/pytorch_triton_v1/serving/receipt.json>`__.
* Cosmos3 Nano DROID: `plan <qualification/nano/pytorch_triton_v1/run/plan.json>`__; `run <qualification/nano/pytorch_triton_v1/run/run.json>`__; `paired CPU replay <current_public_pipeline_v1/snapshot_v5/nano_paired_replay.json>`__; `WS nano-runtime_selected <qualification/nano/pytorch_triton_v1/serving/receipt.json>`__.
* DreamZero DROID: `plan <qualification/dreamzero/run/plan.json>`__; `run <qualification/dreamzero/run/run.json>`__; `paired CPU replay <current_public_pipeline_v1/snapshot_v5/dreamzero_paired_replay.json>`__; `WS dreamzero-dynamic-fp8 <qualification/dreamzero/serving/receipt.json>`__.

* lerobot-pi05: `capture <qualification/foreign/lerobot-pi05/capture.json>`__; `report <qualification/foreign/lerobot-pi05/report.json>`__.
* lerobot-groot: `capture <qualification/foreign/lerobot-groot/capture.json>`__; `report <qualification/foreign/lerobot-groot/report.json>`__.
* lerobot-va: `capture <qualification/foreign/lerobot-va/capture.json>`__; `report <qualification/foreign/lerobot-va/report.json>`__.
* vllm-omni-edge: `capture <qualification/foreign/vllm-omni-edge/capture.json>`__; `report <qualification/foreign/vllm-omni-edge/report.json>`__.
* vllm-omni-nano: `capture <qualification/foreign/vllm-omni-nano/capture.json>`__; `report <qualification/foreign/vllm-omni-nano/report.json>`__.
* vllm-omni-dreamzero: `capture <qualification/foreign/vllm-omni-dreamzero/capture.json>`__; `report <qualification/foreign/vllm-omni-dreamzero/report.json>`__.

Use the portable `reproduction guide <../../REPRODUCE.rst>`_ and `installation guide <../../INSTALL.rst>`_ for fresh environments, pinned checkpoint preparation, paired capture, reporting and serving checks. The installed entrypoints are ``python -m benchmarks.regression.reproduce``, ``python -m benchmarks.regression.serve_smoke`` and ``python -m benchmarks.regression.framework_compare``.

Hash-bound operational receipts retain their original paths. References outside this repository are local-only provenance; they are not requirements for the portable reproduction commands. Source-stage manifest SHA256: ``a316bd918ae4426460f0779f90f34985b31a8639f3b6be7f9dd20e86c9cbdbc4``.

Experimental SDE1/CFG1 recipes are separate from this full UniPC4/CFG3 Cosmos table. See the `SDE1 guide <../../examples/cosmos3_sde1/README.rst>`_; any separately published SDE1 latency receipts carry their own inputs, compiler/cache scope and unqualified task-quality label. No SDE1 value is inferred from this summary.
