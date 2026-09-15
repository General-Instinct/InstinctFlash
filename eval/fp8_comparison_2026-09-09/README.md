# Thor native execution and FP8 engine comparison

This report separates native capture, engine execution and FP8 arithmetic. New measurements use matched staged inputs and fresh processes. They are not complete robot-facing Runtime API latency or new closed-loop quality certificates.

The engine is a frozen copy of the installed Thor build. Its Python source differs from the workspace in six of 143 files (including the pi05 local norm-stat fallback path/unexecuted batched guard and V2 scheduling refactor). Results identify that installed build; they do not certify another build. See [source comparison](engine-source-comparison.json).

V2 required a native working-directory correction before inference. Its primary table uses a complete replacement arm set; the original failed starts, interrupted start and successful engine diagnostic are retained separately. See [setup amendment](SETUP_AMENDMENT.md).

One subsequent capture start needed replacement after a metadata field-name error; its actions and failed log are retained. Only the receipt field changed, not capture admission or computation. See the same amendment.

## Pi05 v044: matched 10-action chunks

Same pinned weights, two 224×224 views, 48 prompt tokens, ten denoise steps. Each invocation rebuilds vision and language prefix. Three fresh processes per arm, eight warmup and 128 measured calls per process. Table values are medians of process percentiles; ranges retain variation between starts.

| Execution | Completed starts | p50 ms (range) | p99 ms | Capture active |
| --- | ---: | ---: | ---: | --- |
| Upstream BF16, chunk 10 | 3/3 | 311.78 (307.92–314.69) | 323.70 | 0/3 |
| Native BF16 capture, chunk 10 | 3/3 | 221.73 (221.72–222.39) | 222.71 | 3/3 |
| Engine FP16, chunk 10 | 3/3 | 79.97 (79.84–80.26) | 80.65 | 3/3 |
| Engine FP8, chunk 10 | 3/3 | 44.89 (44.69–44.92) | 45.05 | 3/3 |

| Paired comparison | p50 speedup | Latency reduction | Maximum observed normalized action delta |
| --- | ---: | ---: | ---: |
| stock10 → capture10 | 1.41× | 28.9% | 0.000000 |
| stock10 → engine16 | 3.90× | 74.4% | 0.037534 |
| stock10 → engine8 | 6.95× | 85.6% | 0.153053 |
| capture10 → engine8 | 4.94× | 79.8% | 0.153053 |
| engine16 → engine8 | 1.78× | 43.9% | 0.143311 |

Engine FP16 is already a NUMERIC execution change relative to the BF16 checkpoint; it is not a bit-exact native baseline. Engine FP16 → FP8 is the closest arithmetic control here. Native capture → engine FP8 includes both engine implementation and precision changes. Numerical deltas are not task success-rate losses. See [raw summary](pi05-summary.json) for A/A, B/B and A/B controls.

The native checkpoint default computes 50-action chunks. Those measurements are separate below; reducing the horizon to 10 changes the computation even when weights and denoise-step count stay fixed. No cross-horizon speedup is presented as a quantization gain.

| Execution | Completed starts | p50 ms (range) | p99 ms | Capture active |
| --- | ---: | ---: | ---: | --- |
| Upstream BF16, chunk 50 | 3/3 | 315.64 (308.92–316.02) | 324.00 | 0/3 |
| Native BF16 capture, chunk 50 | 3/3 | 229.59 (229.55–229.71) | 230.57 | 3/3 |

### Model setup and allocator observations

| Execution | Model load/calibration seconds (median) | Warmup seconds (median) | Largest measured PyTorch allocation peak, GiB |
| --- | ---: | ---: | ---: |
| Upstream BF16, chunk 10 | 166.69 | 2.94 | 8.804 |
| Native BF16 capture, chunk 10 | 166.44 | 2.56 | 8.837 |
| Engine FP16, chunk 10 | 5.58 | 0.64 | 6.422 |
| Engine FP8, chunk 10 | 3.94 | 0.36 | 3.907 |

Fresh-process model setup is not cold-device boot: filesystem caches are not flushed, common interpreter/CUDA initialization precedes this timer, and native loading uses the upstream constructor. Allocation peaks cover steady inference after warmup and exclude external engine allocations and system memory; they are not total device-memory figures.


## LingBot staged comparisons

Both sides recompute vision and prefix every call. VLA4 uses synthetic patches and common BF16-representable state/noise, returning normalized 50×75 tensors before controller slicing to 25×14. V2 uses two recorded real observations sharing a prompt and common actual noise, returning normalized 50×55 tensors. Neither includes raw-camera processing or robot action decoding. These latencies must not be divided into older websocket or whole-policy measurements. The V2 engine uses FP16 vision/prefill, FP8 batched experts and an FP16-source router; “FP8 engine” does not mean every operation runs in FP8.

V2 native capture retains its legacy admission guard. That guard borrows a controller-action tolerance from an older H100 protocol and applies it to denoise velocity. Passing it does not establish a calibrated Thor quality bound. This study records admission and output deltas separately and does not certify that cross-domain threshold.

### vla4

| Execution | Completed starts | p50 ms (range) | p99 ms | Capture active |
| --- | ---: | ---: | ---: | --- |
| stock | 3/3 | 672.59 (666.95–683.49) | 697.61 | 0/3 |
| capture | 3/3 | 350.10 (348.82–350.88) | 354.82 | 3/3 |
| engine8 | 3/3 | 98.94 (98.90–99.06) | 99.17 | 3/3 |
| stock_vendor | 1/1 | 681.67 (681.67–681.67) | 698.88 | 0/1 |
| capture_vendor | 1/1 | 333.79 (333.79–333.79) | 342.28 | 1/1 |

`stock`/`capture` fix matmul TF32 off after loading (three starts). `stock_vendor`/`capture_vendor` are one-start controls that retain the native constructor’s matmul TF32 choice; other settings stay controlled. These are separate observed profiles, not pooled repeats.

| Comparison | p50 ratio | Maximum normalized action delta | Observed action bytes equal |
| --- | ---: | ---: | --- |
| stock → capture | 1.92× | 0.000000 | True |
| stock → engine8 | 6.80× | 1.929688 | False |
| capture → engine8 | 3.54× | 1.929688 | False |
| stock_vendor → capture_vendor | 2.04× | 0.000000 | True |
| stock_vendor → engine8 | 6.89× | 1.929688 | False |
| capture_vendor → engine8 | 3.37× | 1.929688 | False |
| stock → stock_vendor | 0.99× | 0.261719 | False |
| capture → capture_vendor | 1.05× | 0.261719 | False |

Full startup, repeatability and numerical deltas: [vla4 summary](vla4-summary.json).

### v2

| Execution | Completed starts | p50 ms (range) | p99 ms | Capture active |
| --- | ---: | ---: | ---: | --- |
| stock | 3/3 | 704.84 (704.75–705.98) | 709.85 | 0/3 |
| capture | 3/3 | 411.02 (410.70–411.25) | 412.97 | 3/3 |
| engine8 | 3/3 | 181.31 (180.81–184.18) | 182.25 | 3/3 |
| stock_vendor | 1/1 | 694.21 (694.21–694.21) | 713.60 | 0/1 |
| capture_vendor | 1/1 | 369.76 (369.76–369.76) | 372.41 | 1/1 |

`stock`/`capture` fix matmul TF32 off after loading (three starts). `stock_vendor`/`capture_vendor` are one-start controls that retain the native constructor’s matmul TF32 choice; other settings stay controlled. These are separate observed profiles, not pooled repeats.

| Comparison | p50 ratio | Maximum normalized action delta | Observed action bytes equal |
| --- | ---: | ---: | --- |
| stock → capture | 1.71× | 0.046875 | False |
| stock → engine8 | 3.89× | 0.169434 | False |
| capture → engine8 | 2.27× | 0.172852 | False |
| stock_vendor → capture_vendor | 1.88× | 0.046875 | False |
| stock_vendor → engine8 | 3.83× | 0.176758 | False |
| capture_vendor → engine8 | 2.04× | 0.171875 | False |
| stock → stock_vendor | 1.02× | 0.101562 | False |
| capture → capture_vendor | 1.11× | 0.132812 | False |

Full startup, repeatability and numerical deltas: [v2 summary](v2-summary.json).

The LingBot arms do not have a matched full-FP16 engine control. Their native → engine gains combine engine implementation and precision changes; they do not isolate the gain from FP8 alone. Action deltas cover the full normalized model output, including dimensions later discarded by the controller. Observed byte equality on these samples is not a guarantee on every input.

## Deployment interpretation

Keep native precision as the default when its measured end-to-end tail latency meets the application deadline. Native capture is the first option to evaluate. If more headroom is necessary, the pi05 FP16 engine control shows that a substantial part of the speedup comes before FP8 is enabled, although FP16 engine execution is itself a numerical change from BF16. Choose an FP8 profile only with explicit precision permission and a paired closed-loop evaluation of the actual deployed checkpoint, calibration, horizon, preprocessing and action scheduling. The generic pi05 interface gaps below must be closed before it can serve as that deployment profile.

## What is and is not interchangeable

- pi05: generic FP8 Runtime currently drops state processing, sorts the v044 empty camera ahead of real cameras, and returns an action chunk where the native loop returns a buffered single action. The fast staged path does not establish robot-facing API parity. The historical T3 server has a separate state-in-prompt/token-bucket and native-postprocessing implementation.
- The pi05 geometry guard checks action dimension; it does not currently enforce the 10-action horizon against a checkpoint declaring 50. Denoise NFE checks do not close this separate horizon mismatch.
- GR00T: the historical 122 → 42 ms comparison processes vision every call only on the original side. Its auxiliary-feature engine cannot supply a matched whole-policy ratio until visual refresh and its cost are included.
- VLA4 and V2: measured here through standalone engine paths, not newly integrated into the unified FP8 Runtime route.
- VA, Cosmos and DreamZero: no new matched FP8 deployment comparison in this campaign. VA additionally needs an explicit NFE/CFG and history regime.

Reproducible source-contract findings: [audit](contract-audit.json).

## Historical closed-loop evidence

| Historical paired execution | Episodes | Original → candidate success | Observed delta |
| --- | ---: | ---: | ---: |
| pi05 v044, LIBERO Spatial, H100 original / Thor T3 engine | 500 | 85.6% → 84.4% | −1.2 percentage points |
| V2, RoboTwin 50 tasks, historical Thor engine | 1100 | 89.36% → 87.64% | −1.73 percentage points |

The historical reports passed their declared 5-percentage-point non-inferiority tests; this does not establish zero loss. The pi05 success counts were recounted from retained run arrays (428 vs 422; 27 candidate-only and 33 original-only successes). V2 figures above are from its pooled report. These older execution sources/calibrations are not certificates for this campaign or the current generic Runtime. See [historical evidence](historical-evidence.json) and [pi05 recount](pi05-historical-recount.json).

## Measurement limits and reproduction

- All attempts and setup failures remain in the raw campaign; no fastest-start selection or retry-until-admitted policy. Capture fallback stays labelled as fallback.
- Tiny input sets characterize execution and numerical differences, not task coverage. pi05 calibrates on three recorded scenes and evaluates on two other scenes with a fixed synthesized state-containing prompt. VLA4 inputs are synthetic; V2 retains historical calibration artifacts.
- PyTorch allocator peaks and startup/warmup times are in individual receipts. Allocator counters omit external engine CUDA allocations; Thor shares system memory. They do not support a total-memory reduction claim.
- The pi05 frontend internally runs its own action postprocessing before the probe reads normalized actions; its small cost is included. Shared robot pre/postprocessing, transport and sustained control scheduling are not measured.
- H100 and other device measurements are not substituted for Thor results.
- Timings are per action-chunk inference. Their reciprocal is not the robot control frequency; buffered action execution and scheduling are outside this probe.
- p99 is a descriptive percentile from 128 calls per start, not a long-duration tail-latency or thermal-stability guarantee.
- Thor stays in its observed MAXN power mode; energy per action was not measured. The primary native probes disable TF32, cuDNN benchmark and model compilation; constructor-TF32 spot checks are separate. These are declared comparison settings, not a claim to reproduce every vendor default. The model-specific Python environments and installed engine build are recorded separately.

Protocol: [PROTOCOL.md](PROTOCOL.md). Setup and replay instructions: [REPRODUCE.md](REPRODUCE.md). Raw root: `/home/ubuntu/ifl_eval/fp8_comparison_20260909`; Thor root: `/home/guanming/ifl_eval/fp8_comparison_20260909`. Source, artifacts and retained failures are described in [artifact-index.json](artifact-index.json).

```sh
python eval/fp8_comparison_2026-09-09/analyze.py \
  --root /path/to/restored/raw --output eval/fp8_comparison_2026-09-09/pi05-summary.json
python eval/fp8_comparison_2026-09-09/render_report.py \
  --root /path/to/restored/raw --reports eval/fp8_comparison_2026-09-09
```
