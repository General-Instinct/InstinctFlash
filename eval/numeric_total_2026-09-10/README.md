# Explicit numerical and few-step total acceleration

This campaign uses **the direct upstream policy at its original default schedule as the denominator**. It measures selected NUMERIC paths and separately selected few-step schedules; it does not relabel either as BITEXACT or infer task success from latency.

## Measured results

All times are p50 milliseconds. Ratios use the original upstream reference; ranges include both repeats. Action deltas include every saved action and pi05’s full queue. They are not percentages or task-success losses.

| Host | Configuration | Upstream A | Runtime | Upstream B | Total speedup range | Max action delta |
|:--|:--|--:|--:|--:|:--|--:|
| h100 | [VLA-V2 NUMERIC](receipts/h100/vla2-current-comparison.json) | 832.49 | 131.64 | 835.84 | 6.324–6.350× | 0.037102 |
| h100 | [pi05 FP32/TF32](receipts/h100/pi05-current-comparison.json) | 246.44 | 80.95 | 240.10 | 2.966–3.044× | 0.13468 |
| h100 | [VA NUMERIC 25V/50A](receipts/h100/va-current-comparison.json) | 7945.28 | 2364.64 | 7951.82 | 3.360–3.363× | 1.07813 |
| h100 | [VA NUMERIC 2V/4A](receipts/h100/va-current_fewstep-comparison.json) | 7945.28 | 289.24 | 7951.82 | 27.469–27.492× | 0.15625 |
| thor | [VLA-V2 NUMERIC](receipts/thor/vla2-current-comparison.json) | 760.11 | 422.86 | 763.32 | 1.798–1.805× | 0 |
| thor | [VA NUMERIC 25V/50A](receipts/thor/va-current-comparison.json) | 16393.91 | 5604.77 | 16677.27 | 2.925–2.976× | 0.0234375 |
| thor | [VA NUMERIC 2V/4A](receipts/thor/va-current_fewstep-comparison.json) | 16393.91 | 754.01 | 16677.27 | 21.742–22.118× | 1.15625 |
| thor | [VA FP8 2V/4A](receipts/thor/va-current_fewstep_fp8-comparison.json) | 16393.91 | 424.07 | 16677.27 | 38.658–39.326× | 1.10547 |

H100 VLA-V2 upstream A/B itself differs (maximum action delta 0.03515); all other repeated upstream references match exactly. Thor VLA-V2 actions also match across all three measured arms, but its selected implementation remains NUMERIC because its startup gate permits differences on other inputs. Both VLA-V2 candidates actually captured vision, prefill and action graphs.

The pi05 candidate’s returned-action maximum delta is 0.016284, while its queued actions reach 0.134680 against native BF16. The historical 0.000899 TF32 margin used a different FP32 reference and does **not** apply here. VA’s full-step H100 numerical layout change reaches a 1.078125 action delta on these inputs; the measurement does not establish that this change is acceptable for a task. No new closed-loop qualification is claimed for these rows.

Both VA native-parameter candidates installed NDHWC in both streaming VAEs; the per-device cached layout decision is recorded in pass results and `host-evidence/*-autotune.json`. Thor’s default-step speed is almost unchanged from its native result despite the selected encoder layout; a faster submodule need not produce a large full-cycle gain.

## Configurations

| Family / selection | Candidate | Reference |
|:--|:--|:--|
| LingBot-VLA-V2 | Native BF16, `tier_ceiling="numeric"`, CUDA graph admission by the adapter's numerical self-check | Direct upstream, native 10 action steps |
| LingBot-VA NUMERIC | Native precision, numeric ceiling, measured Conv3D layout selection | Direct upstream, 25V/50A |
| LingBot-VA 2V/4A | Above numeric configuration plus explicit `nfe={"video": 2, "action": 4}` | Direct upstream, **25V/50A** |
| LingBot-VA 2V/4A + FP8 (Thor) | Explicit FP8 and 2V/4A | Direct upstream native precision, **25V/50A** |
| pi05 TF32 (H100) | Released `examples/checkpoint/pi05-libero-v044-tf32-h100` declaration, all-FP32 parameters + TF32 + static-KV graph | Direct upstream v044, published BF16 parameter configuration, TF32 off, native 10 action steps |

VA explicitly excludes `cfg_branch_elision`: that analysis has no runtime installer and prior liveness measurements ruled it out. The recorded plan shows the exclusion; no unsupported pass is counted as installed. On native precision, the extra implemented VA pass is `conv_layout_ndhwc`; its per-device autotuner may retain the original layout and records its actual decision.

At this campaign revision, the pi05 TF32 declaration was restricted to SM90 H100. The [subsequent extension](../numeric_extension_2026-09-10/README.md) adds an explicit Thor declaration and paired measurements without transferring H100 margins. Its base-revision field is checked against the cached upstream checkpoint before loading. Simply raising the numerical ceiling does not activate this explicit TF32 declaration. The published v044 config uses BF16 parameters; this explicit TF32 configuration changes them to FP32 and validates every floating parameter. Consequently this is a comparison of complete execution configurations, not an isolated TF32-on/off ablation. Historical action bounds from an FP32 reference do not automatically qualify deltas against this native BF16 reference.

The eight-family planning inventory is in `h100-plan-inventory.json`. VLA-4B, ordinary pi05, GR00T, Cosmos Edge/Nano and DreamZero acquire no additional default numerical pass under this ceiling. Their native speed comparisons remain in the [native total campaign](../native_total_2026-09-10/README.md). This inventory is not a performance prediction for arbitrary compiler/kernel experiments.

## Protocol

Production source: `570b5bf29a6265277dc57a2966266b55ce0b41fe`, plus the frozen benchmark scripts. Each arm runs in a fresh process against an immutable source manifest. Sequence: upstream A, numeric Runtime, upstream B; VA inserts separate 2V/4A (and Thor FP8 2V/4A) processes before upstream B. Baselines retain original native steps, dtype and guidance in every arm.

The fixture, prompts, request seeds, observation preprocessing and timing boundaries match the [native total protocol](../native_total_2026-09-10/README.md). VLA-V2/pi05: 3 warmup and 20 measured generations. VA: four three-cycle episodes, first episode warmup (9 measured cycles); identical recorded executed-action feedback. This measures early-episode latency, not saturated KV-pool latency. Setup, graph admission, warmup and reset are outside the synchronized action-API timer. No simulator or network is in the timer.

VLA upstream constructors enable matmul TF32 in both arms; all other native references begin with it disabled. Only the explicit pi05 TF32 candidate is allowed to change this math setting. cuDNN TF32 and benchmarking must remain matched. Thor VLA-V2 uses the pre-existing SDPA/dense-MoE platform port in all arms, so its upstream reference is that port.

Every generation is saved, including pi05's remaining 49 queued actions. `compare.py` reports action deltas, upstream repeatability and both A/Runtime and B/Runtime latency ratios. It rejects mismatched weights, inputs, shapes, GPU contention, source drift and undeclared math/schedule differences. The few-step ratio is never measured against an already optimized or already shortened baseline. Numerical screens do not manufacture a tolerance from the observed maximum, and changed-step action deltas do not measure task success.

## Reproduction

Prepare `ROOT/source` and `ROOT/source.json` as in the native campaign, including this directory, and adapt the explicit environment paths in `run.py` before freezing it. Checkpoints must already be cached. On an idle device:

```bash
python ROOT/source/eval/numeric_total_2026-09-10/run.py ROOT --device h100 --gpu 6 --families vla2 pi05 va
# On Thor use: --device thor --gpu 0 --families vla2 va
python eval/numeric_total_2026-09-10/compare.py ROOT/h100-6 va --arm current_fewstep --output va-2v4a.json
```

`current` selects the numeric configuration, `current_fewstep` selects native 2V/4A with numeric permissions, and `current_fewstep_fp8` selects Thor FP8 2V/4A. References always use the original checkpoint's defaults. Immutable receipts retain the execution policy and complete pass decisions.

## FP8 extension completion

The first Thor FP8 attempt stopped before loading because the frozen Git archive omitted compiled `flash_rt` extensions. A separate frozen completion checkout adds the same-device extensions from the previous successful FP8 campaign. All original manifest files remain byte-identical; the added binary paths, SHA-256 hashes and sizes are recorded separately. The candidate rerun uses the same interpreter, benchmark script, checkpoint revision and inputs, and is compared against this campaign's original upstream A/B references. The failed receipt is retained.

For a clean reproduction, build the extensions from `serving/` for the selected Thor interpreter **before** freezing the source manifest (`cmake -S . -B build`, then `cmake --build build -j`, with that interpreter active). The completion driver serializes against the same Thor GPU lock. This changes neither an upstream baseline nor a native NUMERIC result. Binary artifacts themselves are not checked into Git.

## Validation

All eight comparisons passed the protocol checks. The 18 successful processes saved 315 generation calls, including pi05's complete queued chunks. The H100 lane verified 1,770 upstream/imported Python files against the earlier same-day inventory; Thor verified 1,454, and its FP8 completion verified 1,429. The original 2,577-file source manifest is unchanged; the FP8 completion adds only four recorded binaries. Every selected receipt has no GPU contenders during its action loop. The H100 observer saw three command-line disappearance flags at known benchmark process exits; preceding samples identify all three as owned PIDs, with no unresolved or confirmed foreign process.

The seven focused tests cover permitted schedule changes, the original denominator, full action queues, disallowed math changes, feedback ordering and loading-helper invariants. Artifact SHA-256 checks and three-way comparisons are rerun from the bundled receipts before publication. Failed FP8 setup evidence remains under `failed-attempts/`; its successful completion replaces only the candidate measurement, retaining the same original upstream A/B references.
