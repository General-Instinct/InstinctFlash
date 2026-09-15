# Thor TF32 support and H100 FP8 few-step extension

Explicit NUMERIC execution on Thor does not inherit any H100 action-error margin. The new pi05 pointer requires FP32 parameters, TF32, an exact graph replay self-check, and a NUMERIC permission. Task quality is unqualified.

This campaign measures fresh upstream A / candidate / upstream B processes using the numeric-total camera fixture, seeds, full action queues and early-episode timing. pi05 compares native BF16 upstream against declared FP32/TF32. H100 VA compares original native 25V/50A against FP8 2V/4A. All speedups use the original upstream default schedule.

## Results

Latency is p50 ms; the range uses both original upstream repeats. These are numerical screens, not new closed-loop certificates.

| Host / configuration | Upstream A | Runtime | Upstream B | Total speedup range | Max action delta |
|:--|--:|--:|--:|:--|--:|
| [Thor pi05 FP32/TF32](receipts/thor/comparison.json) | 419.78 | 287.02 | 410.75 | 1.431–1.463× | 0.110789 |
| [H100 VA FP8 2V/4A](receipts/h100/fp8-comparison.json) | 7809.82 | 496.83 | 7787.10 | 15.674–15.719× | 1.0625 |
| [H100 VA native parameters + NUMERIC 2V/4A](receipts/h100/native-comparison.json) | 7809.82 | 285.61 | 7787.10 | 27.265–27.344× | 0.15625 |

H100 FP8 is **1.74× the latency** of the native-parameter 2V/4A control in this sweep. FP8 execution is supported and active, but it is slower for this configuration. Native parameters remain preferable for speed here.

All original A/B references match finite action bytes exactly. Thor TF32 graph replay also matched TF32 eager on six staged cases, but the candidate differs from native BF16: its full saved chunk reaches a 0.110789 action delta. The pointer therefore retains a null action-error allowance and unqualified task status. The 1.062500 H100 FP8 delta includes the step change versus the original 25V/50A reference; it is not an isolated FP8 quantization error or a task-failure rate.

The Thor VA 2V/4A cells in the main README remain the previous [numeric-total campaign](../numeric_total_2026-09-10/README.md): 21.74× with native parameters, 38.66× with FP8. This extension remeasures the H100 cells with one shared original reference and adds the previously missing Thor pi05 cell.

## Usage

Install the updated pi05 plugin in the model interpreter so its numerical pass entry point is registered:

```bash
python -m pip install --no-deps --no-build-isolation -e examples/pi05_vla
```

```python
from instinctflash import Runtime

# Jetson Thor, v044 weights; explicit unqualified NUMERIC execution.
runtime = Runtime.from_pretrained(
    "examples/checkpoint/pi05-libero-v044-tf32-thor",
    device="cuda:0", tier_ceiling="numeric",
)
runtime.close()

# H100, original VA weights, explicit FP8 and changed step schedule.
runtime = Runtime.from_pretrained(
    "robbyant/lingbot-va-posttrain-robotwin",
    device="cuda:0", precision="fp8", nfe={"video": 2, "action": 4},
)
```

The Thor pointer is separate from the H100 pointer so loading an H100 declaration on Thor cannot silently transfer a hardware-specific numerical contract. Ordinary pi05 loading remains native BF16 with the existing BITEXACT policy. The new declaration has no action-error allowance or closed-loop certificate: `max_abs_action_delta` is null and qualification is explicitly unqualified. Exact replay checks compare the graph with TF32 eager, not with native BF16.

## Reproduction

Freeze production `b70a32d` plus the recorded adapter/pass changes, new pointer and this campaign's harness into `ROOT/source`, hashing every file into `ROOT/source.json`. The runner checks the immutable manifest before and after each lane. The shared input archive is `eval/native_total_2026-09-10/fixtures/va_eval_obs.npz`.

```bash
python ROOT/source/eval/numeric_extension_2026-09-10/run.py ROOT --device thor --gpu 0 --families pi05
python ROOT/source/eval/numeric_extension_2026-09-10/run.py ROOT --device h100 --gpu 6 --families va
```

Use `eval/numeric_total_2026-09-10/compare.py` with `--arm current` for pi05 and `--arm current_fewstep_fp8` for VA. References always retain the original default step count and native precision. Each interpreter/checkpoint path is explicit in `run.py`; adapt paths before freezing. No prior measurement is substituted for these fresh A/B/A comparisons.

## Setup corrections and execution reporting

The initial H100 FP8 run reached inference but its generic object walker did not traverse VA's upstream server, so its post-run packed-weight assertion failed. The final harness directly verifies every recipe entry against an actual registered `H100FP8Linear` and its E4M3 weight shape. It verifies 180 matrices after timing. This changes observation of the model, not its forward computation. The failed receipt is retained.

The initial Thor environment registered the pi05 adapter but omitted its numerical-pass entry point. Runtime refused the declared TF32 configuration before inference. Installing the updated pi05 plugin fixed registration; all three selected Thor arms were then rerun in that environment. The old failed receipt and the new entry-point inventory are retained.

H100 FP8 wraps native Torch execution. Its plan report previously demoted native passes as though a fused pipeline had replaced them. `_mark_plan_engine_executed` now retains installed native passes for `h100_torch_fp8`, preserves exclusions, and keeps the existing fused-engine reporting for Thor. A final H100 sweep uses this reporting fix. H100's actual recipe quantizes attention Q/K/V projections, and is distinct from the Thor fused executor. A native-parameter 2V/4A control in the same sweep measures whether FP8 adds speed.

`frozen-overlay/initial`, `frozen-overlay/thor` and `frozen-overlay/h100` contain the exact files that differ from production `b70a32d` in each immutable checkout. Overlay the selected directory onto a checkout of that commit and verify its corresponding source manifest. Initial failures are not substituted into the published results.

## Validation

The final three comparisons passed the protocol checks, covering seven successful processes and 117 saved generation calls. pi05 saves its returned action and all 49 queued actions. FP8 evidence verifies 180 actual E4M3 projection matrices and checks every registered module against its recipe after timing. Source/manifest verification found no drift across 1,542 H100 and 1,619 Thor upstream/imported Python files. Every selected receipt records no competing GPU process during its action loop. The 52 focused regression tests passed, including hardware isolation, precision permission, FP32 dtype enforcement, missing FP8 evidence, replay policy and preservation of H100 native-pass reporting.
