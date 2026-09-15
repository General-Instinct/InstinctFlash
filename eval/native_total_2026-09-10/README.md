# Total native Runtime acceleration

This campaign compares **direct upstream policy construction → complete default native Runtime**, in fresh upstream / Runtime / upstream processes on H100 and Jetson Thor. It replaces incremental optimizations as the reference for the main speed table.

## Measured results

Milliseconds are p50. The range uses both fresh upstream repeats divided by Runtime latency.

| Host | Family | Upstream A | Runtime | Upstream B | Speedup range | Action check |
|:--|:--|--:|--:|--:|:--|:--|
| h100 | va | 7821.41 | 3161.63 | 8003.94 | 2.474–2.532× | [BITEXACT](receipts/h100/va-comparison.json) |
| h100 | vla4 | 644.75 | 101.13 | 643.61 | 6.364–6.375× | [BITEXACT](receipts/h100/vla4-comparison.json) |
| h100 | vla2 | 848.87 | 840.52 | 907.46 | 1.010–1.080× | [BASELINE_VARIABLE](receipts/h100/vla2-comparison.json) |
| h100 | edge | 989.50 | 1003.02 | 958.62 | 0.956–0.987× | [BITEXACT](receipts/h100/edge-comparison.json) |
| h100 | nano | 1187.61 | 1178.32 | 1186.52 | 1.007–1.008× | [BITEXACT](receipts/h100/nano-comparison.json) |
| h100 | pi05 | 240.83 | 87.06 | 239.58 | 2.752–2.766× | [BITEXACT](receipts/h100/pi05-comparison.json) |
| h100 | groot | 122.35 | 61.90 | 119.07 | 1.924–1.976× | [BITEXACT](receipts/h100/groot-comparison.json) |
| h100 | dreamzero | 3005.17 | 3007.24 | 2995.25 | 0.996–0.999× | [BITEXACT](receipts/h100/dreamzero-comparison.json) |
| thor | va | 16096.69 | 5595.57 | 16161.46 | 2.877–2.888× | [BITEXACT](receipts/thor/va-comparison.json) |
| thor | vla4 | 685.45 | 264.67 | 689.94 | 2.590–2.607× | [BITEXACT](receipts/thor/vla4-comparison.json) |
| thor | vla2 | 760.25 | 752.59 | 733.22 | 0.974–1.010× | [BITEXACT](receipts/thor/vla2-comparison.json) |
| thor | edge | 3438.81 | 3454.48 | 3442.07 | 0.995–0.996× | [BITEXACT](receipts/thor/edge-comparison.json) |
| thor | nano | 10364.87 | 10352.67 | 10337.64 | 0.999–1.001× | [BITEXACT](receipts/thor/nano-comparison.json) |
| thor | pi05 | 408.93 | 312.42 | 413.90 | 1.309–1.325× | [BITEXACT](receipts/thor/pi05-comparison.json) |
| thor | groot | 146.36 | 131.53 | 147.66 | 1.113–1.123× | [BITEXACT](receipts/thor/groot-comparison.json) |
| thor | dreamzero | Load limited | 23411.58 (unpaired) | Load limited | — | [REFERENCE_LOAD_LIMITED](failed-attempts/thor-dreamzero-full-checkpoint/thor-0/dreamzero-current.json) |

H100 VLA-V2 failed upstream repeatability on 1 of 23 calls (maximum absolute action delta 0.03225); Runtime differs from the first upstream run on 4 calls. No tolerance or lossless certificate is assigned. Cosmos and DreamZero H100 apply no additional inference transforms in this native configuration, so their approximately 1× results are expected.

The published Nano H100 row is the complete fresh rerun after the unrelated GPU collision; the failed first attempt is retained under `failed-attempts/h100-nano-collision/`.

## Protocol

- Same checkpoint revision, cameras, prompts, synthetic states, native parameter dtype, guidance and denoising schedule within each pair. The harness starts with TF32 and cuDNN benchmarking disabled. VLA-4B/VLA-V2 upstream constructors then set float32 matmul precision to `high`, enabling matmul TF32 in **all three arms**; their receipts retain that upstream setting. Other families keep matmul TF32 off. cuDNN TF32 and benchmarking remain off throughout all completed pairs. No FP8, distillation, or reduced-step schedule is added.
- Synchronized wall time around the action API, including image/text processing and action decoding; model loading and episode reset excluded. No network or simulator in the timer.
- Stateless policies: 3 warmup calls and 20 measured chunk generations, with changing cameras and prompts. VLA/Cosmos also vary synthetic states; pi05/GR00T use fixed synthetic state fixtures. pi05 saves its entire queued action chunk as well as the returned action.
- History policies: four three-cycle episodes, first episode warmup. VA uses recorded executed actions with the same deferred commit order in both arms.
- Every action is saved. The existing three-arm comparator checks finite action bytes, shapes, protocol identity, artifact hashes, source drift, GPU contention and upstream repeatability. A tolerance is never substituted for a failed byte comparison.
- Published ratios use measured p50 latency, with both upstream repeats reported in the receipts. These are input-specific numerical checks, not simulator success rates.

## Upstream reference construction

`upstream.py` constructs original upstream policies directly, bypassing all adapter performance installers. Shared loop classes only translate observations, prompts and action replies. `Runtime.from_pretrained` resolves the checkpoint declaration before the reference constructor runs; the reference never loads its Runtime backend.

| Family | Upstream entry | Explicit compatibility/configuration |
|:--|:--|:--|
| LingBot-VLA-4B | `LingbotVLAServer` | Published tokenizer and robot normalization; native 10 steps |
| pi05 | `PI05Policy.from_pretrained` | Published pre/post processors on CUDA; `compile_model=False` establishes the eager reference, preserving checkpoint dtype |
| GR00T N1.7 | `Gr00tPolicy` | Skip irrelevant online Mistral tokenizer probe; native 4 steps; no fast decode, backbone fastpath, GPU collate or graph |
| LingBot-VLA-V2 | `LingbotVLAv2Server` | Same offline tokenizer compatibility; native BF16, 10 steps, compilation disabled |
| Cosmos Edge/Nano | `RobolabPolicyService` | Eager DROID service, guardrails disabled in both arms; native 32-action, 4-step, CFG 3 profile |
| LingBot-VA | `VA_Server` | Pinned checkpoint symlinks and single-rank launch; original FSDP, allocator, prefill and KV implementation retained |
| DreamZero | `GrootSimPolicy` | Native encoder compilation and fixed 8-of-16 mask. H100 uses the released loader configuration; Thor uses the native full-checkpoint preload flag, retaining original FP32 initialization and final BF16 conversion. |

The initial Thor DreamZero load failed because redundant base-DiT shards were absent. That attempt remains recorded as a failure. The subsequent attempted replacement pair explicitly uses upstream's `skip_component_loading=True` option for full checkpoints: coverage validation checks every native DiT tensor before skipping base preload. It does not install the Runtime direct-BF16 factory, alter forward methods, change steps, or change final parameter precision. Enable it for reproduction with `IFL_BENCH_DREAMZERO_FULL_CHECKPOINT=1`; this affects loading outside the latency timer. The reference still could not finish loading within the host’s unified-memory budget; only its Runtime arm completed. No paired Thor DreamZero result is claimed. An unchanged default path may correctly measure near 1× and is not replaced by an older result from a different checkpoint, action geometry, precision or schedule.

DreamZero's optimized loader does not preserve the RNG consumption of discarded random initialization. These comparisons explicitly reseed before each request: they certify matched-noise inference, not equality of RNG state immediately after model loading. The loading receipt records this distinction.

## Host reference qualifications

Thor VLA-4B and VLA-V2 use pre-existing platform ports in both arms: VLA-4B replaces FlashAttention with eager/SDPA attention and supplies the missing vision rotary helper; VLA-V2 uses SDPA attention and dense Torch MoE instead of upstream Triton MoE. Byte equality here is against that same Thor port, **not** against the unmodified official FlashAttention/MoE implementation. H100 uses the original VLA source. Both hosts retain the VLA constructor’s `torch.set_float32_matmul_precision("high")` setting. Thus VLA byte checks are relative to the upstream math configuration, not to an independently forced IEEE-FP32 reference. The Runtime does not gain an unmatched TF32 setting.

Cosmos checkouts contain earlier RoboTwin registration and compile-option extensions. This campaign uses the DROID service with compilation disabled, so those performance extensions are inactive in both arms. VA changes only checkpoint-path/config registration outside the upstream server implementation. Git revisions, local diffs, Python source hashes and per-model environment package versions are retained with the host evidence.

## Reproduction

`run.py ROOT --device h100 --gpu 7 --families vla4 pi05 groot` runs the three fresh-process arms. `ROOT/source` is an immutable checkout based on production commit `07e522cfa78c5cd3a557130a7332172a6d92a15d`, plus the recorded benchmark scripts and `ROOT/source.json` maps every included source file to its SHA-256. The runner verifies this manifest before every arm and after the lane completes. The exact input archive is included under `fixtures/`; the published runner uses it directly. Model environment paths are explicit in the runner; adapt them to the host before creating the source manifest.

Create a fresh published-harness checkout (after adapting the environment paths):

```bash
export BENCH_ROOT=/tmp/ifl-native-total-reproduction
mkdir -p "$BENCH_ROOT/source"
git archive HEAD | tar -x -C "$BENCH_ROOT/source"
python - <<'PYTHON'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ["BENCH_ROOT"])
source = root / "source"
manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*") if p.is_file()}
(root / "source.json").write_text(json.dumps(manifest, indent=2))
PYTHON
python "$BENCH_ROOT/source/eval/native_total_2026-09-10/run.py" \
  "$BENCH_ROOT" --device h100 --gpu 7 --families vla4 pi05 groot
```

Use an idle GPU; pass `--device thor --gpu 0` on Thor. Checkpoints must already be cached (`HF_HUB_OFFLINE=1`). Reconstructing the exact campaign instead uses production commit `07e522c` with the chosen frozen protocol overlaid and its archived manifest verified. The published fixture has the identical archive SHA-256 used by every measured arm.

Compare a completed family using `eval/native_coverage_2026-09-10/compare.py ROOT/h100-7 FAMILY --output COMPARISON.json`. The comparator exits nonzero unless the complete three-way check is BITEXACT.

The first batch uses the frozen three-family harness (`protocol-first-batch/`); the remaining families use its extension (`protocol-remaining/`). Thor DreamZero uses the explicitly declared native full-checkpoint option (`protocol-dreamzero-full-checkpoint/`). The published harness restricts precision to native; the frozen versions were also invoked only with native precision. Each pair uses one identical harness, with the source hash recorded in every receipt.

The published harness also refuses an occupied GPU before model loading. In the frozen campaign, an unrelated process briefly entered H100 GPU 6 during Nano loading and caused an OOM; that attempted pair is retained as a setup failure and a complete fresh three-arm rerun completed and supplies the published row. Continuous GPU-process observation was added after the collision.

## Thor DreamZero loading limitation

The native upstream loader initializes FP32 parameters before its final BF16 conversion. Runtime’s direct-BF16 loader completed 12 calls, with 9 measured calls at p50 23411.58 ms. Upstream did not reach inference, so this is **unpaired latency**, not a speedup or numerical certificate.

- The initial attempt failed because four redundant base-Wan DiT shards were absent from the offline cache (`thor-dreamzero-missing-cache/`).
- The next attempt used the upstream full-checkpoint flag. Reference A was stopped at the initial 8 GiB available-memory reserve; this was a conservative monitor cutoff, not an OOM. The reserve was reduced to 2 GiB for reference B, which reached 0 available memory before the monitor terminated it (`thor-dreamzero-full-checkpoint/`).
- A final bounded reference attempt also used `IFL_BENCH_LOAD_HEAP_TRIM=1`: a helper releases unused host heap pages during loading, with no tensor/dtype/forward modification. It joins before warmup or timing. [glibc’s `malloc_trim`](https://man7.org/linux/man-pages/man3/malloc_trim.3.html) only releases free heap memory. This attempt still reached 1,799,344 KiB available memory and was stopped below the 2 GiB reserve. Its controller therefore skipped the remaining arms (`thor-dreamzero-trim/`). No inference measurement used this helper.

All paths above are under `failed-attempts/`; source manifests, progress, memory-stop records and logs are retained. `protocol-dreamzero-heap-trim/` contains the exact final attempted harness. No missing reference timing is inferred from the Runtime-only measurement.

## Evidence validation

The 15 completed three-arm comparisons contain 936 saved generation calls; pi05 additionally saves all 49 queued actions after each returned action. All JSON/NPZ hashes and all three comparisons were recomputed from the bundled artifacts. Fourteen pairs are BITEXACT on these inputs; H100 VLA-V2 is BASELINE_VARIABLE. The additional unpaired Thor DreamZero run contains 12 saved calls.

Post-run upstream/imported-source verification found no changes or cross-report conflicts across 1,912 H100 and 1,831 Thor Python files. Every selected receipt reports no competing GPU process during its measured loop. H100 clock observation is a point sample, not a continuous clock trace. The 19 comparator, feedback-order and checkpoint-loading tests pass. These checks establish evidence integrity and input-specific action equality, not simulator task success or universal equivalence.
