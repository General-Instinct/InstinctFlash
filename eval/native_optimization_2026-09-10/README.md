# Native optimization qualification — 2026-09-10

This campaign compares the existing native Runtime with candidate scheduling,
CUDA Graph and input-staging changes. FP8-to-FP8 comparisons measure reuse of those
changes within the explicitly selected lossy engine. They do not certify FP8 quality.

## Qualified native results

| Model / device | Baseline p50 | Optimized p50 | Speedup | Repeated baseline |
| --- | ---: | ---: | ---: | ---: |
| VLA-4B / H100 | 184.44 ms | 99.18 ms | 1.86× | 185.32 ms |
| VLA-4B / Thor | 363.96 ms | 261.92 ms | 1.39× | 364.07 ms |
| pi05 / H100 | 96.58 ms | 86.41 ms | 1.12× | 96.06 ms |
| pi05 / Thor | 324.50 ms | 309.13 ms | 1.05× | 327.10 ms |
| GR00T / H100 | 66.34 ms | 60.85 ms | 1.09× | 67.01 ms |
| GR00T / Thor | 134.98 ms | 126.84 ms | 1.06× | 139.08 ms |

All listed pairs and repeated baselines have identical finite saved action bytes.
VLA-4B and GR00T H100 pairs use 60 measured calls; the other pairs use 20. Each also
includes three warmup calls in its saved action comparison. GR00T's Thor baseline
varied 3%; the candidate was faster than both baselines. These are generation-latency
pairs on the installed stack, not replacements for historical author-server ratios.
Peak PyTorch allocation changed by approximately 0.01 GiB or less in these pairs.

[Machine-readable comparisons](native-results.json) include actual admission
statistics. Full [receipts](receipts/) include source/input hashes and saved actions.
VLA-4B full capture + preprocessing and GR00T collation now default on for native
H100/Thor; pi05's prefix/full-loop/tables recipe defaults on H100 and Thor. Startup admission
and explicit opt-outs remain active. [Default-selection validation](default-validation.json)
checks actual no-switch native loads and the selected FP8 routes against their paired references.

## Admission and timing protocol

- A and B use the same checkpoint revision, device, upstream environment, recorded
  camera archive, synthetic states, prompt sequence, seed sequence and native schedule.
- Each variant loads in a fresh process. Three warmup calls precede measurement.
  Each timed call includes public Runtime input/output processing and synchronizes CUDA.
  Stateless families reset before each call to force a new action chunk; reset is untimed.
- Comparisons include all saved action bytes, dtype/shape and finiteness. The final pi05
  runner also saves all 49 remaining buffered actions after each first-action response.
  Startup checks independently compare captured computations with the saved upstream.
- Full-graph admission and preprocessing admission must actually pass. An eager fallback
  is recorded as a rejected candidate even when its output matches the baseline.
- `nvidia-smi` checks for other processes on the physical GPU between timed calls.
  These checks are outside latency measurements; this is response latency, not continuous
  throughput. Baselines are repeated after candidates to expose drift.
- Results describe these inputs and this checkpoint, not task-success non-inferiority
  or every possible input. No dtype, denoising count, teacher, or tolerance is changed
  in the native comparisons.

## Reproduction

Use the model's installed upstream environment and an idle H100 or Thor. Set the
family source-root environment variable documented in `examples/<family>/README.md`.
The recorded camera archive is an external fixture; pass its path explicitly. Every
receipt records its SHA-256 and the loaded implementation's source hashes.

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
export IFL_BENCH_TIER=bitexact
IFL_VLA4B_FULL_GRAPH=0 IFL_VLA4B_GPU_PREPROCESS=0 \
  python eval/native_optimization_2026-09-10/benchmark.py \
  vla4 native baseline.json --input-archive "$OBS_ARCHIVE" --iterations 60
IFL_VLA4B_FULL_GRAPH=1 IFL_VLA4B_GPU_PREPROCESS=1 \
  python eval/native_optimization_2026-09-10/benchmark.py \
  vla4 native optimized.json --input-archive "$OBS_ARCHIVE" --iterations 60
python eval/native_optimization_2026-09-10/compare.py baseline.json optimized.json
```

Outputs are JSON receipts and compressed NPZ actions; existing outputs are refused.
The comparator refuses different protocols, nonfinite/mismatched action bytes and
recorded GPU competition. Its `candidate_stats` must be inspected for actual graph
admission; byte equality alone cannot distinguish acceleration from fallback.

## Reuse within explicitly selected FP8

| Model / Thor | FP8 baseline p50 | Same FP8 + shared staging | Repeated baseline | Speedup range |
| --- | ---: | ---: | ---: | ---: |
| VLA-4B | 161.53 ms | 155.34 ms | 161.78 ms | 1.040–1.041× |
| GR00T | 146.61 ms | 132.59 ms | 138.80 ms | 1.047–1.106× |

All saved actions match the corresponding FP8 baseline byte-for-byte, including
its repeated baseline. GR00T's baseline varied 5.3%, so the 11% improvement is not
presented as a stable point estimate; the candidate beat both baselines. Both reuse
paths keep six-live-batch upstream preprocessing checks and explicit opt-outs.
These mechanisms now default on inside the explicitly selected FP8 engine too.
[FP8 reuse receipts and comparisons](fp8-reuse-results.json).

This does **not** establish equivalence between FP8 and native arithmetic. FP8
remains an explicit lossy choice with its own checkpoint-specific quality evidence.
The existing pi05 FP8 engine already contains graph/static-buffer
mechanisms; this campaign adds their native counterparts without importing FP8 math.

## Rejected attempts retained

- pi05 full-loop capture initially assumed the newer six-argument prefix API. Both
  installed LeRobot environments expose the older four-argument API. The attempted
  full-loop arms fell back to eager; their timing is not a graph speed result.
  Compatibility now follows the actual signature, including positional noise.
- A separate InstinctCompress job began on H100 GPU 4 during pi05's initial run.
  The affected timings are excluded; the final pi05 campaign uses GPU 6 and monitors
  competition. VLA-4B uses GPU 5.
- The initial Thor VLA-4B full path assumed FlashAttention vision. Thor uses eager
  vision attention. The new branch calls the original eager attention with cached
  host sequence boundaries; it never substitutes FlashAttention or SDPA.
- GR00T full-backbone capture failed exact admission against unpatched upstream.
  H100 full-path attempts were no faster than repeated baseline. Full capture remains
  unavailable through the public adapter. A follow-up feature-boundary experiment
  encountered a CUDA device assertion; it was not retained in production. GPU
  collation is independent and passed its paired qualification.
- Thor's `nvidia-smi` is under `/usr/sbin`, outside the original pinned benchmark PATH.
  Those failed benchmark attempts are excluded; executable resolution is fixed.

Raw source snapshots, hashes, failed logs and interrupted-run notes are retained in
`/home/ubuntu/ifl_eval/native_optimization_20260910` and the corresponding Thor
`/home/guanming/ifl_eval/native_optimization_20260910` directory. Published results
must be tied to a completed paired receipt, not to a planned optimization.

## Earlier upstream comparisons

These use earlier serving configurations, not the baseline of this campaign.

**BITEXACT** — action bytes matched in the reported benchmark.

| Model | H100: PyTorch → InstinctFlash |
|:--|:--|
| **LingBot-VLA-4B** | 671&nbsp;→&nbsp;185&nbsp;ms,&nbsp;**3.62×** |
| **pi05** | 207&nbsp;→&nbsp;73&nbsp;ms,&nbsp;**2.84×** |
| **GR00T-N1.7-3B** | 115&nbsp;→&nbsp;59&nbsp;ms,&nbsp;**1.94×** |

**Other execution profiles.** NUMERIC allows numerical changes; BEHAVIORAL changes
the model or step schedule; UNVERIFIED means equivalence was not established.

| Model | H100 | Tier | Jetson Thor | Tier |
|:--|:--|:--|:--|:--|
| **LingBot-VA** (5B WAM) | 8448&nbsp;→&nbsp;2583&nbsp;ms,&nbsp;**3.27×**&nbsp;‡ | NUMERIC | 18027&nbsp;→&nbsp;5611&nbsp;ms,&nbsp;**3.21×**&nbsp;‡ | NUMERIC |
| **LingBot-VA @ 2V/4A** (5B WAM) | 8448&nbsp;→&nbsp;360&nbsp;ms,&nbsp;**23×**&nbsp;‡ | BEHAVIORAL | 18027&nbsp;→&nbsp;893&nbsp;ms,&nbsp;**20×**&nbsp;‡ | BEHAVIORAL |
| **LingBot-VLA-V2-6B** (sparse-MoE) | 671&nbsp;→&nbsp;128&nbsp;ms,&nbsp;**5.26×** | NUMERIC | 752&nbsp;→&nbsp;210&nbsp;ms,&nbsp;**3.57×** | NUMERIC (FP8) |
| **Cosmos3-Edge-Policy** (3.86B) | 311&nbsp;→&nbsp;186&nbsp;ms,&nbsp;**1.67×** | NUMERIC | 1158&nbsp;→&nbsp;660&nbsp;ms,&nbsp;**1.75×** | NUMERIC |
| **Cosmos3-Nano-Policy** (15.75B) | 482&nbsp;→&nbsp;325&nbsp;ms,&nbsp;**1.49×** | NUMERIC | 3956&nbsp;→&nbsp;2080&nbsp;ms,&nbsp;**1.90×** | NUMERIC |
| **pi05** | — | — | 255&nbsp;→&nbsp;57&nbsp;ms,&nbsp;**4.49×** | NUMERIC (FP8) |
| **GR00T-N1.7-3B** | — | — | 42.4&nbsp;ms with cached prefix&nbsp;† | UNVERIFIED |
| **DreamZero-DROID** (WAM) | 3227&nbsp;→&nbsp;1843&nbsp;ms,&nbsp;**1.75×** | UNVERIFIED | — | — |

‡ LingBot-VA reports early-episode latency. † GR00T's cached-prefix result excludes
per-call vision processing. These labels describe execution evidence, not task success.
[Historical protocols and reproduction](../../benchmarks/vla/RESULTS_PROTOCOL.md).
