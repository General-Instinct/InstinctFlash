# Native Cosmos scheduler control on Thor

This is an offline investigation of DreamZero's GPU-scheduler idea on the
current Cosmos native BF16 path. No cuDNN attention override is combined with
this experiment, and no production Runtime option is installed.

## Result and decision

**Do not promote this candidate.** It passes full-action byte checks but fails
the 1.02× performance gate. Nano was not benchmarked for this candidate after
the Edge gate failed; the Nano trace audit below is existing-profile evidence.

| Edge arm | p50 (ms) | p95 (ms) | Full action bytes |
| --- | ---: | ---: | --- |
| Native A | 2469.80 | 2473.26 | Reference |
| Host timestep control | 2471.51 | 2475.29 | Identical |
| Native B | 2470.60 | 2472.49 | Identical |

The candidate is 0.04–0.07% slower, with only 0.033% reference drift. All 16
candidate requests actually used CPU timesteps; all baseline requests used CUDA
timesteps. All complete 16×32×8 action arrays are finite and byte-identical,
and the baseline also matches the earlier archived guarded-native receipt.
The [comparison](receipts/edge-comparison.json) deliberately records
`action_bytes_passed=true`, `performance_passed=false` and `passed=false`.
A nonzero comparator exit is expected for these receipts.

This rejects **timestep placement alone** as a meaningful speed improvement.
It does not evaluate precomputed solver coefficients, static packed-input
reuse or removing velocity-mask host branches. Those require independent
experiments; no speed or byte-equivalence claim is made for them here.

## Source findings

Cosmos already performs latent updates on CUDA. Its UniPC implementation retains
sigmas on CPU and constructs step-specific coefficient tensors on the sample's
device. Moving all scalar arithmetic to CUDA would change where logarithms,
exponentials and solver coefficients are computed, so it is not automatically
a BITEXACT transformation.

A narrower redundancy is visible in the pinned source:

1. `FlowUniPCMultistepScheduler.set_timesteps` constructs the schedule on CPU and
   copies the integer timestep grid to CUDA.
2. `UniPCSampler.forward` iterates the grid and passes each timestep to velocity
   evaluation.
3. `OmniMotModel._get_velocity` immediately copies that timestep back to CPU for
   input packing, separately for each CFG branch.
4. First-step scheduler lookup also uses a CUDA `nonzero` and scalar read.

The candidate asks the **original** `set_timesteps` implementation to retain
its control grid on CPU. It preserves sigma/coefficient arithmetic, latent
devices, four denoising steps, full CFG 3 and sampling RNG. The offline subclass
rejects schedules other than four steps and custom `solver_p`. Both baseline
and candidate use the same instrumentation, with the override disabled for the
baselines. Runtime's guarded conditioning cache remains enabled in every arm.

## Interpreting synchronization profiles

| Existing native profile | `cudaStreamSynchronize` count | CPU wait total | Summed CUDA event execution |
| --- | ---: | ---: | ---: |
| Edge | 276 | 1929 ms | 2408 ms |
| Nano | 276 | 6302 ms | 7905 ms |

These columns overlap. CPU waits frequently include the network's necessary GPU
execution, so removing a synchronization cannot be credited with removing that
execution time. Nested `aten::item` and `_local_scalar_dense` rows also overlap
and must not be summed. The retained [Edge](receipts/edge-scheduler-trace-audit.json)
and [Nano](receipts/nano-scheduler-trace-audit.json) trace audits identify the
original Chrome traces by SHA256. The profiles predate this candidate and are
diagnostic, not proof of its speedup.

The velocity function additionally makes GPU-dependent branch decisions when
masking predicted vision/action velocities. Those decisions are outside the
scheduler and are not changed by this candidate. They can still impose barriers
after model execution, limiting the benefit from timestep placement alone.

## Protocol and reproduction

Run native A, host-timestep candidate, native B in three fresh processes on an
idle Thor, holding `/tmp/thor_gpu.lock` for the entire suite. Every arm has six
warmups and ten measured requests with varied recorded observations, prompts,
seeds and resets. Retain all 16 full 32×8 action chunks. Actual timestep devices
and all 16 schedule initializations must match the requested arm; unchanged
checkpoint, source hashes, schedule, precision, cache admission and graph replay
are checked before action/performance comparison.

The promotion gate requires finite identical action bytes, at least 1.02× speed
against **both** references, reference drift at most 5%, and p95 regression at
most 10%. An invalid comparison clears any stale success receipt. No
closed-loop quality certificate is inferred from recorded-input equality.

Using the existing pinned Thor Cosmos environment and checkpoints, put the
Cosmos checkout on PYTHONPATH and run with its Python interpreter:

```bash
python eval/cosmos3_thor_2026-09-12/scheduler/run.py /absolute/new-output --family edge --wait
python eval/cosmos3_thor_2026-09-12/scheduler/compare.py /absolute/new-output edge
```

`--wait` waits for the GPU lock; otherwise an occupied lock fails immediately.
Existing output files are not overwritten. The runner strips inherited Cosmos
optimization flags and enables the previously qualified conditioning cache.
Source and experimental-script hashes are retained in the benchmark receipts.

The GPU suite ran after the existing automatic Cosmos regression released the
lock; that regression was not interrupted or modified.
