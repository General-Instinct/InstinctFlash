# Stronger Thor baseline audit (in progress)

The published 1.33× Edge / 1.18× Nano pairs compare against **eager**
`cosmos-framework` f734253f. They are not comparisons against NVIDIA's latest
default optimized server, and do not establish that native optimization is complete.
The existing constructor explicitly sets `use_torch_compile=False`.

## Missing baseline coverage

Upstream commit `2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce` was copied into an
isolated Thor directory, retaining the installed torch 2.10/cu130 environment.
Its server imports and executes successfully there. The recommended environment
in its documentation is instead `cu130-torch213-train`; that environment has not
yet been reproduced. The effective policy setup enables compilation but disables
CUDA graphs (the general base argument class has a different graph default).

New upstream code includes request-local conditional/unconditional text KV
caching, packed input templates, single-sample dense attention and opt-in
batched CFG. These require measured qualification before adopting them under
InstinctFlash's BITEXACT ceiling. Native BF16 alone does not establish action
identity across upstream versions.

The latest Edge server command also specifies `--guidance-interval 960 1001`.
This changes the guidance computation. The model card's 1.528-second Thor result
does not specify this interval, so its exact provenance remains unresolved.

SeaCache is a separate approximate optimization. Shared setup defaults it off
(the general inference CLI enables it); the audit explicitly disables it.

## Exploratory measurements

Same recorded-input workload as the regression suite, six warmup and ten timed
requests, full 32×8 actions, BF16, four UniPC steps, CFG scale 3, TF32 off and
cuDNN benchmark off. These single arms are diagnostic, not A/current/B certificates.

| Edge path | Warm p50 | Status |
| --- | ---: | --- |
| Latest upstream, eager, CFG every step | 3180.94 ms | Completed; actions differ from old upstream |
| Latest upstream, eager, CFG interval 960–1001 | 2067.79 ms | Completed; 80 velocity calls / 16 requests = 5 per request |
| Latest upstream, eager, batched full CFG | 3416.01 ms | Slower; 64 batched velocity calls / 16 requests; actions match latest sequential run |
| Latest upstream, compiled, CFG every step | 2125.79 ms | Completed; 128 velocity calls; effective CUDA graphs off |
| Latest upstream, compiled, CFG interval 960–1001 | 1386.33 ms | Completed; changed guidance schedule, 5 calls per request |

The interval run confirms template and text-KV eligibility on all 16 requests.
Changing the upstream version alone produced max absolute action delta 0.09865
and mean absolute delta 0.01187 versus the first 16 old-reference requests.
These are raw action deltas, not task-success changes; the responsible upstream
changes have not yet been isolated.

The batched and sequential full-CFG runs have identical action arrays on these
16 requests, but batching is 7.4% slower here. It should not become a default on
this evidence. The interval run differs from latest full CFG (max absolute
delta 0.27857, mean absolute delta 0.03062), confirming it is not a BITEXACT
replacement.

Compilation is 1.496× faster than latest eager here and faster than our published
2554.4-ms path. It changes action bytes versus latest eager (max absolute delta
0.16081, mean absolute delta 0.01722). It remains a native-BF16 baseline, but
cannot be silently admitted as a BITEXACT transform of eager execution.

The compiled interval path is within the performance range needed to explain
the model card's 1.528-second result, but this is not an exact reproduction of
its environment and input workload. Do not use its ratio as a lossless speedup.

## Nano

Latest upstream eager full CFG measures 9714.79 ms p50. Versus the first 16
old-reference requests, max absolute action delta is 1.12710 and mean absolute
delta is 0.08960. This source-version difference needs investigation; it does
not establish a task-quality loss.

Latest compiled full CFG measures **7404.12 ms** p50, versus our published
8672.65-ms old-source optimized path. It is 1.312× faster than latest eager.
All requests return finite full action chunks, but compiled versus latest eager
has max absolute delta **2.17102** and mean absolute delta **0.14339**. This is a
material numerical discrepancy to investigate, not a qualified BITEXACT path or
evidence of unchanged task success.

Next gates: isolate upstream and compiler numerical
changes and port compatible optimizations individually, qualifying exact actions
and speed against the matching baseline. Preserve the accepted nightly baseline until new matched
results justify an explicit replacement. Distillation is deferred during this audit.

Raw arrays, per-request timings, source hashes and instrumentation sidecars are
in [audit-receipts](audit-receipts/). These are exploratory arms and must not be
substituted for the accepted regression receipts. `audit_upstream_executed.py`
archives the instrumentation used by the compiled matrix; the reusable runner
now additionally stamps exploratory status and its own source hash into reports.

`audit_upstream.py` reuses the regression workload and records actual cache
eligibility and velocity call counts. Put the pinned upstream first on
`PYTHONPATH`, hold `/tmp/thor_gpu.lock`, and run:

```bash
python eval/cosmos3_thor_2026-09-11/audit_upstream.py edge baseline_a /absolute/new-result.json --iterations 10
# Separate experiments: AUDIT_BATCHED_CFG=1, AUDIT_COMPILE=1,
# or AUDIT_CFG_INTERVAL=960,1001. Never combine these silently.
```

Sources: [upstream defaults](https://github.com/NVIDIA/cosmos-framework/blob/2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce/cosmos_framework/inference/common/args.py),
[server protocol](https://github.com/NVIDIA/cosmos-framework/blob/2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce/docs/action_policy_droid_server.md),
[Edge model card](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID).
