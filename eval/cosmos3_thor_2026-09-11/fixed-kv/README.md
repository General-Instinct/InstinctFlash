# Fixed interleaved KV storage on Thor

Decision: **do not promote**. Edge action bytes match, but the incremental
speedup fails the 1.02× performance gate and costs another 1.03 GiB of peak
allocated memory. Nano was not measured for this candidate.

| Edge arm | p50 (ms) | p95 (ms) | Full-action bytes |
| --- | ---: | ---: | --- |
| Guarded conditioning Runtime A | 2467.50 | 2469.97 | Reference |
| Same Runtime + fixed KV buffers | 2466.21 | 2467.75 | Identical |
| Guarded conditioning Runtime B | 2472.55 | 2475.68 | Identical |

The candidate is only 1.00052–1.00257× faster; reference drift is 0.205%.
Peak allocated memory grows from 11,716,920,832 to 12,819,521,024 bytes.
The [comparison](receipts/edge-comparison.json) deliberately records
`action_bytes_passed=true`, `performance_passed=false`, and `passed=false`.

## Mechanism and scope

This adapts fixed KV storage to Cosmos's native interleaved text/generation
indices. A contiguous text-prefix layout cannot be substituted without changing
attention token order. Every request fills independent K/V storage for each
conditioning branch and layer using native `get_all_seq`; subsequent decodes
scatter only generation rows with the same native indices. Attention kernels,
RoPE, BF16 precision, four UniPC steps and CFG 3 remain unchanged. Runtime's
first-use reference validation bypasses the experimental buffers.

All three arms enable the guarded conditioning cache, so this experiment measures
only the additional buffer change. The candidate reports 1,792 prefill operations,
672 decode updates and 84 validation bypasses; counters describe Python calls,
not repeated GPU operations inside graph replay. All arms admitted and replayed
the conditioning cache without rejection. Existing graph capture may leave little
allocation overhead to remove; this is an interpretation, not a profiler result.

This is an offline prototype, not a supported Runtime option. The action evidence
covers these recorded inputs only and is not a closed-loop quality certificate.

## Reproduction

The inference implementation is frozen at `e09f698`; the receipts record actual
imported source hashes and candidate hashes. Use the pinned Thor environment and
fixtures described in [Runtime integration](../runtime-conditioning/README.md).
On Thor, with a checkout at an absolute path and no existing output files:

```bash
python eval/cosmos3_thor_2026-09-11/fixed-kv/run.py /absolute/path/to/checkout edge
python eval/cosmos3_thor_2026-09-11/fixed-kv/compare.py /absolute/path/to/checkout edge --minimum-speedup 1.02
```

The runner holds the GPU lock across three fresh processes, each with six warmup
and ten measured requests. It retains complete 16×32×8 action arrays, timing,
source/protocol identities and cache evidence. The comparison checks all finite
action bytes, protocol agreement, drift and latency gates. For the archived
receipts, use this directory's `receipts` as the comparison input; a nonzero exit
is expected because the performance gate failed.
