# Nano original SDE1 BF16 SwiGLU ablation on Thor

Matched complete SDE1 `[1,0]`, CFG1, cuDNN BF16 attention and native prompt
processing; six warmups and ten measurements per process, serialized under the
shared Thor GPU lock. The sole candidate switch is the existing
`IFL_COSMOS3_SWIGLU=1` lookup-table fusion. All 72 text/generation MLPs installed,
with no pointwise numerical rejections or errors.

| Arm | p50 | p95 |
|---|---:|---:|
| Baseline | 773.75 ms | 775.27 ms |
| Shared BF16 SwiGLU | 766.43 ms | 770.64 ms |

All 16 finite 32×8 full actions matched byte for byte on Thor and independently
after transfer. Both arms had zero graph captures/replays and 576 memory bypasses.
Measured speedup is 1.0095×, approximately 1%; this small single-pair difference
is not a robust performance qualification or evidence of a major bottleneck fix.
No production default changes. These are original weights, not trained Nano
students, and no task-quality conclusion follows.

`run_pair.py CHECKPOINT FRESH_RESULTS --fixture FIXTURE` runs both arms and the
paired comparator in the sibling Nano cost-study environment. The worker binds
source/package/fixture identities, actual sampler clocks, full action shape,
loaded external Wan VAE and actual optimization counters. Receipts include both
raw reports and action arrays.
