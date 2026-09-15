# Cosmos Edge compile-shape ablation on Thor

This tests `compile_dynamic=False` on the existing experimental BF16 attention
lane, with pinned upstream `2b6c9a7`, full four steps, CFG 3 and no diffusion cache.
It does not modify the production Runtime or its accepted nightly baseline.

| Arm | p50 ms | Bytes vs dynamic A | Max action delta |
|---|---:|---|---:|
| Dynamic A | 1310.48 | equal | 0 |
| Static shapes | 1314.00 | different | 0.12775 |
| Dynamic B | 1301.04 | equal | 0 |

All arms return finite full 16 × 32 × 8 action arrays and execute 3584 BF16
attention calls. The repeated dynamic arm reproduces its action bytes. Static
shape compilation is 0.3–1.0% slower here and adds numerical differences, so it
is rejected. No Nano sweep was launched for this unpromising candidate.

This comparison is **within a BF16 numerical attention lane**. Even dynamic A
is not a BITEXACT transform of upstream attention and has no closed-loop quality
certificate. Changing compilation shape specialization is not assumed exact.

`run.py ROOT edge` launches three serialized fresh processes under the shared
Thor GPU lock. It uses the existing frozen Cosmos environment and fixture; the
paths are explicit in the script. Place `audit_upstream.py` and the unchanged
`attention-reuse/engine_attention.py` and `attention-reuse/probe.py` beside it.
`compare.py RECEIPTS edge` checks the matched fields, effective compile settings,
nonzero kernel coverage, finite actions, hashes and repeatability.

The six warmup and ten measured cases use two prompts, changing camera/state
inputs and recorded reset/seed behavior from the Cosmos regression workload.
The archived `receipts/` contains all reports, audit sidecars and action arrays.
