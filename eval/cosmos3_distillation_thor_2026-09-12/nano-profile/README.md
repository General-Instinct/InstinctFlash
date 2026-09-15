# Nano SDE1 BF16 diagnostic profile on Thor

Released original Nano weights, complete SDE `[1,0]`, CFG1, cuDNN attention,
plain native prompt processing and zero padding. This is an instrumented profile,
not a replacement latency benchmark or a trained-student quality result.

Six warmups precede ten instrumented requests. All 16 finite 32×8 action arrays
match the saved uninstrumented cuDNN cost baseline byte for byte. Source hashes,
raw artifact identities and the comparison are recorded in `receipt.json`.

| Component | Mean CUDA event span per request |
|---|---:|
| Input preparation | 88.64 ms |
| Attention, all 36 layers | 188.98 ms |
| Text MLP, all 36 layers | 49.01 ms |
| Generation MLP, all 36 layers | 334.28 ms |
| Complete layers | 651.52 ms |
| Complete velocity evaluation | 680.07 ms |

Spans include stream idle time and instrumentation overhead. Layers and velocity
contain other listed groups; do not add these nested measurements. Generation
MLP input is `[3093,4096]`; text MLP input is `[89,4096]`.

Layer graphs are explicitly disabled for attribution. The saved baseline also
performed no captures or replays: all 576 layer calls bypassed the existing
16 GiB free-memory guard. This identifies a memory investigation, not permission
to lower the guard or evidence of a graph speedup.

Prioritize generation MLP projection costs and attention. Existing FP8 merged
projection patterns may inform a BF16 implementation, but changing GEMM geometry
requires separate numerical and full-request latency checks. Measure free,
allocated and reserved memory before deciding whether existing layer graphs can
safely run. Text MLP caching alone cannot remove the dominant generation cost.

Run `profile_original.py CHECKPOINT FRESH_REPORT --attention cudnn --fixture
FIXTURE --allow-unqualified` in the same Thor study environment as the sibling
Nano cost benchmark, with `realtime-deployment-v2` on PYTHONPATH and the shared
Thor GPU lock held. The original cost manifest and external Wan VAE are checked.
