# Nano persistent text cache: Thor diagnostic

Original released Nano, complete SDE `[1, 0]`, CFG1, BF16 with cuDNN attention. This experiment extends the existing conditioning cache across successful model generations. It preserves the plain prompt, image/state inputs, 32×8 actions and step schedule. Production defaults are unchanged.

| Mode | p50 ms | p95 ms |
|---|---:|---:|
| Cache off | 787.636 | 788.899 |
| Recompute and verify | 790.209 | 792.641 |
| Persistent reuse | 723.825 | 726.499 |

The matched primary pair is **1.088×**, or **8.10% lower latency**, with six warmups and ten timed requests. Two prompts alternate across changed images/states and explicit request seeds. All three arrays of 16 complete actions match finite bytes exactly; verification checked 4,536 cached module tensors. See [primary receipts](receipts/comparison.json). This is an original-weight numerical/latency SCREEN, not a trained-student or closed-loop quality certificate.

Successful generations publish complete cache slots. Exceptions discard cached state. Model tensor versions, registered buffers, execution flags and schedule changes invalidate reuse; unsupported generation modes bypass it. The experiment is restricted to fixed geometry. CPU lifecycle tests are in `test_transactions.py`; actual output evidence comes from the GPU runs.

The separate [stress receipts](stress-receipts/comparison.json) cover four prompts competing for two slots, two evictions, an injected layer-5 failure, cleared state and a seeded retry. All three modes retain identical full action bytes. **Graph qualification remains incomplete:** the reuse receipt records PyTorch CUDA allocator assertions during graph capture followed by fallback. Stress timing is diagnostic only and must not be reported as acceleration. Correct fallback output does not establish graph stability. The next regression must isolate graph-pool lifetime across eviction/abort before enabling this combination by default.

`run_all.py CHECKPOINT OUTPUT --fixture FIXTURE` runs primary verification/off/reuse under `/tmp/thor_gpu.lock`; `run_stress.py` runs the failure/eviction diagnostic. Both require the staged Cosmos/Flash dependencies and Thor environment used by the source-bound receipts. Receipt source hashes and all six action archives were independently rechecked after transfer.
