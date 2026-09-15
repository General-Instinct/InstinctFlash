# GR00T native BF16 norm-graph ablation on Thor

Four fresh processes run the same cached `nvidia/GR00T-N1.7-3B` checkpoint through
the native Runtime: current baseline, original norm inside a graph, shared BF16
norm inside a graph, and current baseline repeated. No production adapter is
changed. `norm_candidate.py` is installed only by this benchmark, after loading
and before the first prediction.

Only exact `Qwen3VLTextRMSNorm` instances with width 2048, BF16 weights and the
verified forward source hash are eligible. Training modules, hooks and already
replaced forwards are rejected. Attention, MLPs, precision, preprocessing and
four-step action scheduling are unchanged. Both graph variants return owned
copies so upstream consumers cannot retain overwritten graph buffers.

`StaticTensorGraph` compares the captured candidate to the original upstream
forward on real, flipped and zero inputs. Each report includes every norm's
capture verdict and replay count; an inactive or rejected graph fails the arm.
This is an offline experiment: parameter replacement, training and concurrent
serving are outside its admission scope.

## Protocol

- Shared `/tmp/thor_gpu.lock`; per-request competing GPU PID checks.
- Fixed RNG seeds, 3 warmup + 20 measured requests; 12 recorded camera frames,
  two prompts and synthetic state. Reset before every request.
- Synchronized wall-clock generation latency; all 23 action chunks saved.
- TF32 and cuDNN benchmark disabled in every arm.
- Fresh processes use the same frozen source snapshot and existing GR00T env.
- `compare.py` checks checkpoint revision, input and benchmark hashes, execution
  policy, numeric settings, finite action bytes and active graph evidence.

Run `run.py /absolute/frozen/source/root` on Thor after placing the source and
recorded fixture at the paths expected by the script. The interpreter and
GR00T source paths are recorded explicitly in `run.py`. Existing output is never
overwritten. Run `compare.py /absolute/results/root` after all four arms finish.

The result is action equivalence on this input set, not a simulator success-rate
certificate or a proof covering every fine-tuned checkpoint.

## Results and decision

| Round, p50 ms | Baseline A | Norm graph only | Shared norm + graph | Baseline B |
|---|---:|---:|---:|---:|
| Initial | 119.22 | 112.08 | 112.26 | 113.39 |
| Repeat | 116.17 | 114.61 | 114.25 | 112.89 |

All arms passed finite byte equality on all 23 × 40 × 17 actions within each
round. Each candidate installed 33 norm graphs and recorded 759 replays, with
all capture gates passing. No competing GPU processes were detected at the
per-request checks. Both rounds used the same benchmark and backend sources.

**Do not enable this candidate by default.** In the repeat, shared norm + graph
is 1.2% slower than Baseline B; first-round gains relative to Baseline A are
confounded by baseline drift. The shared fusion also does not establish a
consistent improvement over norm graph alone. Module-level speedups therefore
do not translate into a demonstrated end-to-end gain at this boundary.

Next investigate a larger norm/projection boundary. Per-norm graphs incur input
staging and owned-output copies; their contribution is a hypothesis to profile,
not a measured attribution in these runs. The native plan intentionally leaves
Thor DiT graph capture disabled pending its own device qualification; this
experiment did not override that decision.

Receipts and action arrays are in `receipts/` and `receipts/repeat/`.
`run.py ROOT OUTPUT_DIR` can place a repeat into a new directory without
changing the frozen model/benchmark source or overwriting earlier results.
