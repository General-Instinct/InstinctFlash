# Cosmos BF16 attention scheduling screen on Thor

This is an offline tuning experiment on the existing numeric BF16 attention
kernel. It is not a Runtime registration or a task-quality certificate.

Two candidates were screened:

1. `rejected_q128.cu` changes the query tile from 256 to 128, retaining key tile
   128. It fails to compile against the pinned CUTLASS mainloop: its hard-coded
   128-row fragments after thread partitioning produce non-injective/incompatible
   TMEM copy layouts. No speed or correctness result exists for that candidate.
2. `bf16_fmha.cu` retains the original 256 × 128 tile and changes only the
   scheduler from IndividualTileScheduler to PersistentTileScheduler. It builds
   successfully and matches every tested output byte, but is slower on the real
   Edge/Nano GQA geometries.

| Shape (Q, KV, Q heads, KV heads) | Individual ms (repeat) | Persistent ms |
|---|---:|---:|
| 3094, 3245, 16, 8 | 1.806 | 2.104 |
| 3094, 3112, 16, 8 | 1.741 | 2.024 |
| 3094, 3182, 32, 8 | 3.367 | 3.868 |
| 3094, 3103, 32, 8 | 3.363 | 3.868 |

Persistent scheduling is rejected: kernel latency increases about 15–17%,
equivalent to a speed ratio of 0.86–0.87x. No full-model sweep is justified for
this candidate. These byte comparisons are against the existing experimental
BF16 kernel, not upstream NATTEN or the production BITEXACT baseline.

Reproduction: copy the unchanged `attention-reuse/probe.py` beside the scripts,
set CUTLASS_ROOT to pinned commit `da5e086dab31d63815acafdac9a9c5893b1c69e2`, run
`bash build.sh`, then `compare_tiles.py ROOT` with the existing Cosmos interpreter
under `/tmp/thor_gpu.lock`. The script records both library hashes and compares
individual/persistent/individual timings on synthetic tensors with real shapes.
The original library path is explicit in the script. This screen uses 5 warmup
and 30 timing calls per arm; it is not an end-to-end latency measurement.
