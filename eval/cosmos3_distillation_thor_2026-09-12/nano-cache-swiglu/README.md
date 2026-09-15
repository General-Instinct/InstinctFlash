# Persistent cache plus shared BF16 SwiGLU

Additive matched ablation on original Nano SDE1 CFG1/cuDNN: persistent cache alone (`off.json`) versus the same cache plus the existing72-module BF16 SwiGLU fusion (`reuse.json`). Both arms actually request cache reuse; these filenames are inherited from the runner and explicit arm labels are recorded in the comparison. No FP8 or new step schedule is used.

Each arm declares12 warmups and24 measured requests. Full36×32×8 actions must match finite bytes. Measured requests must have no graph capture/check and exactly36 layer replays. The fused arm must install all72 SwiGLU wrappers. The run completed successfully under `/home/guanming/ifl_eval/cosmos_distill_20260912/nano-cache-swiglu-v1/results/live_status.json`, serialized by the Thor lock. No task-quality certification is implied.


| Arm | p50 ms | p95 ms |
|---|---:|---:|
| Persistent cache | 730.160 | 731.825 |
| Cache plus BF16 SwiGLU | 714.958 | 718.345 |

The [matched pair](receipts/comparison.json) gives1.0213×, or2.08% lower p50 latency. All72 full actions matched finite bytes; each measured request in both arms replayed all36 layers with zero new captures/checks and no graph rejection. After transfer, all source/artifact hashes and measured counter transitions were independently rechecked. This is one ordered pair on original weights, with no FP8 or schedule change. It does not establish a cross-checkpoint quality certificate or remove new-prompt preparation costs.
