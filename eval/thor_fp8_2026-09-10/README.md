# Thor FP8 Runtime results

See the [execution audit](ENGINE_AUDIT.md) for actual FP8 coverage, stage profiles, conversion overhead and the pi05 camera correction.

All nine current pairs passed their checks; pi05 uses the separately validated two-camera float32 input correction. **6 pairs measured more than 1% faster, 2 more than 1% slower, and 1 within 1% of native latency. This is a descriptive band, not a statistical significance test.** Native precision remains the default.

| Model | Native p50 | FP8 p50 | Native / FP8 | E4M3 tensors |
|---|---:|---:|---:|---:|
| LingBot-VA | 5514.01 ms | 2788.15 ms | 1.978× | 180 |
| LingBot-VA @ 2V/4A | 741.91 ms | 421.19 ms | 1.761× | 180 |
| LingBot-VLA-4B | 361.91 ms | 220.08 ms | 1.644× | 360 |
| LingBot-VLA-V2-6B | 424.84 ms | 239.58 ms | 1.773× | 360 |
| Cosmos3-Edge-Policy | 3454.36 ms | 3525.32 ms | 0.980× | 168 |
| Cosmos3-Nano-Policy | 10316.97 ms | 10287.37 ms | 1.003× | 216 |
| pi05 | 321.68 ms | 54.26 ms | 5.928× | 180 |
| GR00T-N1.7-3B | 133.57 ms | 145.47 ms | 0.918× | 214 |
| DreamZero-DROID | 23032.22 ms | 22401.51 ms | 1.028× | 120 |

These are matched public Runtime measurements on Jetson Thor. The pi05 pair uses `lerobot/pi05_libero_finetuned_v044`; GR00T, Cosmos and DreamZero use their DROID checkpoints. Ratios below 1 mean FP8 is slower. Each model uses its existing Thor FP8 implementation, including retained higher-precision components; VLA-4B vision stays BF16. The E4M3 count includes packed weights, runtime buffers and potentially unused frontend weights. It is an inventory, not executed quantization coverage.

Use `Runtime.from_pretrained(checkpoint, precision="fp8")` or `--fp8` to select it explicitly. `precision="native"` stays the default. FP8 is lossy arithmetic; speed does not certify task success. Existing simulator screens are separately scoped to their tested checkpoint, source and protocol. See [VA RoboTwin](../thor_precision_completion_2026-09-09/VA_ROBOTWIN_SCREEN.md), [VLA-4B/V2](../thor_precision_completion_2026-09-09/VLA_JOINT_SCREEN.md), [pi05 LIBERO](../thor_precision_completion_2026-09-09/PI05_PUBLIC_SIM_SMOKE.md) and [GR00T LIBERO fine-tune](../thor_precision_completion_2026-09-09/GROOT_LIBERO_SCREEN.md). The GR00T DROID, Cosmos and DreamZero speed rows have no corresponding closed-loop certificate here.

The original README cells retain their historical protocols. The new column supplies its own matched native/FP8 pair: it does not divide FP8 latency by an unrelated old baseline. See [protocol](PROTOCOL.md), [current checked comparison](current-comparison.json), [hardware](hardware.json), and [frozen source manifest](source-v1.json). Each arm has a `<family>-<precision>.json` receipt with raw timings, action archive hash, quantized module inventory and source hashes.

Reproduction on the measurement host: the frozen `source-v1`, `source-v2` and `source-v3`, compiled libraries and raw action archives are retained at `/home/guanming/ifl_eval/thor_fp8_20260910` on Thor, with receipts mirrored under `/home/ubuntu/ifl_eval/thor_fp8_20260910`. `benchmark.py` runs one arm, `run_sweep.py` schedules pairs under an exclusive GPU lock, and `summarize.py` validates the full sweep. Setup smoke runs and failed attempts are excluded. Other hosts must supply the checkpoint caches, family environments and the recorded camera archive named in the benchmark; this repository does not bundle model weights.

Validation: all 18 arm receipts, source/library hashes, matched inputs/schedules, finite actions and timing summaries checked. Shared runtime regression checks passed before the sweep. This remains a short latency test, not sustained thermal or real-time qualification.

The original pi05 7.41× cell used an unrecognized second camera key and unscaled uint8 images. It is superseded by the [validated two-camera correction](pi05-two-camera/pi05-two-camera-comparison.json); reproduce it with [the corrected benchmark](pi05-two-camera/pi05-two-camera-benchmark.py). The [initial sweep comparison](comparison.json) and receipts are retained for audit, not as the current pi05 result.
