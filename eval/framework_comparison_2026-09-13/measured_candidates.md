# Thor measured candidates

p50 milliseconds; lowest observed completed candidate per cell. History models use continuation calls ([protocol](HISTORY_TIMING.md)). [Quality admission](quality_admission_inventory.json) is separate; these are not all quality-qualified winners.

| Model | LeRobot | vLLM-Omni | InstinctFlash-internal |
| --- | ---: | ---: | ---: |
| LingBot-VA | 1141.35 · native, 2V/4A | Unsupported | 461.12 · FP8, 2V/4A |
| LingBot-VLA-4B | Unsupported | Unsupported | 143.72 · FP8, default schedule |
| LingBot-VLA-V2-6B | Unsupported | Unsupported | 195.82 · FP8, default schedule |
| Cosmos3 Edge DROID | Unsupported | 1083.14 · compiled, UniPC4 | 1364.01 · NUMERIC, default schedule |
| Cosmos3 Nano DROID | Unsupported | 4466.37 · compiled, UniPC4 | 5523.21 · NUMERIC, default schedule |
| pi05 | 93.97 · compiled, NFE1 | Unsupported | 49.47 · FP8, default schedule |
| GR00T N1.7 | 247.95 · native, NFE4 | Unsupported | 109.06 · native, default schedule |
| DreamZero DROID | Unsupported | 8755.35 · compiled, upstream step cache | 21012.75 · FP8, default schedule |

Unsupported = no matching native action policy in the [pinned registry](framework_support_audit.json). Configurations can differ in steps, precision and caching; these are not same-computation speedups. [Setup and compatibility fixes](COMPATIBILITY.md) · [Raw receipts](receipt_inventory.json).
