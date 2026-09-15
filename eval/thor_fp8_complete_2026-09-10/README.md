# Refreshed Thor FP8 Runtime results

All nine matched pairs use full public Runtime generation timing on one Jetson Thor. Native remains the default; select `precision="fp8"` explicitly. FP8 includes retained higher-precision components and does not promise faster execution for every model.

| Model | Native p50 | FP8 p50 | Native / FP8 |
|---|---:|---:|---:|
| LingBot-VA | 5575.59 ms | 2824.01 ms | 1.974× |
| LingBot-VA @ 2V/4A | 750.58 ms | 422.59 ms | 1.776× |
| LingBot-VLA-4B | 360.46 ms | 160.25 ms | 2.249× |
| LingBot-VLA-V2-6B | 423.65 ms | 199.38 ms | 2.125× |
| Cosmos3-Edge-Policy | 3458.48 ms | 3416.68 ms | 1.012× |
| Cosmos3-Nano-Policy | 10337.82 ms | 8901.55 ms | 1.161× |
| pi05 | 318.76 ms | 53.92 ms | 5.912× |
| GR00T-N1.7-3B | 125.41 ms | 143.10 ms | 0.876× |
| DreamZero-DROID | 22906.49 ms | 20256.13 ms | 1.131× |

**Below 1× means slower.** Twelve measured calls per stateless arm, nine early-history cycles per VA/DreamZero arm. These are short-run p50 measurements, not sustained real-time qualification. pi05 uses two active cameras with normalized float32 inputs. Historical H100/Thor cells in the main README retain their original scopes.

The changes add exact-admitted native BF16 vision graphs to VLA-4B/V2, reduce shared FP8 activation-packing overhead, and expand Cosmos dense MLP, GR00T text attention/MLP and DreamZero FFN coverage. VA and pi05 retain their existing fused recipes. See the [per-family coverage and protocol](PROTOCOL.md).

Expanded numerical recipes have **no inherited closed-loop quality certificate**. Action-array deltas in the comparison are numerical screens only. Earlier simulator results remain attached to their original checkpoint/source/recipe.

Evidence: [88 regression checks](regression.json), [validated comparison](comparison.json), [hardware](hardware.json), [final source verification](source-final.json), and per-arm `<family>-<precision>.json` / `.npz` receipts. Frozen source and failed attempts remain in the raw archive named in the protocol. [Earlier sweep and audit](../thor_fp8_2026-09-10/README.md).
