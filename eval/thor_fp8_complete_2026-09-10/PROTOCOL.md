# Thor FP8 completion sweep

Public `Runtime.from_pretrained(..., precision="native" | "fp8")`, same checkpoint revision, recorded cameras, synthetic state, prompts and step schedule. This compares complete runtime implementations, including graph capture and retained high-precision components; it does not isolate the cost of quantization. Native remains the default.

Fresh processes run sequentially under `/tmp/thor_gpu.lock` on one Jetson Thor. Timings synchronize CUDA around each full generation, including preprocessing and CPU action return. Stateless models use three warmups and twelve measured calls. VA and DreamZero use four episodes of three cycles, discard the first episode, and report nine measured cycles. VA uses fixed recorded executed-action feedback; both 25V/50A and 2V/4A have their own matched pair. These are early-history, short-run timings, not thermal or saturated-history qualification.

pi05 uses `lerobot/pi05_libero_finetuned_v044`, two active cameras (`image`, `image2`), float32 CHW images in [0,1], and clears the action queue before each generation. The native checkpoint preserves FP32 vision despite its BF16 configuration. GR00T, Cosmos and DreamZero use DROID checkpoints. Per-call cameras are refreshed, including GR00T.

Changed numerical recipes have no inherited closed-loop certificate. Returned action deltas are recorded only as a numerical screen. DreamZero also has an [unresolved native cross-process repeatability issue](../thor_precision_completion_2026-09-09/COMPARISON.md); its deltas cannot isolate FP8 error. Neither finite outputs nor a faster result establish task success. A byte-exact graph admission check compares graph execution with that component's eager reference on three inputs; it does not establish equivalence between the native and FP8 models.

## Execution coverage

| Family | FP8 execution | Retained precision / change in this sweep |
|---|---|---|
| LingBot-VA, both schedules | Existing fused DiT projection/FFN route | Native T5/VAE, attention and declared fallback families; schedule unchanged |
| VLA-4B | Existing language/action fused route | BF16 native vision now captured after exact admission |
| VLA-V2 | Existing routed expert/action route | BF16 native vision now captured after exact admission; FP16 language stack/routing retained |
| pi05 | Existing fused vision/language/action route | Norms, attention and processor remain recipe-specific; corrected two-camera fixture |
| Cosmos Edge/Nano | Both MoT towers' Q/K/V plus dense MLP | New `cosmos_mot_qkv_dense_mlp_*` recipe; native vision, attention and direct-weight output projections |
| GR00T | Qwen3 text attention/MLP plus existing VLSA | New `groot_qwen3_text_*` recipe; exact-admitted text component graphs with owned output copies, native vision and BF16 DiT |
| DreamZero | Causal self-attention Q/K/V plus DiT FFN | New `dreamzero_causal_qkv_ffn_*` recipe; native cross-attention, T5/CLIP/VAE, history and fixed 8-of-16 mask |

Dynamic per-tensor activation packing now uses a partial reduction and scalar reduction before the existing E4M3 pack kernel. The scale and packed bytes are checked against the previous PyTorch expression, including finite BF16 bit patterns, zero, extreme magnitudes and NaN propagation. `use_fast_accum=False` is retained. Loaded FP8 tensor counts are an inventory, not proof that every buffer was executed.

## Reproduction

`benchmark.py FAMILY PRECISION OUTPUT.json` runs one arm in the corresponding model environment. The runner requires the checkpoint cache, recorded cameras at `/home/guanming/thor_va_engine/va_eval_obs.npz`, the family dependency checkouts and compiled SM110 libraries. These inputs are hashed in the receipts; model weights and camera data are not redistributed here.

Frozen source, manifests, logs, timing receipts and action arrays are retained at `/home/guanming/ifl_eval/thor_fp8_complete_20260910`, mirrored under `/home/ubuntu/ifl_eval/thor_fp8_complete_20260910`. Failed attempts remain separate. [Source-version notes](SOURCE_NOTES.md) identify the snapshot used by each pair. The final comparison is published only after matching checkpoint/input/schedule/source identities and validating all action arrays and p50 calculations.
