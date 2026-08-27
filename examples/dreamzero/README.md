# DreamZero (GEAR-Dreams, Wan2.2-5B WAM), in InstinctFlash

`GEAR-Dreams/DreamZero-DROID` is a causal video-action world-action model: 2 exterior + 1 wrist
camera at 160x320, a prompt, and a KV cache carried **across control cycles** within an episode
(first call warms it with one frame per camera; later calls append four) → a `(24, 8)` action
chunk from 16 scheduler steps at CFG 5.0, of which the shipped fixed mask computes 8 DiT
forwards. The adapter (backbone `dreamzero`) wraps the official serving wrapper in-process —
`DreamZeroWan225BPolicy` over `GrootSimPolicy` from the GEAR-Dreams checkout.

## Run it

```bash
pip install ./examples/dreamzero
export DREAMZERO_ROOT=/path/to/dreamzero-repo      # the GEAR-Dreams checkout
```

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("GEAR-Dreams/DreamZero-DROID")
with runtime.episode(prompt="pick up the banana and place it in the bowl") as episode:
    action = episode.predict(observation)           # -> {"action": (24, 8) float32}
```

Episode boundaries are load-bearing here: the KV cache outlives a cycle, so a new rollout must
be a new `episode()` (the adapter clears upstream's frame buffers and `current_start_frame`).
The Wan components (umt5 text encoder, CLIP, Wan2.2 VAE) resolve through upstream's
`ensure_file` and must be reachable in the HF cache; the load reads ~77 GB of weights.

## DYNAMIC_CACHE_SCHEDULE — a declared option, SCREEN tier, never default-on

Upstream's own velocity-cosine step skipper (`should_run_model`; the exact algorithm and
thresholds vLLM-Omni's `stepcache` vendored — see `step_cache.py` for the file-to-file
provenance). H100 pair: **3226.7 → 1843.1 ms (1.75x)**. It **changes actions by construction**
(measured max |ΔA| 0.288 on identical request streams vs the shipped mask), so:

- it is OFF unless `DYNAMIC_CACHE_SCHEDULE=true` (or the checkpoint declares
  `execution.dynamic_cache_schedule: true`, which the published declaration does not);
- the tier is **SCREEN** — measured deltas without a closed-loop certificate. A closed-loop
  success-rate gate is mandatory before this arm ships as anyone's default, and the adapter
  prints exactly that when the flag is on.

`cfg_batch.py` / `verify_cfg_batch.py` / `diag_batch.py` are the CFG-batching research arm
(gates and honest negative results); they are not part of the served path.

## Reproduce the README H100 row

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> examples/dreamzero/reproduce_h100.sh
```

Both arms are the official websocket server — the only difference between them is the env var —
measured by a byte-identical client (12 calls x 3 warmup, p50, 1-then-4-frame causal protocol).

## Attribution

DreamZero and GEAR-Dreams are © their authors (see the upstream checkout's LICENSE). Nothing is
vendored here beyond a measure client; the adapter imports the checkout and flips one env var
upstream itself defined.
