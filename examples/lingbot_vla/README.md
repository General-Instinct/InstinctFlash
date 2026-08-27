# LingBot-VLA-4B, in InstinctFlash

`robbyant/lingbot-vla-4b-posttrain-robotwin` is a Qwen2.5-VL-based vision-language-action policy
(3 RoboTwin cameras + 14-dim state + prompt → a 50-step action chunk, served at `use_length=25`),
registered from outside the core through the `instinctflash.adapters` entry point. The adapter
wraps the **official serving class** in-process — `deploy.lingbot_vla_policy.LingbotVLAServer`
from the upstream checkout — so preprocessing, tokenization and action un-normalisation stay
byte-identical to upstream's server.

## Run it

```bash
pip install ./examples/lingbot_vla
export LINGBOT_VLA_ROOT=/path/to/lingbot-vla        # the upstream checkout (deploy/, configs/, assets/)
```

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("robbyant/lingbot-vla-4b-posttrain-robotwin")
with runtime.episode(prompt="pick up the block and place it in the tray") as episode:
    action = episode.predict(observation)            # -> {"action": (25, 14) float32}
```

The observation is the model's native RoboTwin format: `observation.images.cam_high` /
`cam_left_wrist` / `cam_right_wrist` as `(480, 640, 3)` uint8 (any size — the server resizes to
its 224 training resolution), `observation.state` as 14 floats, and a `prompt`.

## The T1 arm

One `infer` is 659.8 ms of which the 10-step denoise loop is 547.1 ms (83%) at 54.7 ms/step
(`profile_infer.py`). The loop cannot be replayed stock: `handle_kv_cache` concatenates the
chunk's prefill K/V with the step's suffix K/V per layer per step, so a captured graph would bake
the previous chunk's addresses. `lingbot_vla_iwm/static_capture.py` is the serving-engine fix —
one max-extent K/V buffer per layer, prefix slots refilled outside the graph once per chunk,
suffix slots overwritten inside at fixed addresses.

Gates and the published pair (H100): `verify_static_capture.py` — bitexact (max |d| = 0.0) on the
captured input, three unseen noise/observation cases and two cases on a different prompt with
prefix refill; end-to-end **672.7 → 184.0 ms in-process (3.66x)**, 54.7 → 11.9 ms/step. The
README table's stock arm is the official websocket server (670.9 ms — the ws hop costs ~2 ms).
The backend installs when the plan applies `graph_capture`; `IFL_VLA4B_BACKEND=eager` keeps the
stock loop for A/B runs.

## Reproduce the README H100 row

```bash
IFL_VLA4B_PY=<venv-with-upstream-stack>/bin/python CUDA_VISIBLE_DEVICES=<idle-gpu> \
  examples/lingbot_vla/reproduce_h100.sh
```

## Attribution

LingBot-VLA and its RoboTwin post-train checkpoint are Apache-2.0 (© their authors). Nothing is
vendored here — the adapter imports the upstream checkout and patches one instance-level method
at runtime, gated on bitexactness.
