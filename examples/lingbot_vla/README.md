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

### RTX 5090 full path

On SM120 the adapter upgrades the per-step backend with a vision/prefix graph, one graph for the
complete fixed ten-step Euler schedule, and bitexact GPU image normalization. Fixed-timestep
AdaRMS gamma/beta projections are evaluated once during capture and their original output tensors
are read by the corresponding unrolled step; the live modules are restored after capture.

The committed 5090 harness measures **220.5 → 97.3 ms p50 (2.27x)** with all six
stock-vs-candidate cases exactly equal. Independent repeats measured 97.4 and 98.74 ms. The prior per-step graph
measured 113.0–113.3 ms, so the new full path removes another 13–14%. Full profile and ablations
are committed in `evidence/reproduce_5090_full_path_results.json`.

## Graph capture is the default, and the self-check is the reason it can be

The backend installs when the plan applies `graph_capture` — for every 4B-class checkpoint,
fresh fine-tunes included. H100, Thor and SM120 default to the full path;
other CUDA devices retain the per-step static-KV graph. What makes either default safe is the runtime **self-check**,
not evidence measured on other checkpoints.

The per-step path compares replay against upstream eager `predict_velocity` on fresh actions,
timesteps and a synthetically refilled prefix. The full path instead compares six complete
`sample_actions` calls, including changed image, token, state and noise tensors, against the
true upstream concat-per-step path. GPU preprocessing separately compares six live BF16 image
tensors field-for-field. Exact equality is required everywhere. A mismatch releases only the
affected full graphs and returns to the already-gated per-step backend; serving continues.

Kill-switch: `IFL_VLA4B_NO_CAPTURE=1` serves eager (recorded on the plan, printed).
`IFL_VLA4B_BACKEND=eager` keeps the stock loop for A/B runs.
`IFL_VLA4B_FULL_GRAPH=0` returns to the per-step graph and
`IFL_VLA4B_GPU_PREPROCESS=0` retains upstream CPU preprocessing.
`IFL_VLA4B_SELFCHECK_FAULT=1` drills the loud full-graph fallback.

## Reproduce the README H100 row

```bash
IFL_VLA4B_PY=<venv-with-upstream-stack>/bin/python CUDA_VISIBLE_DEVICES=<idle-gpu> \
  examples/lingbot_vla/reproduce_h100.sh
```

## Native and FP8 on Thor

Use the same Runtime API with `precision="native"` (the default) or explicit
`precision="fp8"`. In `instinctflash serve`, add `--fp8` to opt in. Precision
selection preserves the checkpoint's ten denoise steps and native action
processing; it does not select a shorter schedule.

The corrected FP8 route keeps vision in BF16. The old FP16-vision speed result
is invalid for deployment because real images exposed overflow and
image-insensitive actions. Current short Runtime measurements are 355.00 ms
for native capture and 217.97 ms for FP8 per 25-action chunk (1.63×).
The 40-pair RoboTwin screen measured clean 15/20 → 14/20 and randomized
15/20 → 16/20 successes; all four `handover_block` scenes succeeded only
with native. These samples do not establish non-inferiority or sustained
real-time performance. [Current evidence](../../eval/thor_precision_completion_2026-09-09/COMPARISON.md).

The same six-case harness reproduces the SM120 full arm with an explicit full-path selection:

```bash
IFL_VLA4B_VERIFY_FULL=1 \
LINGBOT_VLA_CHECKPOINT=/path/to/checkpoint \
LINGBOT_VLA_NORM=/path/to/robotwin_50.json \
IFL_VLA4B_PY=<venv>/bin/python CUDA_VISIBLE_DEVICES=0 \
  examples/lingbot_vla/reproduce_h100.sh
```

## Attribution

LingBot-VLA and its RoboTwin post-train checkpoint are Apache-2.0 (© their authors). Nothing is
vendored here — the adapter imports the upstream checkout and patches one instance-level method
at runtime, gated on bitexactness.

The 2026-09-10 native Runtime pairs measured H100 184 → 99 ms and Thor 364 → 262 ms, with identical finite action bytes. Thor retains upstream eager vision attention. [Protocol and receipts](../../eval/native_optimization_2026-09-10/README.md).

The same checked image preprocessor also defaults on inside the explicitly selected Thor FP8 engine (161.5 → 155.3 ms; identical FP8 action bytes). `IFL_VLA4B_GPU_PREPROCESS=0` disables it in either precision mode. This is not native/FP8 quality equivalence.
