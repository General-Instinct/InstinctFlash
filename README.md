<div align="center">

<img src="assets/iFlash.png" alt="InstinctFlash" width="360"/>

**A high-performance serving framework for robotics models.**

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

---

## What's new 🔥

- **Seven model families served, measured, and tiered.** LingBot-VA 3.22×, LingBot-VLA-4B 3.66×,
  **LingBot-VLA-V2-6B (sparse-MoE) 4.45×**, Cosmos3-Edge-Policy 1.64×, Cosmos3-Nano-Policy 1.51×,
  pi05 1.65× — each against its authors' own serving code on the same H100, identical requests
  per arm. DreamZero-DROID ships with its upstream dynamic step-cache surfaced as a declared
  configuration (1.74×, screen-tier).
- **Declared operating points, and a few-step distillation framework on the way.** A checkpoint
  can declare a few-step denoise schedule and the runtime serves it: LingBot-VA at its 2V/4A
  point runs the full pipeline in **360 ms** on the same H100 — **23×** end to end. A changed
  schedule never inherits a serving tier: the point ships with its own paired closed-loop
  evidence (600 RoboTwin episode pairs, −2.3 pp against a pre-registered −0.05 non-inferiority
  margin; certification extension to n=1200 in progress). The distillation framework that
  trains the few-step gap back out is under development as an InstinctFlash component.
- **Static-KV replay-safe CUDA-graph capture.** A preallocated max-extent KV buffer makes the
  denoise loop capturable with **bit-exact replay on unseen inputs** — verified on three model
  families and on two GPU architectures (H100/SM90 and Jetson Thor/SM110, same certificate).
  Sparse-MoE routing is capture-safe: it re-executes per replay, never baked.
- **[InstinctCompress](https://github.com/General-Instinct/InstinctCompress)** (customer access),
  the companion compression toolkit: a fine-tuned pi05 checkpoint 8.7 GB → 3.2 GB with the
  accuracy trained back on your own demonstrations and verified, then served through the same
  stack unchanged.

## Architecture

A checkpoint carries a short declaration of what it is. The runtime reads the declaration, decides
which optimizations are provably valid for those weights, applies them, and shows its work:

```
checkpoint ─▶ adapter          ─▶ planner            ─▶ engine passes        ─▶ actions
              declares what        decides what          apply and measure
              the model is         is valid (no GPU,     each optimization
                                   no weights needed)
```

Optimization is organized in six layers, by what each one changes:

| layer | | changes |
|:--|:--|:--|
| 1 | **MODEL** | what is computed — distillation, step reduction, checkpoint compression ([InstinctCompress](https://github.com/General-Instinct/InstinctCompress), [instinct-pdd](https://github.com/General-Instinct/instinct-pdd)) |
| 2 | **GRAPH** | when work is issued — prefill extraction, CUDA-graph capture, memory planning |
| 3 | **CACHE** | what is recomputed — KV reuse, cross-attention and episode caches |
| 4 | **ATTENTION** | how tokens mix — FlashAttention, hybrid and linear attention |
| 5 | **KERNEL** | how a kernel is written — backend and layout dispatch, fusion |
| 6 | **HARDWARE** | what it executes on — fp8/int8, TensorRT, Jetson-class edge devices ([`serving/`](serving/)) |

Layer 1 changes the *weights* and produces a checkpoint; it lives in the companion repos. Layers
2–6 change *how the weights execute* and produce a plan; they are the runtime in this repo. The
layers are not a priority order — the runtime measures where the time actually goes and starts
there.

## Using it

```bash
git clone https://github.com/General-Instinct/InstinctFlash && cd InstinctFlash
pip install -e ".[runtime,diffusion]"
```

Python 3.10+. `pip install -e .` alone needs no GPU and is enough to inspect checkpoints;
LingBot-VA needs a CUDA GPU with ~30 GB free.

Serve actions:

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("robbyant/lingbot-va-posttrain-robotwin")

with runtime.episode(prompt="put the bottle in the dustbin") as episode:
    while not done:
        action = episode.predict(observation)
```

`observation` is a dict in the model's own format. That is the whole API — no server to start, no
optimization to choose. If a safety layer changed the action before it reached the robot, pass
`executed_action=...` and the model conditions on what actually happened.

Or from the command line:

```bash
instinctflash devices                 # what machine am I on, and what can it do
instinctflash describe  <model-id>    # what a checkpoint declares — no weights downloaded
instinctflash plan      <model-id>    # what the runtime would do to it, and why
instinctflash run       <model-id>    # load it and produce real actions
```

`describe` and `plan` need no weights and no GPU: they answer *will this machine serve it* before
you commit to a download.

Compress a checkpoint before serving it — see
[InstinctCompress](https://github.com/General-Instinct/InstinctCompress) (customer access;
contact founders@general-instinct.com):

```bash
pip install git+https://github.com/General-Instinct/InstinctCompress   # with granted access
pi05-compress compress <checkpoint> out/ --tasks tasks.txt --dataset <your_demonstrations>
```

| model | pytorch vs InstinctFlash | tier |
|:--|:--|:--|
| **LingBot-VA** (14B WAM) | 8308 → 2580 ms, **3.22×** | NUMERIC |
| ↳ **@ 2V/4A** (declared operating point) | 8308 → 360 ms, **23×** | OPERATING-POINT |
| **LingBot-VLA-4B** | 673 → 184 ms, **3.66×** | BITEXACT |
| **LingBot-VLA-V2-6B** (sparse-MoE) | 840 → 184 ms, **4.45×** | NUMERIC |
| **Cosmos3-Edge-Policy** (3.86B) | 300 → 184 ms, **1.64×** | NUMERIC |
| **Cosmos3-Nano-Policy** (15.75B) | 480 → 318 ms, **1.51×** | NUMERIC |
| **pi05** | 299 → 181 ms, **1.65×** | BITEXACT |
| **DreamZero-DROID** (Wan2.2-5B WAM) | 3117 → 1787 ms, **1.74×** | SCREEN |

The Cosmos3 arms are measured on repeated single-prompt serving; multi-prompt serving currently
runs the pipeline arm. Tiers are derived from what a pass can prove, never asserted. **BITEXACT** means identical actions,
**NUMERIC** means a declared-margin result, **SCREEN** means measured deltas without a closed-loop
certificate. **OPERATING-POINT** means a declared few-step schedule — changed computation,
carried by its own paired closed-loop evidence rather than a serving tier. `runtime.explain()` prints the chain and its tier for the checkpoint you loaded,
including the passes it declined and why. Protocols and per-pass results are in [`eval/`](eval/).

To add your own model family, declare an `instinctflash.adapters` entry point and `pip install`
your package — see [`examples/external_plugin/`](examples/external_plugin/).
