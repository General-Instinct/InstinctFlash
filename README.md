<div align="center">

<img src="assets/iFlash.png" alt="InstinctFlash" width="360"/>

**A high-performance serving framework for robotics models.**

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

---

## Install

```bash
git clone https://github.com/General-Instinct/InstinctFlash && cd InstinctFlash
pip install -e ".[runtime,diffusion]"
```

Python 3.10+. `pip install -e .` alone needs no GPU and is enough to inspect checkpoints;
LingBot-VA needs a CUDA GPU with ~30 GB free.

## Load a model

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("robbyant/lingbot-va-posttrain-robotwin")
```

## Get actions

```python
with runtime.episode(prompt="put the bottle in the dustbin") as episode:
    while not done:
        action = episode.predict(observation)
```

`observation` is a dict in the model's own format. That is the whole API — no server to start, no
optimization to choose. If a safety layer changed the action before it reached the robot, pass
`executed_action=...` and the model conditions on what actually happened.

## What's new 🔥

- **Eight model families served, measured, and tiered — on two device classes.** LingBot-VA,
  LingBot-VLA-4B, LingBot-VLA-V2-6B (sparse-MoE), Cosmos3 Edge and Nano policies, pi05,
  GR00T-N1.7, and DreamZero-DROID — each measured against its authors' own serving code on the
  same device, identical requests per arm, on H100 **and on Jetson Thor**. The two device
  classes need opposite optimizations (H100 is launch-bound at batch 1, Thor is
  bandwidth-bound), which is why the planner measures before it applies. The numbers live in
  the results table below.
- **Declared few-step schedules, and a few-step distillation framework on the way.** A checkpoint
  can declare a few-step denoise schedule and the runtime serves it: LingBot-VA at its 2V/4A
  point runs the full pipeline at twenty-three times its upstream end-to-end serving
  cost — the measured row is in the results table below. A changed
  schedule never inherits a serving tier: the point ships with its own paired closed-loop
  certificate — **non-inferior** over 1153 pre-registered RoboTwin episode pairs, with the
  most-conservative interval clearing the declared margin with room to spare. The distillation
  framework that trains the few-step gap back out is under development as an InstinctFlash
  component.
- **Static-KV replay-safe CUDA-graph capture.** A preallocated max-extent KV buffer makes the
  denoise loop capturable with **bit-exact replay on unseen inputs** — verified on three model
  families and on two GPU architectures (H100/SM90 and Jetson Thor/SM110, same certificate).
  Sparse-MoE routing is capture-safe: it re-executes per replay, never baked.
- **[InstinctCompress](https://github.com/General-Instinct/InstinctCompress)** (customer access),
  the companion compression toolkit: a fine-tuned pi05 checkpoint 8.7 GB → 3.2 GB with the
  accuracy trained back on your own demonstrations and verified, then served through the same
  stack unchanged.

## Framework overview

InstinctFlash keeps model declarations, optimization planning, runtime execution, and evidence in
one inspectable path, whether it is called from Python or the command line.

# Architecture

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

## Results

| model | PyTorch → InstinctFlash (H100) | PyTorch → InstinctFlash (Jetson Thor) | tier |
|:--|:--|:--|:--|
| **LingBot-VA** (5B WAM) | 8448 → 2583 ms, **3.27×** | 18027 → 5611 ms, **3.21×** | NUMERIC |
| **LingBot-VA @ 2V/4A** (5B WAM) | 8448 → 360 ms, **23×** | 18027 → 893 ms, **20×** | OPERATING-POINT (certified) |
| **LingBot-VLA-4B** | 671 → 185 ms, **3.62×** | 696 → 97.5 ms, **7.13×** | BITEXACT / engine |
| **LingBot-VLA-V2-6B** (sparse-MoE) | 829 → 183 ms, **4.54×** | 752 → 210 ms, **3.57×** | NUMERIC / engine |
| **Cosmos3-Edge-Policy** (3.86B) | 311 → 186 ms, **1.67×** | 1158 → 660 ms, **1.75×** | NUMERIC |
| **Cosmos3-Nano-Policy** (15.75B) | 482 → 325 ms, **1.49×** | 3956 → 2080 ms, **1.90×** | NUMERIC |
| **pi05** | 207 → 73 ms, **2.84×** | 255 → 57 ms, **4.49×** | BITEXACT / engine |
| **GR00T-N1.7-3B** | 115 → 59 ms, **1.94×** | 122 → 42 ms, **2.88×** | BITEXACT / engine |
| **DreamZero-DROID** (Wan2.2-5B WAM) | 3227 → 1843 ms, **1.75×** | — | SCREEN |

**BITEXACT** means identical actions, **NUMERIC** means a declared-margin result, **SCREEN**
means measured deltas without a closed-loop certificate, and **OPERATING-POINT** means a declared
few-step schedule — changed computation, carried by its own paired closed-loop certificate.


## Shipped configuration

`shipped_configuration()` is the source of truth for the production LingBot serving flags:

```text
--no-fsdp --no-empty-cache --no-debug-dump --conditioning-prefill --ring-kv --conv-layout
```

This serves P001 (substrate elision), P002 (conditioning prefill), P003 (ring KV addressing), and
P007 (convolution layout) at an overall NUMERIC tier. P005 (`--graph-blocks`) and P006
(`--stable-pools`) remain available for measurement but are **NOT RECOMMENDED** in the shipped
configuration.

## Or serve it over the network

```bash
instinctflash serve robbyant/lingbot-va-posttrain-robotwin
```

This is the same msgpack-over-websocket wire protocol the pi0/openpi ecosystem's robot-side
clients already speak, so existing clients connect unchanged:

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # pip install openpi-client

client = WebsocketClientPolicy("my-server", 8000)
action = client.infer(observation)      # dict in, dict out — "prompt" rides in the observation
```

For stateful models a changed prompt starts a new episode server-side, and a client can say it
explicitly with `{"reset": True, "prompt": ...}`. `--serve.viz=true` streams observations, actions
and latency to a [Rerun](https://rerun.io) viewer.

Or from the command line, two verbs:

```bash
instinctflash serve    <model-id>   # preflight (device + declaration + plan), then serve on :8000
                                    #   --serve.dry_run=true   preflight only — no download, no GPU
                                    #   --serve.smoke=true     load, produce one action, exit
instinctflash validate <dir>        # is this directory a publishable checkpoint; with
                                    #   --validate.teacher_outcomes / .student_outcomes / .margin it
                                    #   also certifies and stamps the certificate into the package
```

The preflight fetches one metadata file — no weights, no GPU: it answers *will this machine serve
it* before you commit to a download.

Compress a checkpoint before serving it — see
[InstinctCompress](https://github.com/General-Instinct/InstinctCompress) (customer access;
contact founders@general-instinct.com):

```bash
pip install git+https://github.com/General-Instinct/InstinctCompress   # with granted access
pi05-compress compress <checkpoint> out/ --tasks tasks.txt --dataset <your_demonstrations>
```


To add your own model family, declare an `instinctflash.adapters` entry point and `pip install`
your package — see [`examples/external_plugin/`](examples/external_plugin/).
