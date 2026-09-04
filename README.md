<div align="center">

<img src="assets/iFlash.png" alt="InstinctFlash" width="360"/>

**A high-performance serving framework for robotics models.**

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

---

## What's new 🔥

- **Eight model families, two device classes.** Each measured against its authors' own serving
  code on the same device — up to 5.26x on H100 (LingBot-VLA-V2) and 7.13x on Jetson Thor
  (LingBot-VLA-4B). The two devices need opposite optimizations — the planner measures before
  it applies. Full table in the results section below.
- **Declared few-step schedules.** LingBot-VA's 2V/4A schedule runs at twenty-three times its
  upstream serving cost, certified non-inferior over 1153 pre-registered RoboTwin episode
  pairs. A few-step distillation framework is under development as an InstinctFlash component.
- **Bit-exact CUDA-graph capture, on by default.** Static max-extent KV buffers make denoise
  loops capturable with bit-exact replay on unseen inputs. pi05, LingBot-VLA-4B, LingBot-VLA-V2
  and GR00T-N1.7 — fresh fine-tunes included — serve their family's verified capture arm by
  default, gated by a startup self-check vs upstream eager (exact equality; V2: its recorded
  stock-vs-stock envelope) that falls back loudly. Kill-switch: `IFL_<FAMILY>_NO_CAPTURE=1`.
- **[InstinctCompress](https://github.com/General-Instinct/InstinctCompress)** (customer
  access): a fine-tuned pi05 checkpoint 8.7 GB → 3.2 GB, accuracy trained back on your own
  demonstrations and verified, served through the same stack unchanged.

## Results

| model | PyTorch → InstinctFlash (H100) | PyTorch → InstinctFlash (Jetson Thor) | tier |
|:--|:--|:--|:--|
| **LingBot-VA** (5B WAM) | 8448&nbsp;→&nbsp;2583&nbsp;ms,&nbsp;**3.27×**&nbsp;‡ | 18027&nbsp;→&nbsp;5611&nbsp;ms,&nbsp;**3.21×**&nbsp;‡ | NUMERIC |
| **LingBot-VA @ 2V/4A** (5B WAM) | 8448&nbsp;→&nbsp;360&nbsp;ms,&nbsp;**23×**&nbsp;‡ | 18027&nbsp;→&nbsp;893&nbsp;ms,&nbsp;**20×**&nbsp;‡ | OPERATING-POINT (certified) |
| **LingBot-VLA-4B** | 671&nbsp;→&nbsp;185&nbsp;ms,&nbsp;**3.62×** | 696&nbsp;→&nbsp;97.5&nbsp;ms,&nbsp;**7.13×** | BITEXACT / engine |
| **LingBot-VLA-V2-6B** (sparse-MoE) | 671&nbsp;→&nbsp;128&nbsp;ms,&nbsp;**5.26×** | 752&nbsp;→&nbsp;210&nbsp;ms,&nbsp;**3.57×** | NUMERIC / engine |
| **Cosmos3-Edge-Policy** (3.86B) | 311&nbsp;→&nbsp;186&nbsp;ms,&nbsp;**1.67×** | 1158&nbsp;→&nbsp;660&nbsp;ms,&nbsp;**1.75×** | NUMERIC |
| **Cosmos3-Nano-Policy** (15.75B) | 482&nbsp;→&nbsp;325&nbsp;ms,&nbsp;**1.49×** | 3956&nbsp;→&nbsp;2080&nbsp;ms,&nbsp;**1.90×** | NUMERIC |
| **pi05** | 207&nbsp;→&nbsp;73&nbsp;ms,&nbsp;**2.84×** | 255&nbsp;→&nbsp;57&nbsp;ms,&nbsp;**4.49×** | BITEXACT / engine |
| **GR00T-N1.7-3B** | 115&nbsp;→&nbsp;59&nbsp;ms,&nbsp;**1.94×** | 122&nbsp;→&nbsp;42&nbsp;ms,&nbsp;**2.88×** | BITEXACT / engine |
| **DreamZero-DROID** (Wan2.2-5B WAM) | 3227&nbsp;→&nbsp;1843&nbsp;ms,&nbsp;**1.75×** | — | SCREEN |

**BITEXACT** means identical actions, **NUMERIC** means a declared-margin result, **SCREEN**
means measured deltas without a closed-loop certificate, and **OPERATING-POINT** means a declared
few-step schedule — changed computation, carried by its own paired closed-loop certificate.
Every row is reproducible: `examples/<family>/reproduce_h100.sh` reruns its pair with the exact
protocol.

H100 cells are the 2026-08-24 sweep; LingBot-VLA-V2 is the 2026-08-28 re-sweep on a different
H100-80GB box (4xH100 host), both cells remeasured there, after its default arm gained the
vision/prefill graphs and GPU preprocessing. The re-swept VLA-4B, GR00T and pi05 pairs
reproduced their rows' class on that box and keep their published cells.

‡ LingBot-VA cycle latency has two regimes within an episode: early (the ring-KV pool still filling,
cycles 1–36; a typical RoboTwin episode ends before the pool saturates) and saturated (pool full,
cycles ≥ 37). The LingBot-VA cells above are the early regime (cycles 2–8 on Thor, 2–12 on H100).
Measured to saturation (48-cycle episodes × 3 runs, run 0 discarded, real message order), the Thor
cells read 31533 → 8250 ms, 3.82× and 31533 → 1382 ms, 22.8×; on the 4xH100 host the same-computation
ratio is 4.10× and 2V/4A is 27.7×. Every ratio holds or rises at saturation — the vendor server's cycle
grows more with pool size than ours — while the Thor absolute ms above understate a saturated episode
by 1.5–1.8×; on H100 our default-schedule chain is regime-flat and the vendor server is +20 %.

## Install

```bash
git clone https://github.com/General-Instinct/InstinctFlash && cd InstinctFlash
pip install -e ".[runtime,diffusion,serve]"
```

Python 3.10+. `pip install -e .` alone needs no GPU and is enough to inspect checkpoints and
plan; serving needs a CUDA GPU (memory varies by family — `--serve.dry_run` tells you before
you download anything).

## Load a model

**Your fine-tuned checkpoint** — the expected case. Point `serve` at the training output; it
detects the family, writes the small `instinctflash.json` declaration from what the checkpoint
itself proves, and starts serving. One command:

```bash
instinctflash serve /path/to/your/checkpoint
```

Anything the checkpoint cannot prove is asked for explicitly, never guessed. Once the
declaration exists (serve writes it on first run), the same directory also loads in Python:

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("/path/to/your/checkpoint")
```

**A stock release** — by Hub id, zero setup, any of the eight families:

```python
runtime = Runtime.from_pretrained("robbyant/lingbot-va-posttrain-robotwin")
```

| family | model id |
|:--|:--|
| LingBot-VA (5B WAM) | `robbyant/lingbot-va-posttrain-robotwin` |
| LingBot-VLA-4B | `robbyant/lingbot-vla-4b-posttrain-robotwin` |
| LingBot-VLA-V2-6B | `robbyant/lingbot-vla-v2-6b-robotwin` |
| pi0.5 | `lerobot/pi05_base` · `lerobot/pi05_libero_finetuned_v044` |
| GR00T-N1.7-3B | `nvidia/GR00T-N1.7-3B` |
| Cosmos3 policies | `nvidia/Cosmos3-Edge-Policy-DROID` · `nvidia/Cosmos3-Nano-Policy-DROID` |
| DreamZero | `GEAR-Dreams/DreamZero-DROID` |

Either way, a fine-tune inherits every optimization and proof tier of its family automatically.

## Get actions

**In process** — this is the whole Python API:

```python
with runtime.episode(prompt="put the bottle in the dustbin") as episode:
    while not done:
        action = episode.predict(observation)
```

`observation` is a dict in the model's own format; `action` comes back ready for the robot. No
server to start, no optimization to choose. If a safety layer changed the action before it
reached the robot, pass `executed_action=...` and the model conditions on what actually
happened.

**Over the network** — the `serve` command above hosts the same runtime behind the
msgpack-over-websocket wire protocol the pi0/openpi ecosystem already speaks, so existing
robot-side clients connect unchanged (`pip install openpi-client`):

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy("my-server", 8000)
action = client.infer(observation)
```

The prompt rides in the observation; a changed prompt starts a new episode, and a client can
say it explicitly with `{"reset": True, ...}`. Four flags cover the rest:

- `--serve.dry_run` — preflight only: device, declaration, plan. No weights, no GPU.
- `--serve.smoke` — load, produce one action, exit.
- `--serve.seed` — seed the model's noise, for value-for-value A/B between two servers.
- `--serve.viz` — stream observations, actions and latency to a [Rerun](https://rerun.io) viewer.

The second verb, `instinctflash validate <dir>`, checks a checkpoint is publishable; given
`--validate.teacher_outcomes/.student_outcomes/.margin` it also certifies non-inferiority and
stamps the certificate into the package.

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

## Compress a checkpoint before serving

See
[InstinctCompress](https://github.com/General-Instinct/InstinctCompress) (customer access;
contact founders@general-instinct.com):

```bash
pip install git+https://github.com/General-Instinct/InstinctCompress   # with granted access
pi05-compress compress <checkpoint> out/ --tasks tasks.txt --dataset <your_demonstrations>
```

To add your own model family, declare an `instinctflash.adapters` entry point and `pip install`
your package — see [`examples/external_plugin/`](examples/external_plugin/).

## Roadmap

- [ ] **Few-step distillation** — train back the small quality cost of declared few-step
      schedules, starting with LingBot-VA's 2V/4A point (−1.65 pp against its teacher today);
      reach schedules untrained reduction cannot, each with its own paired certificate.
- [ ] **LingBot-VA on the edge engine** — extend the fp8 engine tier that serves pi05,
      LingBot-VLA-4B and LingBot-VLA-V2 on Jetson Thor to the VA world-action model.
- [ ] **Attention upgrades** — a faster NUMERIC-tier attention arm beside the BITEXACT default
      for pi05-class models; hybrid and linear attention for long-context world models.
- [ ] **Cosmos3 engine serving, DreamZero on the edge** — each after its groundwork
      (reference-implementation alignment; memory footprint on unified-memory devices).
- [ ] **Device-conditional serving defaults** — encode the measured capture-vs-pipeline law as
      automatic per-device defaults for every model family, so no operator flag is ever the
      difference between the right arm and the wrong one.
