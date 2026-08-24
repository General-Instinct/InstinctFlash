<div align="center">

<img src="assets/iFlash.png" alt="InstinctFlash" width="360"/>

**A high-performance serving framework for robotics models.**

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

---

You give InstinctFlash a robotics checkpoint — a world-action model or a VLA policy. It works out
how to run that checkpoint quickly and correctly, and can show its reasoning.

## What's new 🔥

- **pi05 (VLA) support.** Load a pi05 checkpoint and serve actions. The runtime measures the
  model's cost profile and reports exactly which optimizations apply — and which it declined, and why.
- **[InstinctCompress](https://github.com/General-Instinct/InstinctCompress)**, included as a
  submodule: compresses a fine-tuned pi05 checkpoint 8.7 GB → 3.2 GB with the accuracy trained
  back on your own demonstrations and verified, then serves through the same stack unchanged.
- **LingBot-VA at 2.88× bit-exact**, plus 1.405× from convolution-layout selection under a paired
  non-inferiority certificate (555 episodes, identical seeds, one-sided p = 0.00031).
  **Cosmos3-Edge at 2.33×** on the control step.
- **Load straight from the Hub, or bring your own model.** `Runtime.from_pretrained("org/model")`
  resolves everything; an external package adds a new model family through an entry point — no fork.

## Architecture

A checkpoint carries a short declaration of what it is. The runtime reads the declaration, decides
which optimizations are provably valid for those weights, applies them, and shows its work:

```
checkpoint ─▶ adapter          ─▶ planner            ─▶ engine passes        ─▶ actions
              declares what        decides what          apply and measure
              the model is         is valid (no GPU,     each optimization
                                   no weights needed)
```

Every shipped optimization carries a proof tier, derived rather than asserted: **BITEXACT**
(identical actions), **NUMERIC** (a paired non-inferiority certificate at a pre-declared margin),
or it does not ship. `runtime.explain()` prints the chain chosen for your checkpoint, including
the passes it declined and why. Protocols and per-pass results are in [`eval/`](eval/).

Nothing about how a model was trained reaches the runtime, so a new training method needs no
changes here.

## Using it

```bash
git clone --recurse-submodules https://github.com/General-Instinct/InstinctFlash && cd InstinctFlash
pip install -e ".[runtime,diffusion]"
```

Python 3.10+. `pip install -e .` alone needs no GPU and is enough to inspect checkpoints;
LingBot-VA needs a CUDA GPU with ~30 GB free.

Serve actions:

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("general-instinct/lingbot-va")

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

Compress a checkpoint before serving it:

```bash
cd InstinctCompress   # see its README
pi05-compress compress <checkpoint> out/ --tasks tasks.txt --dataset <your_demonstrations>
```

| model | speedup | tier |
|:--|:--|:--|
| **LingBot-VA** | 2.88×, plus 1.405× from convolution layout | BITEXACT / NUMERIC (certified) |
| **Cosmos3-Edge** | 2.33× on the control step | no accuracy claim — tested on random weights |
| **pi05** | serving + compression via InstinctCompress | verified per checkpoint |

To add your own model family, declare an `instinctflash.adapters` entry point and `pip install`
your package — see [`examples/external_plugin/`](examples/external_plugin/).
