<div align="center">

<img src="assets/instinctwm_2.png" alt="InstinctWM" width="360"/>

**One runtime for robot world-action models.**

[Architecture](ARCHITECTURE.md) · [Publishing checkpoints](CHECKPOINTS.md) · [Examples](examples/)

</div>

---

InstinctWM runs world-action models — robot policies that predict what happens next *and* what to do
about it. You give it a checkpoint; it works out how to run that checkpoint quickly and correctly.

A checkpoint carries a short declaration of what it is. The runtime reads the declaration, decides
which optimizations are valid for those weights, applies them, and can show its reasoning. Nothing
about how the model was trained reaches the runtime, so a new training method needs no changes here.

## Install

```bash
git clone https://github.com/general-instinct/InstinctWM && cd InstinctWM
pip install -e ".[runtime,diffusion]"
```

Python 3.10+. Running LingBot-VA needs a CUDA GPU with about 30 GB free; it will not fit on a 24 GB
card. `pip install -e .` on its own has no GPU or torch requirement and is enough to inspect
checkpoints. For a pinned serving environment, use [`requirements-serving.txt`](requirements-serving.txt).

LingBot-VA also needs its upstream serving code, which InstinctWM patches rather than copies:

```bash
git clone https://github.com/robbyant/lingbot-va ~/.cache/instinctwm/lingbot-va
```

Not on PyPI yet — install from the clone.

## Load a model

```python
from instinctwm import Runtime

runtime = Runtime.from_pretrained("general-instinct/lingbot-va")
```

The default uses the verified `shipped` optimization pipeline. You can choose the bit-exact chain,
the unmodified baseline, or your own YAML composition:

```python
runtime = Runtime.from_pretrained("general-instinct/lingbot-va", optimization_config="bitexact")
runtime = Runtime.from_pretrained("general-instinct/lingbot-va", optimization_config="stock")
runtime = Runtime.from_pretrained("general-instinct/lingbot-va", optimization_config="optimizations.yaml")
```

The YAML format, pass modes, dependencies and plugin entry point are documented in
[ARCHITECTURE.md](ARCHITECTURE.md#yaml-optimization-pipelines). `runtime.explain()` includes the
resolved pass order and configuration fingerprint.

To see what a checkpoint is before downloading its weights:

```python
from instinctwm import describe

describe("general-instinct/lingbot-va")
```

## Get actions

```python
with runtime.episode(prompt="put the bottle in the dustbin") as episode:
    while not done:
        action = episode.predict(observation)
```

`observation` is a dict in the model's own format. For LingBot-VA that is one entry per camera,
holding the frames observed since your last call:

```python
observation = {
    "obs": [{"observation.images.cam_high": frame, ...}],
    "prompt": "put the bottle in the dustbin",
}
```

If a safety layer changed the action before it reached the robot, say so, and the model will condition
on what actually happened:

```python
action = episode.predict(observation, executed_action=clipped)
```

That is the whole required API. There is no server to start, and optimization configuration is
optional.

---

## Supported models

| model | notes |
|:--|:--|
| **LingBot-VA** | Full support. 3.38× faster than the reference implementation with bit-identical actions. |
| **Cosmos3-Edge** | Runs; 2.33× on the control step. No accuracy claim — tested on random weights. |

Measurement protocols and per-pass results are in [`eval/`](eval/).

## Adding your own model

An external package can add a model family without changing InstinctWM. Write an adapter with two
methods and declare one entry point:

```toml
[project.entry-points."instinctwm.adapters"]
my_backbone = "my_package.adapter:MyAdapter"
```

After `pip install`, any checkpoint declaring `"backbone": "my_backbone"` loads through the same
`Runtime`. [`examples/external_plugin/`](examples/external_plugin/) is a complete working one, in 125
lines, for a model deliberately unlike LingBot-VA.

Publishing a checkpoint is covered in [CHECKPOINTS.md](CHECKPOINTS.md), including how to ship weights
without shipping your training recipe.

## How it works

`Runtime.from_pretrained` reads the checkpoint's declaration, finds the adapter for its backbone,
derives a set of capabilities, and compiles a plan from them. `runtime.explain()` prints every
decision, including the passes it declined and why. [ARCHITECTURE.md](ARCHITECTURE.md) covers the
design; the optimization passes live in [`instinctwm/passes/`](instinctwm/passes/).

## Development

```bash
./scripts/task.sh test        # core: no GPU required
./scripts/task.sh test-all    # adds the torch-dependent suites
```

The core deliberately has no torch or CUDA dependency, so reasoning about a checkpoint works on a laptop.
Tests that need torch, diffusers or a GPU skip rather than fail.

## Citation

```bibtex
@software{instinctwm2026,
  title  = {InstinctWM: One Runtime for Robot World-Action Models},
  author = {General Instinct},
  year   = {2026},
  url    = {https://github.com/General-Instinct/InstinctWM}
}
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
