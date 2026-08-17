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

That is the whole API. There is nothing to configure, no server to start, and no optimization to
choose.

---

## What's new 🔥

- **Load from a Hub repo id.** `Runtime.from_pretrained("org/model")` resolves the declaration,
  the adapter and the weights; `describe("org/model")` reads the metadata without downloading them.
- **Episodes and closed-loop control.** `runtime.episode()` scopes a rollout and `episode.predict()`
  is callable in a loop, so multi-phase models no longer leak their phases to callers.
- **Bring your own model family.** Declare an `instinctwm.adapters` entry point and `pip install`;
  no fork and no pull request. See [`examples/external_plugin/`](examples/external_plugin/).
- **Report what the robot actually did.** `episode.predict(obs, executed_action=...)` for when a
  safety layer changed the action before it reached the hardware.

## Supported models

| model | speedup | actions |
|:--|:--|:--|
| **LingBot-VA** | 2.88× bit-exact, plus 1.405× from convolution-layout selection | see below |
| **Cosmos3-Edge** | 2.33× on the control step | no accuracy claim — tested on random weights |

The layout pass changes the accumulation order of every 3D convolution, so the chain LingBot-VA ships
with is **NUMERIC**, not bit-exact. It carries a non-inferiority certificate: 555 paired episodes on
identical seeds, 0.9081 against 0.9117, margin −0.05 declared before the run, one-sided p = 0.00031.
Drop `--conv-layout` and the chain is bit-exact at 2.88×.

Tiers are derived from what a pass can prove, never asserted. **BITEXACT** means identical actions,
**NUMERIC** means a declared-margin non-inferiority result, **BEHAVIORAL** means neither.
`runtime.explain()` prints the chain and its tier for the checkpoint you loaded, including the passes
it declined and why. Protocols and per-pass results are in [`eval/`](eval/).

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

The core is deliberately dependency-free so that reasoning about a checkpoint works on a laptop.
Tests that need torch, diffusers or a GPU skip rather than fail.

## Acknowledgements

InstinctWM learns from other projects rather than reinventing their work. Three categories, kept
distinct on purpose, because they carry different obligations:

**Inspiration — ideas studied, nothing copied.**

- [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0) — the standard for what loading a
  robot policy should feel like. `Runtime.from_pretrained` and the model-card-generated-from-artifact
  habit come from studying it. Where we differ: LeRobot loads a model *class*, InstinctWM loads a
  checkpoint *declaration* and derives what is legal from it.
- [FlashRT](https://github.com/gugudeshubao/FlashRT) (Apache-2.0) — hand-tuned realtime inference for
  small-batch embodied workloads. The reference for hardware as a first-class dimension: strict
  architecture detection that refuses unknown devices rather than falling back, and buffers
  pre-allocated to a maximum so a captured graph stays valid. Its per-SM dispatch informed the
  hardware vocabulary in [ARCHITECTURE.md](ARCHITECTURE.md); we keep a capability *predicate* instead
  of a `(model, framework, arch)` table.
- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) — request-scoped sessions with cache and
  scheduling state kept entirely out of the user's API. `Episode` exists because of it.

**Adapted implementations — none currently.** When we adapt code, the file will carry the origin, the
upstream license header, and a note on what changed.

**Directly reused code — none currently.**

License compatibility is checked before reuse, not after. Apache-2.0 permits incorporation into an
AGPL-3.0 work with notices preserved; the reverse does not hold, so code cannot flow from InstinctWM
back into those projects under their licenses.

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
