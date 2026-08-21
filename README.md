<div align="center">

<img src="assets/iFlash.png" alt="InstinctFlash" width="360"/>

**One runtime for robot world-action models.**

[Architecture](#architecture) · [Publishing checkpoints](#publishing-checkpoints) · [Examples](examples/)

</div>

---

InstinctFlash runs world-action models — robot policies that predict what happens next *and* what to do
about it. You give it a checkpoint; it works out how to run that checkpoint quickly and correctly.

A checkpoint carries a short declaration of what it is. The runtime reads the declaration, decides
which optimizations are valid for those weights, applies them, and can show its reasoning. Nothing
about how the model was trained reaches the runtime, so a new training method needs no changes here.

## Install

```bash
git clone https://github.com/general-instinct/InstinctFlash && cd InstinctFlash
pip install -e ".[runtime,diffusion]"
```

Python 3.10+. Running LingBot-VA needs a CUDA GPU with about 30 GB free; it will not fit on a 24 GB
card. `pip install -e .` on its own has no GPU or torch requirement and is enough to inspect
checkpoints. For a pinned serving environment, use [`requirements-serving.txt`](requirements-serving.txt).

LingBot-VA also needs its upstream serving code, which InstinctFlash patches rather than copies:

```bash
git clone https://github.com/robbyant/lingbot-va ~/.cache/instinctflash/lingbot-va
```

Not on PyPI yet — install from the clone.

## Load a model

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("general-instinct/lingbot-va")
```

To see what a checkpoint is before downloading its weights:

```python
from instinctflash import describe

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

## Or without writing Python

```bash
instinctflash devices                 # what machine am I on, and what can it do
instinctflash describe  <model-id>    # what a checkpoint declares — no weights downloaded
instinctflash validate  <dir>         # is this a publishable checkpoint
instinctflash plan      <model-id>    # what the runtime would do to it, and why
instinctflash run       <model-id>    # load it and produce real actions
```

`describe` and `plan` need no weights and no GPU, which is the point: they answer *what is this, and
will this machine serve it* before you commit to a download. `run` uses zero-filled observations and
says so — it proves a checkpoint loads here and returns finite actions of the right shape, which is a
smoke test, not an evaluation.

---

## What's new 🔥

- **Load from a Hub repo id.** `Runtime.from_pretrained("org/model")` resolves the declaration,
  the adapter and the weights; `describe("org/model")` reads the metadata without downloading them.
- **Episodes and closed-loop control.** `runtime.episode()` scopes a rollout and `episode.predict()`
  is callable in a loop, so multi-phase models no longer leak their phases to callers.
- **Bring your own model family.** Declare an `instinctflash.adapters` entry point and `pip install`;
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

An external package can add a model family without changing InstinctFlash. Write an adapter with two
methods and declare one entry point:

```toml
[project.entry-points."instinctflash.adapters"]
my_backbone = "my_package.adapter:MyAdapter"
```

After `pip install`, any checkpoint declaring `"backbone": "my_backbone"` loads through the same
`Runtime`. [`examples/external_plugin/`](examples/external_plugin/) is a complete working one, in 125
lines, for a model deliberately unlike LingBot-VA.

Publishing a checkpoint is covered in the [Publishing checkpoints](#publishing-checkpoints) section, including how to ship weights
without shipping your training recipe.

## How it works

`Runtime.from_pretrained` reads the checkpoint's declaration, finds the adapter for its backbone,
derives a set of capabilities, and compiles a plan from them. `runtime.explain()` prints every
decision, including the passes it declined and why. the [Architecture](#architecture) section covers the
design; the optimization passes live in [`instinctflash/passes/`](instinctflash/passes/).

## Development

```bash
./scripts/task.sh test        # core: no GPU required
./scripts/task.sh test-all    # adds the torch-dependent suites
```

The core is deliberately dependency-free so that reasoning about a checkpoint works on a laptop.
Tests that need torch, diffusers or a GPU skip rather than fail.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

---

# Architecture

**One Runtime. Many Checkpoints. Shared Infrastructure.**

This document describes how InstinctFlash is organized and why. It describes the code on `main`; for how
it got there, read the git history.

---

## The organizing principle

A world-action model gets faster in two fundamentally different ways, and conflating them is the
central design mistake this repository is arranged to prevent.

| | **Training** | **Runtime** |
|:--|:--|:--|
| Layer | 1 | 2–6 |
| Changes | the weights | how the weights are executed |
| Produces | a **checkpoint** | a **plan** |
| Cost | GPU-weeks, a dataset, a teacher | milliseconds, at load time |
| Verified by | held-out accuracy, task success | bit-exactness or a declared margin |
| Lives in | `train/`, and in separate repos per recipe | `runtime/`, `planners/`, `passes/`, … |

Training recipes — PDD, DMD2, LCM, DreamZero, rCM, sCM — produce **different checkpoints**. They do
not produce different runtimes. There is exactly one InstinctFlash runtime, and it serves every
compatible checkpoint by reading what that checkpoint declares about itself.

This is **enforced, not aspirational**. `tests/test_runtime_boundary.py` parses every module under
`runtime/`, `planners/`, `executors/`, `descriptors/` and `backends/` and fails if any of them can
reach `instinct_pdd` or `instinctflash.train` — transitively, and including function-local imports,
because the violation it was written for was function-local. It also fails if any of them names a
provenance key such as `coverage_gate_pass` as a live string. The gate is self-checking: it plants the
original violation in a temporary module each run and confirms it is caught, because a gate that
cannot fail on the bug it gates is worse than no gate.

### There is no fast runtime and no quality runtime

A checkpoint declares its step schedule in `execution.nfe`. Two checkpoints that differ only there
are the same weights served two ways, and it is tempting to make each one a runtime with its own
pass list.

That would bury the reason for a decision inside a branch. Whether a pass pays depends on the
declared schedule, so the test belongs where the schedule can be read:

```
pass.admit(spec, deployment)   ->  is this legal here, and does it pay here?
```

Graph capture is the case that makes this concrete. It trades a fixed per-cycle capture cost for a
saving on every forward, so its profitability depends entirely on how many forwards a cycle runs. At
the schedule LingBot-VA ships with, it loses: capture measures **1.43× slower** than not capturing,
so `admit()` declines it and the plan says why. A runtime that hardcoded a pass list per mode would
have shipped that regression instead of reporting it.

Adding a third schedule needs no new code path.

---

## The two seams

Everything in the repository sits on one side or the other of two boundaries.

### Seam 1 — WHERE vs WHAT (adapters vs passes)

Every optimization we wrote before this seam existed began the same way:

```python
import modules.model as M
M.WanAttention.forward = my_replacement
```

That line fuses two responsibilities with different owners. *Where* the attention module lives is a
fact about one model. *What* to do there is a fact about an optimization. Because they were fused,
the passes were adapters wearing pass clothing: on Cosmos3-Edge all of them were no-ops, not because
the optimizations were inapplicable but because the symbols had different names.

So an **adapter publishes sites**, and a **pass consumes sites and returns rewrites**:

```
adapter.sites(SiteKind.INVARIANT_CONDITIONING)  ->  [Site(...), ...]
pass.plan_rewrites(sites, device)               ->  [Rewrite(...), ...]
executor.apply(adapter, rewrites)               ->  installed, recorded, gated
```

A pass never imports a model module. It reads `site.attrs["scope"] == Scope.MODEL` and decides to
hoist; it never learns which tensor that was.

### Seam 2 — DECIDE vs ACT (planners vs executors)

Planning is pure analysis: no torch, no checkpoint, no GPU.

```python
model = load("wan_va")                            # states facts; touches nothing
plan  = Optimizer(tier_ceiling=Tier.BITEXACT).compile(model.spec())
print(plan.explain())                             # what fired, and why
```

If you need the weights loaded before the framework can tell you what it would do, it is a runtime
wearing a framework's clothes. Being able to answer "what would you do to this model, and why" on a
laptop is what makes the optimizer something you can argue with.

This is the internal seam, not the way anyone serves a model. `Runtime.from_pretrained` runs the same
compile step and then acts on it, which is why `runtime.explain()` can print the plan.

---

## Directory layout

```
instinctflash/
  descriptors/     what a checkpoint declares            capabilities, never recipes
  adapters/        WHERE things are, per backbone        publish sites
  passes/          WHAT to do there                      consume sites, return rewrites
    lingbot/         backbone-coupled implementations    direct-install, LingBot-specific
    contract.py      the five questions every pass answers
  planners/        which passes are legal + profitable   declarations only, no GPU
  executors/       apply a plan to a live server         the only layer that touches the model
  backends/        kernels                               triton / torch, chosen by measurement
  runtime/         load, install, serve                  the one runtime
    state/           state manifests and scratch arenas
  verify/          gates, certificates, release registry
  train/           Layer 1 — recipes that MAKE checkpoints
```

Two notes on this tree, because both are load-bearing:

**`passes/` has two generations, and that is deliberate.** `passes/*.py` are generic and site-based;
they work on any adapter that publishes the sites they ask for. `passes/lingbot/*.py` are the
LingBot-coupled direct-install implementations that produced every measured number we have. The
generic versions are the architecture; the coupled versions are what currently ships. Naming them
honestly is better than pretending the port is finished. Where both exist (`graph_capture`,
`stable_pools`, `hoist_invariant`), the generic one is the target and the coupled one is the
reference.

**`train/` is not part of serving.** It appears in the tree because Layer 1 is part of the product,
but nothing under `runtime/`, `planners/`, or `passes/` imports it. The one recipe we have
implemented in full, PDD, lives in its own repository ([`instinct-pdd`](instinct-pdd), Apache-2.0)
and is consumed here as a submodule — it is backbone-agnostic infrastructure, not an InstinctFlash
research project.

---

## The optimization stack

Six layers, ordered by *what they change*.

| Layer | | Changes |
|:--|:--|:--|
| 1 | **MODEL** | what is computed — step reduction, distillation, latent compression |
| 2 | **GRAPH** | when work is issued — prefill extraction, graph capture, memory planning |
| 3 | **CACHE** | what is recomputed — KV reuse, cross-attention cache, episode cache |
| 4 | **ATTENTION** | how tokens mix — FlashAttention, hybrid and linear attention (design) |
| 5 | **KERNEL** | how a kernel is written — backend/layout dispatch first, then fusion and Triton (design) |
| 6 | **HARDWARE** | what it executes on — TensorRT, FP8/INT8, Jetson, Thor |

Layer 1 is the training side of the seam. Layers 2–6 are the runtime.

The layers are **not** a priority order, and treating them as one is how time gets wasted. Attention
is Layer 4 and looks like the obvious first move; it is 7% of GPU-busy time here.

Priority comes from decomposing the cycle at the schedule you actually serve. At LingBot-VA's
shipped 2 video / 4 action steps — 10 transformer forwards per cycle, once the two cache-only
forwards and the two KV-refresh forwards are counted — two components are 99% of it:

```
transformer forwards   80.8%
keyframe VAE encode    17.7%
everything else        < 1%     schedulers, prepare, pre/postprocess
```

Measure the phases directly. Inferring a fixed overhead by regressing cycle time against forward
count across configurations does not work here, because per-forward cost is not constant across
them.

---

## What a pass must answer

Five questions, enforced by `passes/contract.py`:

1. **Detection** — can the optimizer find the opportunity by itself?
2. **Applicability** — is it legal for this model, on this hardware, right now?
3. **Correctness** — what does it do to the outputs, and how is that proven?
4. **Performance** — does it actually make *this* model faster, measured?
5. **Hardware** — where does it run?

Correctness and performance are separate gates because they fail independently. A pass can be
perfectly accuracy-neutral and still be a regression: on pi-0's real shapes, swapping eager
attention for SDPA while keeping the mask measures 133.5 → 144–184 µs. A correctness-only gate
certifies the numerics of the slower variant and ships it.

Accuracy tiers do not compose upward. A plan containing one `BEHAVIORAL` pass is `BEHAVIORAL` no
matter how many `BITEXACT` passes sit beside it, and `Plan.tier()` enforces that rather than leaving
it to discipline — quoting a bit-exactness claim for a plan that contains a lossy pass is exactly
the kind of error that survives review and then invalidates a benchmark.

---

## Where each responsibility lives

The two seams say what is separated. This says where a given decision belongs, and what must never
appear in a layer. Where a placement was
contested, the reason it lives where it does is given.

| layer | belongs here | must not be here |
|:--|:--|:--|
| checkpoint / adapter | capacity, lifetime, addressing, window, commit points, phase order, guidance, purity, attention *semantics*, shape envelope; quantization format and scale provenance | the calibration recipe, integer layer indices, kernel names, mode flags |
| planner / passes | legality gates, tier derivation, refusal records with reasons, composition at the weakest tier, hardware-requirement enforcement | anything that changes numerics without changing the declared tier |
| backend selection | measured choice over *declared semantics*, incumbent kept on ties, every rejection recorded | a model name or an architecture string as the dispatch key |
| hardware kernels | arithmetic a vendor library declines, with shape legality proved and the tier derived from structure | a compile-time bound behind a runtime launcher with no check |
| graph / capture | a whole declared phase program under one launch footprint, with caller-owned buffers and stable pointers | a capture key derived from a shape that changes per cycle |
| quantization / calibration | format, granularity, scheme and scale provenance in the execution namespace | a host-local mutable file as the source of deployed numerics |

Two of these are load-bearing and worth stating as prohibitions.

**A dispatch key must not name the model.** The alternative is a table indexed by
`(model, framework, architecture)`, which materialises the cross-product by hand: a new architecture
then costs one entry per model, and a new model one entry per architecture. Ours is a predicate —
a backend declares `HardwareReq(min_capability, requires, excludes)` and the planner asks whether the
probed device satisfies it — so a new target costs one probe extension and a new model costs one
adapter. That is the difference between adding a target and forking per model.

**A registry may name what it ships; nothing else may name a model.** `runtime/loader.py` knows which
adapters are bundled and `passes/registry.py` knows which passes are, because that is a registry's job.
Planners, descriptors, executors and verifiers must not: the planner used to import one model's pass
list as its default, so every family in the ecosystem inherited passes written for a world model and no
new family could add one without editing the planner. Both now discover from entry points, so the
in-tree path and the third-party path are the same path. `tests/test_core_stability.py` counts
model-name references in the executable code of those layers and fails above zero.

**A capability the probe cannot name is a capability no backend may require.** Both halves of that
vocabulary have to be closed together. `DeviceProfile.probe()` reports what a device actually has,
`KNOWN_FEATURES` bounds the names, and `tests/test_hardware_probe.py` fails if a backend requires
something outside it. Without that closure a requirement is unsatisfiable everywhere, which looks
exactly like a typo and is equally silent: the shipped convolution-layout selection declared
`requires={"cudnn"}` against a probe that never emitted `cudnn`, and the contradiction was invisible
only because nothing called the probe.

## Measurement protocol

Every latency claim in this repository is measured under **ABBA ordering** — base, treat, treat,
base — because within a session latency drifts monotonically and a sequential A/B charges the whole
drift to the treatment. Numbers taken before this became the default are labelled
*pre-order-control* and are not comparable to numbers taken after.

A performance gate has two verdicts and they are reported separately:

- **Correctness** is unconditional. Numerics do not care who else is on the box.
- **Speed** is conditional. On a contended device the honest answer is **NOT EVALUATED**, never
  PASS and never FAIL. A 48%-busy neighbour was once enough to turn a genuine 1.20× into a reported
  kernel regression, and the regression was investigated before the contention was noticed.

The same rule applies to any gate that runs zero checks: it reports NOT EVALUATED. A gate that
cannot fail on the bug it is gating is worse than no gate, because it produces a signed-off feeling.

---

## Further reading

- [the Publishing checkpoints chapter](#publishing-checkpoints) — what a checkpoint declares, and why the training method is
  deliberately absent from it
- [`instinctflash/passes/`](instinctflash/passes/) — every pass, with its legality argument in its docstring
- [`instinctflash/verify/released.py`](instinctflash/verify/released.py) — the release ledger: what shipped,
  at what tier, on what evidence
- [eval/lingbot_va_robotwin/README.md](eval/lingbot_va_robotwin/README.md) — measured numbers and
  protocols

## Shipped configuration

`instinctflash.verify.released.shipped_configuration()` is the single source of truth. The launch
scripts, `serve_variant.py` and this table all derive from it, and `tests/test_shipped_config.py`
fails if they drift apart. Add a flag there, not in four places.

```
--no-fsdp --no-empty-cache --no-debug-dump --conditioning-prefill --ring-kv --conv-layout
```

The served chain is **NUMERIC** — the weakest link, not the best member. `conv_layout_ndhwc` is
NUMERIC, so the chain is not bit-exact end to end; its non-inferiority certificate over 555 episodes
is what backs it.

| pass | tier | disposition | flags |
|:--|:--|:--|:--|
| `substrate_elision` | BITEXACT | shipped | `--no-fsdp --no-empty-cache --no-debug-dump` |
| `conditioning_prefill` | BITEXACT | shipped | `--conditioning-prefill` |
| `ring_kv_addressing` | BITEXACT | shipped | `--ring-kv` |
| `conv_layout_ndhwc` | NUMERIC | shipped | `--conv-layout` |
| `hoist_invariant_casts` | BITEXACT | available | `--hoist-casts` |
| `graph_block_stack` (P005) | BITEXACT | not recommended | `--graph-blocks` |
| `stable_state_pools` (P006) | BITEXACT | not recommended | `--stable-pools` |

Released is not the same as recommended. `RELEASED` is a frozen ledger of what shipped, at what tier,
on what evidence. `DISPOSITIONS` states what should run today. P005 and P006 forced the distinction:
both were released and verified, and at the schedule LingBot-VA ships with CUDA graph capture measures 1.43×
*slower* than not capturing, so they stay in the ledger and are marked not recommended.


---

# Publishing checkpoints

**One Runtime. Many Checkpoints. Shared Infrastructure.**

A checkpoint is a set of weights plus a declaration of what those weights need in order to run
correctly. The runtime reads the declaration and derives everything else. This document specifies
what a checkpoint declares, and — more importantly — what it must *not*.

---

## The rule

> **The runtime must never branch on the training method.**

Nothing under `runtime/`, `planners/`, `passes/`, or `executors/` may ask whether a checkpoint came
from PDD, DMD2, LCM, DreamZero, rCM, sCM, or a recipe that does not exist yet. Not in a conditional,
not in a lookup table, not in a pass's `admit()`.

This is not stylistic. It is what makes the platform claim true.

### Why the method name is poison

Consider what a runtime would actually do with `training_method: "pdd"`. It would look up a pass set,
or a step schedule, or a sampler. Every one of those is a **capability the checkpoint could have
declared directly** — and the moment it is inferred from a name instead, three things break:

1. **A new recipe requires a runtime change.** Adding DMD2 means editing a branch in the serving
   path, which means the runtime has an opinion about research it has never seen. The whole point of
   one runtime is that a recipe is a *training-time* artifact.

2. **The branch encodes a correlation, not a cause.** PDD checkpoints tend to tolerate few steps.
   *Tolerating few steps* is the fact that matters, and it is measurable and declarable. A
   distillation run that failed to converge is still "PDD" and would inherit the aggressive schedule.
   We have this exact case in our own results: a heads-only PDD student measured 0.920 against an
   untrained control that also measured 0.920 — identical, 3/3 discordant, p = 1.0. Branching on
   `"pdd"` would have served it aggressively on the strength of its name.

3. **It hides the reason.** When a pass fires because the checkpoint declares
   `nfe: {video: 2, action: 4}`, the log says so and you can argue with it. When it fires because
   the method was PDD, the reason is a string.

So the declaration carries **capabilities and structure**. The method name may appear in the model
card, the citation, and the training provenance block — anywhere a human reads it. Never where the
runtime reads it.

---

## What a checkpoint declares

Two kinds of facts, and the split is load-bearing.

### Checkpoint-scoped — `AdapterSpec`

Immutable, identical on every box the checkpoint runs on. Implemented in
[`instinctflash/adapters/base.py`](instinctflash/adapters/base.py).

| Field | Meaning | What reads it |
|:--|:--|:--|
| `model_id` | identity | the loader |
| `param_bytes` | size | memory planning |
| `streams` | per-stream KV specs: lifetime, ownership, addressing | `ring_kv`, `stable_pools` |
| `phases` | the control step as an ordered list of phases, each with an NFE count | the whole planner |
| `guidance` | per-stream guidance rule (`POSITIVE_ONLY`, CFG scale) | `cfg_elision` |
| `purity` | which computations are loop-invariant, and at what scope | `conditioning_prefill`, `hoist_invariant_casts` |
| `obs_decode_modules` | modules needed only when the caller wants predicted pixels | `obs_decode_elision` |
| `notes` | free text for humans | nothing |

`phases` is doing most of the work. It is what makes the step schedule a declared fact rather than
a flag: `total_forwards()` and `forwards_breakdown()` come straight out of it, and those are the
numbers profitability is computed against.

### Site-scoped — `DeploymentSpec`

The same checkpoint is one GPU here and eight there, and one caller wants predicted pixels while the
next wants only actions. Implemented in
[`instinctflash/descriptors/deployment.py`](instinctflash/descriptors/deployment.py).

| Field | Meaning |
|:--|:--|
| `world_size` | how many ranks — `FSDPElision` guards on it |
| `want_pixels` | does the caller need decoded video — `ObsDecodeElision` guards on it |

A pass that read a deployment fact off `AdapterSpec` would be asking the model author to declare
something they cannot know. Fields are added here only when a pass reads one.

---

## The serialized form

> **Status: the declaration and the package layer are implemented; full `AdapterSpec` construction is
> not.** [`descriptors/checkpoint.py`](instinctflash/descriptors/checkpoint.py) reads this schema:
> `load_declaration()` returns the `execution` block **only** and has nowhere to put a training
> method, `provenance_of()` is a separate deliberate call, and a provenance key found inside
> `execution` is a load error. [`descriptors/package.py`](instinctflash/descriptors/package.py) adds the
> published form — `from_pretrained()` (local path, or Hub repo id when `huggingface_hub` is
> installed), `validate_package()`, `publishability()` and `migrate_legacy()`, with a CLI at
> `python -m instinctflash.descriptors.package <dir>`. `Checkpoint.capabilities()` is what the planner
> receives; `Optimizer.compile(..., capabilities=...)` admits a pass only if the tokens it requires
> are declared.
>
> What is **still not** built: construction of a full `AdapterSpec` from the file — adapters are
> registered in Python, so `backbone` selects an adapter rather than describing one. A checkpoint for
> a backbone with no registered adapter is not servable, however well it declares itself.

### Scope: many checkpoints today, arbitrary backbones later

**Supported now.** Many checkpoints per backbone, where the backbone has a registered adapter. Step
schedules, recipes, fine-tunes and output projections all declare capabilities and plan from them with
no runtime change. The adapter does not have to live here: declare an entry point in your own package
and `pip install` it, and any checkpoint naming that backbone resolves.

```toml
[project.entry-points."instinctflash.adapters"]
my_backbone = "my_package.adapter:MyAdapter"
```

[`examples/external_plugin/`](examples/external_plugin/) is a complete one for a model unlike
LingBot-VA; [`examples/tiny_wam/`](examples/tiny_wam/) shows the same on real weights.

**Not yet.** An arbitrary new backbone with *no* adapter. `execution.backbone` must resolve to one,
because the adapter supplies the shape of a control step and the declaration cannot express that.
Writing an adapter needs no change to InstinctFlash, but it is code rather than JSON. Deriving a full
`AdapterSpec` from the declaration would close the gap and is deliberately not built: the schema
would have to describe an execution graph, which is a much larger contract to freeze.

### The published package

```
my-checkpoint/
  instinctflash.json          REQUIRED   the declaration
  config.json              REQUIRED   the backbone's own config, as its modelling library expects it
  model.safetensors        REQUIRED   or a sharded set + model.safetensors.index.json
  README.md                optional   model card
  tokenizer/ scheduler/ …  optional   whatever the backbone needs
```

**Minimal serving metadata** is three fields — `model_id`, `backbone`, `servable`. Everything else has
a defensible default, and a checkpoint declaring only those three is servable if an adapter exists for
its backbone.

**Publishing without training internals** is a property you can check rather than a promise:
`publishability()` strips the `provenance` block, re-loads the declaration, and confirms the runtime
would still serve it. If that fails, a fact the runtime needs is in the wrong namespace. A complete
worked example lives in [`examples/checkpoint/wm-blockheads-2v4a/`](examples/checkpoint/wm-blockheads-2v4a/),
and `tests/test_checkpoint_platform.py` validates that directory rather than a fixture invented inside
the test.

For a checkpoint published on the Hub, the declaration ships as `instinctflash.json` beside the weights.
The runtime resolves it in this order, first hit wins:

1. `instinctflash.json` in the checkpoint directory or repo
2. a registered adapter for `model_id` (`instinctflash.register`)
3. refuse — an unrecognized checkpoint is not served on guessed facts

```jsonc
{
  "instinctflash_schema": 1,

  // ==========================================================================
  // EXECUTION -- everything the runtime may read, and nothing else.
  // Adding a key here requires naming the pass or planner that reads it.
  // ==========================================================================
  "execution": {
    "model_id": "lingbot-va-robotwin-blockheads-2v4a",
    "backbone": "wan-va",                  // which adapter publishes the sites
    "param_bytes": 24696061952,

    // The control step. A reduced schedule is a
    // different phases block, not a mode flag and not a second runtime.
    "phases": [
      {"name": "kv_refresh", "nfe": 1},
      {"name": "video",  "nfe": 2, "stream": "video"},
      {"name": "action", "nfe": 4, "stream": "action"}
    ],

    "streams": [
      {"name": "video",  "kv_lifetime": "episode", "addressing": "ring", "commit": "once_per_cycle"},
      {"name": "action", "kv_lifetime": "cycle",   "addressing": "dense"}
    ],

    "guidance": {
      "video":  {"mode": "folded", "scale": 5.0},
      "action": {"mode": "positive_only"}
    },

    "purity": [
      {"key": "text_conditioning", "scope": "episode"},
      {"key": "rope_tables",       "scope": "model"}
    ],

    "obs_decode_modules": ["vae_decoder", "obs_head"],

    // Layer 4. A checkpoint-scoped fact: the runtime may choose any backend
    "attention": {"semantics": "softmax_full", "mask": "none", "layout": "bshd"},

    // CAPABILITIES of the output projection -- what replaces "this is a PDD
    // checkpoint". Any recipe producing per-interval velocity heads declares the
    // same numbers and is served by the same code, with no runtime change.
    "output_projection": {
      "kind": "per_interval_velocity_heads",
      "n_intervals": 256,
      "block": 128,
      "velocity_convention": "sigma_descending",
      "foldable": true
    },

    // Fit to serve? A BOOLEAN, recipe-agnostic. `verify/` decides it, the
    // publisher sets it, the runtime only refuses when it is false.
    "servable": true
  },

  // ==========================================================================
  // PROVENANCE -- for humans, model cards, and reproduction.
  // THE LOADER DOES NOT HAND THIS TO THE PLANNER. A pass that read it is a bug.
  // ==========================================================================
  "provenance": {
    "training_method": "parallel_decoding_distillation",
    "recipe_repo": "https://github.com/General-Instinct/instinct-pdd",
    "teacher": "lingbot-va-posttrain-robotwin",
    "trainable": "output heads only; trunk frozen",
    "solver": "euler",
    "dataset": "robotwin-2.0-reset-contexts-50task",
    "optimizer": {"name": "adamw", "lr": 1e-5, "weight_decay": 0.0},

    "training_diagnostics": {
      "coverage_gate_pass": true,
      "min_updates_per_head": 1,
      "endpoint_rmse": 0.153
    },

    "certification": {
      "protocol": "paired non-inferiority, exact McNemar, identical seeds",
      "margin": -0.05, "pairs": 566,
      "reference_success": 0.929, "candidate_success": 0.910, "p_value": 0.0085
    }
  }
}
```

Note where `training_method` sits: inside `provenance`, which the loader does not hand to the
planner. The separation is structural, not a convention to remember.

Three properties of this shape are load-bearing:

1. **`output_projection.kind` is the capability that replaces the method name.** DMD2, LCM, or a recipe
   nobody has written yet, producing per-interval velocity heads, declares the same numbers and is
   served by the same code. That is the platform claim made concrete rather than asserted.
2. **`servable` is a boolean, not a diagnostic.** The runtime asks one recipe-agnostic question, and
   a recipe-specific reason for refusing lives under `provenance.training_diagnostics`, where the
   runtime cannot reach it. `tests/test_runtime_boundary.py` fails if a serving module names such a
   key.
3. **`velocity_convention` closes a real trap declaratively.** A double sign flip here once produced
   0/100 on RoboTwin against a 92/100 control. It is currently a comment in
   [`runtime/block_heads.py`](instinctflash/runtime/block_heads.py); a comment cannot be checked.

---

## A step schedule is a declared field

A step schedule is a field in the declaration, not a mode in the runtime. Two checkpoints whose
`execution.nfe` differ are two declarations over the same weights:

```json
"execution": { "nfe": { "video": 2, "action": 4 } }
```

Publish a second checkpoint that says something else, or override the declared field at load:

```python
Runtime.from_pretrained("org/model", nfe={"video": 25, "action": 50})
```

There is no `Fast` or `Quality` preset in the runtime, and there will not be one: a preset table here
would be per-checkpoint tuning living in the wrong repository. For sweeping schedules during
evaluation, the eval server takes `--degrade-nfe 2,4`.

The planner re-derives which passes are legal *and* profitable from the declared schedule. That
matters because profitability genuinely inverts. Graph capture trades a fixed per-cycle capture cost
against a saving on every forward, so it depends on how many forwards a cycle runs; at the schedule
LingBot-VA ships with it measures 1.43× slower than not capturing, and `admit()` declines it. Had the
schedule been a runtime mode with its own hardcoded pass list, that would have shipped as a
regression with no record of why.

---

## Adding a recipe

Nothing in the runtime changes. In full:

1. Train, in your own repository. If it is reusable, keep it standalone and backbone-agnostic — PDD
   lives in [`instinct-pdd`](instinct-pdd) under Apache-2.0 for exactly this reason.
2. Publish the checkpoint with an `instinctflash.json` declaring its `phases`, `streams`, `guidance`,
   and `purity`.
3. Certify it: paired episodes on identical seeds, exact McNemar, a declared margin.
4. `load()` it. The runtime plans against the declaration it finds.

If step 4 requires a code change in `runtime/` or `planners/`, something is wrong — either the
capability is not yet declarable, in which case add the field and say which pass reads it, or the
recipe is trying to be a runtime.
