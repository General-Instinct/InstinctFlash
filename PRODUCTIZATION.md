# Learn from LeRobot while productizing LingBot-VA

**A product-design proposal. No performance work, no new training algorithm, no broadening of
`AdapterSpec`.**

Everything attributed to LeRobot below was read from primary sources on 2026-08-09 and is cited. Where
I could not verify something I say so rather than assert it.

---

## 1. The gap, in one page

Two facts about InstinctWM, both verified on the tree today, define the milestone.

**The public API is entirely implementation vocabulary.** `instinctwm.__all__` is:

```
AdapterSpec, BackendAdapter, CommitMode, DeploymentSpec, GuidanceMode, GuidanceRule, KVLifetime,
KVStreamSpec, OptimizationPass, Optimizer, PassResult, PhaseSpec, Plan, PurityKey, Tier,
available_models, default_passes, load, register
```

There is **no `Runtime` class** and **no top-level `from_pretrained`**. Loading today returns a
`Checkpoint` — metadata, not something that runs — and the user must then resolve an adapter, call
`spec()`, call `Optimizer().compile(...)`, call `install(...)`, and drive inference themselves. Every
one of the eighteen exported names is a thing the product owner has said users should never need to
know.

**The flagship checkpoint fails our own validator.** The real LingBot-VA weights — 23 GB, the primary
supported model — are diffusers-style multi-folder:

```
lingbot-va-posttrain-robotwin/
  README.md   assets/
  transformer/    config.json + diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors + index
  vae/            config.json + diffusion_pytorch_model.safetensors
  text_encoder/   config.json + model-0000{1,2,3}-of-00003.safetensors + index
  tokenizer/
```

```
$ python -m instinctwm.descriptors.package /home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin
  servable package: NO
  MISSING  instinctwm.json
  PROBLEM  no declaration: this directory is not a checkpoint
```

`package.py` requires a root `config.json` and a root weights file. That assumption came from writing
the validator against a single-component example. Real world models are multi-component, and the one
checkpoint this project exists to serve is the counter-example.

So the gap is not subtle and it is not architectural. **The pipeline works; the front door is
missing**, and the flagship package does not fit through it.

---

## 2. What to adopt from LeRobot

Only items that survived scrutiny. Each row is a verified LeRobot behaviour, the UX property it
produces, and the concrete InstinctWM change.

| LeRobot behaviour (verified) | UX property | InstinctWM change |
|:--|:--|:--|
| Loading names a **repo, not a class**: `--policy.path=${HF_USER}/my_policy`, resolved via `get_policy_class(cfg.type)` in `policies/factory.py` | you need to know *what you have*, not *what class implements it* | `Runtime.from_pretrained(repo)`. We already have the mechanism — `execution.backbone` → registered Adapter is exactly `type` → policy class |
| **Quick Start is README section 1**, with `pip install lerobot` and a runnable snippet before any architecture. The GitHub README has *no architecture section at all* | a newcomer runs something in the first minute | Move Quick Start to the top; Optimization Stack (currently §3) and Shipped Configuration move to `ARCHITECTURE.md` / stay linked |
| Model card body order: **Model description → Quick start (inference) → Training → Fine-tune → Real-world eval** (`lerobot/smolvla_base`) | the card answers "how do I run this" second, not tenth | Adopt verbatim for the LingBot-VA card |
| Front-matter `library_name`, `pipeline_tag: robotics`, `tags`, `inference: false` | Hub filtering, discovery, no broken inference widget | Same, with `library_name: instinctwm`, `tags: [world-action-model, robotics, instinctwm]` |
| `hf upload <repo> <dir>` is the whole publish step | no bespoke CLI to learn | Document exactly this. Do not write a `instinctwm push` |
| **Revision pinning**: `--policy.pretrained_revision` accepting commit / branch / tag, with checkpoints tagged by step | you can name the exact artefact a number came from | `Runtime.from_pretrained(..., revision=...)` — the parameter already exists on `package.from_pretrained`, surface it and *document it*, because this project's whole culture is reproducible numbers |
| **Separate preprocessor/postprocessor artefacts** (`policy_preprocessor.json` + `.safetensors`) rather than baking normalization into the weights | the transform is inspectable and swappable | Not needed today, but it is the right precedent if LingBot-VA ever ships normalization stats: a sibling artefact, not a new `execution` key |
| **`model-index`** in front-matter, parsed by the Hub into an eval widget ([Hub model-card docs](https://huggingface.co/docs/hub/en/model-cards)) | benchmark results render natively | Carry the RoboTwin success rate here. This is how the benchmark becomes first-class instead of buried in `eval/` |
| **`base_model` + `base_model_relation`** (`finetune` / `quantized` / `adapter` / `merge`) | lineage is Hub-native and filterable | Declare the teacher here — **card metadata, not `execution`**. Lineage becomes visible without the runtime being able to read it |

### The one LeRobot solves that we should copy wholesale: failure that teaches

`get_policy_class` resolves a registered name. Ours must do the same, and the error is the product.
When `execution.backbone` names something unregistered, the message must say what was declared, what
is registered, and what to do — because "arbitrary backbone without an adapter" is our documented
boundary, and a user will hit it.

---

## 3. What stays deliberately different

Three places where InstinctWM should **not** follow LeRobot, and one where it is already ahead.

### LeRobot loads a model class. InstinctWM loads a declaration.

`make_policy(cfg)` instantiates a Python class chosen by `cfg.type`. The class *is* the contract: to
support a new policy you add a module to LeRobot.

InstinctWM loads a declaration and derives capabilities from it. The adapter supplies execution
*shape*; the checkpoint supplies execution *facts*. That is why a DMD2 or LCM checkpoint needs no
runtime change — it declares `output_projection.kind` and the planner does the rest.

**What it buys:** many checkpoints per backbone, planning that cannot see the recipe, and a
byte-identical-plan test to prove it.
**What it costs:** an extra concept. A LeRobot user thinks "which policy class"; an InstinctWM user
must think "which backbone, and what does my checkpoint declare". The README must earn that concept,
not assume it.

### LeRobot's `config.json` mixes execution and training. Ours must not.

This is the sharpest contrast and it is in our favour. `lerobot/smolvla_base/config.json` carries, in
one flat namespace:

```
inference:  input_features, output_features, n_action_steps, n_obs_steps, normalization_mapping, ...
training:   optimizer_lr, optimizer_betas, optimizer_weight_decay, optimizer_grad_clip_norm,
            scheduler_warmup_steps, scheduler_decay_steps, freeze_vision_encoder, train_expert_only
publishing: push_to_hub, repo_id, private, tags, license
```

That is precisely the `delta.json` mistake this project diagnosed and fixed: one namespace, so a
serving path reading a training key is a one-line slip rather than a boundary violation. InstinctWM
enforces the split at load and refuses a provenance key found under `execution`.

**Keep this. Do not flatten the schema to look more like LeRobot's.** It is the difference between a
convention and an invariant, and `publishability()` — strip provenance, reload, confirm it still
serves — is a guarantee LeRobot cannot currently offer a checkpoint author.

### LeRobot is CLI-first. InstinctWM should be API-first.

LeRobot's primary surface is `lerobot-train`, `lerobot-record`, `lerobot-rollout`,
`lerobot-teleoperate`; Python appears as "API examples". That fits a project whose users are
teleoperating arms and recording datasets.

InstinctWM's user is integrating a policy into a control loop. `Runtime.from_pretrained(...)` then
`runtime.predict(obs)` is the right shape, and the target API the product owner specified is
**simpler than LeRobot's own Python path**, which needs a config object and two calls. We should ship
one CLI — `instinctwm validate` / `instinctwm serve` — and not build a command surface we do not need.

### Do not hide the honest reporting behind the facade

This project reports `NOT EVALUATED` on a contended device, refuses `servable=false` checkpoints, and
gates NUMERIC passes on a certificate. A facade that swallowed those to look smooth would be a
regression dressed as usability. `Runtime` must surface them — as exceptions that explain, and as a
readable `runtime.explain()` — while not *requiring* the user to understand tiers to get an action.

---

## 4. The `Runtime` facade

```python
from instinctwm import Runtime

runtime = Runtime.from_pretrained("general-instinct/lingbot-va")
action = runtime.predict(observation)
```

### API

```python
class Runtime:
    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        revision: str | None = None,        # commit / branch / tag -- reproducibility
        operating_point: str | None = None, # "fast" | "quality" | None = the declared default
        device: str | None = None,          # None -> cuda if available, else cpu
        strict: bool = True,                # False downgrades refusals to warnings, for inspection
    ) -> "Runtime": ...

    def predict(self, observation: Mapping[str, Any]) -> Any: ...
    def reset(self) -> None: ...
    def serve(self, port: int = 29056, **kwargs) -> Any: ...

    # escape hatches -- present, documented, never required
    @property
    def plan(self) -> Plan: ...             # read-only; what was applied and why
    @property
    def checkpoint(self) -> Checkpoint: ... # .execution, .capabilities()
    def explain(self) -> str: ...           # human-readable, for a bug report
```

`from_pretrained` performs, in order — and this order *is* the platform pipeline, which is why the
facade does not weaken it:

1. resolve the repo (local path, or Hub via `huggingface_hub`)
2. `validate_package` → refuse a directory that is not a checkpoint
3. `load_declaration` → **execution only**
4. `require_servable` → refuse `servable=false`, without asking why
5. resolve `execution.backbone` → registered Adapter, or raise the teaching error below
6. `adapter.spec()`
7. `checkpoint.capabilities()` → `Optimizer().compile(spec, capabilities=...)`
8. `adapter.install(...)` → and every applied pass must be installed or shown vacuous

### The error that teaches

```
UnknownBackboneError: checkpoint 'general-instinct/lingbot-va' declares
  execution.backbone = "wan_va"
but no adapter is registered for it.

Registered backbones: lingbot-va-posttrain-robotwin

An adapter tells InstinctWM the SHAPE of a control step -- streams, phases, guidance. It is a
small Python class, it lives in your project (not in InstinctWM), and you register it with:

    instinctwm.register("wan_va", MyAdapter)

Worked example: examples/tiny_wam/adapter.py
Why this is required, and what would remove the requirement: CHECKPOINTS.md, "Scope".
```

### Operating points belong here

Operating points are already descriptor deltas, not code paths. `operating_point="fast"` is the
natural product surface for that, and the declaration already carries `nfe` per stream. This adds a
*name* for something that exists; it does not add a mode flag or a second runtime.

### What breaks

Nothing. `Runtime` is additive. The existing exports stay for the people who want them — but
`__all__` is reordered so `Runtime` and `from_pretrained` come first, and the implementation
vocabulary moves behind a documented "advanced" heading.

---

## 5. The package layout

### Revised rules

The current rules assume one component. Replace with:

```python
REQUIRED_ROOT = ("instinctwm.json",)          # the declaration is the only unconditional file

# a package is EITHER single-component OR multi-component; both are first class
SINGLE_COMPONENT = root has config.json AND one of WEIGHTS_ANY
MULTI_COMPONENT  = execution.components names >= 1 subdirectory, each of which
                   has config.json AND one of WEIGHTS_ANY
```

and add one optional key to `execution` — the only schema change this proposal makes:

```jsonc
"components": {
  "transformer":   "transformer",
  "vae":           "vae",
  "text_encoder":  "text_encoder",
  "tokenizer":     "tokenizer"
}
```

**Why this is an execution key and not provenance:** which subdirectory holds which component is a
fact the runtime needs in order to load the weights. It says nothing about training. It passes the
`FORBIDDEN_IN_EXECUTION` test by construction.

**Why declare it rather than infer it:** inference by globbing would make the layout an accident of
directory naming. This project's own history says a fact worth depending on should be declared and
checkable, not guessed — the same argument that put `velocity_convention` in the schema.

`validate_package` then reports per component, and `publishability()` is unchanged.

### What LingBot-VA specifically needs

To become a first-class package, added to the existing 23 GB directory:

| file | content |
|:--|:--|
| `instinctwm.json` | `execution`: `model_id`, `backbone: "wan_va"`, `servable: true`, `guidance {video: cfg, action: positive_only}`, `nfe {video: 2, action: 4}`, `components {...}`. `provenance`: optional |
| `README.md` | model card, front-matter below, body in LeRobot's order |
| `LICENSE` | explicit |
| `instinctwm_benchmark.json` | *optional* — RoboTwin success, protocol, seeds |
| `instinctwm_certificate.json` | *optional* — the NUMERIC certificate for P007 |

And one adapter change: `backbone: "wan_va"` must resolve. Today the registered id is
`lingbot-va-posttrain-robotwin` — a *checkpoint* id used as a *backbone* id. That conflation is
exactly what the platform claim forbids, and it should be fixed by registering the backbone under
`wan_va` and letting many checkpoints declare it.

---

## 6. Publishing and the model card

Publishing is `hf upload`, as in LeRobot. No bespoke tool.

```bash
python -m instinctwm.descriptors.package ./lingbot-va     # validate first
hf upload general-instinct/lingbot-va ./lingbot-va
```

### Front-matter

```yaml
---
library_name: instinctwm
pipeline_tag: robotics
license: apache-2.0
tags:
  - world-action-model
  - robotics
  - instinctwm
inference: false
base_model: general-instinct/lingbot-va-teacher     # lineage, HUB-NATIVE
base_model_relation: finetune
model-index:
  - name: lingbot-va
    results:
      - task: {type: robotics}
        dataset: {name: RoboTwin 2.0, type: robotwin}
        metrics:
          - {name: success rate, type: success_rate, value: 0.9081}
---
```

Two things to note. `base_model` puts the teacher where the Hub can render it **without putting it in
`execution`** — lineage becomes visible while remaining unreadable by the runtime, which is exactly
the split the project wants. And `model-index` is parsed by the Hub into an eval widget, so the
benchmark renders natively instead of living only in `eval/`.

### The optional artefacts

`instinctwm.json` stays the declaration. Benchmark and certificate ride alongside as **separate
optional files**, because they are evidence about a checkpoint rather than facts needed to run it:

```
instinctwm_benchmark.json     protocol, seeds, arms, result, and what was NOT evaluated
instinctwm_certificate.json   tier, margin declared BEFORE the run, paired episodes, exact McNemar
```

**The certificate is a differentiator, not clutter.** No other robotics checkpoint on the Hub ships
"this optimization is NUMERIC, here is a 555-episode paired non-inferiority test with the margin
declared in advance and exact McNemar p = 0.897". The card should render a short table and link the
file. Absent → the card simply does not have that section; nothing refuses to load.

---

## 7. README rewrite

The current README is 407 lines with **Optimization Stack at §3 (line 69)** and **Quick Start at line
252**. Implementation sits above usage and there are two competing entry points ("Serve a checkpoint"
at line 100, "Quick Start" at 252).

### New section order

| # | section | from |
|--:|:--|:--|
| 1 | hero + one-paragraph value proposition | rewritten |
| 2 | **Quick Start** — install, load, predict | merge of "Serve a checkpoint" + "Quick Start" |
| 3 | Supported models | keep |
| 4 | Operating points | keep, shortened |
| 5 | Publish a checkpoint | keep, trimmed to a pointer + example |
| 6 | Results | keep |
| 7 | Documentation | keep, now the gateway to everything below |
| 8 | Citation / License | keep |

**Moved out of the README entirely:** Optimization Stack, Shipped Configuration, Scope, How a new
recipe plugs in → `ARCHITECTURE.md` and `CHECKPOINTS.md`, linked from §7. Retire "What's New 🔥".

Note `tests/test_shipped_config.py` asserts the README lists the shipped flags and mentions P005/P006.
Moving Shipped Configuration means **updating that test to assert against the new home**, not deleting
the check.

### First screen, drafted

> # InstinctWM
> **One runtime for robot world-action models.**
>
> ```bash
> pip install instinctwm
> ```
>
> ```python
> from instinctwm import Runtime
>
> runtime = Runtime.from_pretrained("general-instinct/lingbot-va")
> action = runtime.predict(observation)
> ```
>
> That is the whole API. InstinctWM reads what the checkpoint declares about itself, works out which
> optimizations are legal for it, applies them, and reports what each one cost. You never choose a
> pass, a backend, or a kernel.
>
> **Publishing your own checkpoint** is a JSON file next to your weights, and `hf upload`. Your
> training recipe stays yours — the runtime has nowhere to read it, and `publishability()` proves it.
>
> [Quick Start](#quick-start) · [Supported models](#supported-models) · [Publish a checkpoint](CHECKPOINTS.md) · [Architecture](ARCHITECTURE.md)

---

## 8. Migration and sequencing

Ordered smallest-first. **B** = behaviour change, **D** = documentation only.

| # | item | breaks | test |
|--:|:--|:--|:--|
| 1 | **D** README rewrite + move Optimization Stack / Shipped Config out | `test_shipped_config.py` README assertions | update that test to the new location |
| 2 | **B** `execution.components` optional key; `validate_package` accepts multi-component | nothing — optional key, old packages still single-component | extend `test_checkpoint_platform.py` with a multi-component fixture |
| 3 | **B** register the `wan_va` **backbone** (distinct from the checkpoint id) | the current registration name; keep it as an alias for one release | `test_checkpoint_platform.py` resolves `wan_va` |
| 4 | **B** `Runtime` facade + top-level `from_pretrained`, `__all__` reordered | nothing removed | new `test_runtime_facade.py`: the eight-step order, the teaching error, `plan` read-only |
| 5 | **B** stamp `instinctwm.json` onto the real LingBot-VA directory | nothing | `python -m instinctwm.descriptors.package` on the real 23 GB dir must report YES |
| 6 | **D** LingBot-VA model card, front-matter, optional benchmark + certificate files | nothing | a card-lint test: front-matter keys present, body sections in order |
| 7 | **B** `instinctwm validate` / `instinctwm serve` console entry points | nothing | smoke test |

Items 1–2 are independent and can go first. Item 5 depends on 2 and 3. Item 4 is the milestone's
headline and depends on 3.

---

## 9. Risks and open questions

- **`library_name: instinctwm` is not a Hub-registered library.** The docs say an unregistered
  `library_name` still displays and filters, but the auto-generated "Use this model" snippet comes
  from Hub-side library integration. Getting that requires a PR to `huggingface/hub-docs`. Until then
  the card's Quick Start section carries the snippet manually. *Unverified: whether the Hub currently
  accepts new library registrations for niche robotics libraries.*
- **`Runtime.predict()` for LingBot-VA is not obviously in-process.** Today LingBot-VA serves over a
  websocket with a separate client interpreter (deliberately — the two envs are
  dependency-incompatible). `predict()` may have to be a thin client, or `serve()` may be the honest
  primary. **This needs deciding before item 4**, and it is the largest unknown in the proposal.
- **`base_model` may not accept a private or non-existent repo** without rendering an error. Verify
  before putting a teacher id in the card.
- **23 GB is not `hf upload`-in-one-go friendly.** Chunked upload and LFS behaviour on a repo this
  size should be rehearsed before it is documented as a one-liner.
- **The `wan_va` rename touches a frozen surface.** The registered id appears in `released.py`'s
  world and in launch scripts. Item 3 must keep the old name as an alias for a release, and
  `test_shipped_config.py` should assert both resolve.
- **Not verified:** whether LeRobot has added a generic `AutoPolicy`-style Python entry point since
  the sources I read. I checked `policies/factory.py` and the docs, and found `make_policy(cfg)` with
  `get_policy_class(cfg.type)` — a factory, but one that still takes a config object rather than a
  repo id. If they have since added a one-liner, our design is unaffected: it is already simpler.

---

## Appendix: sources

- `lerobot/smolvla_base` file listing and `config.json` — https://huggingface.co/lerobot/smolvla_base
- Model card front-matter and body order — https://huggingface.co/lerobot/smolvla_base/raw/main/README.md
- Policy factory / type dispatch — https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/factory.py
- CLI, `--policy.path`, `--policy.pretrained_revision`, `hf upload` — https://huggingface.co/docs/lerobot/en/il_robots
- GitHub README section order — https://raw.githubusercontent.com/huggingface/lerobot/main/README.md
- `model-index`, `base_model`, `library_name` semantics — https://huggingface.co/docs/hub/en/model-cards
