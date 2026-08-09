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

`package.py` requires a root `config.json` and a root weights file, and this directory has neither.

**The fix is not to widen the validator** — see §5. LeRobot publishes this same model flat, as a single
10.2 GB file plus a repo-id pointer to the frozen stack. The 23 GB directory is a *training output*,
not a published package, and treating it as one is the actual mistake.

So the gap is not subtle and it is not architectural. **The pipeline works; the front door is
missing**, and nobody has yet converted the flagship weights into something to put through it.

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
| **The model card is GENERATED** from a Jinja template at publish time (`generate_model_card()`), from the config alone | cards never drift from the artifact | Generate the InstinctWM card from `instinctwm.json`. We already generate nothing and hand-write everything |
| The generated card ships Evaluation as an **explicit honest hole**: `_No evaluation results have been provided._` | absence is visible rather than implied | Exactly this project's `NOT EVALUATED` discipline, applied to the card |
| **Metadata fetchable without weights** — `config.json` alone via `hf_hub_download` | inspect before you download 10 GB | `describe(repo)`, see §5 |
| **Typed migration errors carrying the exact fix command** (`ProcessorMigrationError`) | a format change is a one-liner, not an archaeology exercise | Adopt the pattern for any future `instinctwm_schema` bump |
| **Third-party policies discovered by distribution-name prefix** (`lerobot_policy_*`), no entry points, no core PR | an outside author ships a package, not a patch | Consider `instinctwm_backbone_*` later; `instinctwm.register` already covers the near term |
| `lerobot.__all__ == ["__version__", "available_extras"]` — the top level exports **almost nothing** | no vocabulary to learn before the first call | Shrink `__all__` to `Runtime`, `from_pretrained`, `describe`, `register`, `__version__`. The other fourteen move behind `instinctwm.advanced` |

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
        nfe: Mapping[str, int] | None = None,  # explicit override of a DECLARED field; see below
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

### Operating points: what I proposed, and why it was wrong

I first proposed `operating_point="fast" | "quality"`. **Do not build that.** `grep -rn
operating_point instinctwm/` returns nothing: Fast and Quality are not declared execution facts, they
are `--degrade-nfe 2,50` flags in the eval harness. A named preset resolved inside `Runtime` would put
a per-checkpoint tuning table *in the runtime* — precisely the branch that
`checkpoint → declaration → capabilities → planner` exists to prevent, and it would have been a
platform violation shipped inside the usability milestone.

The honest forms, both of which keep the declaration authoritative:

- **publish a second checkpoint** whose `execution.nfe` is the reduced schedule — an operating point
  *is* a descriptor delta, so it is a different declaration, not a mode flag; or
- **`nfe={"video": 2, "action": 4}`** as an explicit, visible override of a field the checkpoint
  already declares.

If presets are wanted later they belong in the *declaration* — e.g. an `execution.operating_points`
map the author publishes — never as a table inside `Runtime`.

### What breaks

Nothing. `Runtime` is additive. But `__all__` should also SHRINK, following LeRobot's two-name top level:
export `Runtime`, `from_pretrained`, `describe`, `register`, `__version__`, and move the other
fourteen to `instinctwm.advanced` with a one-line re-export shim so nothing existing breaks.

---

## 5. The package layout

**This section was rewritten after the study. My first answer was wrong, and the evidence is a repo
that already exists.**

### LeRobot already solved this, with our checkpoint, and did not widen its loader

`lerobot/lingbot_va_robotwin` is on the Hub. It is the same model as our 23 GB multi-folder directory,
and it is published **flat**:

```
lerobot/lingbot_va_robotwin/
  config.json                                              2.67 kB
  model.safetensors                                        10.2 GB    <- ONE file, not three folders
  policy_preprocessor.json / policy_postprocessor.json
  policy_postprocessor_step_0_unnormalizer_processor.safetensors
  README.md   assets/
```

and its `config.json` contains:

```jsonc
"type": "lingbot_va",
"wan_pretrained_path": "robbyant/lingbot-va-posttrain-robotwin",   // a REPO ID, not a subfolder
"pretrained_path": null
```

So the frozen stack — VAE, text encoder, tokenizer — is **referenced by repo id, not vendored**. The
servable artifact is the trainable part alone, 10.2 GB instead of 23 GB. They also never shard:
`_SINGLE_FILE_SHARD_SIZE = "1TB"` is passed as `max_shard_size` to every save, deliberately.

**This is better than the subdirectory map I proposed, for three reasons.** The published artifact is
less than half the size. Ten fine-tunes of the same backbone reference one upstream copy of the frozen
stack instead of each duplicating it. And the loader stays simple — no component-walking, no
per-component validation.

### Revised proposal: convert at publish time, point at the rest

Keep `REQUIRED = ("instinctwm.json", "config.json")` and a root weights file. Add **one** optional
`execution` key — a pointer, not a map:

```jsonc
"execution": {
  "backbone": "wan_va",
  "base_weights": "robbyant/lingbot-va-posttrain-robotwin",   // frozen stack, resolved at load
  ...
}
```

**Why this is an execution key:** the runtime cannot load the model without it. It names weights, not
a recipe, and passes `FORBIDDEN_IN_EXECUTION` by construction.

**What this makes the LingBot-VA work:** a *conversion*, not a validator change. Produce a flat
servable package — trainable weights in one `model.safetensors`, `instinctwm.json`, `config.json`,
card — that points at the upstream repo for the frozen stack. That is a script in `eval/` or
`examples/`, and it is the honest scope of "make LingBot-VA a first-class package".

**What I got wrong and why it matters:** I proposed widening `validate_package` to walk declared
subdirectories. That would have made every published LingBot-VA fine-tune carry its own 13 GB copy of
frozen weights, and added a component-validation path to the loader for a problem that publish-time
conversion removes. The study found the counter-example by looking at what LeRobot actually shipped
for this exact checkpoint.

### Fetch metadata without the weights

`PreTrainedConfig.from_pretrained` downloads only `config.json` via `hf_hub_download`. Nobody should
pull 10 GB to find out whether a checkpoint is servable. Add:

```python
from instinctwm import describe
describe("general-instinct/lingbot-va")   # execution declaration + capabilities, no weights
```

backed by a single `hf_hub_download(repo_id, "instinctwm.json")`. This is a small function and it is
the highest ratio of usefulness to effort in the whole proposal.

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

**Moved out of the README entirely:** Optimization Stack, Scope, How a new recipe plugs in →
`ARCHITECTURE.md` and `CHECKPOINTS.md`, linked from §7. Retire "What's New 🔥".

**Shipped Configuration must NOT simply move.** `README.md` is the only place that states the served
chain is **NUMERIC, not bit-exact end to end** — `served_tier()` returns `Tier.NUMERIC` because P007 is
served — and the only place naming the 555-episode certificate and the two NOT RECOMMENDED passes.
Deleting it and leaving "3.38× bit-exact" anywhere near the first screen would be a false claim with
its correction moved somewhere nobody in the quick-start path looks. That is the certificate gate
turned into a facade, which is the one thing this milestone must not do. Replace it with a short
generated block in **Results**:

> Shipped chain: P001 P002 P003 P007 · tier **NUMERIC** (P007 conv layout; 555-episode paired
> non-inferiority certificate) · the other three are bit-exact · 3.38× end-to-end in episode mode.

and repoint `test_shipped_config.py` at `README.md ∪ docs/models/lingbot_va.md`, adding an assertion
that the literal `served_tier().name` appears in the README.

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

## 8b. Two design rules the adversarial review produced

Both are general, and both cost nothing to follow.

**A closed vocabulary can legalize the complement of a forbidden key.** A reviewer proposed a
per-component `frozen: true` flag. `trainable` is already in `FORBIDDEN_IN_EXECUTION` — and "which
components were frozen" is the same training fact stated the other way round. The reader would raise
on one word and accept its complement, and the byte-identical-plan test could not catch it because
that test varies only *provenance*. **Rule: before adding an `execution` key, check whether its
negation is already forbidden.** If a planner genuinely needs the fact later, it arrives as a
capability token plus a test that two checkpoints differing *only* in that field produce **different**
plans — proving it is live rather than decorative.

**Do not promote a coincidence to an invariant.** `adapters/lingbot_va.py:36` declares
`param_bytes=10_179_017_396`, commented "transformer safetensors, bf16". That is the *file* size and
it matches the diffusers index `total_size` exactly; actual tensor-data bytes are 10,178,931,132, the
86,264-byte difference being the safetensors header. The number is fine as a file size, the field name
implies parameters, and any future check written against the apparent equality would break the first
time the transformer is re-sharded.

## 9. Risks and open questions

**One recommendation in this document was already revised by its own research.** §5 originally
proposed widening `validate_package` to walk declared subdirectories. A parallel study of what LeRobot
actually shipped for *this exact checkpoint* found `lerobot/lingbot_va_robotwin` — flat, 10.2 GB, with
the frozen stack referenced by repo id — which is strictly better, and §5 now says so. That is the
process working, and it is a reason to check the remaining unverified items below before building.


- **`library_name: instinctwm` is not a Hub-registered library — but the path is now known.**
  `lerobot` is registered in `huggingface.js`, `packages/tasks/src/model-libraries`, so registration is
  a PR to that repo rather than an unknown. Worth knowing before relying on it: the study found
  LeRobot's own Hub snippet is stale and only fires for one model family, so the auto-snippet is a
  nice-to-have, not the reason to register. Carry the snippet in the card body meanwhile.
- **`model-index` may not be the current mechanism.** The study found `.eval_results/*.yaml` files in
  `lerobot/pi05_base` — a newer, decentralized eval-results format. Both appear to work. *Decide which
  before writing the card; I verified `model-index` in the Hub docs but did not verify which the Hub
  now prefers.*
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
