# Checkpoints

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
[`instinctwm/adapters/base.py`](instinctwm/adapters/base.py).

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

`phases` is doing most of the work. It is what makes the operating point a declared fact rather than
a flag: `total_forwards()` and `forwards_breakdown()` come straight out of it, and those are the
numbers profitability is computed against.

### Site-scoped — `DeploymentSpec`

The same checkpoint is one GPU here and eight there, and one caller wants predicted pixels while the
next wants only actions. Implemented in
[`instinctwm/descriptors/deployment.py`](instinctwm/descriptors/deployment.py).

| Field | Meaning |
|:--|:--|
| `world_size` | how many ranks — `FSDPElision` guards on it |
| `want_pixels` | does the caller need decoded video — `ObsDecodeElision` guards on it |

A pass that read a deployment fact off `AdapterSpec` would be asking the model author to declare
something they cannot know. Fields are added here only when a pass reads one.

---

## The serialized form

> **Status: implemented for the block-heads serving path; not yet for `AdapterSpec` as a whole.**
> [`descriptors/checkpoint.py`](instinctwm/descriptors/checkpoint.py) reads this schema today:
> `load_declaration()` returns the `execution` block **only** and has nowhere to put a training
> method, `provenance_of()` is a separate deliberate call, and a provenance key found inside
> `execution` is a load error. `runtime/block_heads.py` consumes it, and the LingBot heads trainer
> writes it alongside the legacy `delta.json`. What is **not** built: Hub resolution by `model_id`,
> and construction of a full `AdapterSpec` from the file — adapters are still registered in Python.

For a checkpoint published on the Hub, the declaration ships as `instinctwm.json` beside the weights.
The runtime resolves it in this order, first hit wins:

1. `instinctwm.json` in the checkpoint directory or repo
2. a registered adapter for `model_id` (`instinctwm.register`)
3. refuse — an unrecognized checkpoint is not served on guessed facts

```jsonc
{
  "instinctwm_schema": 1,

  // ==========================================================================
  // EXECUTION -- everything the runtime may read, and nothing else.
  // Adding a key here requires naming the pass or planner that reads it.
  // ==========================================================================
  "execution": {
    "model_id": "lingbot-va-robotwin-blockheads-2v4a",
    "backbone": "wan-va",                  // which adapter publishes the sites
    "param_bytes": 24696061952,

    // The control step. This IS the operating point: a reduced schedule is a
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
    // implementing THIS function and no other. See ATTENTION.md.
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
2. **`servable` is a boolean, not a diagnostic.** The runtime asks one recipe-agnostic question. The
   audit found the serving path reading PDD's `coverage_gate_pass` directly
   ([AUDIT.md](AUDIT.md) F2) — right intent, wrong layer. The PDD-specific reason now lives under
   `provenance.training_diagnostics`, where the runtime cannot reach it.
3. **`velocity_convention` closes a real trap declaratively.** A double sign flip here once produced
   0/100 on RoboTwin against a 92/100 control. It is currently a comment in
   [`runtime/block_heads.py`](instinctwm/runtime/block_heads.py); a comment cannot be checked.

The audit and the staged migration to this schema are in [AUDIT.md](AUDIT.md).

---

## Operating points are descriptor deltas

An operating point is a checkpoint declaration with a different `phases` block. Nothing else.

```python
fast = spec.with_phases(video=2, action=4)     # PROPOSED API -- not implemented yet
plan = Optimizer(tier_ceiling=Tier.BEHAVIORAL).compile(fast, deployment)
```

Today the same thing is expressed by `--degrade-nfe 2,4` on the eval server, which is how every Fast
number in this repository was measured. `with_phases` is the shape it should take once operating
points are first-class; the planner already computes profitability from `phases`, so the missing part
is only the ergonomics of producing the delta.

The planner re-derives which passes are legal *and profitable* from that. This is not theoretical
tidiness — it is the only reason we caught the following.

**Graph capture inverts between operating points.** `P005` saves ~17 ms per forward but costs a fixed
~700 ms per cycle to capture, breaking even near **41 forwards per cycle**:

| Operating point | forwards/cycle | graph capture |
|:--|--:|:--|
| Quality (25 video / 50 action) | 75 | **profitable** — 1.205× |
| Fast (2 video / 4 action) | 6 | **unprofitable** |

If Fast were a separate runtime with a hardcoded pass list, this would have shipped as a 2× latency
regression with the reason buried in a branch. Because profitability is computed from declared
`phases`, `admit()` answers it directly and the log says why.

The same arithmetic explains why further step reduction is not worth pursuing at Fast. The warm cost
model is `FIXED 1164 ms + 15.5 ms/forward` (R² = 0.994), so at 6 forwards **93% of latency is fixed
overhead**. Cutting steps further optimizes 7% of the problem.

---

## Adding a recipe

Nothing in the runtime changes. In full:

1. Train, in your own repository. If it is reusable, keep it standalone and backbone-agnostic — PDD
   lives in [`instinct-pdd`](instinct-pdd) under Apache-2.0 for exactly this reason.
2. Publish the checkpoint with an `instinctwm.json` declaring its `phases`, `streams`, `guidance`,
   and `purity`.
3. Certify it: paired episodes on identical seeds, exact McNemar, a declared margin.
4. `load()` it. The runtime plans against the declaration it finds.

If step 4 requires a code change in `runtime/` or `planners/`, something is wrong — either the
capability is not yet declarable, in which case add the field and say which pass reads it, or the
recipe is trying to be a runtime.
