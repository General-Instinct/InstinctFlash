# Repository audit: One Runtime, Many Checkpoints, Zero Runtime Forks

Audited 2026-08-06 against one principle:

> **There is only one InstinctWM Runtime. Different models are different checkpoints. Every checkpoint
> declares its capabilities through descriptors. The runtime never depends on how a checkpoint was
> trained.**

Method: grep the tree for method names, conditional branches, profile terminology, and training
artifacts read at serve time; then check whether each hit is a *branch* (a violation), a *declaration*
(correct), or *prose* (harmless). Findings are ordered by severity. Every one cites file and line.

**Headline: the runtime does not branch on training method — but it does something worse in one
place.** When serving a distilled checkpoint, the serving path imports the training library of one
specific distillation method and reads that method's training diagnostics. Three findings, one
structural, are real. The rest is terminology.

---

## Findings

### F1 — CRITICAL: the serving path imports a training library — **RESOLVED (Stage 1)**

`instinctwm/runtime/block_heads.py:137`

```python
# as found:
from instinctwm.adapters.lingbot_velocity import LingBotChunk0Video          # -> imports instinct_pdd
# after Stage 0 -- same dependency, now impossible to miss:
from instinctwm.train.oracles.lingbot_velocity import LingBotChunk0VideoOracle
```

`lingbot_velocity.py:55` does `from instinct_pdd import Grid, MultiHeadStudent`. So serving a
heads checkpoint pulls in **the training library of one distillation method**. The import is
function-local, so it lands only on that path — which makes it invisible rather than absent.

What it is used for is one line, `block_heads.py:150`:

```python
adapter = LingBotChunk0VideoOracle(server, guidance=...)
grid = adapter.grid(n_intervals, block)
```

It builds a *training oracle over a live server* to obtain interval widths. Those widths are
`sigma_{n+1} - sigma_n` on the scheduler the server already owns — an execution fact the runtime can
compute or the checkpoint can declare. Nothing about it requires PDD.

**Severity: critical.** This is the principle's exact prohibition. It also means a checkpoint from any
future recipe cannot be served without `instinct_pdd` installed, which makes the training method a
runtime dependency.

### F2 — HIGH: the runtime refuses to serve based on a training diagnostic — **RESOLVED (Stage 2)**

`instinctwm/runtime/block_heads.py:141`

```python
if not meta.get("coverage_gate_pass", False):
    raise RuntimeError(f"{d}: delta.json says the coverage gate FAILED, ...")
```

`coverage_gate_pass` is a *PDD training* statistic — whether every one of 256 heads received a
minimum update count. The serving path reads it and refuses.

The intent is right and should be kept: an undertrained checkpoint should not produce a benchmark
number. The *layer* is wrong. "Was this checkpoint trained adequately" is a question for `verify/`
or for the publisher, and expressing it as a recipe-specific key means every future recipe must
either invent its own key or be served ungated.

**Severity: high.** Behaviour is correct; the dependency is not.

### F3 — HIGH: a training module was filed as a runtime adapter — **fixed in Stage 0**

Found at `instinctwm/adapters/lingbot_velocity.py`, now
`instinctwm/train/oracles/lingbot_velocity.py` — *"LingBot-VA chunk-0 video stream, exposed as a PDD
velocity oracle"*

It defines `teacher()`, `student()`, and `grid()`. It implements **zero** methods of the
`BackendAdapter` protocol — no `spec()`, no `sites()`, no `apply()`. It is a Layer 1 training oracle
sitting in a Layer 2–6 concept directory, which is what let F1 look reasonable when it was written.

**Severity: high**, because the mislabelling is what hides F1. Moved in Stage 0 and the class renamed
to `LingBotChunk0VideoOracle`; F1 itself is untouched by that and remains open.

### F4 — HIGH: `delta.json` mixes execution facts with training provenance — **RESOLVED (Stage 2)**

`eval/lingbot_va_robotwin/train_pdd_heads.py:693-706` and `instinctwm/train/trainer.py:261-264`

One flat namespace, no separation:

| key | kind |
|:--|:--|
| `nfe`, `n_intervals`, `block`, `guidance`, `shapes`, `dtypes` | **execution** — the runtime needs these |
| `recipe`, `trainable`, `solver`, `coverage_gate_pass`, `min_updates_per_head`, `head_updates_min`, `endpoint_rmse`, `note` | **provenance** — the runtime must never read these |

Because they share a namespace, F2 was a one-line mistake rather than a boundary violation. `recipe:
"parallel_decoding_distillation"` sits in the same dict the serving path indexes.

**Severity: high.** This is the root cause of F2 and the thing the schema below fixes.

### F5 — MEDIUM: `--pdd-heads` names a training method in a serving interface — **RESOLVED (Stage 3)**

`eval/lingbot_va_robotwin/serve_variant.py:91, 140, 146, 168`, and the internal `_pdd_dir`

The flag selects a *capability* — serve a checkpoint whose output projection is per-interval velocity
heads — but is named for the method that happened to produce one. `run_pdd_cert.sh:52` passes it.

**Severity: medium.** It is the eval harness, not the runtime, and the harness is allowed to be
research-shaped. But it is the user-facing name for the capability, so it teaches the wrong model.

### F6 — LOW: "Fast profile" / "Quality profile" used as runtime terminology

`instinctwm/verify/released.py:105-108`

```python
MEASUREMENT_PROTOCOL = "sequential A/B, pre-order-control; Quality profile (25/50)"
```

As a *label on a measurement* this is legitimate and should stay — those numbers really were taken at
25/50. The word "profile" is the problem: it reads as a runtime mode. The correct noun is **operating
point**, which names a declared step schedule rather than a configuration of the engine.

**Severity: low**, terminology only.

### F7 — LOW: one documentation line implies two execution stacks

`README.md:149` — *"One Plan runs under both executors"*

True and load-bearing (eager and graph-replay executors, one plan), but "both executors" scans as
"two execution stacks" to a first-time reader on a page whose thesis is one runtime.

**Severity: low**, wording.

---

## What is NOT a violation

Recorded so it is not re-flagged:

- **No conditional anywhere branches on a training method.** The audit's grep for
  `if .*(pdd|dmd2|lcm|dreamzero|distill|method ==)` across `instinctwm/` returns nothing in the
  runtime path. `serve_variant.py:140` (`if args.pdd_heads`) branches on a *CLI flag*, not on
  checkpoint metadata.
- **`DreamZero` in `runtime/state/types.py:34,46,82` and `adapters/base.py:41`.** These cite a *model
  family* whose KV lifetime motivated an enum member. Provenance comments on a vocabulary, not
  branches.
- **`teacher`/`student` in `runtime/block_heads.py` prose.** Docstring commentary explaining the sign
  convention. One code comment should be reworded (below), the rest is fine.
- **`instinct_pdd` bootstrap in `instinctwm/__init__.py:47`.** A `sys.path` fallback for the submodule.
  It imports nothing and branches on nothing.
- **"fast path" in `passes/lingbot/ring_kv.py`, `runtime/lingbot_install.py`.** Means a branch inside
  a kernel. Unrelated to operating points.
- **`train/` containing recipe names.** That is its job. Layer 1 is *allowed* to know it is PDD.

---

## Proposed `instinctwm.json`

Two top-level namespaces, and the loader hands only the first to the planner. The separation is
structural, not a convention to remember.

```jsonc
{
  "instinctwm_schema": 1,

  // ============================================================================
  // EXECUTION -- everything the runtime may read, and nothing else.
  // Adding a key here requires naming the pass or planner that reads it.
  // ============================================================================
  "execution": {
    "model_id": "lingbot-va-robotwin-blockheads-2v4a",
    "backbone": "wan-va",                 // which adapter publishes the sites
    "param_bytes": 24696061952,

    // The control step. This IS the operating point: a reduced schedule is a
    // different phases block, not a mode flag.
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
      "video":  {"mode": "folded", "scale": 5.0},   // folded => no CFG batch at serve time
      "action": {"mode": "positive_only"}
    },

    "purity": [
      {"key": "text_conditioning", "scope": "episode"},
      {"key": "rope_tables",       "scope": "model"}
    ],

    "obs_decode_modules": ["vae_decoder", "obs_head"],

    // Attention semantics, per Layer 4. A checkpoint-scoped fact: the runtime may
    // choose any backend implementing THIS function and no other.
    "attention": {"semantics": "softmax_full", "mask": "none", "layout": "bshd"},

    // CAPABILITIES the output projection provides. This is what replaces
    // "this is a PDD checkpoint". Any recipe producing per-interval velocity
    // heads declares the same three numbers and is served by the same code.
    "output_projection": {
      "kind": "per_interval_velocity_heads",
      "n_intervals": 256,
      "block": 128,
      "velocity_convention": "sigma_descending",   // fixes F1's sign question declaratively
      "foldable": true                             // affine heads collapse to one Linear
    },

    // Is this checkpoint fit to serve? A BOOLEAN, recipe-agnostic. Replaces the
    // runtime reading `coverage_gate_pass` (F2). Whoever publishes sets it;
    // `verify/` decides it; the runtime only refuses when it is false.
    "servable": true
  },

  // ============================================================================
  // PROVENANCE -- for humans, model cards, and reproduction.
  // THE LOADER DOES NOT HAND THIS TO THE PLANNER. A pass that read it is a bug.
  // ============================================================================
  "provenance": {
    "training_method": "parallel_decoding_distillation",
    "recipe_repo": "https://github.com/General-Instinct/instinct-pdd",
    "teacher": "lingbot-va-posttrain-robotwin",
    "trainable": "output heads only; trunk frozen",
    "solver": "euler",
    "dataset": "robotwin-2.0-reset-contexts-50task",
    "optimizer": {"name": "adamw", "lr": 1e-5, "weight_decay": 0.0},
    "paper": "Parallel Decoding Distillation",

    "training_diagnostics": {
      "coverage_gate_pass": true,
      "min_updates_per_head": 1,
      "head_updates_min": 0,
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

Three properties worth stating:

1. **`output_projection.kind` is the capability that replaces the method name.** DMD2, LCM, or a recipe
   nobody has written yet, producing per-interval velocity heads, declares the same three numbers and
   is served by the same code with no runtime change. That is the whole claim, made concrete.
2. **`servable` is a boolean, not a diagnostic.** The runtime asks one recipe-agnostic question. The
   PDD-specific reason lives in `provenance.training_diagnostics`, where the runtime cannot reach it.
3. **`velocity_convention` closes a real trap declaratively.** A double sign flip here once produced
   0/100 on RoboTwin against a 92/100 control. It is currently a comment; it should be a field.

---

## Proposed renames

| # | From | To | Kind |
|:--|:--|:--|:--|
| R1 | `instinctwm/train/oracles/lingbot_velocity.py` | `instinctwm/train/oracles/lingbot_velocity.py` | pure move |
| R2 | `LingBotChunk0VideoOracle` | `LingBotChunk0VideoOracle` | class rename — it is an oracle, not an adapter |
| R3 | `delta.json` | `instinctwm.json`, two namespaces | schema |
| R4 | `--pdd-heads` | `--block-heads` (keep `--pdd-heads` as a deprecated alias) | CLI |
| R5 | `_pdd_dir` | `_heads_dir` | local variable |
| R6 | "Fast profile" / "Quality profile" | "Fast / Quality **operating point**" | terminology |
| R7 | `# the teacher's own proj_out` | `# the checkpoint's own proj_out` | comment |
| R8 | `run_pdd_cert.sh` | `run_blockheads_cert.sh` | script name |

R1, R2, R5, R6, R7 are behaviour-preserving. R3 and R4 are staged below.

---

## Migration plan

Ordered so that nothing is half-migrated at any point, and so no stage depends on a stage that has not
shipped. **Stage 0 is applied in this commit; stages 1–4 are not.**

### Stage 0 — make the violation legible (no behaviour change) ✅ applied

- R1, R2: move the training oracle out of `adapters/` into `train/oracles/`, update its 5 importers.
- R6, R7: terminology.
- R7b: reword `README.md:149` so "both executors" cannot read as two stacks.

Moving the oracle does **not** fix F1 — `block_heads.py` then reads
`from instinctwm.train.oracles...`, which is a *more* obvious smell inside `runtime/`. That is the
point: the mislabelling as an adapter is what made the dependency look acceptable. Make it ugly, then
remove it.

### Stage 1 — remove the training dependency from serving (fixes F1)

Replace the oracle round-trip with the two facts it was standing in for:

```python
# now:    adapter = LingBotChunk0VideoOracle(server, guidance=...); grid = adapter.grid(N, L)
# after:  widths = interval_widths(server.scheduler.sigmas, n_intervals, block)
```

Interval widths are differences of the scheduler's own sigmas. The runtime already owns the scheduler.
Gate with `tests/test_pdd_serve_parity.py`, which already asserts the folded weights equal the 2-step
`dsigma` at `0.00e+00` — so this is verifiable as bit-exact before and after.

Removes: the `instinct_pdd` import from serving, and the "build a training oracle over a live server"
construction. **Behaviour must be bit-identical; the parity test is the gate.**

### Stage 2 — split the declaration (fixes F4, then F2)

1. Trainers write `instinctwm.json` with both namespaces, *and* keep writing `delta.json` unchanged.
   Purely additive; nothing reads the new file yet.
2. `block_heads.py` prefers `instinctwm.json` when present, falls back to `delta.json`. Reads
   `execution.*` only, and `servable` instead of `coverage_gate_pass`.
3. Once every live checkpoint under `/home/ubuntu/iwm_results/` has been re-stamped, delete the
   `delta.json` fallback.

Step 2 is where F2's behaviour changes: the refusal becomes recipe-agnostic. Same protection, and the
PDD reason moves to `provenance.training_diagnostics`.

### Stage 3 — CLI (R4, R8)

Add `--block-heads`; keep `--pdd-heads` as a deprecated alias that warns, because
`run_pdd_cert.sh:52` and queued chain scripts pass it. Remove the alias only after those are updated.

### Stage 4 — enforce the boundary mechanically

A test that fails if the runtime path can reach a training module or a provenance key:

```python
# tests/test_runtime_is_method_agnostic.py
FORBIDDEN = ("instinct_pdd", "instinctwm.train")
for mod in walk("instinctwm/runtime", "instinctwm/planners", "instinctwm/executors"):
    assert not imports_any(mod, FORBIDDEN)
for key in ("recipe", "training_method", "coverage_gate_pass", "endpoint_rmse", "teacher"):
    assert key not in source_of("instinctwm/runtime/**")
```

This is the deliverable that keeps the audit from being needed twice. It should land with Stage 2, not
before — it fails today, correctly.

---

## Scorecard

**All stages applied 2026-08-06.** 22 passed, 4 skipped, 0 failed.

| Area | State |
|:--|:--|
| Conditional branching on training method | **clean** — none anywhere |
| Method names in `planners/`, `executors/`, `descriptors/`, `backends/` | **clean** — zero hits |
| Method names in `runtime/` | **clean** — F1 and F2 resolved; enforced by `tests/test_runtime_boundary.py` |
| Training modules in runtime concept dirs | **clean** — moved in Stage 0 |
| Checkpoint metadata vs provenance | **clean** — two namespaces, `load_declaration()` returns execution only |
| Docs implying multiple runtimes | **clean** — reworded in Stage 0 |
| Operating-point terminology | **clean** — "operating point" throughout |
| Attention layer (Layer 4) | **clean** — semantics are checkpoint-declared by construction |

The claim *"the runtime never depends on how a checkpoint was trained"* is now true of the code and
not only of the documentation, and `tests/test_runtime_boundary.py` keeps it true.

**One thing found while doing the work, worth recording.** `block_heads.py` claimed its grid
equivalence was "verified in tests/test_pdd_serve_parity.py"; that file did not exist, and this
document repeated the claim in its Stage 1 plan. The equivalence the entire serving path rests on was
argued in a docstring and checked nowhere. `tests/test_serve_parity.py` now exists and holds it at
`0.00e+00`.

**And one arithmetic subtlety the gate caught.** The retired `Grid` computed widths as
`(1 - sigma[l+1]) - (1 - sigma[l])`; the obvious replacement `sigma[l] - sigma[l+1]` is algebraically
identical, cheaper, and slightly more accurate — and differs in fp64 at ~5.6e-16, because subtracting
from 1.0 rounds. Every published number, including the Fast operating point's certification over 566
matched pairs, was measured with the old form baked into the folded heads. `interval_widths()`
therefore preserves it exactly. An improvement that silently invalidates "this refactor changed
nothing" is not an improvement.
