<div align="center">

<img src="assets/instinctwm_2.png" alt="InstinctWM" width="360"/>

### One runtime for robot world-action models

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

[Architecture](ARCHITECTURE.md) •
[Checkpoints](CHECKPOINTS.md) •
[Quick Start](#quick-start) •
[Results](eval/lingbot_va_robotwin/RESULTS.md)

</div>

---

InstinctWM is an open platform for **world-action models** — robot policies that predict what
happens next *and* what to do about it in one model.

**If you have trained a checkpoint, this runtime will serve it.** You publish weights plus a short
declaration of what those weights *are*; the runtime reads the declaration, works out which
optimizations are legal, applies them, and reports what each one cost. You do not publish your
recipe, your dataset, or your training code — and there is no place in the runtime where they could
be read even if you did.

## One Runtime. Many Checkpoints. Shared Infrastructure.

Training recipes — PDD, DMD2, LCM, DreamZero — produce different **checkpoints**. They do not produce
different runtimes. There is one InstinctWM runtime, and it serves every compatible checkpoint by
reading what that checkpoint declares about itself.

There is no "fast runtime" and no "quality runtime". An operating point is a descriptor delta, not a
code path, and **the runtime never branches on the training method** — it consumes capabilities, so a
recipe that does not exist yet needs no change here. [How, and why →](ARCHITECTURE.md)

> **Status: early.** The evaluation pipeline, the measurement tooling, and the graph and cache
> layers are real and reproducible. The kernel and hardware layers are designed and being built.
> Every number here is measured on our own hardware with the scripts in [`eval/`](eval/).

---

## What's New 🔥

- **[2026/08]** Repository reorganized around the architecture rather than the build order: one
  directory per concept, [P001–P006 moved out of the public narrative](HISTORY.md), and the
  checkpoint contract written down. [Architecture →](ARCHITECTURE.md)
- **[2026/08]** **Fast operating point certified** at 2 video / 4 action steps: 566 matched pairs,
  0.910 vs a 0.929 teacher, non-inferior at a −0.05 margin (p = 0.0085). It is a descriptor delta,
  not a second checkpoint and not a second runtime.
- **[2026/08]** Graph capture **inverts between operating points** — profitable at Quality, a
  regression at Fast, breaking even near 41 forwards/cycle. The strongest argument yet for one
  runtime that computes profitability instead of two that hardcode it. [Why →](CHECKPOINTS.md)
- **[2026/08]** Step-allocation response surface mapped over 7 operating points, 3500 paired
  episodes, 50 tasks. Both streams tolerate 2 steps and both cliff at 1: **79 forwards/cycle down to
  4–6 is nearly free with no retraining at all.**
- **[2026/08]** **3.38× bit-exact** on LingBot-VA in episode mode: 9585 → 2832 ms per control
  cycle, at `max |Δ action| = 0`. [Protocol and full chain →](eval/lingbot_va_robotwin/RESULTS.md)
- **[2026/08]** Remaining cost profiled. LingBot-VA *was* launch-bound; after graph capture it is
  GPU-bound again, which re-ranks every layer below.
- **[2026/08]** Canonical RoboTwin 2.0 baseline: **91.6% macro**, 50 tasks, 2500 episodes.
- **[2026/07]** Evaluation pipeline for LingBot-VA on RoboTwin 2.0, including a prompt-parity gate
  that closes a silent train/serve mismatch.

---

## Optimization Stack

Six layers, ordered by *what they change*. **Layer 1 is training; Layers 2-6 are the runtime.**
Layer 1 produces checkpoints. Layers 2-6 produce plans. The layer number is not a priority order --
attention is Layer 4 and looks like the obvious first move, but it measures 7% of GPU-busy time.

| **MODEL** | **GRAPH** | **CACHE** | **ATTENTION** | **KERNEL** | **HARDWARE** |
|:--|:--|:--|:--|:--|:--|
| *what it computes* | *when work is issued* | *what is recomputed* | *how tokens mix* | *how a kernel is written* | *what it executes on* |
| Step Reduction | **Prefill Extraction** | **KV Reuse** | FlashAttention | **Operator Fusion** | TensorRT |
| Parallel Decoding Distillation | **Execution Graph Rewrite** | **Cross-Attention Cache** | FlashInfer | *Fused AdaLN* | FP8 |
| rCM | **CUDA Graph Capture** | **Episode Cache** | Sana-Video Hybrid | *Triton Kernels* | INT8 |
| sCM | **Persistent State Analysis** | TeaCache | LongSana | Fused CFG | INT4 |
| DMD2 | **Static Memory Planning** | XCache | Linear Attention | Fused Scheduler | Jetson |
| DreamZero-Flash | Stream Overlap | SeaCache | Mamba / DeltaNet | Fused VAE | Thor |
| Latent Compression | ~~CFG Parallelization~~ | Window Cache | | Paged KV Kernels | Snapdragon |
| DC-AE / DC-VE | ~~Whole-Cycle Capture~~ | Energy-based Cache | | | |

**bold** shipped, gated bit-exact, measured end to end · *italic* implemented but not on the shipped
path · ~~struck~~ rejected *by measurement*, kept so it is not re-proposed · plain designed only.

**All 3.38× of measured speedup comes from GRAPH and CACHE.** The other four layers are either
unbuilt or deprioritized *by profile*. Priority comes from the measurement at the operating point:
a direct phase decomposition at Fast attributes 99% of the cycle to two components — transformer
forwards (81%) and the keyframe VAE encode (18%) — so those are the only two things worth optimizing.
An earlier regression-derived cost model claimed 93% fixed overhead; it was wrong, and
[PROFILE.md](PROFILE.md) says why. Per-pass measurements, protocols, and the
full chain are in [Results](eval/lingbot_va_robotwin/RESULTS.md).

---

## Serve a checkpoint

```python
from instinctwm.descriptors.package import from_pretrained

ckpt = from_pretrained("example-org/wm-blockheads-2v4a")   # Hub id, or a local path
print(ckpt.capabilities())
# frozenset({'servable', 'backbone:wan_va',
#            'output_projection:per_interval_velocity_heads',
#            'output_projection:foldable',
#            'guidance:video=cfg', 'guidance:action=positive_only'})
```

Those tokens are **the only thing the planner is told about your checkpoint.** A pass is admitted when
the capabilities it requires are declared, and skipped when they are not. A pass that requires nothing
composes with every checkpoint — which is the default, and the reason adding a checkpoint does not
mean re-testing the optimization stack.

## Publish a checkpoint

```
my-checkpoint/
  instinctwm.json          REQUIRED  the declaration — two namespaces, below
  config.json              REQUIRED  your backbone's own config
  model.safetensors        REQUIRED  or a sharded set + model.safetensors.index.json
  README.md                optional  model card
```

The minimal `execution` block is three fields — `model_id`, `backbone`, `servable`. Everything else
has a defensible default. A complete worked example is in
[`examples/checkpoint/wm-blockheads-2v4a/`](examples/checkpoint/wm-blockheads-2v4a/).

Check it before you push:

```bash
python -m instinctwm.descriptors.package my-checkpoint/
#   servable package: YES
#   publishable without training internals: YES
```

### Two namespaces, and the runtime only ever sees one

```jsonc
{
  "instinctwm_schema": 1,
  "execution":  { /* what the runtime may read: capabilities and structure */ },
  "provenance": { /* how it was trained. FOR HUMANS. Never returned to the runtime. */ }
}
```

`load_declaration()` parses `provenance` only to drop it. Keys like `recipe`, `teacher`, `dataset`,
`optimizer` or `coverage_gate_pass` are **rejected at load** if they appear under `execution`, with an
error telling you where to move them.

**You can delete `provenance` entirely and the checkpoint still serves.** `publishability()` verifies
exactly that, by stripping the block and re-loading. If it fails, something the runtime needs is in
the wrong namespace — which is the mistake the split exists to catch.

## How a new recipe plugs in

Suppose you distil with DMD2, LCM, consistency training, or something not yet written. **No
infrastructure changes.** You declare what your weights *are*, not how they were made:

```jsonc
"output_projection": {
  "kind": "per_interval_velocity_heads",
  "n_intervals": 8, "block": 4,
  "velocity_convention": "sigma_descending",
  "foldable": true
}
```

That says: *L linear heads per block over an N-interval grid, emitting velocity in the σ-descending
convention, foldable into one affine map at load.* Every one of those is checkable by looking at the
weights. A DMD2 checkpoint and an LCM checkpoint that both produce that shape declare the same four
values and take the same code path.

If your recipe produces a genuinely new output structure, you add a **capability** — a new
`output_projection.kind` and the code that consumes it — and every existing pass keeps working, because
none of them asked what recipe you used. What you never add is a branch on the method name.

> **Why the method name is poison.** `if recipe == "pdd"` is a runtime that supports one recipe and
> claims to support many. It cannot be tested against a recipe that does not exist yet, and the next
> author has to send a PR to the runtime instead of publishing a checkpoint.
> `tests/test_checkpoint_platform.py` asserts the property directly: two checkpoints with identical
> `execution` and opposite `provenance` produce a **byte-identical plan**.

## Shipped Configuration

**Single source of truth: `instinctwm.verify.released.shipped_configuration()`.** The launch scripts,
this table and `serve_variant.py` all derive from it, and `tests/test_shipped_config.py` fails if they
drift apart. Add a flag there, not in four places.

```
--no-fsdp --no-empty-cache --no-debug-dump --conditioning-prefill --ring-kv --conv-layout
```

Served chain tier: **NUMERIC** — the weakest link, not the best member. P007 is NUMERIC, so
the chain is *not* bit-exact end to end; that is what its 555-episode non-inferiority certificate is
for.

| pass | name | tier | disposition | flags |
|:--|:--|:--|:--|:--|
| P001 | `substrate_elision` | BITEXACT | **shipped** | `--no-fsdp --no-empty-cache --no-debug-dump` |
| P002 | `conditioning_prefill` | BITEXACT | **shipped** | `--conditioning-prefill` |
| P004 | `hoist_invariant_casts` | BITEXACT | available | `--hoist-casts` |
| P005 | `graph_block_stack` | BITEXACT | **not recommended** | `--graph-blocks` |
| P006 | `stable_state_pools` | BITEXACT | **not recommended** | `--stable-pools` |
| P003 | `ring_kv_addressing` | BITEXACT | **shipped** | `--ring-kv` |
| P007 | `conv_layout_ndhwc` | NUMERIC | **shipped** | `--conv-layout` |

Historical material is split in [HISTORICAL.md](HISTORICAL.md): **negative results** (tested and refuted) versus **archived implementations** (correct, not enabled).

**Released is not the same as recommended.** `RELEASED` is a ledger of what shipped, at what tier, on
what evidence, at the time — it is frozen and never rewritten. `DISPOSITIONS` states what should run
*today*, with the measurement behind it. P005 and P006 are the case that forced the distinction: both
were released and verified, and at the current operating point CUDA graph capture measures **1.43×
slower** than not capturing ([LAYER5_GRAPH_PERSISTENCE_RESULT.md](LAYER5_GRAPH_PERSISTENCE_RESULT.md)),
so they are kept in the ledger and marked not recommended.

## Quick Start

```bash
git clone https://github.com/general-instinct/InstinctWM && cd InstinctWM
pip install -e .                # analysis only: no torch, no GPU required
pip install -e ".[runtime]"     # to actually apply a plan and serve
```

Deciding *which* optimizations are legal is dependency-free by design, so you can inspect a plan on
a laptop. Only applying one needs torch.

```python
from instinctwm import load, Optimizer, Tier

model  = load("lingbot-va-posttrain-robotwin")
plan   = Optimizer(tier_ceiling=Tier.BITEXACT).compile(model.spec())
print(plan.explain())                    # what fired, and why
server = plan.serve(model, port=29056)   # deploy
```

`plan.explain()` reports every decision, including the ones it declined:

```
InstinctWM plan for lingbot-va-posttrain-robotwin      plan tier: BITEXACT

  APPLY  fsdp_elision            [BITEXACT] world_size=1, so every FSDP all-gather
                                            is identity while still paying a
                                            flat-param copy and stream sync
  APPLY  allocator_churn_elision [BITEXACT] closed-loop serving has a stable
                                            working set
  ...
```

---

## Operating Points

Two published points, same weights, same runtime. Improvements flow Quality → Fast.

| | steps | forwards/cycle | success | latency |
|:--|:--|--:|:--|:--|
| **Quality** | 25 video / 50 action | 75 | 0.929 | reference |
| **Fast** | 2 video / 4 action | 6 | 0.910, non-inferior at −0.05 (p = 0.0085) | ~2.3× faster warm |

Fast is **not** a distilled checkpoint and **not** a second runtime — it is the same weights with a
different declared step schedule, and the planner re-derives which passes are profitable from it.
That is why graph capture is enabled at Quality and disabled at Fast without a branch anywhere in
the serving path. [The mechanism →](CHECKPOINTS.md)

---

## Supported Models

| model | status |
|:--|:--|
| **LingBot-VA** | Full runtime support. Primary optimization target and evaluation benchmark: 3.38× bit-exact in episode mode, with multi-episode bit-exactness, reset isolation, and pointer-stability gates. |
| **Cosmos3-Edge** | Runtime support, measured at the shipped geometry against a fresh upstream checkout: 5,320 ops, 624 external reads, **0 unnamed**, capturable. GraphExecutor is **2.33× bit-exact** on the control step (2.30–2.36× over five runs), with an EagerExecutor control at 1.002× showing the engine's own dispatch costs nothing. Real cuDNN attention on both paths, no shim. **No accuracy claim** — random weights, no checkpoint. [Results](eval/cosmos3_edge/RESULTS.md) |

Additional world-action models will be added over time. We would rather have two models fully
verified than six partly claimed.

---

## Documentation

**Start here:**

| | |
|:--|:--|
| [**Architecture**](ARCHITECTURE.md) | How the repository is organized: training vs runtime, the two seams, and why there is only one runtime |
| [**Checkpoints**](CHECKPOINTS.md) | What a checkpoint declares, the `instinctwm.json` schema, and why the training method is deliberately absent from it |
| [**Attention Backends**](ATTENTION.md) | Layer 4: how a checkpoint avoids depending on any attention implementation. Architecture only — selection is not implemented |
| [**Layer 5**](LAYER5.md) | Backend and layout selection before kernels — the flow every Layer 5 backend follows, with P007 as the reference implementation. Next decision: [LAYER5_NEXT.md](LAYER5_NEXT.md) |
| [**Graph persistence**](LAYER5_GRAPH_PERSISTENCE.md) | The design that follows: one captured graph surviving ring advancement, and why `start` is the only blocker after cycle 36 |
| [**Critical path**](LAYER5_CRITICAL_PATH.md) | Why three region-level wins moved nothing: the cycle is host-dispatch bound, and 66% of the path launches no kernel |
| [**Profile**](PROFILE.md) | Where the Fast cycle actually goes — 81% forwards, 18% keyframe VAE encode — and the retraction of the cost model that said otherwise |
| [**Audit**](AUDIT.md) | Where the repository does and does not yet live up to *One Runtime, Many Checkpoints, Zero Runtime Forks* — findings, renames, migration plan |
| [Results](eval/lingbot_va_robotwin/RESULTS.md) | Measured chain, per-pass numbers, protocols |
| [Evaluation harness](eval/lingbot_va_robotwin/README.md) | Running the RoboTwin pipeline, and seven ways it can silently produce a plausible wrong number |

**Source layout** — one directory per concept, described in [ARCHITECTURE.md](ARCHITECTURE.md):

| | |
|:--|:--|
| [`descriptors/`](instinctwm/descriptors/) | What a checkpoint declares — capabilities, not recipes |
| [`adapters/`](instinctwm/adapters/) | Where things are, per backbone: publish sites |
| [`passes/`](instinctwm/passes/) | What to do there: consume sites, return rewrites |
| [`planners/`](instinctwm/planners/) | Which passes are legal and profitable — no GPU required |
| [`executors/`](instinctwm/executors/) | Apply a plan to a live server |
| [`backends/`](instinctwm/backends/) | Kernels, chosen by measurement |
| [`runtime/`](instinctwm/runtime/) | Load, install, serve |
| [`train/`](instinctwm/train/) | Layer 1 — recipes that make checkpoints |

Implementation milestones P001–P006 are in [HISTORY.md](HISTORY.md). They are how the project was
built, not how it is organized — [ARCHITECTURE.md](ARCHITECTURE.md) is the one to read first.

---

## Evaluation

Accuracy is gated separately from speed, and neither gate trusts the other.

- **Bit-exact optimizations** are gated at `max |Δ action| = 0` on paired seeded rollouts.
- **Behavior-changing optimizations** are gated by paired non-inferiority on pinned seeds, with the
  margin declared *before* the run, exact McNemar on discordant pairs, and a per-task table.

```bash
cd eval/lingbot_va_robotwin && source ./env.sh

IWM_FA_SHIM=1 ./servers.sh start 8         # one policy server per GPU
$IWM_SERVER_PY check_prompt_parity.py ...  # correctness gate — run it first
./run_eval.sh myrun 50 adjust_bottle ...   # fan tasks across the fleet
$IWM_CLIENT_PY aggregate.py $IWM_RESULT_DIR/myrun --expect-episodes 50
```

`aggregate.py` prints `REPORTABLE: NO` and refuses to give a number when a run is incomplete or
internally inconsistent. That is deliberate: **a number you cannot defend is worse than no number.**

The gate that earns its keep most often is `check_prompt_parity.py`. LingBot-VA never runs T5 during
training — it reads a precomputed embedding baked into the dataset, while the server recomputes it
live, and nothing upstream checked that the two agree. An earlier project lost a 22.7-hour run to
exactly that class of bug.

---

## Examples

| | |
|:--|:--|
| [`probe_latency.py`](eval/lingbot_va_robotwin/probe_latency.py) | Short-horizon latency A/B across cumulative pass configurations |
| [`probe_episode.py`](eval/lingbot_va_robotwin/probe_episode.py) | Episode mode — consecutive cycles, one reset. The reporting standard |
| [`probe_bitexact.py`](eval/lingbot_va_robotwin/probe_bitexact.py) | Paired seeded rollouts at zero action delta |
| [`probe_cfg_liveness.py`](eval/lingbot_va_robotwin/probe_cfg_liveness.py) | Two-axis liveness test that ruled out CFG elision |
| [`serve_variant.py`](eval/lingbot_va_robotwin/serve_variant.py) | A/B policy server; every variant applies the same installers production does |
| [`certify_run.py`](eval/lingbot_va_robotwin/certify_run.py) | Paired non-inferiority certificate from per-episode JSONL |

---

## Citation

```bibtex
@software{instinctwm2026,
  title  = {InstinctWM: One runtime for robot world-action models},
  author = {General Instinct},
  year   = {2026},
  url    = {https://github.com/general-instinct/InstinctWM}
}
```

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

AGPLv3 is a network-copyleft licence: if you run a modified InstinctWM as a service, you must offer
the modified source to its users. Third-party components keep their own terms (vLLM-Omni is
Apache-2.0, compatible in this direction).
