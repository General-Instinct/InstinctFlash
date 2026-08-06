# Architecture

**One Runtime. Many Checkpoints. Shared Infrastructure.**

This document describes how InstinctWM is organized and why. It is about the architecture, not the
order in which the pieces were built — for that see [HISTORY.md](HISTORY.md).

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
not produce different runtimes. There is exactly one InstinctWM runtime, and it serves every
compatible checkpoint by reading what that checkpoint declares about itself.

### There is no "Fast Runtime" and no "Quality Runtime"

This is worth stating as a prohibition because we nearly built one.

We operate two published operating points. **Quality** runs the teacher's full step schedule.
**Fast** runs a reduced one — currently 2 video / 4 action steps, certified non-inferior at a −0.05
margin over 566 matched pairs (p = 0.0085). It is tempting to describe these as two runtimes, ship
two entry points, and let each pick its own pass set.

That would have been wrong, and we have the measurement that proves it. Graph capture
(`P005`) is **profitable at Quality and unprofitable at Fast**: it saves roughly 17 ms per forward
pass but costs a fixed ~700 ms per cycle to capture, so it breaks even near 41 forwards per cycle.
Quality runs 75. Fast runs 6.

A "Fast Runtime" would have hardcoded that as two pass lists, and the *reason* would have been lost
in a branch. Instead the profitability test belongs where it can be evaluated from declared facts:

```
pass.admit(spec, deployment)   ->  is this legal, and is it profitable AT THIS OPERATING POINT?
```

Fast is not a checkpoint and not a runtime. It is a **descriptor delta** — the same weights, a
different declared step schedule — and the planner re-derives the pass set from it. Adding a third
operating point requires no new code path.

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
model = load("lingbot-va-posttrain-robotwin")     # states facts; touches nothing
plan  = Optimizer(tier_ceiling=Tier.BITEXACT).compile(model.spec())
print(plan.explain())                             # what fired, and why
server = plan.serve(model, port=29056)            # the first line that needs a GPU
```

If you need the weights loaded before the framework can tell you what it would do, it is a runtime
wearing a framework's clothes. Being able to answer "what would you do to this model, and why" on a
laptop is what makes the optimizer something you can argue with.

---

## Directory layout

```
instinctwm/
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
and is consumed here as a submodule — it is backbone-agnostic infrastructure, not an InstinctWM
research project.

---

## The optimization stack

Six layers, ordered by *what they change*.

| Layer | | Changes |
|:--|:--|:--|
| 1 | **MODEL** | what is computed — step reduction, distillation, latent compression |
| 2 | **GRAPH** | when work is issued — prefill extraction, graph capture, memory planning |
| 3 | **CACHE** | what is recomputed — KV reuse, cross-attention cache, episode cache |
| 4 | **ATTENTION** | how tokens mix — FlashAttention, hybrid and linear attention |
| 5 | **KERNEL** | how a kernel is written — operator fusion, Triton |
| 6 | **HARDWARE** | what it executes on — TensorRT, FP8/INT8, Jetson, Thor |

Layer 1 is the training side of the seam. Layers 2–6 are the runtime.

The layers are **not** a priority order, and treating them as one is how time gets wasted. Attention
is Layer 4 and looks like the obvious first move; on our profile it is 7% of GPU-busy time. At the
Fast operating point the picture is starker — the warm cost model is

```
FIXED 1164 ms  +  15.5 ms/forward        (R² = 0.994)
```

so at 6 forwards per cycle, **93% of latency is fixed overhead** and everything Layer 4 could win
lives inside the other 7%. Priority comes from the profile at the operating point, never from the
layer number.

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

- [CHECKPOINTS.md](CHECKPOINTS.md) — what a checkpoint declares, and why the training method is
  deliberately absent from it
- [HISTORY.md](HISTORY.md) — P001–P006 implementation milestones
- [eval/lingbot_va_robotwin/RESULTS.md](eval/lingbot_va_robotwin/RESULTS.md) — measured numbers and
  protocols
