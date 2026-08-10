# Architecture

**One Runtime. Many Checkpoints. Shared Infrastructure.**

This document describes how InstinctWM is organized and why. It describes the code on `main`; for how
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
not produce different runtimes. There is exactly one InstinctWM runtime, and it serves every
compatible checkpoint by reading what that checkpoint declares about itself.

This is **enforced, not aspirational**. `tests/test_runtime_boundary.py` parses every module under
`runtime/`, `planners/`, `executors/`, `descriptors/` and `backends/` and fails if any of them can
reach `instinct_pdd` or `instinctwm.train` — transitively, and including function-local imports,
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
- [`instinctwm/passes/`](instinctwm/passes/) — every pass, with its legality argument in its docstring
- [`instinctwm/verify/released.py`](instinctwm/verify/released.py) — the release ledger: what shipped,
  at what tier, on what evidence
- [eval/lingbot_va_robotwin/RESULTS.md](eval/lingbot_va_robotwin/RESULTS.md) — measured numbers and
  protocols

## Shipped configuration

`instinctwm.verify.released.shipped_configuration()` is the single source of truth. The launch
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
