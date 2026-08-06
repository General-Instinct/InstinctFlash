# Attention Backends

**One Runtime. Many Checkpoints. Shared Infrastructure.** — Layer 4.

A checkpoint must never depend on a specific attention implementation. This document defines the
abstraction that makes that true: what a checkpoint declares about its attention, what a backend
declares about itself, and how the planner intersects the two without either side learning anything
about the other.

> **Status: architecture only.** Interfaces, capability descriptors, legality, and the profitability
> model are complete and tested. **Selection raises `NotImplementedError` and no backend is installed
> anywhere.** Nothing in this document changes runtime behaviour. PR #2 is unrelated and stays
> separate — see [Relationship to PR #2](#relationship-to-pr-2).

---

## The distinction everything rests on

The list of things called "attention backends" contains two different kinds of thing, and conflating
them is the one mistake that would make this layer unsafe.

| | |
|:--|:--|
| **Same function, different implementation** | PyTorch SDPA, FlashAttention, FlashInfer, cuDNN SDPA |
| **Different function** | sliding window, Sana Hybrid, LongSana, linear attention, Mamba, DeltaNet |

The runtime may substitute freely within the first group. Softmax attention computed two ways gives
the same answer to within reduction order, so the choice is a legality-plus-profitability question
exactly like any other pass.

The runtime may **never** substitute across the second group. A checkpoint trained with full softmax
attention does not compute the same thing under a sliding window. No measured speedup makes that a
valid swap — it is a different model that happens to load. So:

> `AttentionSemantics` is a property of the **checkpoint**, declared by the adapter. A backend declares
> which semantics it *implements*. Selection intersects the two, and an empty intersection is a
> structural refusal, not a review comment.

Ring Attention is a third case worth naming: **same function, different distribution**. It is legal
only when the deployment has ranks to distribute over — a `DeploymentSpec` fact, which the checkpoint
author cannot know. Hence `Distribution` is its own axis.

This is also how Mamba/DeltaNet fit without special-casing. They declare `STATE_SPACE` semantics; a
checkpoint trained that way declares it too, and the same intersection does the work. Nothing about
the interface changes to admit them.

---

## The five questions, again

An attention backend is a pass with a narrower job, so it answers the five questions from
[`passes/contract.py`](instinctwm/passes/contract.py):

| | | Where |
|:--|:--|:--|
| 1 | **Detection** — the adapter publishes `ATTENTION` sites; the backend never goes looking | `site.py` |
| 2 | **Applicability** — declared envelope ∩ site facts, pure and GPU-free | `capabilities.py` |
| 3 | **Correctness** — tier *derived* from declared numerics, then verified | `capabilities.tier_ceiling()` |
| 4 | **Performance** — measured on the site's real shapes | `backend.measure()` |
| 5 | **Hardware** — `capabilities().hardware` | reuses `HardwareReq` |

A backend may not import a model module, may not decide when it is used, and may not change the
function being computed.

---

## How adapters expose attention sites

```
the adapter says   "layer 7 self-attention computes SOFTMAX_FULL over BSHD, ring-addressed KV,
                    40 heads of dim 128, seq_q 1 and seq_kv growing to 9792, bf16, in the video
                    phase which runs 25 forwards per cycle"

the backend says   "I implement SOFTMAX_FULL over BSHD with causal or windowed flags, head_dim a
                    multiple of 8 up to 256, and I cannot take a data-dependent dense mask"

the planner        intersects the two, and neither side has learned anything about the other
```

`SiteKind.ATTENTION` is the new site kind; `attention_site()` and `read_site()` are the constructor
and its typed inverse, so no backend indexes `attrs` by hand and no adapter spells a key differently.
A missing `semantics` raises rather than defaulting — defaulting it is the one failure that could let
a site be served by a backend computing a different function.

The adapter's rewrite `handle` rides on the site and is **opaque to every backend and to the
planner**. Only the executor passes it back to the adapter. That is what keeps backends free of model
symbols and adapters free of backend knowledge.

### Why `forwards_per_cycle` is on the site

Because profitability is meaningless without it. Attention is a `PER_STEP` cost, so what a backend can
win scales with how often the site is entered per control cycle — 75 at Quality, 6 at Fast. That
number comes from the checkpoint's declared `phases`, which is exactly why an operating point is a
descriptor delta and not a runtime mode. A site that omitted it would force the planner to guess the
denominator, and guessing it is how graph capture came to be enabled at an operating point where it is
a 2× regression.

---

## Legality — a pure predicate

`legality()` is a total function of (site facts, capabilities, deployment, device) with no GPU access.
"Which backends could serve this checkpoint" is answerable on a laptop before any weights exist. It
returns the *first* failing reason, because a plan explanation listing six is read by nobody.

Checked in order: semantics → tier ceiling → mask → layout → KV addressing → shapes → deployment →
hardware.

**Layout mismatch is not illegal.** BSHD and BHSD differ by one permute. Modelling the mismatch as
illegal would exclude torch SDPA from every BSHD site, which is wrong; modelling it as free would hide
a per-call copy of three tensors. So it is legal with a declared `layout_adaptation_us` and the
adaptation is recorded in `verdict.params` for the planner to charge. `PACKED_VARLEN` is not reachable
by a permute — it needs `cu_seqlens` — so adaptation is offered only between the two strided layouts.

### What this predicate says about our actual model

Running it against LingBot-VA's video self-attention site, **exactly 3 of 7 declared backends are
legal**:

| Backend | Verdict |
|:--|:--|
| `adapter_native` | legal, BITEXACT — the incumbent |
| `flash_attn` | legal, NUMERIC |
| `torch_sdpa` | legal, NUMERIC, needs a BSHD→BHSD transpose |
| `flashinfer` | **refused** — KV addressing `ring` unsupported (wants `paged`/`dense`) |
| `cudnn_sdpa` | **refused** — KV addressing `ring` unsupported |
| `ring_attn` | **refused** — needs `world_size ≥ 2` |
| `sana_hybrid` | **refused** — different semantics |

Two findings fell out of writing this down, and neither was expected:

**1. Ring-interval KV is the binding constraint at Layer 4.** It is what makes flash attention legal
*and* what excludes every paged-KV backend. A Layer 3 decision set the Layer 4 menu — so the honest
prerequisite for this layer already shipped as P003, and the paged-KV half of the ecosystem is
unavailable to us until KV addressing becomes a per-site choice rather than a global one.

**2. The site needs no mask at all.** I first wrote the example with `BLOCK_STATIC`. That is wrong:
under P003 the live KV set is the contiguous interval `[start, start+count)`, so the site attends over
a **slice** — `MaskKind.NONE`. The mask only ever existed to select the live set out of a padded
buffer, and the stock path built it with `mask.nonzero()` per layer per forward:
`DENSE_DATA_DEPENDENT`, which rules out every flash-family backend and is also what raised
`cudaErrorStreamCaptureInvalidated` under capture. One addressing change moved this site from "no fast
attention backend is legal" to "almost all are".

---

## Profitability — where the interesting failure lives

The naive model is `saving = forwards_per_cycle × Δ per-forward`. It is wrong in three ways we have
already been bitten by, and the model makes all three explicit rather than leaving them to a reviewer.

### 1. The operating point sets the denominator

Graph capture is profitable at Quality (75 forwards/cycle) and a regression at Fast (6): it trades
~17 ms/forward against ~700 ms/cycle of fixed capture cost, breaking even near 41 forwards. Any
backend with a `host_setup_us` term has the same shape. **FlashInfer is the likeliest in this set to
invert**, because its per-forward host `plan()` is amortised over 75 forwards and is not amortised
over 6.

### 2. Attention's share is small, and shrinks

Attention is 7% of GPU-busy time at Quality. The measured warm cost model at Fast is
`FIXED 1164 ms + 15.5 ms/forward` (R² = 0.994), so at 6 forwards/cycle **93% of latency is fixed
overhead no attention kernel touches**. Attention is ≈2.2 ms per forward, so:

| Operating point | attention/cycle | a backend that **halves** it wins |
|:--|--:|--:|
| Quality, 75 forwards | ~163 ms of 2325 ms | ~81 ms → **3.5%** |
| Fast, 6 forwards | ~13 ms of 1257 ms | ~7 ms → **0.5%** |

Both are real. Neither is what the layer's reputation suggests. This arithmetic belongs in the planner
so it can stop a kernel being written, rather than in a retrospective.

### 3. A backend can be a net loss while being strictly faster at attention

If it is not `capture_safe`, selecting it forfeits graph capture — 1.205× at Quality. On a 2325 ms
cycle that is a **396 ms penalty**, against an attention budget of 163 ms in total. So a
capture-hostile attention backend **cannot win at Quality even if it made attention free.**

That is not a caveat, it is the answer, and it is why `plan_penalty_ms()` is part of the model rather
than a footnote. A per-kernel microbenchmark reports this backend as a success. The plan-level term is
the only thing that catches it.

---

## Planner integration

Attention backend choice is a `Rewrite` on an `ATTENTION` site, applied by the executor through the
adapter's handle. It enters the existing pipeline with no new machinery:

```
adapter.sites(SiteKind.ATTENTION)          ->  [Site, ...]         WHERE
REGISTRY.candidates(**read_site(site))     ->  [Candidate, ...]    legal set + every refusal reason
   (ranking: not implemented)              ->  one backend         WHAT
executor.apply(adapter, Rewrite(...))      ->  installed, gated    ACT
```

`candidates()` deliberately **retains** illegal backends with their reasons, because a backend
silently missing from a list is indistinguishable from one that was never written — the same reason
`plan.explain()` reports the passes it declined.

The incumbent (`adapter_native`) is always a candidate. It is trivially legal, bit-exact by
construction, and the only honest baseline for `max_abs_delta`. Its presence means `candidates()` is
never empty, selection never has to invent a fallback, and *"we measured four backends and kept the
original"* is an expressible outcome rather than a bug.

### Tiers

`tier_ceiling()` is **derived** from declared numerics, never claimed:

| | Tier | Why |
|:--|:--|:--|
| `adapter_native` | BITEXACT | identity substitution |
| `flash_attn`, `torch_sdpa`, `flashinfer`, `cudnn_sdpa` | NUMERIC | online softmax / dispatch changes reduction order |
| any non-deterministic backend | BEHAVIORAL | `max|Δ| = 0` is unavailable, so it needs paired non-inferiority |

This has a consequence worth stating plainly: **every real attention backend swap costs us
bit-exactness.** Our whole gating regime for Layers 2–3 is `max |Δ action| = 0`. Layer 4 cannot be
gated that way, so any attention change must go through the paired non-inferiority protocol —
identical seeds, declared margin, exact McNemar — the same regime the Fast operating point was
certified under. Budget the episodes before starting, not after.

---

## What is deliberately not built

| | Why |
|:--|:--|
| `select()` — ranking | needs measured numbers on an idle fleet; raises `NotImplementedError` |
| `measure()`, `bind()` on every backend | would change runtime behaviour |
| adapter wiring — `lingbot.sites()` is unchanged | the example site is a worked example, not a graft |
| any kernel | this is an abstraction, not an implementation |

The declared envelopes in [`reference.py`](instinctwm/backends/attention/reference.py) answer
`capabilities()` honestly and raise from everything else. That is the deliverable: the envelopes are
what the planner reasons over, and declaring seven of them is what proves the vocabulary
discriminates. A capability model that accepts everything would be decoration.

## Relationship to PR #2

PR #2 is a separate change and stays that way. It is a concrete attention integration; this is the
abstraction that any such integration should arrive through. Nothing here depends on it, it is not a
prerequisite for it, and merging it is a separate decision made on its own measurements.

When attention work does resume, the order this design implies is:

1. Publish real `ATTENTION` sites from the LingBot adapter — replaces the example, no behaviour change
2. Implement `measure()` for `adapter_native` and one challenger, and get a number on an idle fleet
3. Only then implement `select()`, with ties going to the incumbent

Step 2 is where the 3.5%-at-Quality / 0.5%-at-Fast arithmetic gets tested against reality. It may end
the layer, and that would be a good outcome to reach cheaply.

## Further reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — the two seams this layer sits on
- [CHECKPOINTS.md](CHECKPOINTS.md) — why a checkpoint declares capabilities and never a method
- [`tests/test_attention_backend.py`](tests/test_attention_backend.py) — the refusals, as executable
  assertions
