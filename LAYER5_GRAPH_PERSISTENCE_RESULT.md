# Graph persistence: a correct design that does not pay — frozen

**Status: NEGATIVE RESULT. Frozen 2026-08-08. Not shipped, not to be resumed without new evidence.**

[LAYER5_GRAPH_PERSISTENCE.md](LAYER5_GRAPH_PERSISTENCE.md) is the design. This document is the outcome.
The design was implemented to its stated scope, every correctness gate passed, and it was rejected by the
latency gate. Both halves matter: the design's central claim turned out to be **true**, and the
optimization it justified turned out to be **worthless**. Those are separable, and keeping them separate is
the point of writing this down.

## What was claimed, and what happened to each claim

| claim from the design | verdict |
|:--|:--|
| `start` changes only *addressing*, `count` changes *shape* | **confirmed** |
| the write offset can move into a device-resident buffer with byte-identical writes | **confirmed** — 0 differing words, every pool shape / key size / offset |
| exact KV semantics survive without recapture | **confirmed** — `max \|Δ action\| = 0` over 45 cycles spanning saturation |
| removing `start` from the key substantially reduces recapture | **confirmed** — 270 → 238 captures over 45 cycles |
| a fully-replayed cycle becomes device-bound at 196 ms, i.e. **1.72×** | **REFUTED at the cycle level** — see below |

## The measurement that rejected it

Episode mode, post-saturation steady state, ABBA-ordered (base, treat, treat, base), 2V/4A, P007
conv-layout installed in every arm so only graph behaviour differs.

| variant | cycle | vs shipped |
|:--|--:|--:|
| **capture OFF — what ships today** | **351.4 ms** | — |
| capture ON, no plan buffer | 936.1 ms | 2.66× slower |
| capture ON + plan buffer | 503.5 ms | **1.43× slower** |

Drift on the repeated base arms 0.5% / 2.0%, so the ordering is doing its job and the gaps are far outside
it.

The plan buffer recovers **432 of the 585 ms** capture penalty. That is a large, real, correctly-predicted
improvement to graph capture. It is also irrelevant, because graph capture starts 585 ms in deficit and
ends 152 ms in deficit. **Fixing the thing that was wrong with the mechanism did not make the mechanism
profitable.**

## Why the 1.72× prediction failed

The prediction came from the critical-path model: 94.3% of dispatcher operations occur inside the region a
graph replaces, the cycle is host-bound, therefore replay removes ~319 ms of a 338 ms host chain and the
cycle falls to the 196 ms device floor.

Two explanations were open at the time of the commit, and they were distinguishable by one measurement —
the dispatcher-op count with capture on and the plan buffer engaged:

- **(a) surviving recaptures swamp the gain.** 5.3 captures/cycle at ~111 ms each is 588 ms — by itself
  enough to account for the entire deficit.
- **(b) replay does not remove the host cost attributed to it**, in which case the ~3.2 µs/op model that
  underwrites all of Layer 5 and Layer 6 is wrong.

**(a) is the explanation.** The arithmetic settles it without a further run: 5.3 × 111 ms = 588 ms of
capture cost per cycle against a 351 ms baseline. A mechanism whose *setup* cost exceeds the entire cycle it
is optimizing cannot be net-positive no matter how well replay works, and the plan buffer reduced the
*number* of captures (270 → 238, 12%) without reducing the *cost* of the ones that remain. The recaptures
that survive are the pre-saturation cycles and the shape-driven `count` transitions, and those are intrinsic
under bit-exactness — the design said so and was right.

So the host-dispatch model is **not** impeached by this result. It was never given a cycle in which its
prediction could be observed: no cycle in the ABBA run was fully replayed with zero captures. That is a
weaker statement than "the model is confirmed", and it is the honest one.

### The consequence for the model

The model remains **retrodictive only**. It has explained four outcomes after the fact (P007's 1.405% from
~56,600 ops, the cast hoist's 0.66% from 1,740, RoPE and fused QKV's ~0% from 0) and has now made two
prospective predictions: fused QKV (1.9% predicted, 0.2% slower) and graph persistence (1.72× predicted,
1.43× slower). **Prospectively it is 0 for 2.** Both failures share a shape: the model priced the ops
removed and ignored the cost of the mechanism that removed them. Layer 6 must price both.

## The bug the gate caught

Worth preserving because the class recurs. Plan buffers are keyed `(cache_name, key_size)` — their *length*
differs, 240 slots for a video write and 32 for an action write — but `head` depends only on ring state and
not on `key_size`. I refreshed only the buffer belonging to the commit that had just run, leaving the other
holding the previous cycle's head, so the next write of the other size landed on live slots.

The gate localised it exactly: cycles 0–35 exact, **36–44 wrong, max |Δ| 0.453** — error appearing precisely
where the plan path engages. Fixed by refreshing every buffer for the cache unconditionally after the
commit, and by initialising a lazily-created buffer to the current head rather than to zero.

**One buffer refreshed is not "the buffer refreshed."** A gate that ran only post-saturation would have
caught this; a gate that ran only pre-saturation would have passed a wrong implementation. Spanning the
transition is what made it visible.

## What is left in the tree, and in what state

| artefact | state |
|:--|:--|
| `passes/lingbot/ring_kv.py` | plan buffer is **opt-in**, `_iwm_use_plan_buffer` defaults **False**; byte-identical to v1.0.0 when off, so the frozen pass is unchanged |
| `passes/lingbot/graph_capture.py` | key collapses to `("saturated", total)` when `count >= total` **and** the plan buffer is on |
| `passes/lingbot/persistent_graph.py` | opt-in pass, verdict NOT SHIPPED recorded in its docstring |
| `eval/.../probe_persistent_graph.py` | the five gates, runnable |
| `verify/released.py` | **no entry** — this was never released |

Nothing defaults on. The scope held: pre-saturation untouched, the full-pool read path untouched, no `count`
bucketing, no NUMERIC behaviour.

## Two harness facts that cost time

- **Both graph arms cannot be built in one process.** `graph_capture` rewrites
  `WanTransformer3DModel.forward` by source and the class is shared, so the second install searches an
  already-patched forward and raises. One arm per process, always.
- The saturation check read `total` before the ring dict existed, comparing against a sentinel.

## A second, independent reason: capture does not survive a 50-task run

The 1.43× is a latency argument. There is an operational one, from the pre-reorg `--graph-blocks`
fleet runs recorded in PR #2, and it is worth separating because it does not depend on the cycle time
at all.

**Evicting a captured graph does not return its private memory pool.** Over a 50-task run the teacher
servers climbed from 24 GB to the 80 GB ceiling and all 8 OOMed. `IWM_MAX_GRAPHS` was added to bound
the held set and the experiment falsified the idea:

```
cap=32 -> gpu0: captures=523 replays=20881 held=32 evicted=461 fallbacks=1   (all 8 still OOMed)
```

461 evictions for 523 captures. Capping the *held* set does not help when the leak is in *eviction* —
it only raises the eviction count. One A100 entered `GPU requires reset` and was lost for the session.

**The contrast that matters, and it is a memory claim, not a latency one:** the `--ring-attention`
arm, which holds 6 graphs and evicts none, ran the same workload at a flat **41–42 GB with zero
fallbacks**. That is the only surviving argument for the ring-attention branch, and it should never be
cited as support for its 1.32×/episode latency figure, which is stale (see
[SALVAGE_PR2.md](SALVAGE_PR2.md)).

So capture fails twice over, for unrelated reasons: it is slower at this operating point, and a
non-converging capture key cannot finish a 50-task evaluation on this box. *(Fleet numbers collected
pre-reorganization on 8×A100; the mechanism claim stands, the hardware context is dated.)*

## Why this is frozen rather than parked

Resuming would require one of: a mechanism whose per-capture cost is not ~111 ms, or an operating point
where captures are amortised over enough cycles to matter. Neither is in view at Fast — 2V/4A runs 10
forwards per cycle and a ~53-cycle episode, so there is no long tail to amortise against. At Quality
(25V/50A, 79 forwards/cycle) the arithmetic differs and the question could legitimately reopen; that is not
the operating point being optimized.

**The generalisable lesson, and it redirected the project:** graph capture and `torch.compile` both attack
host dispatch by *building a persistent artefact*, and the artefact's construction cost is charged against
the same budget as the dispatch it eliminates. Layer 6 therefore prefers transformations that **remove**
dispatch outright — a view built once and held, a composite op replaced by its leaf — over transformations
that **replace** dispatch with a compiled or captured object. See [LAYER6.md](LAYER6.md).

## Further reading

- [LAYER5_GRAPH_PERSISTENCE.md](LAYER5_GRAPH_PERSISTENCE.md) — the design, unamended; read it for the
  field-by-field analysis of what forces a capture
- [LAYER5_CRITICAL_PATH.md](LAYER5_CRITICAL_PATH.md) — where the 1.72× came from
- [LAYER6.md](LAYER6.md) — the direction taken instead
- commit `db8a86c` — the implementation and its gates
