# Is there a family of scope-lifting casts? **No.**

Measured 2026-08-07, 2V/4A warm past ring saturation, P007 applied, 20 forwards over 2 cycles.
Probe: [`probe_cast_lifetime.py`](eval/lingbot_va_robotwin/probe_cast_lifetime.py).

**Conclusion: 292 of 6,101 cast calls per cycle are removable — 4.8% — and 290 of those are a single
callsite.** P004's weight-cast family does not have an activation-cast counterpart. Candidate 4 is one
bug, not the first instance of an abstraction, and it should be fixed as one bug.

---

## The measurement that decides it

Every `_to_copy` callsite at ≥95% coverage, with the coarsest scope at which its input value is
constant. "Calls/fwd" is how often it runs; "distinct value/fwd" is how many *different* values it
actually casts.

| callsite | calls/fwd | distinct storage | distinct **value** | current scope | minimal legal scope | calls removed /cycle | tier |
|:--|--:|--:|--:|:--|:--|--:|:--|
| `model.py:524` `temb.float()` | 30 | 1.0 | **1.0** | LAYER | **STEP** | **290** | BITEXACT |
| `model.py:565` | 60 | 4.0 | 60.0 | LAYER | LAYER (minimal) | 0 | — |
| `model.py:548` | 60 | 4.2 | 60.0 | LAYER | LAYER (minimal) | 0 | — |
| `ring_kv.py:154` | 60 | **1.2** | 60.0 | LAYER | LAYER (minimal) | 0 | — |
| `ring_kv.py:153` | 60 | 4.5 | 60.0 | LAYER | LAYER (minimal) | 0 | — |
| `model.py:564` | 30 | 1.6 | 30.0 | LAYER | LAYER (minimal) | 0 | — |
| `model.py:560` | 30 | 1.6 | 30.0 | LAYER | LAYER (minimal) | 0 | — |
| `model.py:558` | 30 | 1.6 | 30.0 | LAYER | LAYER (minimal) | 0 | — |
| `model.py:544` | 30 | 1.9 | 30.0 | LAYER | LAYER (minimal) | 0 | — |
| `model.py:543` | 30 | 3.2 | 30.0 | LAYER | LAYER (minimal) | 0 | — |
| `model.py:536` | 30 | 1.8 | 30.0 | LAYER | LAYER (minimal) | 0 | — |
| `model.py:534` | 30 | 3.2 | 30.0 | LAYER | LAYER (minimal) | 0 | — |
| `ring_kv.py:170`, `:171` | 30 | 30.0 | 30.0 | LAYER | LAYER (minimal) | 0 | — |
| `normalization.py:91`, `:92` | 30 | 30.0 | 30.0 | LAYER | LAYER (minimal) | 0 | — |
| `wan_va_server.py:356`, `:358` | ~1.5 | — | 1.0 | — | STEP | 2 | BITEXACT |
| 6 sites at `model.py:280-283, 872, 874` | 1 | 1.0 | 1.0 | STEP | STEP (minimal) | 0 | — |

Expected launch reduction: **292 of ~15,000 launches per cycle, ~2%.** Expected cycle impact:
**~4.5 ms of 330.2 ms = 1.4%**, all of it at `model.py:524`.

---

## Why the family does not exist

**Twenty-one of twenty-two sites cast a genuinely different value on every call.** They are inside the
transformer block, and the block's activations change per block by definition — that is what a block
does. The one exception is `temb`, which is computed once per forward at `model.py:861` and passed
unchanged into all 30 blocks.

The asymmetry with P004 is structural, not accidental:

| | P004 (weights) | here (activations) |
|:--|:--|:--|
| what is cast | `FP32LayerNorm.weight`, `scale_shift_table` | block activations |
| lifetime | **EPISODE** — a parameter does not change | **LAYER** — that is the point of a layer |
| redundancy | 7,110 casts/cycle of a constant | 290 |

A parameter is invariant by definition, so P004 had a whole population to harvest. An activation is
variant by definition, so there is no population — only the handful of *conditioning* values that
happen to be computed outside the block loop and consumed inside it. There are two such values in this
model (`temb`, and a pair in `_encode_obs`), and one of them is already minimal.

### The trap that almost produced the opposite answer

The first version of this probe classified on a **storage digest** — `(data_ptr, storage_offset, shape,
stride, dtype, version)` — reasoning that identical storage proves identical input. It reported
**4,484 calls/cycle removable**, a 15× overstatement, and would have "proved" the family exists.

Look at the storage and value columns together. `ring_kv.py:154` shows **1.2** distinct storages per
forward against **60** distinct values. `model.py:534` shows 3.2 against 30. The **caching allocator
reuses addresses**: each block frees its temporaries before the next allocates, so a handful of
addresses are recycled through dozens of genuinely different tensors. Same pointer, same shape, same
stride — different data.

So a storage digest measures *allocator behaviour*, not value identity. It can corroborate a hoist when
it agrees with the value digest (as at `model.py:524`, where both read 1.0, proving the 30 calls
received the same object), but it can never establish one. The probe now classifies on value and reports
storage alongside so the disagreement is visible.

---

## The generic legality rule, written down anyway

Worth recording even for a single instance, because it is the rule a planner would need and because it
is what makes the *absence* of a family checkable rather than asserted:

> A cast at site `s` may be hoisted from scope `inner` to scope `outer` iff
>
> 1. **`outer` strictly encloses `inner`** — `Scope.STEP < Scope.LAYER`, already expressible via
>    `Site.is_hoistable()` in `passes/interface.py`.
> 2. **The cast's input is value-invariant across `outer`.** Established by measurement, not by
>    inspection: distinct values per `outer` == 1. A storage digest may corroborate but may not
>    establish, because the allocator recycles addresses.
> 3. **The cast is pure** — no `OpKind.EFFECTFUL` op between the hoisted evaluation and every consumer,
>    or the hoist reorders an effect.
> 4. **The consumer does not mutate the result in place.** One cast shared by 30 consumers is only
>    valid if all 30 treat it as read-only; otherwise block *k* sees block *k−1*'s writes.
> 5. **Tier is BITEXACT** when 1–4 hold, because the same input under the same rounding evaluated once
>    yields bit-identical output. This is the only class of Layer 5 optimization so far that is
>    bit-exact *by construction* rather than by measurement.

Condition 4 is the one a naive implementation gets wrong, and it is why this is a pass with a gate
rather than a one-line edit.

---

## Correction to LAYER5_NEXT.md

That proposal valued Candidate 4 at **11.37 ms/cycle**, combining the `_to_copy` (4.69 ms) and the `add`
(6.68 ms) at `model.py:524`. **The `add` is not hoistable.** `scale_shift_table` is an
`nn.Parameter` declared per block (`model.py:512`), so `scale_shift_table[None] + temb.float()` produces
a genuinely different result in each of the 30 blocks. Only the cast of `temb` is redundant.

**Candidate 4 is therefore ~4.5 ms of 330.2 ms = 1.4%, not 3.4%.** That puts it at rough parity with
Candidate 3 (fused QKV, ~4 ms plus 600 launches) rather than ahead of it on raw impact.

It still wins, on the two axes that were always the real argument:

- **BITEXACT by construction.** Gate is `max |Δ action| = 0` on paired seeded cycles — hours. Candidate
  3 is NUMERIC and costs a 555-episode certification plus a permanent entry in the "chain is no longer
  bit-exact" ledger.
- **It removes work rather than accelerating it**, and it composes with P004's existing vocabulary
  with no new machinery.

But the margin is now thin enough to state plainly: this is a **1.4% optimization**. It is worth doing
because it is cheap and provably safe, not because it is large.

---

## Recommendation

1. **Do not build a `StepInvariantCastHoisting` abstraction.** An abstraction over one instance is
   speculation, and the measurement says there is exactly one instance in this model. If a second
   backbone shows the same pattern, generalize then — with two data points instead of one.
2. **Implement Candidate 4 as a single narrow pass**, gated at `max |Δ action| = 0`, with legality
   conditions 1–4 checked explicitly rather than assumed.
3. **Keep the legality rule and this table.** They are what makes "we looked for a family and there
   isn't one" a finding rather than an omission — and what a second model can be tested against cheaply.
4. **Reconsider Candidate 3 next**, not third. With Candidate 4 at 1.4%, the fused QKV projection's
   600 removed launches in a 42%-idle cycle is the larger remaining prize, and its NUMERIC cost is now
   the thing to weigh rather than dismiss.
