# Layer 6: host dispatch, measured — and why removing it does not pay

**One Runtime. Many Checkpoints. Shared Infrastructure.**

> **Read section H first.** This document was written to rank host-dispatch eliminations by operations
> removed per cycle. It ends by **measuring that operations removed per cycle does not predict cycle time**:
> the top candidate removed 12,190 of 105,123 dispatches bit-exactly and the cycle got 3.4 ms *slower*. The
> inventory in sections A–G is sound and worth keeping. The ranking is recorded, not recommended.

Layer 5 asked which kernel to write. The answer, four times running, was *none of them*: the cycle appeared
not to be waiting on the device. So Layer 6 began from the opposite premise — **the host issues the work,
and the host is the clock** — and the job of this document was to test that premise hard enough to act on
it. It did not survive.

| | |
|:--|--:|
| shipped cycle, 2V/4A post-saturation (capture off, P007 on) | **351.4 ms** |
| device busy, interval union | 196 ms |
| **device slack** | **~155 ms** |
| profiler-visible aten events per cycle | **105,123** |
| dispatcher-level operations per cycle | 38,562 |
| fraction of aten events that launch **no kernel** | **66%** |

Everything below is ranked by **host operations removed per cycle**. GPU time does not appear as a ranking
term, and a proposal that only makes an existing kernel faster is rejected on sight — that is the class that
already failed four times.

---

## The scoreboard this layer inherits

| attempt | class | host ops removed | predicted | measured |
|:--|:--|--:|--:|--:|
| **P007 conv layout** | backend dispatch | **~56,600** | — | **1.405×** |
| step-scope cast hoist | dispatch removal | 1,740 | 1.6% | 0.66% |
| RoPE Triton kernel | faster kernel | 0 | 1.10× region | 0.3% |
| fused QKV | fewer, bigger kernels | 0 | 1.9% | **0.2% slower** |
| graph persistence | replace dispatch with a captured artefact | ~36,000 | 1.72× | **1.43× slower** |

Two lessons, and they set this layer's rules.

**Only op-count reduction has ever moved the cycle.** The one shipped win removed ~56,600 host operations.
Everything that left the op count unchanged measured nothing, in both directions.

**The mechanism's own cost is charged to the same budget.** Graph capture and `torch.compile` both attack
dispatch by *building a persistent artefact*, and construction is paid from the cycle it is meant to
shorten. Graph persistence removed the right ops and still lost, because 5.3 surviving captures at ~111 ms
each exceed the entire 351 ms cycle. See
[LAYER5_GRAPH_PERSISTENCE_RESULT.md](LAYER5_GRAPH_PERSISTENCE_RESULT.md). So this layer prefers
transformations that **remove** dispatch outright — a view built once and held, a composite replaced by its
leaf, a no-op module deleted — over transformations that **replace** dispatch with something that must first
be built.

### The model, and its status — now refuted

When host-bound, `cycle ≈ (host op count) × ~3.2 µs`, calibrated against profiler-visible aten events.

It **retrodicts** all five rows above, which is how it earned its authority. It made two **prospective**
predictions before this document — fused QKV and graph persistence — and got both wrong, each time by
pricing the operations removed while ignoring the cost of the mechanism that removed them. That left it
plausible: the mechanism cost was a sufficient excuse both times.

**Section H removes the excuse.** A transformation with no mechanism cost — no artefact, no kernel change,
no shape change, `max |Δ| = 0` — removed 12,190 operations and the cycle did not move. Prospectively the
model is **0 for 3**, and the third failure has no confound to hide behind.

The number was never measured in the first place. `3.2 µs` came from *dividing* 338 ms by 105,130
operations, which presumes the cycle is the sum of per-operation host costs. Section I measures the
derivative instead of assuming it.

---

## A. Where the host time goes

One saturated cycle, `eval/lingbot_va_robotwin/profile_host_dispatch.py`. Self-CPU sums to 501 ms against a
351 ms cycle because the profiler inflates host time ~1.43×; **read the ratios, not the milliseconds.**

| category | ops/cycle | % ops | self-CPU | % CPU | µs/op | launches a kernel? |
|:--|--:|--:|--:|--:|--:|:--|
| **tensor metadata** | **47,020** | **44.7%** | 75.7 ms | 15.1% | 1.6 | no |
| framework work | 24,994 | 23.8% | 330.8 ms | 66.1% | 13.2 | **yes** |
| dispatcher bookkeeping | 15,995 | 15.2% | 34.0 ms | 6.8% | 2.1 | no |
| object allocation | 14,482 | 13.8% | 57.7 ms | 11.5% | 4.0 | no |
| synchronization | 2,632 | 2.5% | 2.5 ms | 0.5% | 1.0 | no |

**Two thirds of the operations do no arithmetic.** The three no-kernel categories — metadata, bookkeeping,
allocation — are 77,497 ops, 73.7% of the cycle's dispatches, and 33.4% of its host CPU time. That was taken
to be the Layer 6 surface.

**The `µs/op` column is the tell, and I read it too late.** Operations that launch a kernel cost **13.2 µs**
each; operations that do not cost **1.6–4.0 µs**. If the host were throughput-bound, every dispatch would
cost roughly its own dispatch overhead and the two groups would be within a factor of ~2. A 3–8× premium
concentrated on exactly the ops that call into the driver is the signature of `cudaLaunchKernel` **blocking
on a full queue** — that is, the host waiting for the device. Read that way, 330.8 ms of the 501 ms of
self-CPU is not host work at all; it is device wait booked to whichever host operation happened to be
holding the thread. Section H measures the consequence.

```
metadata      as_strided 12082  view 10796  transpose 4858  slice 4244  reshape 3185
              t 2450  narrow 2154  squeeze 1823  flatten 1226
bookkeeping   to 8097  linear 2450  type_as 2110  dropout 920  layer_norm 910
              rms_norm 900  scaled_dot_product_attention 602
allocation    empty_strided 6279  empty 5986  lift_fresh 1297  empty_like 694  clone 138
framework     copy_ 6385  _to_copy 5543  add 2534  addmm 2444  mul 2010  fill_ 1361
sync          item 1306  _local_scalar_dense 1306  nonzero 14  is_nonzero 6
```

**`bookkeeping` is composite ops that exist only to redispatch.** `aten::linear` dispatches `aten::t` then
`aten::addmm`; `aten::type_as` dispatches `aten::to`, which dispatches `aten::_to_copy`. Each level is a
separately-counted profiler event doing no work of its own.

### Correction: the `synchronization` category is misnamed

1,306 `item` calls at **1.0 µs each** cannot be device syncs — a real host←device round trip is ~10 µs
minimum. They are `.item()` on tensors already resident on the CPU.

The genuine syncs were localised separately: **16 host←device scalar reads per cycle**, all in
`FlowMatchScheduler.step` (`scheduler.py:82`, `:83`, `:87`), plus 195 `cudaMemcpy` at 15.3 µs total.
**Synchronization is not a Layer 6 target on this model.** It is 2.5 ms of a 351 ms cycle and the arithmetic
was never going to justify the work. Recorded here so it is not proposed again.

## B. Python and framework overhead above the dispatcher

cProfile shares (absolute seconds inflated; shares are the signal):

| stratum | share | calls | µs/call |
|:--|--:|--:|--:|
| built-in / C, includes aten dispatch | 56.9% | 55,072 | 4.8 |
| **model Python** (`wan_va/modules/model.py`) | **26.4%** | 3,520 | **34.9** |
| `nn.Module.__call__` machinery | 7.3% | 29,256 | 1.2 |
| torch Python, other | 4.1% | 13,620 | 1.4 |
| diffusers Python | 3.1% | 2,115 | 6.8 |

**This is a stratum the op count misses entirely.** `nn.Module._call_impl` runs 29,256 times per cycle and
dispatches nothing — it is 97 module invocations per block execution, each ~1.2 µs of hook checks and
attribute lookups, and none of it appears in the 105,123. Any ranking done purely on aten events
undercounts the removable surface by roughly 29,000 Python frames.

## C. Where the work is: 300 executions of one function

| | |
|:--|--:|
| dispatcher ops inside the 30 transformer blocks | **36,360 of 38,562 = 94.3%** |
| dispatcher ops per block execution | **121** |
| profiler aten events per block execution | **331** |
| block executions per cycle | 300 (30 blocks × 10 forwards) |
| eager block wall time | **1.175 ms** (spread 3.5%) |
| × 300 | **353 ms** — the entire 351 ms cycle |
| device time per block | 0.65 ms → **~0.52 ms per block is host** |

Inside a block, by category: metadata 18,480 (50.8%), framework 16,620 (45.7%), allocation 1,260 (3.5%).

**The block forward × 300 accounts for the whole cycle.** That is the single most useful fact in this
document: Layer 6 has one target, not five, and any transformation is worth `300 ×` its per-block effect.
Removing one dispatch from the block forward removes 300 from the cycle and ~1 ms from the clock.

### What the block forward actually does

`WanVATransformerBlock.forward` (`wan_va/modules/model.py:515`) and `WanAttention.forward` (`:414`),
read for removable structure rather than for arithmetic:

- **Modulation prep, ~14 metadata ops per block.** `scale_shift_table[None]` (unsqueeze) `+ temb.float()`
  (cast), then `rearrange('b l n c -> b n l c')` (permute), `.chunk(6, dim=1)` (6 narrows), then **six
  separate `.squeeze(1)`**. Every call rebuilds twelve view descriptors whose shapes and strides are
  identical on every one of the 300 executions.
- **Four fp32 round trips on the full hidden-states tensor.** `hidden_states.float()` … `.type_as(...)`
  appears four times, each pair being `to` → `_to_copy` at both ends: ~8 casts per block, 2,400 per cycle,
  on the largest tensor in the block. The fp32 arithmetic between them *is* the numerics, so the casts
  cannot simply be deleted — but they are the bulk of `to 8097` / `type_as 2110` / `_to_copy 5543`.
- **Three `nn.Dropout(0.0)` per block** — `attn1.to_out[1]`, `attn2.to_out[1]`, `ffn.net[1]`. In eval mode
  `F.dropout(x, 0.0, False)` returns its input unchanged, yet `aten::dropout` is dispatched every time.
  30 blocks × 3 × 10 forwards = **900 dispatches per cycle that provably do nothing**, which is exactly the
  measured `dropout 920`.
- **`unflatten` ×3, `flatten(2,3)`, `type_as`** per attention, plus the RoPE block's `to(float64)`,
  `reshape`, `view_as_complex`, `view_as_real`, `flatten(3)`, `to(dtype)` on both q and k.

## D. `torch.compile` is available here, and it is slower

The obvious mechanism for eliminating dispatch wholesale. It works, and it does not pay.

**P003 is what makes the block compilable.** `torch._dynamo.explain` on the block forward:

| | graphs | breaks | ops captured |
|:--|--:|--:|--:|
| with `ring_kv` (P003) installed | **1** | **0** | 55 |
| without it | 10 | 9 | — |

The nine breaks without P003 are "Dynamic shape operator" ×4 — `mask.nonzero()` at `model.py:451`, the
data-dependent free-slot search P003 replaced with interval arithmetic — plus a `generic_jump
TensorVariable`. **The ring rewrite bought compilability as a side effect of buying capturability**, which
is the same property twice and worth stating once.

Compiled output is **bit-exact** against eager at the served shape (0 of 1,474,560 bf16 words differ).

### It removes zero host operations. Measured, not predicted.

`eval/lingbot_va_robotwin/probe_compile_stack.py`, one process, warm post-saturation state, the same
captured block arguments fed to every arm.

| arm | aten events | kernels | wall | spread | vs eager | numerics |
|:--|--:|--:|--:|--:|--:|:--|
| eager, 1 block | 331 | 58 | 1.259 ms | 3.5% | 1.00× | — |
| **compiled, 1 block** | **331** | 56 | 1.523 ms | **3702%** | 0.83× | BITEXACT |
| eager, 30 blocks | 9,930 | 1,740 | 37.205 ms | 3.1% | 1.00× | — |
| **compiled, 30 blocks** | **9,930** | 1,711 | 42.198 ms | **36.3%** | 0.88× | BITEXACT |

**Not one operation removed** — 331 → 331 and 9,930 → 9,930, and the per-operator counts are identical
line for line (`as_strided` 37, `view` 35, `to` 26, `empty_strided` 20 …). The wall times are unusable at
3702% and 36.3% spread and are **NOT EVALUATED**; the op count is the number that isn't noisy, and it is
zero. On this layer's only ranking term, `torch.compile` scores nothing.

**Why: the ring's `next_id` is a static guard.** Dynamo reported the reason itself —

```
torch._dynamo hit config.recompile_limit (8)
  function: 'stack'
  last reason: G['blocks']._modules['0']._modules['attn1']
                 .attn_caches['pos']['_ring']['next_id'] == 1255
               # kvc["id"][sl] = r["next_id"]   ring_kv.py:192
  HINT: torch.compile considers integer attributes of the nn.Module to be static.
```

`next_id` increments once per commit, so **every cycle fails the guard, recompiles, and after eight
recompiles dynamo gives up and falls back to eager for the rest of the process.** `unique_graphs` reached
8, then 16 for the stack. The 3702% spread is a recompile landing inside the timing window; the earlier
"9,960 events / 2,143 kernels" that looked like a recompile storm was this fallback, and the number was the
30-block count, not a single block's.

So `dynamo.explain`'s clean "1 graph, 0 breaks" was true of one invocation and false of the workload. **A
single-call trace cannot detect a guard that fails on the second call**; the tenth recompile is invisible to
`explain` and fatal in production.

**This is the graph-key problem again, in a different mechanism.** P005 failed because ring state was in the
CUDA-graph key; `torch.compile` fails because ring state is in the guard set. The same Python int, `head`/
`start`/`next_id`, defeats both artefact-building mechanisms for the same reason.

**And even if the guard were fixed, the construction cost decides it.** Compiling the 30-block stack took
**318.9 s** of warmup. A RoboTwin episode is ~53 cycles at 351 ms, i.e. **~19 s of inference**. The artefact
costs 17× the entire episode it would accelerate. That is the graph-persistence arithmetic verbatim, and it
is why this layer's rules put *removal* ahead of *replacement*.

`torch.compile` is therefore **rejected**, on measurement, with `allow_unspec_int_on_nn_module` and warm
inductor caches left as the only untested repairs — neither of which changes the 318.9 s.

### One useful by-product

Eager 30 blocks is **37.205 ms** per forward; × 10 forwards = **372 ms** against a 351 ms cycle, and
331 × 300 = 99,300 of the cycle's 105,123 aten events = **94.5%**. Two independent instruments now agree
that the block stack *is* the cycle. Anything outside it is rounding.

*(the `device` column of that probe double-counts composite parents and their kernels — 40.7 ms of device
time against a 37.2 ms wall is impossible — so no device figure from it is quoted here.)*

## F. Callsite attribution for the metadata population

The project's rule — *no operator is optimized before its callsite distribution is known* — applies to
Layer 6 unchanged. `profile_attribution.py --watch-set metadata`, 2 warm cycles past saturation:

| operator | calls/cyc | coverage | dominant callsites |
|:--|--:|--:|:--|
| **`t`** | **2,450** | **100%** | ring_kv:146 **900**, ring_kv:221 300, lingbot_install:117 300, :120 300, activations.py:88 300, attention.py:1741 300 |
| **`squeeze`** | **1,823** | **100%** | model.py:**527–532**, 300 each = **1,800** |
| `unsqueeze` | 445 | 100% | model.py:524 **300** |
| `permute` | 359 | 100% | einops `_backends.py:261` **324** |
| `view` | 8,025 | 74% | ring_kv:146 1,800, lingbot_install:117 900, ring_kv:153/154 600+600, ring_kv:221 600 |
| `expand` | 96 | 61% | autoencoder_kl_wan:201 44 |
| `select` | 42 | 66% | model.py:280–282 |
| `transpose` | 9,716 | **49%** | `[PARTIAL, NOT RANKABLE]` — attributed part is all `custom_sdpa` model.py:38–40 |
| `slice` | 8,428 | **50%** | `[PARTIAL, NOT RANKABLE]` — attributed part is ring_kv:184/185/191/192/193 |
| `detach` | 4 | **200%** | `[NON-STATIONARY, NOT RANKABLE]` |
| `as_strided`, `reshape`, `narrow`, `flatten`, `unflatten` | — | **0%** | **not attributable at all** — see below |

**`t` is the finding.** 2,450 calls at 100% coverage, and every dominant row is an `nn.Linear`: three
projections at `ring_kv.py:146`, self-attention output at `:221`, cross-attention query and output in
`lingbot_install.py`, and the two feed-forward projections in diffusers. **2,400 of 2,450 are the eight
Linear modules inside the block** — 8 × 300 executions, exactly.

**`squeeze` is the second finding.** 1,800 of 1,823 are the six `.squeeze(1)` at `model.py:527–532`, one per
modulation component, 300 times each. Nothing else in the cycle squeezes anything meaningful.

### The instrument's limit, stated plainly

`as_strided` (12,082), `reshape` (3,185), `narrow`, `flatten` and half of `transpose` and `slice` attribute
to **no Python callsite at all**, because they are emitted from inside C++ composite implementations. A
Python-stack attributor cannot see them, and no amount of warmup fixes that.

**A second instrument recovers exactly that population**: microbenchmark the composite and read its
children. That is what the two probes below do, and it is how `as_strided` and `reshape` get accounted for
without guessing.

## G. Two composites, priced directly

Standalone, real shapes, no checkpoint — `/tmp/linear_dispatch.py` and `/tmp/modulation_dispatch.py`.

**`nn.Linear` on a 3-D input costs 8 aten events; the arithmetic needs 3.**

| shape | incumbent | direct `addmm` | removed | bit-exact |
|:--|--:|--:|--:|:--|
| to_q video (2,512,3072)→3072 | 8 | 2 | 6 | **YES** |
| to_q action (2,32,3072)→3072 | 8 | 2 | 6 | **YES** |
| ffn.proj (2,512,3072)→14336 | 8 | 2 | 6 | **YES** |
| ffn.out (2,512,14336)→3072 | 8 | 2 | 6 | **YES** |

```
incumbent:  view×2  linear  reshape  t  transpose  as_strided  addmm      = 8
direct:     addmm  view                                                   = 2   (+1 input collapse = 3)
```

Zero differing words on all four. **The transposed weight is a view of a frozen parameter — a stride swap,
computed once and held.** The `2` above hoists the input collapse out of the timed region because that is
the claim under test; a real pass still pays one `view(-1,C)` per call, so budget **5 removed of 8**.

This also closes the `as_strided`/`reshape` gap from section F: `nn.Linear` emits one `as_strided` and one
`reshape` per call, so **2,450 of 3,185 `reshape` (77%)** and 2,450 of 12,082 `as_strided` are Linear's, and
they are removable with it.

**The modulation prep costs 41 aten events per block; `unbind` produces the identical six tensors in 20.**

| form | events | shapes + bits match reference |
|:--|--:|:--|
| incumbent `rearrange → chunk(6) → 6× squeeze` | **41** | — |
| `unbind(permute(...), 1)` | 22 | **YES** |
| **`unbind(tbl, 2)`** — no permute at all | **20** | **YES** |

```
incumbent:  as_strided×14  narrow×6  slice×6  squeeze×6  chunk  split  permute  unsqueeze
            to  _to_copy  empty_strided  copy_  add                              = 41
unbind:     as_strided×7  select×6  unbind  unsqueeze  to  _to_copy
            empty_strided  copy_  add                                            = 20
```

`chunk(6, dim=1)` followed by `squeeze(1)` **is** `unbind(1)`, and the `permute` exists only to move the
6-axis into position — `unbind` takes the axis as an argument. Six outputs, all `(2,240,3072)`, all
bit-identical.

**And `nn.Dropout(0.0)` in eval mode is provably the identity, and dispatches anyway.**

```
nn.Dropout(0.0).eval() : events=1.0 {'dropout': 1.0}  returns_input_identity=True
```

`F.dropout(x, 0.0, training=False)` returns its argument — verified by object identity, not by comparing
values. Three per block (`attn1.to_out[1]`, `attn2.to_out[1]`, `ffn.net[1]`) × 30 × 10 = **900**, which is
the measured `dropout 920`.

---

## H. The candidate was built as a probe, and it refuted the ranking term

Before writing a pass, I ran the falsification step the four preceding rounds skipped: monkeypatch the
transformation in, measure the **cycle**, and check the arithmetic against the clock.
`eval/lingbot_va_robotwin/probe_prebound_projection.py`, one process, both arms reversible, 314 `nn.Linear`
modules patched.

**The op count fell exactly as predicted.**

| | |
|:--|--:|
| baseline | **105,123** aten events/cycle |
| prebound projection | **92,933** |
| removed | **12,190 (11.6%)** — predicted 12,250 |

| operator | base | prebound | delta |
|:--|--:|--:|--:|
| `aten::linear` | 2,450 | 6 | **−2,444** |
| `aten::t` | 2,450 | 6 | **−2,444** |
| `aten::as_strided` | 12,082 | 9,638 | **−2,444** |
| `aten::transpose` | 4,858 | 2,414 | **−2,444** |
| `aten::reshape` | 3,185 | 771 | **−2,414** |
| `aten::view` | 10,796 | 10,796 | +0 |
| `aten::addmm` | 2,444 | 2,444 | **+0** |

`addmm` unchanged at 2,444 and `view` unchanged at 10,796 is the exactness argument made visible: **same
kernel, same count, same operands.** The six surviving `linear` calls are the fail-closed fallback firing on
non-viewable inputs, which is the guard working. And:

**`max |Δ action| = 0` over 8 paired seeded cycles.** BITEXACT on the real served path, not just on four
microbenchmark shapes.

**Then the cycle gate, ABBA-ordered (base, treat, treat, base), 12 cycles per arm:**

| arm | median |
|:--|--:|
| base | 411.3 ms |
| treat | 415.2 ms |
| treat | 412.3 ms |
| base | 409.3 ms |
| **base mean** | **410.3 ms** (drift 0.5%) |
| **treat mean** | **413.8 ms** (drift 0.7%) |
| | **0.992× — 3.4 ms SLOWER** |

**Predicted −39.0 ms. Measured +3.4 ms.** Drift 0.5% on the repeated base arm, so the measurement is clean
and the gap is 11× outside it.

### Why this is different from the previous four failures

Every earlier miss had a confound that let the model survive: fused QKV changed the GEMM shape, graph
persistence and `torch.compile` had to build an artefact, the RoPE kernel touched a 0.3% region. **This one
has none of them.** No artefact is constructed at runtime. No kernel changes — `addmm` count is identical.
No shape changes. No numerics change — `max |Δ| = 0`. The *only* difference between the arms is 12,190
fewer host dispatches, and the cycle did not move.

**So the `~3.2 µs/op` model is refuted, and this is the finding of this document.** It was never a
measurement: it was obtained by *dividing* cycle time by operation count, which presumes the cycle is the
sum of per-operation host costs — the very claim at issue. It retrodicted five results because op count and
device time happened to fall together in all of them.

**Including P007, which I had attributed to the wrong term.** Its certificate says the cause plainly:
`slow_conv_dilated3d` at 2.659 ms → `cudnn_convolution` at 0.581 ms, ~2.1 ms saved on each of 62
convolutions ≈ **130 ms**, against a measured **+150 ms/cycle**. P007's gain is accounted for by the
device-side kernel change alone. The 56,600 vanished host operations were a *side effect* — `vol2col`
lowering disappearing with the fallback — not the mechanism. I credited the side effect.

### What this does to the ranking below

It invalidates it. Candidates 2 and 3 remove 6,300 and 920 operations by the same means as candidate 1,
which removed 12,190 and bought nothing; there is no reason to expect a smaller version of a null result to
be positive. **The ranking is kept below as recorded work, not as a recommendation.**

I stopped the ranking workflow mid-flight for this reason rather than spend two dozen more agents
adversarially verifying an ordering whose ranking term had just been measured to be non-predictive.

## I. What one host operation actually costs — measure the slope, don't divide

`eval/lingbot_va_robotwin/probe_host_op_slope.py`. Inject K extra no-kernel dispatches
(`torch.as_strided` on an existing tensor: pure stride arithmetic, no kernel, no allocation, result
discarded) into every block execution, sweep K, and fit the derivative. The injection makes the runtime
strictly *slower* on purpose — this measures a cost, it does not optimize anything.

The injection lands exactly where intended, which also re-confirms 300 block executions per cycle:

| K per block | aten events/cycle | extra |
|--:|--:|--:|
| 0 | **105,123** | — |
| 10 | 108,123 | +3,000 |
| 25 | 112,623 | +7,500 |
| 50 | 120,123 | +15,000 |
| 100 | 135,123 | +30,000 |

| extra ops/cycle | Δ cycle | µs per host op |
|--:|--:|--:|
| 0 | +1.3 ms | — *(noise floor)* |
| 3,000 | +7.5 ms | 2.515 |
| 7,500 | +6.2 ms | 0.820 |
| 15,000 | +15.4 ms | **1.025** |
| 30,000 | +30.4 ms | **1.012** |

**Least-squares through the origin: 1.017 µs per host operation.** The k=0 arms held to 1.7% across the
whole sweep, and the two largest levels agree to 1.3%, so the fit is solid where it matters.

**The assumed 3.2 µs is 3.1× too high.** That alone rewrites every estimate this project has made about host
work.

### And the two results together explain each other exactly

Injected Python-level dispatches cost **1.02 µs**. The dispatches `prebound_projection` removed cost
**nothing** — the cycle got *slower*. Both are true, and the reason is that **the 105,123 aten events are not
fungible: cost depends on where the call originates.**

```
incumbent   self.to_q(q)  ->  1 Python-level call (aten::linear)
                              + 7 C++-internal children (t, transpose, as_strided, reshape, view x2, addmm)
prebound    addmm form    ->  3 Python-level calls (view, addmm, view)
                              + 0 children
```

A C++-internal redispatch never crosses the Python boundary: no GIL round trip, no argument parsing, no
`THPVariable` wrapping. The profiler counts it identically to a Python call, and it costs a small fraction
as much.

**So the pass traded 12,190 cheap operations for 4,888 expensive ones.** Python-level calls went
2,444 → 7,332 across the 2,444 projections. At the measured 1.02 µs that predicts **+5.0 ms**; the ABBA
measurement found **+3.4 ms**, inside the 1.3 ms noise floor. The null result is not a mystery — it is the
corrected model getting the sign and the magnitude right.

### The corrected model

> **cycle host cost ≈ (Python-originated aten entries) × 1.02 µs.**
> C++-internal redispatch is effectively free and must not be counted.

This is measured rather than divided, and it retrodicts the failures without excuses: the RoPE kernel and
fused QKV changed no Python-level call count; the cast hoist removed 1,740 Python-level casts → 1.8 ms
predicted, 2.3 ms measured (0.66%); `prebound_projection` **added** 4,888 → +5.0 ms predicted, +3.4 ms
measured.

**And it puts a ceiling on this entire layer.** cProfile counts **55,072** built-in/C calls per cycle, which
bounds the Python→C entries from above. At 1.02 µs that is **≤56 ms of a 351 ms cycle — a hard ceiling of
about 1.16×, if every Python-originated dispatch in the runtime were eliminated.** Not 155 ms, and not the
73.7% of operations the inventory pointed at.

Re-priced against the corrected model, the candidates below are worth:

| candidate | Python-level calls removed/cyc | at 1.02 µs | % of cycle |
|:--|--:|--:|--:|
| `prebound_projection` | **−4,888** (adds them) | **+5.0 ms** | −1.4% |
| `modulation_unbind` | ~2,100 | ~2.1 ms | 0.6% |
| `dropout_elision` | ~900 (+900 `Module.__call__`) | ~1–2 ms | 0.4% |

**Single-digit milliseconds each, and the largest one is negative.** That is the answer to whether Layer 6 is
a direction.

### The one mechanism that targets the right term is priced out

Under the corrected model the correct transformation is obvious: collapse thousands of Python-level calls
into one guarded entry — which is exactly what `torch.compile` does. It is the only mechanism here that
addresses the cost term that actually exists.

It is also the one measured at **318.9 s of construction** for a 30-block region, defeated by a ring-state
integer guard, against a ~19 s episode (section D). **The right mechanism is unaffordable and the affordable
mechanisms are the wrong term.** That is a coherent stopping point rather than a list of things left to try.

---

## The ranking, as produced — superseded by section H

Ranked by **host operations removed per cycle**. Section H measured that this term does not predict cycle
time, so the `ms` column below is arithmetic that has now been falsified at the top of the list. Kept for
the record.

| # | candidate | class | ops removed/cyc | % of 105,123 | ms | tier | effort | evidence |
|--:|:--|:--|--:|--:|--:|:--|:--|:--|
| **1** | **`prebound_projection`** | dispatch elimination | **12,250** | **11.7%** | 39 | **BITEXACT** | medium | **measured**, 4/4 shapes |
| **2** | **`modulation_unbind`** | dispatch elimination | **6,300** | **6.0%** | 20 | **BITEXACT** | low | **measured**, bit-identical |
| **3** | **`dropout_elision`** | dead-code removal | **920** (+920 Python frames) | 0.9% | 3 | **BITEXACT** | trivial | **measured**, identity |
| | **1+2+3 together** | | **19,470** | **18.5%** | **62** | BITEXACT | | → 351 → ~289 ms, **1.21×** |
| 4 | `ring_bookkeeping_elision` | dead-store removal | ~1,800 | 1.7% | 6 | BITEXACT | medium | **blocked** — live reader |
| 5 | `sdpa_head_view` | view construction | ≤2,400 | 2.3% | 8 | NUMERIC risk | high | no free mechanism |
| — | `torch.compile` | replace dispatch | **0** | 0% | 0 | BITEXACT | — | **measured: rejected** |
| — | CUDA graph capture | replace dispatch | ~36,000 claimed | — | — | BITEXACT | — | **frozen: 1.43× slower** |

Candidates 1–3 are **disjoint** — Linear's `t`/`reshape`/`as_strided`, the modulation's
`narrow`/`squeeze`/`chunk`, and `aten::dropout` are different operators at different callsites — so they
compose additively rather than competing for the same ops.

## The recommendation, as written before section H ran

`prebound_projection` was the recommendation: precompute each `nn.Linear`'s transposed weight view once at
install and call `addmm` directly. It was chosen on the strongest reasoning available at the time, and the
reasoning is preserved here because **the way it was wrong is the useful part.**

The argument had four legs. Three were correct.

- **Largest by the ranking term** — 12,250 operations, confirmed at 12,190. *Correct, and irrelevant: the
  ranking term does not price the cycle.*
- **BITEXACT by construction, not observation** — `aten::linear` itself computes
  `addmm(bias, input.reshape(-1,C), weight.t())`, so calling that expression invokes the same kernel on the
  same operands. Unlike fused QKV, whose certificate found M=7 differing in 55 of 64,512 words because
  concatenation changed `M`, nothing about this GEMM changes. *Correct — `addmm` count unchanged at 2,444 and
  `max |Δ action| = 0` on the served path.*
- **A runtime capability, not a LingBot patch** — `nn.Linear` is a PyTorch class, so the transformation
  applies to every checkpoint with nothing branching on training method. *Correct.*
- **"It removes work rather than replacing it, so the failure mode that killed the last two predictions
  cannot apply."** ***This was the error.*** It does replace work: one Python-level `linear` call becomes
  three Python-level calls. I checked that no *artefact* was built and concluded no cost was added, without
  asking what the removed and added operations each cost. Section I shows the added ones cost 1.02 µs and
  the removed ones cost ~0.

**The lesson is about the unit, not the candidate.** Four rounds ranked host work by counting dispatches,
and a dispatch is not a unit of cost — a Python-level call and a C++-internal redispatch differ by more than
an order of magnitude and the profiler reports them identically. Any future host-side proposal has to price
Python-boundary crossings, and the cheapest way to get that number is the injection sweep in section I,
which takes one probe.

## The falsification step, and why it belongs first

The plan was: patch it in, count the cycle's events, run one ABBA pair, and only then decide whether to
write a pass. That is what sections H and I are. It cost two probes and about an hour, it produced a
BITEXACT null result, a corrected cost model, and a ceiling for the whole layer — and it prevented a fifth
pass being written against a discredited unit.

**Run this before the pass, not after.** The four preceding rounds each wrote the pass first.

## Rejected, and why

**`torch.compile` — rejected on measurement.** Zero operations removed (331 → 331, 9,930 → 9,930), because
the ring's `next_id` int is a dynamo guard that fails every cycle until the recompile limit forces a
permanent eager fallback. Even repaired, 318.9 s of construction against a ~19 s episode. Section D.

**CUDA graph capture — frozen.** All correctness gates passed; 1.43× slower than not capturing.
[LAYER5_GRAPH_PERSISTENCE_RESULT.md](LAYER5_GRAPH_PERSISTENCE_RESULT.md).

**`ring_bookkeeping_elision` — blocked, and I was wrong about it first.** `ring_kv.py:191–193` writes
`mask`, `id` and `is_pred` on every commit, 900 slice-assignments per cycle, and the interval read path
never consults them — which reads like ~1,800 dead operations. **It is not dead.** `clear_pred_cache` chains
to `_orig_clear_pred`, which reads `is_pred` and writes `mask`; the pass's own docstring records that an
earlier draft dropping this bookkeeping was *unsafe*. Removing the stores requires removing that reader in
the same change, which converts a 1.7% cleanup into a correctness-critical edit to a frozen pass. Not worth
it at that price. Recorded here so the same claim is not made a third time.

**`sdpa_head_view` — no free mechanism.** `custom_sdpa` (`model.py:38–40`) does four transposes per call,
2,400 per cycle at 49% coverage. They are metadata, not copies: the projections naturally produce
`(B,S,H,D)` and SDPA wants `(B,H,S,D)`. Making the tensor arrive pre-transposed means a real copy, which is
strictly worse, and materialising it contiguously would change the layout SDPA reduces over. Ruled out on
mechanism, not on size.

**Synchronization — not a target.** 2.5 ms of a 351 ms cycle, and only 16 genuine device round trips.
Section A.

**Every GPU-kernel proposal — rejected by rule at the time.** The stated reason was that the device chain
carries ~155 ms of slack, so a faster kernel shortens nothing. **Section I weakens that reason
considerably.** With the host budget capped at ~56 ms and P007's 1.405× now attributable to its *kernel*
change rather than to its op-count reduction, the device side is where the only demonstrated win in this
project came from. Nothing here reinstates the three rejected kernels — RoPE was 0.3% of the cycle, fused
QKV changed nothing measurable — but "the device has slack, therefore device work cannot matter" no longer
follows from the measurements.

## What these measurements do NOT support

- **The corrected 1.02 µs/op figure is measured on ONE kind of operation** — a Python-level `as_strided` with
  a tuple of shape and stride to parse. A Python-level call with fewer arguments, or one that allocates,
  will differ. The slope is a good estimate for metadata dispatch and should not be applied to
  kernel-launching operations, whose cost is dominated by device wait.
- **The re-priced candidate figures in section I are arithmetic, not measurements.** Only
  `prebound_projection` was actually run end to end. `modulation_unbind`'s ~2.1 ms and `dropout_elision`'s
  ~1–2 ms are the corrected model applied to hand-counted Python-level calls, and the model has now earned
  exactly one prospective success.
- **The ~56 ms ceiling rests on cProfile's 55,072 built-in/C calls**, which is an upper bound on Python→C
  aten entries — it includes non-aten builtins. The true count is lower, so the ceiling is generous rather
  than conservative.
- **The compiled-stack wall times are NOT EVALUATED** — 3702% and 36.3% spread, contaminated by recompiles
  landing inside the timing window. Only the op counts (0 removed) are quoted, and only because a count is
  not noisy.
- **`allow_unspec_int_on_nn_module` and a warm inductor cache are untested.** Either could remove the guard
  failures, and under the corrected model a *working* `torch.compile` is the one mechanism aimed at the right
  cost term. Neither changes the 318.9 s construction cost, which is why it stays rejected — but this is the
  most defensible thing left on the list, not a dead end.
- **`as_strided` is only ~20% accounted for.** 2,444 of 12,082 are Linear's. The other ~9,600 attribute to no
  Python callsite and no composite I have priced. Under the corrected model most of them are probably
  C++-internal and therefore free, which would make the unexplained population harmless — but that is an
  inference, not a measurement.
- **`transpose` (49%) and `slice` (50%) are `[PARTIAL, NOT RANKABLE]`** and were not used to choose anything.
- **Self-CPU milliseconds are inflated ~1.43×** by the profiler (501 ms attributed to a 351 ms cycle). Only
  ratios and op counts are load-bearing.
- **The probe harness cycle is ~410 ms, not the shipped 351 ms**, because the timed region includes
  generating 5.5 MB of random keyframes per cycle in numpy. It inflates both arms equally and does not
  rescue a 39 ms prediction, but percentages computed against 410 ms are not percentages of the served cycle.
- **P007's re-attribution to the device is arithmetic, not a re-run.** 62 convolutions × ~2.1 ms ≈ 130 ms
  against a measured +150 ms is a good fit, not a proof. Re-measuring P007 with the op count held fixed would
  settle it.

## Where this leaves the work

**Layer 6 as a direction is bounded at ~1.16× and realistically at a few percent.** The mechanism that would
claim the ceiling — collapsing Python-level dispatch wholesale — exists and is measured at 318.9 s of
construction per region, defeated by a ring-state guard. The mechanisms that are affordable address a cost
term worth single-digit milliseconds each.

Two things follow, and the second is a question for whoever reads this rather than a decision I should make
alone.

1. **`modulation_unbind` and `dropout_elision` are still worth their price.** Both are BITEXACT, both are
   trivial, together they are worth ~3–4 ms (~1%), and `dropout_elision` in particular is deleting a
   provable no-op from an inference path. That is small but honest, and neither needs new infrastructure.
2. **The premise that sent the work here was that the host is the clock, and it is now measured not to be.**
   The instruction for this layer was to eliminate dispatch rather than accelerate kernels, on the strength of
   the critical-path analysis. That analysis divided where it should have differentiated. With the host
   capped at ~56 ms, a 351 ms cycle, and 196 ms of device work, the remaining 155 ms is neither host
   throughput nor kernel duration — **it is gaps on the device timeline**, and nothing in this document
   measures what creates them. That is the next question, and it is a different question from both "which
   kernel" and "which dispatch".

## Further reading

- [LAYER5_CRITICAL_PATH.md](LAYER5_CRITICAL_PATH.md) — where `3.2 µs/op` came from, and why it was circular
- [LAYER5_GRAPH_PERSISTENCE_RESULT.md](LAYER5_GRAPH_PERSISTENCE_RESULT.md) — the frozen negative result
- [LAYER5.md](LAYER5.md) — backend/layout selection, and P007 as the reference flow
- [`verify/attribution.py`](instinctwm/verify/attribution.py) — the coverage gate, and its C++ blind spot
