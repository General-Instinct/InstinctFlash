# Layer 5, next decision: a ranked proposal

> **SUPERSEDED.** This ranked the next Layer 5 decision by exclusive device time. That ranking
> term was later refuted: device time in the transformer returns **0.145 ms of cycle per ms**
> ([LAYER6_REGIMES.md](LAYER6_REGIMES.md)), so the ordering here does not price anything. Layer 5 is
> closed — see [LAYER5_COMPLETE.md](LAYER5_COMPLETE.md). Kept for the attribution method and the
> coverage gate that excluded `cat` at 121%.

Source: `profile_attribution.py`, 2V/4A warm past ring saturation, conv-layout=ndhwc (P007 applied).
Method per [LAYER5.md](LAYER5.md): attribute → exclude the unrankable → classify → estimate → pick one.

**No code. This is the proposal.**

---

## Exclusions first

| operator | ms/cyc | coverage | why excluded |
|:--|--:|--:|:--|
| `copy_` | 2.22 | **10%** | `[PARTIAL, NOT RANKABLE]`. Its callsites are inside C++. Also now trivial — P007 removed 82% of the population, so there is little left to want. |
| `cat` | 21.85 | **121%** | `[NON-STATIONARY, NOT RANKABLE]` — see below. |

### `cat` is disqualified, and this is the most important finding in this report

Two instruments disagreed. `probe_cat_sites` (one counted cycle) found **zero** cats in `ring_kv`;
attribution (two counted cycles) found **60** at `ring_kv.py:198-199`, 10.21 ms each.

Neither is wrong. The span shape is `(2, 512, 24, 128)` — one span of a two-span read totalling
~114 MiB per call — so the ring-wrap materialization is real. But that branch is reachable *only during
the wrap transition*: before the ring wraps the read is a slice view, and once `count >= total` it is a
whole-pool view. Its call count therefore depends on ring position, and the two instruments sampled
different ring positions.

**That is what 121% coverage means.** Coverage cannot exceed 100% for a stationary workload, because the
attributed pass and the ground-truth pass run the same code. When it does, the operator's count depends
on state that advanced between them. My tool did not flag this — `MIN_COVERAGE` only caught
under-attribution — so `MAX_COVERAGE = 1.10` and a `[NON-STATIONARY, NOT RANKABLE]` label now exist.
Without that, `cat` would have ranked 4th and 5th here on a number that describes one ring position.

`cat` may still be a real target. It cannot be ranked until it is measured across a full ring period
(~64 cycles), reporting the distribution rather than a sample.

---

## The five rankable candidates

All at ≥99% coverage, ranked by exclusive device time. Every one is shape-stable (1–2 shapes).

| # | callsite | ms/cyc | calls/cyc | MiB/cyc | class |
|:--|:--|--:|--:|--:|:--|
| 1 | `addmm` @ `activations.py:88` — FFN up-proj + GELU | 14.18 | 300 | 25,613 | operator fusion |
| 2 | `addmm` @ `attention.py:1741` — output projection | 13.66 | 300 | 27,092 | operator fusion |
| 3 | `addmm` @ `ring_kv.py:146` — `to_q`/`to_k`/`to_v` | 12.01 | **900** | 17,420 | operator fusion |
| 4 | `add` + `_to_copy` @ `model.py:524` — one line, two ops | **11.37**<br>*(only 4.69 hoistable — see [correction](LAYER5_CAST_FAMILY.md))* | 600 | 7,301 | **materialization removal** |
| 5 | `mul` + `_to_copy` @ `ring_kv.py:153-154` — RoPE | 8.25 | 1,800 | 5,357 | operator fusion |

### 1–2. FFN and output projections — `operator fusion`

`self.proj(hidden_states)` followed by `F.gelu`, and the attention output projection. Already on cuBLAS
(`nvjet_*`), so dispatch is exhausted: the only win is a fused epilogue.

- **cycle impact:** low. Fusing GELU into the epilogue removes 300 `gelu` launches (1.85 ms total) and
  one round-trip of the 14336-wide intermediate. Optimistically 3–5 ms of 330 ms = **1–1.5%**.
- **cost:** high. cuBLASLt epilogue plumbing or a CUTLASS path; PyTorch will not do it for us here.
- **tier:** NUMERIC. A fused epilogue keeps the product at accumulator width instead of rounding at the
  materialization boundary.
- **composes?** New machinery. `backends/registry.py` has fusible regions but nothing that owns a GEMM
  epilogue, so this introduces a special case.

### 3. Fused QKV projection — `operator fusion`

Three GEMMs per attention, 900 calls/cycle, on the same input for self-attention. Concatenating the
three weight matrices at load time makes it one GEMM: **900 → 300 calls.**

- **cycle impact:** moderate. ~4 ms of GEMM time plus 600 fewer launches. In a cycle that is **42%
  GPU-idle**, the launch reduction may be worth more than the GPU time.
- **cost:** moderate. A load-time weight transform plus a forward rewrite; needs care where q ≠ k ≠ v
  (cross-attention must keep the split path).
- **tier:** NUMERIC, probably. K is unchanged at 3072, but N goes 3072 → 9216, which can change
  cuBLAS tile/split-K selection and therefore the accumulation order. Might measure bit-exact; cannot
  be *claimed* so.
- **composes?** Partly. It is a weight-level rewrite, which the pass framework supports, but "fuse
  three ops into one with a new weight layout" has no existing home.

### 4. Hoist the timestep-modulation cast — `materialization removal`

> **CORRECTED after the family analysis** ([LAYER5_CAST_FAMILY.md](LAYER5_CAST_FAMILY.md)): the `add` at
> this line is **not** hoistable — `scale_shift_table` is a per-block `nn.Parameter`, so the sum differs
> in every block. Only the cast of `temb` is redundant. Candidate 4 is **~4.5 ms = 1.4%** of the cycle,
> not 3.4%. It still wins on tier and cost, but the margin over Candidate 3 is thin and the figure below
> overstates it.

```python
# model.py:524, executed once per block per forward
temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
```

Two operators at **one source line**: `_to_copy` 4.69 ms (300 calls, `(2,32,6,3072)`) and `add` 6.68 ms
(300 calls, `(1,1,6,3072)` broadcast) — **11.37 ms/cycle**, the largest single-line cost in the report.

**`temb` is the same tensor for all 30 blocks within a forward.** `model.py:861` computes
`timestep_proj` once and passes it to every block, so `temb.float()` produces 30 *identical* fp32
tensors per forward and is discarded 30 times. 300 casts per cycle where **10** would do.

- **cycle impact:** ~4.5 ms of the cast (290 of 300 removed) and 290 fewer launches. If the fp32
  materialization is also hoisted rather than just the cast, more. **~1.5–3%**, and the *launch*
  saving lands on the idle half of the cycle.
- **cost:** **low.** This is a scope error, and the vocabulary for it already exists:
  `Scope.STEP` vs `Scope.LAYER`, `Site.is_hoistable()`, `SiteKind.INVARIANT_CONDITIONING`. P004
  (`hoist_invariant_casts`) is the same transform applied to weights instead of activations.
- **tier:** **BITEXACT.** Computing one cast and reusing it yields bit-identical values — the same
  input, the same rounding, evaluated once. Gate is `max |Δ action| = 0` on paired seeded cycles: hours,
  not a 555-episode certification.
- **composes?** **Yes, entirely.** It is P004's pass generalized from MODEL scope to STEP scope. No new
  concept, no special case.

### 5. RoPE — `operator fusion`

`ring_kv.py:153-154`, 1,800 calls/cycle, 8.25 ms combined (`mul` 3.04 + `_to_copy` 2.80 + 2.41).

**Already attempted and rejected.** The kernel exists, is bit-exact at these shapes, and measured 1.10×
at region scale — 0.3% of the cycle. The attribution now shows why: 8.25 ms of 330 ms is 2.5%, and a
1.10× kernel captures a tenth of that. Listed for completeness; the earlier rejection stands and the
numbers are better, not worse, than when it was rejected.

---

## Recommendation: candidate 4, hoist the timestep-modulation cast

It does not have the largest raw number — the FFN projection does — and it dominates anyway, on four
axes at once:

**1. It removes work rather than accelerating it.** 290 of 300 casts per cycle produce values that are
already in memory. Candidates 1, 2, 3 and 5 all make necessary work faster; this deletes unnecessary
work. That is the P007 lesson stated as a preference, and P007 outperformed the hand-written kernel by
a factor of 14 on exactly this distinction.

**2. It is the only BITEXACT candidate.** Every other candidate is NUMERIC and therefore costs a paired
non-inferiority certification: ~6 hours of fleet time, 555+ episodes, a margin declared in advance, and
a permanent entry in the "chain is no longer bit-exact" ledger. Candidate 4 needs `max |Δ| = 0`. Per
unit of verification cost it is not close.

**3. It composes with no new machinery.** `Scope`, `is_hoistable()` and
`SiteKind.INVARIANT_CONDITIONING` already exist and P004 is the template. Candidates 1 and 2 need a
GEMM-epilogue owner that does not exist; candidate 3 needs a fused-weight concept that does not exist.

**4. Its saving lands where the bottleneck is.** GPU busy is 191.7 ms of a 330.2 ms cycle — **42%
idle**. The cycle is launch-bound. Candidate 4 removes 290 launches for ~4.5 ms of GPU time, and in a
dispatch-bound regime the launch count is the lever. Candidates 1 and 2 remove almost no launches.

**And it is probably a family, not a single site.** `_to_copy` is 5,493 calls/cycle at 99% coverage, and
five of the top six sites are per-block casts of per-forward values (`model.py:524`, `:548`, `:564`,
`:565` = 1,800 calls between them). If they share the scope error, one scope-directed pass addresses
~3,000 launches rather than 290. That is worth *measuring* before building, and it is the natural first
step of the work.

### What would change this recommendation

- If the scope analysis shows `temb` is *not* block-invariant, the whole basis is gone. **Verify that
  first** — cheaply, by comparing the 30 casts within one forward for bit-equality. One probe.
- If the launch-bound hypothesis is wrong and removing 290 launches buys nothing measurable, candidate
  3's 600 launches plus 4 ms of GEMM becomes the better bet.
- If `cat`, re-measured over a full ring period, turns out to fire on most cycles rather than a few,
  its 20.4 ms of pure materialization would outrank everything here — and it would be *materialization
  removal*, the same preferred class. It is excluded today for lack of a trustworthy measurement, not
  for lack of promise.
