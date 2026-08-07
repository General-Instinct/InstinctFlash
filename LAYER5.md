# Layer 5: backend and layout selection, then kernels

**One Runtime. Many Checkpoints. Shared Infrastructure.**

Layer 5 is where a kernel gets chosen or written. The temptation is to start by writing one. Two
attempts here say don't:

| attempt | outcome |
|:--|:--|
| Fused RoPE Triton kernel — bit-exact, 1.10× at region scale | **rejected**: the region was 0.3% of the cycle |
| Conv **layout dispatch** — no kernel written | **shipped as P007**: 1.405× end to end |

The difference was not skill or effort. The kernel was written against a callsite chosen from a sample
covering 12% of the operator's calls; the dispatch fix was found by asking why an operator was on a
fallback path at all. So Layer 5's order of operations is:

> **backend dispatch → layout planning → materialization removal → operator fusion → custom kernel.**
> The cheapest class that applies wins, and nothing justifies a kernel until the four cheaper classes
> are ruled out.

---

## The canonical example: P007, `conv_layout_ndhwc`

Every 3×3×3 convolution in LingBot-VA's two observation VAEs was declining cuDNN and landing on
`slow_conv_dilated3d`. The op's name is misleading — `WanCausalConv3d` pads explicitly and never sets
`dilation`, so nothing is dilated. `slow_conv_dilated3d` is simply where PyTorch lands when its 3D
backends decline.

**The cause was memory layout, and nothing else.**

| input | as-is (NCDHW) | NDHWC | |
|:--|--:|--:|--:|
| (1,160,8,128,160) k3 | 2.659 ms `slow_conv_dilated3d` | 0.581 ms `cudnn_convolution` | 4.58× |
| (1,320,8,64,80) k3 | 2.702 ms | 0.540 ms | 5.00× |
| (1,12,8,128,160) k3 | 1.336 ms | 0.307 ms | 4.35× |
| (1,160,8,64,80) k3 | 2.593 ms | 0.358 ms | 7.24× |

`cudnn.benchmark=True` changes **nothing** (1.00× on all four), so this is not heuristic search
failing: there is no NCDHW bf16 3D kernel for these shapes on H100 / torch 2.9 / cuDNN 9.10. The 1×1×1
convolutions already reached cuDNN in either layout, which is why 16 of 62 were never slow — and why a
summary that dropped the kernel size would have hidden the whole effect.

**Result:** episode mode, post-saturation steady state, **ABBA-ordered** (base, treat, treat, base):
baseline 519.2 / 522.7 → **521.0 ms**, conv-layout 358.8 / 382.8 → **370.8 ms** = **1.405×**,
+150.2 ms/cycle. Drift on the repeated base arm 0.7%.

Note the asymmetry, because it is a property of the change and not of the measurement: the two
treatment arms differ by **6.4%** while the base arms differ by 0.7%, so the converted path is the
noisier of the two — plausibly cuDNN re-selecting a kernel between runs. The ABBA mean is the number.
An earlier 1.45× came from the first arm pair before the ordering completed; quoting it would have
meant picking the better of two treatment arms.

Corroborated in-process at 490.4 → 330.2 ms, 1.49×, by an independent harness — slightly higher there
because it excludes websocket transport.

**Side effect that closed an older mystery.** `aten::copy_` fell 34,710 → 6,385 calls and `fill_`
29,681 → 1,361, because **82% of the copy population was `vol2col` lowering inside the fallback**.
`copy_` had been the largest line in the profile for weeks. A copy kernel would have been wasted work.

---

## The required flow: planner → backend → verification

Every Layer 5 backend follows this. P007 is the reference implementation of each step.

### 1. Planner — capability declared, legality pure

`instinctwm/backends/conv/` mirrors `backends/attention/`, with one structural difference that is the
whole lesson: **a conv backend's answer depends on the layout it is offered, and the layout is ours to
choose.** So `legality()` returns a verdict per **(backend, layout) pair** and selection ranges over
that product.

Three declarations carry measured behaviour rather than documentation:

```python
pointwise_only_off_preferred_layout = True   # 3x3x3 declined in NCDHW; 1x1x1 served in either
amortises_over = 62                          # a conversion must amortise over a subgraph
layout_changes_reduction_order = True        # so the pair is NUMERIC, not BITEXACT
```

**The tier is a property of the pair, not the backend.** The same cuDNN backend is BITEXACT when a
tensor already arrives channels-last and NUMERIC when reaching it required a conversion. A
backend-only tier would have been wrong in one of the two cases — which is the concrete reason layout
cannot stay an implementation detail.

### 2. Backend — selection needs measurement *and* consent

`select()` takes `measured: {(backend, layout) -> ms}`. Absent measurement it returns the incumbent;
a *guessed* speed term would be the reputation-ranking this project keeps rejecting. And with the
default ceiling it returns the incumbent **even when handed measurements showing a 10× win** — only an
explicit `prefer_bitexact=False` lets a NUMERIC pair through, because Layers 2–3 are gated at
`max |Δ action| = 0` and a silent downgrade would invalidate that.

Applying is separate from choosing (`backends/conv/apply.py`), the same decide/act seam as the rest of
the stack. A plan can be inspected — and refused — before anything is mutated.

**Every subgraph must be converted, not the obvious one.** LingBot-VA runs two VAEs: full-resolution
for the head camera, half-resolution for the two wrist cameras. Converting only the first leaves two
thirds of the encode on the fallback path, which reads as the optimization underperforming rather than
as a missed subgraph.

### 3. Verification — the evidence must match the tier

| tier | evidence | why |
|:--|:--|:--|
| BITEXACT | `max |Δ action| = 0` on paired seeded cycles | outputs are identical by construction |
| **NUMERIC** | **paired non-inferiority, margin declared first** | outputs differ; equality is unavailable |
| BEHAVIORAL | paired non-inferiority, and it cannot be gated by `max|Δ|` at all | non-deterministic |

`Released.is_verified()` now **refuses** a non-BITEXACT pass that has no `certificate` field. The
failure mode it prevents is specific: a lossy pass inheriting the credibility of six bit-exact ones.
`summary()` prints the evidence kind beside every pass and warns that the chain is no longer bit-exact
end to end.

P007's certificate: margin **−0.05 declared before the run**, both arms at 2V/4A on identical pinned
seeds so only the layout differs, **555 paired episodes** — baseline 0.9117, conv-layout 0.9081, delta
**−0.0036**, exact McNemar p = 0.897, one-sided non-inferiority **p = 0.00031**. Latency measured under
**ABBA** ordering (base, treat, treat, base) so within-session drift cancels.

---

## Attribution comes before all of it

`instinctwm/verify/attribution.py` reports, per **(operator, callsite)**: calls, bytes, shapes, and
measured exclusive device time. An operator total is not a target — `copy_` proved that twice.

**Coverage is the load-bearing feature.** Four attribution attempts failed, and the dangerous one did
not look like failure: dispatch-mode counting produced a confident table from 12% of the calls, its top
row read "47.4% of watched calls," and a kernel was written against it. So `coverage()` compares
attributed calls against the profiler's count from an *uninstrumented* pass, and any operator below 60%
is stamped **`[PARTIAL, NOT RANKABLE]`**.

The tool is validated on a synthetic distribution with a deliberately inverted ranking, not on real
data — if the instrument is wrong, real numbers are wrong in a way that looks plausible. That test
caught three bugs in the tool, each of which would have produced a believable wrong table.

---

## Checklist for the next Layer 5 backend

1. **Attribute first.** Operator × callsite, with coverage ≥ 60%. Do not choose a target otherwise.
2. **Classify the callsite** into dispatch / layout / materialization / fusion / kernel. Take the
   cheapest class that applies.
3. **Declare the capability**, including layout, and let the tier be *derived* rather than claimed.
4. **Make selection require measurement**, and require explicit consent to leave BITEXACT.
5. **Measure at three scales**: region, block/subgraph, and full cycle. A region win is not a win —
   the RoPE kernel was 1.10× at region scale and 0.3% of the cycle.
6. **Gate with evidence matching the tier.** NUMERIC means a certificate, not a bit-exactness check.
7. **Register it** in `verify/released.py` with the certificate, so `is_verified()` can refuse it.

## Current Layer 5 state

| | ms/cycle | share of GPU busy |
|:--|--:|--:|
| elementwise / layout | 74.0 | 38.6% |
| matmul / projections | 60.2 | 31.4% |
| attention | 44.4 | 23.2% |
| normalisation | 10.1 | 5.3% |
| **GPU busy** | **191.7** | of a 330.2 ms cycle → **42% idle** |

Forwards are now 87.4% of the cycle, up from 60.4%. The next candidate is `cat` — 19.68 ms, 172 calls,
114.4 µs each, one shape signature, unchanged by P007 — and per step 1 it does not get touched until
its callsite distribution is measured with adequate coverage.

## Further reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — the two seams this layer sits on
- [PROFILE.md](PROFILE.md) — the measurements, and the retractions behind them
- [ATTENTION.md](ATTENTION.md) — Layer 4, same planner/backend shape
- [`verify/released.py`](instinctwm/verify/released.py) — P007 and its certificate
