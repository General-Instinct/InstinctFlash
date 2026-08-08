# The ~139 ms of device-timeline gaps: diffuse, and it is the eager-runtime floor

**One diagnostic, no optimization. Verdict: DIFFUSE. Stop chasing micro-optimizations at this operating
point.**

Probe: [`probe_device_gaps.py`](eval/lingbot_va_robotwin/probe_device_gaps.py). 2V/4A, warm 70 cycles past
ring saturation, shipped stack (P003 ring KV + P007 conv layout), keyframes generated *outside* the measured
window so 5.5 MB/cycle of numpy randomness is not mistaken for runtime cost.

## The number, and how the instrument nearly ruined it

| instrument | window | vs unprofiled | device busy | measured gaps |
|:--|--:|--:|--:|--:|
| **unprofiled** (median of 12) | **330.7 ms** | 1.00× | — | — |
| CUDA activities only | 428.0 ms | 1.29× | 191.8 ms | 236.2 ms |
| CPU + CUDA | 590.6 ms | 1.79× | 191.4 ms | 399.2 ms |
| CPU + CUDA + 1,200 named scopes | 668.3 ms | 2.02× | 190.9 ms | 477.5 ms |

**The profiler manufactures the very quantity under study.** It instruments the host, the host falls behind,
and every microsecond of that lands in a device gap. A first attempt at this probe reported 486 ms of gaps
and would have been reported as fact; the control above is what caught it.

**Device busy is undistorted** — 190.9 to 191.8 ms across all three instruments, and consistent with the
191.7 ms measured independently in Layer 5. So the true total comes from subtraction, not from the trace:

> **true gap total = 330.7 − 191.8 = 138.9 ms**, of which the CUDA-only pass measures 236.2 ms and the fully
> instrumented pass 477.5 ms. Absolute per-gap milliseconds below are inflated by ~1.7×; **counts are not.**

## 1. Concentrated or diffuse — the decision

**18,605 device events → 18,589 gaps, identically in every pass.** The count is a property of the workload,
not of the instrument.

> **138.9 ms / 18,589 gaps = 7.5 µs mean gap.**

| bucket | count | measured ms | mean µs |
|:--|--:|--:|--:|
| < 5 µs | 6,767 | 9.8 | 1.4 |
| 5–20 µs | 8,531 | 101.8 | 11.9 |
| 20–100 µs | 3,258 | 114.0 | 35.0 |
| 100 µs – 1 ms | **32** | 6.7 | 208.9 |
| 1–10 ms | **1** | 3.9 | 3860 |

**Only 33 gaps in the entire cycle are ≥ 100 µs, and they measure 10.6 ms under an instrument that
inflates.** So at most 10.6 of 138.9 ms — **≤ 7.6%** — sits in gaps large enough to have an individual
cause. **At least 92% is in gaps under 100 µs**, and 15,298 of them are under 20 µs.

That is diffuse by any reading of the data, and the conclusion does not depend on the inflated
milliseconds: 33 gaps cannot hold 139 ms.

## 2. Why the device was idle — launch timing of the kernel that ends each gap

Every device event carries a `correlation` id shared with the runtime call that issued it, so for the kernel
that *ends* a gap we can ask when its launch was actually made.

| | gaps | measured ms | % of gap |
|:--|--:|--:|--:|
| **host had not launched it yet** | 9,033 | 163.5 | **69.2%** |
| already enqueued, device still idle | 6,496 | 11.2 | **4.8%** |
| no correlation recorded | 3,060 | 61.5 | 26.0% |

**Dependency and serialization stalls are 4.8%.** The device is not waiting on the device. In 69% of gap
time the next kernel had not been issued yet — the queue was empty because the host had not got there.

## 3. Cause classification

| cause | gaps | % of gap | mean µs |
|:--|--:|--:|--:|
| LAUNCH-STARVED (only `cudaLaunchKernel` in the gap) | 12,595 | **63.7%** | 24.2 |
| HOST-ONLY (dispatches, no launch at all) | 2,672 | **31.1%** | 55.6 |
| HOST-ONLY (no dispatch at all) | 3,209 | 3.0% | 4.4 |
| SYNCHRONIZATION | 117 | **2.2%** | 88.7 |
| ALLOCATOR (driver call) | **0** | **0%** | — |

**Three mechanisms that are commonly blamed are absent here.** Not one `cudaMalloc`/`cudaFree` appears in
any gap — the caching allocator is doing its job. Synchronization is 2.2%, consistent with the 16 genuine
device→host reads found earlier in `FlowMatchScheduler.step`. Pure CPU work with no dispatch at all is 3.0%.

**95% of gap time is the host walking the eager dispatch path between kernel launches.**

## 4. Where the gaps are

| scope | gaps | % of gap | cumulative | mean µs |
|:--|--:|--:|--:|--:|
| `attn.self` | 6,020 | 33.5% | 33.5% | 26.5 |
| `block.forward` | 8,061 | 33.3% | 66.8% | 19.7 |
| `attn.cross` | 2,005 | 15.9% | 82.6% | 37.8 |
| `ffn` | 900 | 6.4% | 89.1% | 34.2 |
| `transformer.forward` | 616 | 6.3% | **95.4%** | 48.7 |
| `encode_obs` | 799 | 2.1% | 97.4% | 12.4 |
| `infer` | 70 | 1.1% | 98.5% | 73.2 |
| `prepare_latent_input` | 62 | 0.8% | 99.3% | 64.5 |
| `compute_kv_cache` | 42 | 0.3% | 99.6% | 29.0 |
| `action_scheduler.step` | 8 | 0.2% | 99.8% | 130.5 |
| `postprocess_action` | 2 | 0.1% | 99.9% | 212.0 |

**95.4% inside the transformer stack**, which agrees with every other instrument: 94.3% of dispatcher ops,
94.5% of aten events, and 300 block executions accounting for the whole cycle. There is no gap hiding
outside the model.

The three top scopes are 82.6% cumulatively — but that is *not* concentration in the actionable sense. They
are 16,086 separate gaps averaging 26 µs. A scope containing a third of the gap time in six thousand pieces
offers nothing to optimize.

## 5. The largest gaps, individually

The one gap worth naming is the largest in the cycle, and it is not in the model:

```
gap 1: 3,414 µs   [SYNCHRONIZATION]   scope=encode_obs
  prev device : vectorized_elementwise_kernel
  next device : Memcpy HtoD (Pageable -> Device)
  host op live: aten::upsample_bilinear2d
  inside      : sync=1 memcpy=1 | dispatches=98 python-originated=23 metadata=18
```

**A pageable host→device copy of the raw camera frames**, serialized because the source is pageable rather
than pinned. It is 3.9 ms of a 331 ms cycle — **1.2%** — and it is the single most concentrated thing in the
entire gap inventory. Pinning that staging buffer is a real, small, self-contained fix. It is also the whole
list.

Below it, the next 11 gaps are 300–610 µs each, sum to ~4.5 ms, and are ordinary launch starvation inside
`attn.self` and `block.forward` with 1–78 dispatches and a single launch inside them. Nothing recurs.

## 6. Dispatch origin, now measured rather than estimated

The trace settles the distinction that Layer 6 section I had to infer from cProfile. A `cpu_op` nested
inside another `cpu_op` on the same thread is a C++-internal redispatch; one with no `cpu_op` parent was
entered from Python.

| | count | at 1.02 µs |
|:--|--:|--:|
| `cpu_op` events in the window | 105,130 | — |
| **Python-originated** | **34,635** | **35.3 ms** |
| C++-internal redispatch | 70,495 | ~0 |

**35.3 ms, not the ~56 ms estimated in LAYER6.md section I.** So Python-originated dispatch accounts for
**25% of the 138.9 ms**, and the ceiling on all Python-level tidying is lower than previously stated. The
remaining ~104 ms is spread across 18,605 launches at ~5.6 µs each: argument marshalling, kernel
configuration, and the `cudaLaunchKernel` driver path itself, none of which Python-level changes touch.

## The verdict

**Diffuse.** Per the decision rule set for this pass: **this is the practical eager-runtime floor at 2V/4A,
and micro-optimization stops here.**

The floor is structural and its shape is now known: **~7.5 µs of host issue cost per kernel launched, times
18,605 launches.** It is not synchronization (2.2%), not the allocator (0%), not dependency stalls (4.8%),
and not concentrated in any callsite (33 gaps ≥ 100 µs, ≤7.6% of the total).

Two levers exist against a per-launch floor, and both were already measured and rejected on their own costs:

| lever | status |
|:--|:--|
| **fewer kernels** (fusion) | fused QKV removed ~1,200 of 18,605 launches, predicted 1.9%, **measured 0.2% slower** — the fused GEMM lost more on-device than the launches saved |
| **no launches** (graph replay, `torch.compile`) | graph capture **1.43× slower** ([result](LAYER5_GRAPH_PERSISTENCE_RESULT.md)); `torch.compile` removed **zero** ops and costs **318.9 s** to build ([LAYER6.md](LAYER6.md) §D) |

So the mechanism class that could claim the 139 ms is exactly the class already priced out, and the
mechanisms that are affordable are worth single-digit milliseconds each. That is a floor, not a to-do list.

### The one exception, and it is small

**Pin the observation staging buffer.** 3.9 ms, 1.2% of the cycle, one pageable H2D copy in `encode_obs`,
independent of everything above, and the only gap in the cycle with a nameable individual cause. Worth doing
on its own merits; not worth calling an optimization programme.

### What was not measured, deliberately

- **The marginal cost of one kernel launch.** Section I measured 1.017 µs for a *no-kernel* dispatch by
  injection. The equivalent sweep for a launch-producing op would price fusion exactly, and it was not run:
  this pass was scoped to explain the gaps, not to build another cost model.
- **Per-gap milliseconds are inflated ~1.7×.** Only the 138.9 ms total (by subtraction), the gap counts, and
  the bucket counts are load-bearing. Bucket *shares* could shift; the ≤7.6% concentration bound cannot,
  because it rests on a count of 33.
- **26% of gap time has no launch correlation recorded**, so the 69/5 split between "host late" and
  "already enqueued" is a split of the attributable 74%, not of the whole.
- **This is 2V/4A only.** At Quality (25V/50A, 79 forwards per cycle) the device term grows ~8× while the
  per-launch floor grows with kernel count; the balance is different and this verdict does not transfer.

## Further reading

- [LAYER6.md](LAYER6.md) — the dispatch-elimination inventory, its refutation, and the 1.017 µs slope
- [LAYER5_CRITICAL_PATH.md](LAYER5_CRITICAL_PATH.md) — the retracted host-throughput model
- [LAYER5_GRAPH_PERSISTENCE_RESULT.md](LAYER5_GRAPH_PERSISTENCE_RESULT.md) — the frozen replay attempt
- [LAYER5.md](LAYER5.md) — P007, correctly attributed to backend/layout on the device
