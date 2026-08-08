# Making one captured graph survive ring advancement

Design only. Measured 2026-08-08, 2V/4A, P007 applied.
Probes: [`probe_graph_scope.py`](eval/lingbot_va_robotwin/probe_graph_scope.py),
[`probe_critical_path.py`](eval/lingbot_va_robotwin/probe_critical_path.py).

P005 was measured as a device optimization and rejected at Fast on that basis. From the critical-path
result it is not primarily a device optimization: **graph replay eliminates host dispatch, and host
dispatch is the binding chain.** So the question is what stops one graph from serving every cycle.

**Answer: exactly one quantity, and only after cycle 36.** `count` changes the read *shape* until the
ring saturates and nothing can absorb that. `start` changes only *addressing*, and it is the sole reason
captures never stop afterwards. Moving the write offset into a device-resident buffer takes
post-saturation captures from 6/cycle to **zero**.

---

## 1. Which fields force a new graph

The key is `(hidden.shape, tproj.shape, update_cache, cache_name, ring_signature)` where
`ring_signature = (start, count)` (`graph_capture.py:163-174`).

| field | values seen | forces recapture because |
|:--|:--|:--|
| `hidden.shape` | 2 (240 / 32 tokens) | genuine shape change |
| `tproj.shape` | follows `hidden` | genuine shape change |
| `update_cache` | 3 (0, 1, 2) | control flow: `update_cache=0` rolls the write back |
| `cache_name` | 1–2 | selects the pool |
| **`count`** | **36 distinct, then pinned** | **read extent — a SHAPE** |
| **`start`** | **unbounded, advances every cycle after 36** | **read + write OFFSET — addressing only** |

Together these give the measured **6 captures/cycle** — and they never stop, which is the anomaly.

## 2. Measured ring progression: the two regimes

| cycle | `start` | `count` |
|--:|--:|--:|
| 1 | 0 | 544 |
| 10 | 0 | 2,992 |
| 30 | 0 | 8,432 |
| 34 | 0 | 9,520 |
| **36** | **272** | **9,792 = total** |
| 50 | 4,080 | 9,792 |
| 70 | 9,520 | 9,792 |

**`count` stops growing at cycle 36 and `start` starts advancing.** These are disjoint regimes, and that
is the whole design:

- **Cycles 1–35:** `start = 0` throughout, `count` grows. The key varies *only* by `count`.
- **Cycles 36+:** `count` pinned at `total`, `start` advances. The key varies *only* by `start`.

So "captures never stop" is caused **entirely by `start`**, and only in the second regime.

## 3. Shape versus addressing, per use site

Read (`ring_kv.py:191-199`):

```python
start, count = r["start"], r["count"] + key_size
if count >= total:            key_all = kp                              # whole pool: FIXED shape, no start
elif start + count <= total:  key_all = kp[:, start:start+count]        # shape=count, offset=start
else:                         key_all = cat([kp[:,:end], kp[:,start:]]) # shape=count, and a copy
```

Write (`ring_kv.py:167-171`):

```python
head = (r["start"] + r["count"]) % total
kvc["k"][:, head:head+key_size] = key                                   # offset=head, shape FIXED
```

| quantity | affects tensor shape | affects addressing only | device-resident candidate |
|:--|:--|:--|:--|
| `count`, pre-saturation | **yes** — attention K/V length | via `head` | **no** |
| `count`, post-saturation | no (== `total`) | via `head` | n/a, constant |
| `start`, post-saturation | **no** — the read takes the `count >= total` branch and never mentions `start` | **yes** — write offset via `head` | **yes** |
| `head` | no — extent is `key_size` | **yes** | **yes** |
| `pred`, `next_id`, `mask`, `id`, `is_pred` | no | no | already host-deferred by P005 v1.0.1 |

**The decisive line:** post-saturation the read is `key_all = kp` — the whole pool, a view with a fixed
pointer and fixed shape. It does not reference `start` at all. So after cycle 36 the *only* graph-relevant
use of ring state is the write offset.

## 4. The minimal design: a device-resident Plan Buffer for the write offset

Replace the baked slice-write with an indexed write whose index lives in a fixed-address device buffer:

```python
# today -- `head` is a Python int, so the destination address is baked at capture
head = (r["start"] + r["count"]) % total
kvc["k"][:, head:head+key_size] = key

# proposed -- the ADDRESS of write_idx is baked; its CONTENTS are read at replay
kvc["k"].index_copy_(1, plan.write_idx, key)      # plan.write_idx: int64[key_size], fixed allocation
kvc["v"].index_copy_(1, plan.write_idx, value)
```

`plan.write_idx` is allocated once per (layer, cache_name) at install and never reallocated. Between
replays, `_commit_all` — which already runs on the host after every replay — refreshes its contents:

```python
plan.write_idx.copy_(arange_buffer + head)        # one small H2D or device-side add, outside the graph
```

A CUDA graph bakes the pointer, not the payload, so mutating the buffer's contents redirects the write
with no recapture. This is the same mechanism P006 already relies on for pool stability: the graph holds
addresses, and the pass guarantees the addresses do not move.

The graph key then drops `start` in the saturated regime:

```python
sig = (count,) if count < total else ("saturated",)
```

### Is exact KV semantics preserved?

Yes, and for a reason rather than by measurement:

- **The write.** `index_copy_` into `total`-length dim 1 with a contiguous ascending index range writes
  the same bytes to the same slots as the slice assignment. It is a copy, not arithmetic — no rounding, no
  reduction, and the destinations are disjoint so write order is irrelevant. Bit-exact by construction.
- **The read.** Unchanged. Post-saturation it was already `kp`, the whole pool in ascending slot order,
  which is the order stock's `mask.nonzero()` produces and the order P003 preserves for bit-exactness.
- **The metadata.** `mask`, `id`, `is_pred`, `next_id`, `pred` already live on the host after every replay
  (P005 v1.0.1). They need the same `head`, which the host still computes — nothing moves into the graph.
- **`update_cache == 0` rollback.** Still host-side, still after replay, still using the host's `head`.

The one thing that must be gated rather than argued: `index_copy_` and slice-assignment must dispatch to
kernels that produce identical bytes. That is a `max |Δ| = 0` check on the pool after a write, not an
argument — and it is cheap.

## 5. Captures per episode after each change

A RoboTwin episode is ~53 cycles; saturation is at 36.

| variant | captures/episode | fully-replayed cycles |
|:--|--:|--:|
| today | 53 × 6 = **318** (and 204 evictions — the cache thrashes) | 0 |
| **write offset → Plan Buffer** | 36 × 6 = **216** | **18 of 53** |
| + also bucket `count` (pad the read, **NUMERIC**) | ~5 × 6 = **30** | ~48 of 53 |
| + smaller `kv_slots` so saturation comes earlier | fewer | more |

Only the first row is BITEXACT. Bucketing `count` means attending over a padded window, and `-inf`
masking to a longer sequence changes the softmax reduction order — that is a NUMERIC change and a
separate decision.

## 6. Predicted host-op reduction

Measured directly: **36,240 of 38,442 dispatcher operations per cycle (94.3%) occur inside the 30-block
stack** — the exact region `graph_capture._stack` wraps. Outside it are 2,202 ops: the VAE encode,
schedulers, and server bookkeeping.

| | |
|:--|--:|
| host ops removable by replay | **94.3%** |
| host time removed (338 ms cycle, host-bound) | ~319 ms |
| remaining host time | ~19 ms + replay launches |
| device work (unchanged) | 196 ms |
| **cycle on a fully-replayed cycle** | **→ 196 ms, device-bound** |
| **speedup on those cycles** | **1.72×** |
| episode average, 18 of 53 cycles replayed | **~1.17×** |

The per-cycle ceiling is 1.72× because the cycle becomes device-bound at 196 ms; that is the same
`max(host, device)` bound the critical-path analysis established, now reached from the other side.

**Note the asymmetry this creates.** Once fully-replayed cycles are device-bound, every rejected Layer 5
kernel becomes relevant *for those cycles* — attention's 44 ms, the `cat`'s 21 ms and the GEMM time are
all on the critical path again once the host stops being the constraint. The correct order is therefore
graph persistence first, kernels second, and the three rejections should be revisited only after the host
chain is shortened, not before.

## Correction to my own arithmetic

`probe_graph_scope.py` printed "removing the inside set saves ~116 ms" by multiplying 36,240 dispatcher
ops by 3.2 µs. That mixes two bases: **3.2 µs/op was calibrated against the profiler's 105,130 `cpu_op`
events, not against 38,442 dispatcher calls.** The profiler counts composite ops and their children
separately (`aten::linear` plus the `aten::t` and `aten::addmm` it dispatches); dispatch mode sees only
the dispatcher level. On a consistent basis the dispatcher rate is 338 ms / 38,442 = 8.8 µs/op, giving
36,240 × 8.8 µs ≈ 319 ms — which is what the table above uses, and which agrees with the 94.3% share
applied to the whole 338 ms. The 116 ms figure understates it by 2.75×.

## What to build, in order

1. **Verify `index_copy_` is byte-identical to the slice write** on the real pool shapes. One probe, and
   it gates everything downstream.
2. **Add the Plan Buffer** and route the write through it, with the host still computing `head`.
3. **Drop `start` from the graph key** in the saturated regime only, keeping `count` while it varies.
4. **Gate:** `max |Δ action| = 0` over paired seeded cycles spanning saturation (so both regimes and the
   transition are covered), then the cycle-level ABBA — measured on post-saturation cycles specifically,
   since that is where the effect lives and an episode-average would dilute it by 3×.

Nothing here proposes touching `count`. That is the shape term, it is genuinely intrinsic under
bit-exactness, and it is the reason the ceiling is 18 cycles rather than 53.
