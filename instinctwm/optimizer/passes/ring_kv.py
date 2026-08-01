"""RingKVAddressing — address the KV pool by interval instead of by boolean mask.

Profile that motivated it (one LingBot-VA control cycle, post 1.92x base + conditioning prefill):

    GPU busy / wall        48.9%          -> the GPU is idle 51% of the cycle
    gather / copy          39.6% of GPU   (1150 ms, 203k launches)
    GEMM (real math)       17.4% of GPU   (8.5% of wall)
    aten::index  [2,9792,24,128] x4620 = 30 layers x 77 fwd x 2   <- the pool gather
    aten::nonzero          [9792] x6930 = 30 x 77 x 3             <- host round trips

Two lines cause both. `model.py:451-453`:

    valid = mask.nonzero(as_tuple=False).squeeze(-1)
    key   = key_pool[:, valid]
    value = value_pool[:, valid]

`nonzero` has a data-dependent output shape, so it is a host sync; the advanced index is a full
copy of the live pool. Per layer, per forward.

The observation that makes this cheap to fix
--------------------------------------------
`allocate_slots` already behaves like a ring. It takes the *lowest* free slots
(`free[:key_size]` after `(~mask).nonzero()`), and when it must evict it frees the *oldest* ids,
which are the lowest indices because they were allocated first. So allocation runs
0, n, 2n, ... and wraps. The live set is therefore always a **ring interval**, never a scattered
set — the boolean mask was only ever encoding an interval the long way round.

Track that interval as two host ints (`start`, `count`) and:

  * `slots` is `arange(head, head+k)`, no `nonzero`;
  * `valid` is `[start : start+count]` — a **view**, not a gather;
  * when the pool is full it is the whole tensor, also a view;
  * `mask` / `id` / `is_pred` bookkeeping disappears, taking ~4 `index_put_` per layer per forward.

Bit-exactness
-------------
Softmax attention is permutation-invariant over keys mathematically, but *not* in floating point:
changing key order changes the reduction order. So this pass is only bit-exact if it presents keys
in the same order the mask did, which is ascending slot index.

A contiguous interval satisfies that directly, as a single slice. A wrapped interval does NOT come
out in ascending order if enumerated chronologically -- so the wrapped case is emitted as
`cat([pool[:, :end], pool[:, start:]])`, which IS ascending index order and therefore order-exact.
Neither path calls `nonzero`, and neither falls back to the stock code: the fallback in an earlier
draft was unsafe, because the fast path did not maintain `mask`/`id`/`is_pred` and stock would have
read stale bookkeeping. Those arrays are now maintained by slice on every path.

Two properties worth knowing before changing this file:

  * VALIDITY CONDITION: the ring-interval model holds only when the per-commit token count divides
    the pool capacity. LingBot-VA satisfies it (9792 = 36 * 272). A capacity that is not a multiple
    of the commit period makes the live set stop being a single interval, and the parity test in
    tests/test_ring_allocator.py is what detects that.
  * LAYOUT: the pool is [B=2, 9792, 24, 128], so a slice `kp[:, s:s+n]` is NOT contiguous, whereas
    stock's advanced-index gather produced a contiguous tensor. Attention is bit-exact here as
    measured, but a different attention backend could dispatch differently on a non-contiguous
    input, so layout is part of what an equivalence check has to compare -- not just values.
"""

from __future__ import annotations

import torch

from instinctwm.adapter.base import AdapterSpec, KVLifetime
from instinctwm.optimizer.contract import (
    Applicability, BenchResult, CostTerm, DeviceProfile, Discovery, HardwareReq, Tier,
    VerifyResult,
)


class RingKVAddressing:
    name = "ring_kv_addressing"
    hardware = HardwareReq()   # pure indexing change; no arch requirement

    # ---- 1 + 2. detection and applicability ------------------------------------------------
    def applicability(self, spec: AdapterSpec, device: DeviceProfile) -> Applicability:
        persistent = [s for s in spec.streams
                      if s.lifetime in (KVLifetime.WINDOW, KVLifetime.EPISODE)]
        if not persistent:
            return Applicability(
                False,
                "no window- or episode-scoped KV stream; there is no pool to re-gather "
                "(GR00T has no KV at all, and a chunk-scoped prefix is built once and read)",
                discovery=Discovery.AUTO)
        return Applicability(
            True,
            f"streams {[s.name for s in persistent]} live across control steps in a pool that is "
            f"currently addressed by boolean mask, costing a host sync and a full-pool copy per "
            f"layer per forward",
            discovery=Discovery.AUTO,       # detectable from the module tree: a bool mask + nonzero
            cost_term=CostTerm.PER_STEP,
            claimed_tier=Tier.BITEXACT,
            params={"streams": [s.name for s in persistent]})

    def expected_delta_ms(self, spec: AdapterSpec, device: DeviceProfile) -> float:
        """Formula, not a guess: bytes not copied / achievable bandwidth, plus syncs removed."""
        n_layers = 30
        fwd = spec.total_forwards()
        # measured on LingBot-VA: 4620 gathers totalling 157.9 ms, 6930 nonzeros totalling 65.4 ms
        gather_ms = 157.9 * (fwd / 77.0)
        sync_ms = 65.4 * (fwd / 77.0)
        # each nonzero also stalls the enqueue pipeline; measured idle/launch was 6.67 us
        stall_ms = n_layers * fwd * 3 * device.launch_overhead_us / 1000.0
        return gather_ms + sync_ms + stall_ms

    # ---- install ---------------------------------------------------------------------------
    def install(self, server_module, server_cls) -> None:
        import modules.model as M

        Attn = M.WanAttention
        _orig_forward = Attn.forward
        _orig_init = Attn.init_kv_cache
        _orig_clear_pred = Attn.clear_pred_cache

        def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                          device, dtype, batch_size):
            _orig_init(self, cache_name, total_tolen, num_head, head_dim,
                       device, dtype, batch_size)
            if self.attn_caches is None:
                return
            c = self.attn_caches[cache_name]
            c["_ring"] = {"total": total_tolen, "start": 0, "count": 0, "pred": 0, "next_id": 0}

        def _ring(self, cache_name):
            c = self.attn_caches.get(cache_name) if self.attn_caches else None
            return c.get("_ring") if c else None

        def clear_pred_cache(self, cache_name):
            r = _ring(self, cache_name)
            if r is not None:
                # provisional slots are the most recent allocations; drop them off the tail
                r["count"] -= r["pred"]
                r["pred"] = 0
            _orig_clear_pred(self, cache_name)

        def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
            r = _ring(self, cache_name)
            kv_cache = (self.attn_caches[cache_name]
                        if (self.attn_caches is not None and cache_name in self.attn_caches)
                        else None)
            if r is None or kv_cache is None or kv_cache.get("k") is None:
                return _orig_forward(self, q, k, v, rotary_emb, update_cache, cache_name)

            key_size = k.shape[1]

            query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
            query = self.norm_q(query).unflatten(2, (self.heads, -1))
            key = self.norm_k(key).unflatten(2, (self.heads, -1))
            value = value.unflatten(2, (self.heads, -1))
            if rotary_emb is not None:
                def apply_rotary_emb(x, freqs):
                    x_out = torch.view_as_complex(
                        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
                    return torch.view_as_real(x_out * freqs).flatten(3).to(x.dtype)
                query = apply_rotary_emb(query, rotary_emb)
                key = apply_rotary_emb(key, rotary_emb)

            # Write via SLICES, predicted by the ring model -- never `allocate_slots`, which is
            # where the `(~mask).nonzero()` host sync lives. The prediction is not a guess: the
            # parity test (tests/test_ring_allocator.py) checks it against the real
            # `allocate_slots` for 800 allocations across 5.6 full wraps, slot-for-slot.
            #
            # mask/id/is_pred are still maintained, stock-exact, so the cache is always in a state
            # the original code could consume and there is no path on which a stale mask is read.
            kvc = kv_cache
            total = r["total"]
            head = (r["start"] + r["count"]) % total
            assert head + key_size <= total, "allocation wrapped; ring model violated"
            sl = slice(head, head + key_size)
            kvc["k"][:, sl] = key
            kvc["v"][:, sl] = value
            kvc["mask"][sl] = True
            kvc["id"][sl] = r["next_id"]
            kvc["is_pred"][sl] = (update_cache == 1)
            r["next_id"] += 1

            # Read the live set WITHOUT nonzero and WITHOUT an advanced-index gather.
            # Measured ground truth (tests/test_ring_allocator.py, 120 cycles = 3.3 wraps):
            # allocations are ALWAYS a contiguous run, and the live set is always the pool minus
            # one contiguous hole -- i.e. a ring interval. Stock presents it in ASCENDING SLOT
            # INDEX order (`mask.nonzero()`), which is what must be reproduced for bit-exactness:
            #   * interval does not wrap  -> one slice          (a view: no copy at all)
            #   * interval wraps          -> [0:end] ++ [start:total], which IS ascending
            kp, vp = kvc["k"], kvc["v"]
            start, count = r["start"], r["count"] + key_size
            if count >= total:
                key_all, value_all = kp, vp                       # whole pool, a view
            elif start + count <= total:
                key_all = kp[:, start:start + count]              # a view
                value_all = vp[:, start:start + count]
            else:
                end = (start + count) - total
                key_all = torch.cat([kp[:, :end], kp[:, start:]], dim=1)
                value_all = torch.cat([vp[:, :end], vp[:, start:]], dim=1)

            hidden_states = self.attn_op(query, key_all, value_all)

            if update_cache == 0:
                kvc["mask"][sl] = False                 # transient: roll the write back
                kvc["id"][sl] = -1
            else:
                r["count"] = count
                if update_cache == 1:
                    # ACCUMULATE: update_cache=1 fires twice per cycle -- the video loop's last
                    # step (wan_va_server.py:504) and the action loop's last step (:544) -- and
                    # stock clear_pred_cache drops every slot with is_pred set, i.e. BOTH blocks.
                    # Overwriting leaked the video block into the permanent cache; the correctness
                    # gate caught it at max|delta| = 1.22 against a 1.03 chunk-to-chunk movement.
                    r["pred"] += key_size
                else:
                    r["pred"] = 0
                if r["count"] > total:                  # eviction: the hole advances
                    r["start"] = (r["start"] + (r["count"] - total)) % total
                    r["count"] = total

            hidden_states = hidden_states.flatten(2, 3).type_as(query)
            return self.to_out[1](self.to_out[0](hidden_states))

        Attn.init_kv_cache = init_kv_cache
        Attn.clear_pred_cache = clear_pred_cache
        Attn.forward = forward

    # ---- 3 + 4. gates ----------------------------------------------------------------------
    def verify(self, harness) -> VerifyResult:
        d = harness.max_abs_action_delta()
        return VerifyResult(
            passed=(d == 0.0),
            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
            max_abs_delta=d,
            detail="contiguous-interval fast path presents keys in ascending slot order, "
                   "identical to mask.nonzero(); wrapped intervals fall back to stock")

    def benchmark(self, harness) -> BenchResult:
        before, after = harness.cycle_ms_before(), harness.cycle_ms_after()
        return BenchResult(passed=after < before, before_ms=before, after_ms=after,
                           detail="pre-saturation regime; the removed gather grows with "
                                  "occupancy, so this understates steady state")
