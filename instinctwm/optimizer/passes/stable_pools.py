"""StableStatePools (E1 / P006) -- reset clears logical state without reallocating device buffers.

WHY

`_reset` re-runs `init_kv_cache`, which allocates fresh pool tensors for all 30 layers. Any CUDA
graph captured in the previous episode still writes to the OLD addresses, so P005 has to throw its
graphs away at every reset and recapture ~60 of them. Measured cost of that recapture:

    graphs dropped per reset (P005 as shipped) : 1842.0 ms
    graphs surviving resets (unsafe, for size) : 1208.2 ms
    ------------------------------------------------------
    recoverable                                : ~634 ms

This pass makes the pools address-stable. It does NOT yet cash in the 634 ms, because keeping
graphs across a reset is still not provably correct -- see STATUS below.

STATUS (2026-08-02): DONE. Graph preservation is ON by default, gated by the certificate.

    probe_latency          : 2539.9 -> 1211.3 ms  = 2.10x  (repeats 3, spread 0.0%)
    probe_bitexact         : max|delta action| = 0.000e+00, run after 5 resets
    probe_reset_isolation  : max|delta action| = 0.000e+00
    runtime                : resets_survived=5, pool_reuses=150, cross_stable=30

Three episode-scoped dependencies had to be removed to get here. The first two were found by hand
after a wrong number; the THIRD was found automatically by `engine/deps.py`, which reported that
90 of the captured region's read buffers had no name and 89 of them moved across a reset:

  1. FIXED -- `clear_cache` sets `attn_caches[name] = None` right before `create_empty_cache`, so
     `init_kv_cache` never saw anything to reuse (pool_reuses=0 on the real server while the unit
     test happily reported 12/12). Now the buffers are parked and restored.
  2. FIXED -- P002's cross-attention K/V is read inside the captured region and was rebuilt into
     fresh tensors every episode. Stabilizing only the self-attention pools left episode 2
     returning `nan`. Now repopulation copies into the same storage.
  3. FIXED -- P004's hoisted fp32 parameter casts (`_iwm_w32`, `_iwm_b32`, `_iwm_sst32`) are read
     inside the captured region, and `hoist_invariant_casts._reset` deleted them, so all 90 were
     reallocated every episode. They are now refreshed IN PLACE with `copy_`, which keeps the
     storage and still propagates a genuine weight change.

KNOWN GAP: the `_hoisted` arm of the certificate reports 0 buffers in the running server, so it is
covering nothing. P004 creates those casts lazily and neither bind point (reset, first capture) has
been made to see them yet -- unresolved. Correctness does not currently depend on it (the casts are
address-stable by construction now, and all three gates pass), but the certificate is weaker than
it reads, which is precisely the failure mode this file is supposed to end. Fix before relying on
the certificate for a new pass.

The lesson is the reusable part: a pointer-stability certificate is only as good as its coverage,
and both (1) and (2) were cases of certifying a subset of the state the graph actually touches. The
certificate must fail CLOSED for anything it does not cover, which is why preservation is gated
behind an explicit flag instead of being inferred from a passing check.

WHAT "CLEAR LOGICAL STATE" MEANS HERE

The pools are (k, v, mask, id, is_pred) per layer plus the ring's (start, count, pred, next_id).
On reset:

  * k / v are NOT zeroed. 30 layers x 9792 slots x 24 heads x 128 dims x 2 tensors x 2 bytes is
    ~3.5 GB of pointless writes per reset, and the ring only ever reads [start, start+count), which
    is empty after a reset. That is a claim about reachability, so it is GATED, not assumed:
    `probe_reset_isolation.py` runs prompt B on a server that already ran prompt A and requires
    bitwise equality with a server that only ever ran B.
  * mask / id / is_pred ARE cleared in place, because `clear_pred_cache` and the stock fallback
    path read them and would otherwise see the previous episode's slots.
  * the ring is reset to the values `init_kv_cache` would have produced.

WHY POINTER STABILITY IS CHECKED AND NOT TRUSTED

A stale graph does not fail loudly; it computes on whatever now occupies the address. That is how
the previous round produced `nan` on episode 2 rather than an exception. So this pass records every
pool's `data_ptr()` and exposes `pointers_stable()`; P005 keeps its graphs only when that returns
True, and drops them otherwise. The safe behaviour is the default, and preserving graphs is earned.

Tier: BITEXACT.
"""

from __future__ import annotations

import torch

from instinctwm.optimizer.contract import (
    Applicability, BenchResult, CostTerm, DeviceProfile, Discovery, HardwareReq, Tier,
    VerifyResult,
)

_POOL_KEYS = ("k", "v", "mask", "id", "is_pred")


class StableStatePools:
    name = "stable_state_pools"
    hardware = HardwareReq()

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.n_reuses = 0
        self.n_allocs = 0
        #: cache_name -> {(layer_index, pool_name): data_ptr} recorded at first allocation
        self._ptrs: dict[str, dict] = {}
        self._modules: list = []
        self._cross: list = []
        self._cross_ptrs: dict = {}
        self._has_cross = False
        self._hoisted: list = []

    def applicability(self, spec, device: DeviceProfile) -> Applicability:
        return Applicability(
            True,
            "the KV pools are EPISODE-scoped in content but MODEL-scoped in shape; reallocating "
            "them per episode invalidates every captured graph",
            discovery=Discovery.AUTO,
            cost_term=CostTerm.FIXED,
            claimed_tier=Tier.BITEXACT)

    def expected_delta_ms(self, spec, device: DeviceProfile) -> float:
        return 634.0        # measured recapture gap, amortized per cycle by the caller

    # ---- install -----------------------------------------------------------------------------
    def install(self, server_module, server_cls) -> list[str]:
        import modules.model as M

        pass_self = self
        Attn = M.WanAttention
        _orig_init = Attn.init_kv_cache
        _orig_clear = Attn.clear_cache

        def clear_cache(self, cache_name):
            """PARK the pools instead of dropping them.

            Stock `clear_cache` does `attn_caches[cache_name] = None`, and `_reset` calls it just
            before `create_empty_cache`. That is what defeated the first version of this pass:
            `init_kv_cache` always saw None and always allocated (measured: pool_reuses=0 on the
            real server, while the unit test with 12/12 reuses passed happily).

            Setting the entry to None is externally visible -- `ring_kv.forward` treats a None
            cache as "no KV pool, use the stock path" -- so that behaviour is preserved exactly.
            The buffers just move to a side table where `init_kv_cache` can find them again.
            """
            if self.attn_caches is None:
                return
            parked = getattr(self, "_iwm_parked", None)
            if parked is None:
                parked = self._iwm_parked = {}
            cur = self.attn_caches.get(cache_name)
            if isinstance(cur, dict) and cur.get("k") is not None:
                parked[cache_name] = cur
            return _orig_clear(self, cache_name)

        def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                          device, dtype, batch_size):
            existing = (self.attn_caches or {}).get(cache_name)
            if existing is None:
                existing = getattr(self, "_iwm_parked", {}).get(cache_name)
            want_kv = (batch_size, total_tolen, num_head, head_dim)

            # `torch.device("cuda") != torch.device("cuda:0")`, so comparing the raw arguments
            # makes every reset look like a device change and reuse never fires. Normalize
            # through an actual allocation's device instead.
            want_dev = torch.empty(0, device=device).device
            reusable = (
                existing is not None
                and existing.get("k") is not None
                and tuple(existing["k"].shape) == want_kv
                and existing["k"].dtype == dtype
                and existing["k"].device == want_dev
            )
            if not reusable:
                _orig_init(self, cache_name, total_tolen, num_head, head_dim,
                           device, dtype, batch_size)
                pass_self.n_allocs += 1
                # deliberately NOT recording pointers here -- see `_record`
                return

            # Same buffers, cleared logical state. k/v are deliberately left alone; see module
            # docstring, and probe_reset_isolation.py for the gate that proves it is safe.
            c = existing
            self.attn_caches[cache_name] = c        # un-park
            c["mask"].fill_(False)
            c["id"].fill_(-1)
            c["is_pred"].fill_(False)
            r = c.get("_ring")
            if r is not None:
                r.update(start=0, count=0, pred=0, next_id=0)
            pass_self.n_reuses += 1

        Attn.init_kv_cache = init_kv_cache
        Attn.clear_cache = clear_cache
        self._attn_cls = Attn

        # ---- P002's cross-attention K/V is ALSO read inside the captured region ---------------
        # Stabilizing only the self-attention pools was not enough: `populate_cross_cache` builds
        # fresh tensors every episode, so the graphs still read freed memory and episode 2 still
        # produced nan. The lesson generalizes -- a stability certificate has to cover ALL device
        # state the captured region touches, not the subset that was top of mind.
        Model = M.WanTransformer3DModel
        if hasattr(Model, "populate_cross_cache"):
            _orig_populate = Model.populate_cross_cache

            def populate_cross_cache(self, text_emb):
                parked = [getattr(b.attn2, "_iwm_parked_cross", None) for b in self.blocks]
                _orig_populate(self, text_emb)
                for b, old in zip(self.blocks, parked):
                    new = getattr(b.attn2, "_iwm_cross_kv", None)
                    if old is None or new is None or len(old) != len(new):
                        continue
                    if any(o.shape != n.shape or o.dtype != n.dtype for o, n in zip(old, new)):
                        continue
                    for o, n in zip(old, new):
                        o.copy_(n)                       # same storage, new values
                    b.attn2._iwm_cross_kv = old          # publish the STABLE tensors
                for b in self.blocks:
                    kv = getattr(b.attn2, "_iwm_cross_kv", None)
                    if kv is not None:
                        b.attn2._iwm_parked_cross = kv

            Model.populate_cross_cache = populate_cross_cache
            pass_self._has_cross = True
        return ["stable_state_pools"]

    # ---- pointer bookkeeping -----------------------------------------------------------------
    def bind(self, model) -> None:
        """Remember every piece of device state a captured graph reads."""
        self._modules = [b.attn1 for b in model.blocks if getattr(b.attn1, "attn_caches", None)]
        for m in self._modules:
            for name in list(m.attn_caches or {}):
                self._record(m, name)
        # cross-attention K/V (P002) is read inside the captured region too
        self._cross = [b.attn2 for b in model.blocks
                       if getattr(b.attn2, "_iwm_cross_kv", None) is not None]
        self._cross_ptrs = {id(a): tuple(t.data_ptr() for t in a._iwm_cross_kv)
                            for a in self._cross}
        # P004's hoisted fp32 casts are read inside the captured region too. Derived, not
        # remembered: engine/deps.py reported 90 unnamed read buffers, 89 of which moved.
        hoisted = []
        for blk in getattr(model, "blocks", []) or []:
            for _n, mod in blk.named_modules():
                for attr in ("_iwm_w32", "_iwm_b32", "_iwm_sst32"):
                    t = getattr(mod, attr, None)
                    if t is not None:
                        hoisted.append((mod, attr, t.data_ptr()))
        # P004 creates these lazily on the first forward, and `bind()` runs right after a reset --
        # so a bind that finds none must not erase a previously discovered set, or the certificate
        # silently covers zero buffers while reporting success.
        if hoisted or not self._hoisted:
            self._hoisted = hoisted

    def _record(self, module, cache_name) -> None:
        """Snapshot pool addresses. ONLY called from `bind()`.

        It used to be called from `init_kv_cache` too, which made the detector fail OPEN: a reset
        that reallocated would re-record the new pointers before anyone compared them, so
        `pointers_stable()` cheerfully returned True while every captured graph pointed at freed
        memory. A detector that fails open is worse than no detector, because it converts a loud
        crash into wrong actions. The baseline must only move when someone explicitly re-binds.
        """
        c = (module.attn_caches or {}).get(cache_name)
        if c is None or c.get("k") is None:
            return
        book = self._ptrs.setdefault(cache_name, {})
        book[id(module)] = tuple(
            c[k].data_ptr() for k in _POOL_KEYS if isinstance(c.get(k), torch.Tensor))

    def pointers_stable(self, model=None) -> tuple[bool, str]:
        """True iff every recorded pool still lives at the address it was recorded at.

        This is what P005 consults before keeping graphs across a reset. A False here is not an
        error -- it just means the graphs must be dropped, which is the pre-E1 behaviour.
        """
        mods = self._modules or ([b.attn1 for b in model.blocks] if model else [])
        if not mods:
            return False, "no pools bound; refusing to certify stability"
        for m in mods:
            for name, c in (m.attn_caches or {}).items():
                if c.get("k") is None:
                    continue
                want = self._ptrs.get(name, {}).get(id(m))
                got = tuple(c[k].data_ptr() for k in _POOL_KEYS
                            if isinstance(c.get(k), torch.Tensor))
                if want is None:
                    return False, f"pool {name!r} on a layer was never recorded"
                if want != got:
                    return False, (f"pool {name!r} moved: {want[:2]} -> {got[:2]} "
                                   f"(a reallocation happened; captured graphs are stale)")
        # cross-attention K/V. Omitting this is what let the first E1 attempt certify stability
        # while episode 2 still returned nan.
        for a in self._cross:
            kv = getattr(a, "_iwm_cross_kv", None)
            if kv is None:
                return False, "cross-attention K/V is absent on a layer that had it at bind time"
            if tuple(t.data_ptr() for t in kv) != self._cross_ptrs.get(id(a)):
                return False, "cross-attention K/V moved (P002 repopulated into new tensors)"
        for mod, attr, want in self._hoisted:
            t = getattr(mod, attr, None)
            if t is None or t.data_ptr() != want:
                return False, f"hoisted cast {attr} moved or was dropped (P004 reset behaviour)"
        return True, (f"all {len(mods)} layers' KV pools, {len(self._cross)} cross-attn caches and "
                      f"{len(self._hoisted)} hoisted casts stable across reset")

    def stats(self) -> str:
        return (f"pool_allocs={self.n_allocs} pool_reuses={self.n_reuses} "
                f"cross_stable={len(self._cross)}")

    # ---- gates -------------------------------------------------------------------------------
    def verify(self, harness) -> VerifyResult:
        d = harness.max_abs_action_delta()
        return VerifyResult(passed=(d == 0.0),
                            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
                            max_abs_delta=d,
                            detail="reset clears mask/id/is_pred and the ring in place; k/v are "
                                   "unreachable after a reset because the ring live set is empty")

    def benchmark(self, harness) -> BenchResult:
        b, a = harness.cycle_ms_before(), harness.cycle_ms_after()
        return BenchResult(passed=a < b, before_ms=b, after_ms=a)
