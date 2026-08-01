"""ForwardScratchArena (L3-P8) — stop allocating per-call scratch inside the denoise loop.

Derived, not configured. The adapter states a fact:

    SliceSpec(name="attn_reassembly_scratch", scope=Scope.FORWARD,
              residency=MANAGED, commit_mode=TRANSIENT, extent=...)

A FORWARD-scoped, MANAGED, TRANSIENT slice with a host-evaluable extent is by definition
scratch that is created and destroyed inside the loop. The optimizer reads that shape and
derives "preallocate it into a scope-bumped arena". Nothing in the manifest mentions arenas.

Cosmos3-Edge is the motivating case: 896 allocations and ~2.08 GB of alloc-and-scatter traffic
per control step. But the pass keys on the declaration, not on Cosmos3 — it will fire on any WAM
whose adapter declares per-call scratch inside the loop.

SAFETY. The two results of the targeted call site are live simultaneously as the K and V
arguments of one attention() call, so a naive shared buffer would alias them and silently compute
attention against the wrong values. `ScratchArena` makes that impossible structurally: its bump
pointer only advances within a scope and resets only at a scope boundary, so two acquires in one
scope can never return the same storage. There is no capacity to size wrong and no wraparound.
"""

from __future__ import annotations

from instinctwm.optimizer.contract import (
    Applicability, BenchResult, CostTerm, DeviceProfile, Discovery as PassDiscovery, HardwareReq,
    Tier, VerifyResult,
)
from instinctwm.state.types import CommitMode, Discovery, Residency, Scope, StateManifest, applies_to


class ForwardScratchArena:
    name = "forward_scratch_arena"
    hardware = HardwareReq()

    def applicability_l3(self, manifest: StateManifest):
        def _pred(m: StateManifest):
            hits = tuple(
                s.name for s in m.slices
                if s.scope is Scope.FORWARD
                and s.residency is Residency.MANAGED
                and s.commit_mode is CommitMode.TRANSIENT
                and s.extent is not None)
            if not hits:
                return False, ("no FORWARD-scoped MANAGED TRANSIENT slice with a host-evaluable "
                               "extent; nothing is being allocated per call inside the loop"), ()
            return True, (f"slices {list(hits)} are allocated and destroyed inside the denoise "
                          f"loop with a bounded, host-known extent"), hits
        return applies_to(manifest, detects=Discovery.NONE, predicate=_pred)

    def applicability(self, spec, device: DeviceProfile) -> Applicability:
        manifest = getattr(spec, "state_manifest", None)
        if manifest is None:
            return Applicability(False, "no StateDescriptor attached", discovery=PassDiscovery.AUTO)
        a = self.applicability_l3(manifest)
        return Applicability(a.applies, a.reason, discovery=PassDiscovery.AUTO,
                             cost_term=CostTerm.PER_STEP, claimed_tier=Tier.BITEXACT,
                             params={"slices": list(a.targets)})

    def expected_delta_ms(self, spec, device: DeviceProfile) -> float:
        calls = getattr(spec, "scratch_calls_per_step", 896)
        bytes_per = getattr(spec, "scratch_bytes", 567 * 2048 * 2)
        bw = max(device.hbm_bandwidth_gbps, 1.0) * 1e9
        # the zero-fill disappears; the scatter remains
        return calls * bytes_per / bw * 1e3

    # ---- install -----------------------------------------------------------------------------
    def install(self, runtime_module=None, attention_module=None):
        """Route get_all_seq's output buffer through a scope-bumped arena."""
        import torch

        from instinctwm.state.scratch import ScratchArena

        if runtime_module is None:
            from cosmos_framework.data.generator.sequence_packing import runtime as runtime_module
        if attention_module is None:
            from cosmos_framework.model.generator.mot import attention as attention_module

        arena = ScratchArena("cosmos3_get_all_seq")
        runtime_module._iwm_scratch = arena

        orig_get = getattr(runtime_module, "_iwm_orig_get_all_seq", None)
        if orig_get is None:
            orig_get = runtime_module.get_all_seq
            runtime_module._iwm_orig_get_all_seq = orig_get

        def get_all_seq(pack):
            if "all_seq" in pack:
                return pack["all_seq"]
            runtime_module._ensure_core_metadata(pack)
            if pack["is_sharded"]:
                raise AssertionError("get_all_seq is not supported in context parallel sharded mode")
            ci, fi = pack["_causal_indices"], pack["_full_indices"]
            cs = pack["causal_seq"]
            shape = (int(ci.shape[0] + fi.shape[0]), *cs.shape[1:])
            out = arena.acquire(shape, cs.dtype, cs.device)
            # Every element is written by exactly one of the two scatters (the index sets
            # partition the sequence), so reusing a buffer needs no zero-fill -- which is also
            # what removes the new_zeros. If that partition property ever fails, the parity gate
            # sees stale data from the previous scope immediately.
            if cs.shape[0] > 0:
                out[ci] = cs[: ci.shape[0]]
            if pack["full_only_seq"].shape[0] > 0:
                out[fi] = pack["full_only_seq"][: fi.shape[0]]
            return out

        runtime_module.get_all_seq = get_all_seq

        # Scope boundary: entry to the attention call that CONSUMES the buffers. Placing it here
        # is what makes cross-scope reuse safe -- the results never outlive this call.
        orig_attn = getattr(attention_module, "_iwm_orig_two_way_attention", None)
        if orig_attn is None:
            orig_attn = attention_module.two_way_attention
            attention_module._iwm_orig_two_way_attention = orig_attn

        def two_way_attention(*a, **kw):
            arena.begin_scope()
            return orig_attn(*a, **kw)

        attention_module.two_way_attention = two_way_attention
        # the module-level import in attention.py binds get_all_seq by value, so rebind it too
        if hasattr(attention_module, "get_all_seq"):
            attention_module.get_all_seq = get_all_seq
        return arena

    @staticmethod
    def uninstall(runtime_module, attention_module) -> None:
        g = getattr(runtime_module, "_iwm_orig_get_all_seq", None)
        if g is not None:
            runtime_module.get_all_seq = g
            if hasattr(attention_module, "get_all_seq"):
                attention_module.get_all_seq = g
        a = getattr(attention_module, "_iwm_orig_two_way_attention", None)
        if a is not None:
            attention_module.two_way_attention = a

    def verify(self, harness) -> VerifyResult:
        d = harness.max_abs_delta()
        return VerifyResult(passed=(d == 0.0),
                            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
                            max_abs_delta=d,
                            detail="same scatter, same order, same kernels; only the destination "
                                   "buffer's provenance changes")

    def benchmark(self, harness) -> BenchResult:
        b, a = harness.ms_before(), harness.ms_after()
        return BenchResult(passed=a < b, before_ms=b, after_ms=a)
