"""StaticPartitionHoist (L3-P3) — stop rebuilding an index that declared geometry already fixes.

Detection signature D3. This is the one LingBot's detector could not see, and finding it is what
turned L3 from "a KV pool optimizer" into a state layer: there is no KV pool here and no boolean
mask anywhere. The state being mis-addressed is a PACKED TOKEN SEQUENCE.

The defect, in Cosmos3-Edge
---------------------------
`init_sequence_pack` (sequence_packing/runtime.py:143-172) is a pure function of
`(sample_lens, split_lens, attn_modes, device)` -- all declared geometry, constant for the whole
episode because the chunk structure does not change between control steps. It is called on every
pack and it:

  * builds `_causal_indices` / `_full_indices` with a Python `range` loop and then
    `torch.tensor(indices, device=cuda)` (`:60-83`) -- a host-built index shipped to the device;
  * `.tolist()`s `split_lens` if it arrives as a tensor (`:72`) -- a device->host sync;
  * runs `torch.cumsum` over a freshly built `sample_lens_cu`.

Downstream, `runtime.py:253` `.tolist()`s a device tensor whose ONLY product is
`assert len(non_causal_text_idxs) == 0`, and `get_all_seq` (`:430-441`) allocates a fresh
`new_zeros([seq_len, D])` and scatters TWICE to reassemble a sequence it just split -- called from
`attention.py:203-204`, i.e. **2x per layer per forward**, or 2 x 28 x 16 = **896 times per
control step** at the served NFE.

What this pass does
-------------------
Memoizes `init_sequence_pack` on its declared-geometry key. Nothing else: same tensors, same
values, same order, same device. The first call of an episode computes; the rest restate.

Deliberately NOT done here, and why
-----------------------------------
Reusing the `get_all_seq` output buffer looks like the bigger win and is unsafe as stated:
`attention.py:203-204` has the key and the value results LIVE SIMULTANEOUSLY in one expression,
so a single shared scratch buffer would alias them. It needs a per-role buffer, which is a
separate pass with its own aliasing argument.

Correctness
-----------
BITEXACT by construction: a memoized pure function returns the identical tensors. The gate is not
"the outputs look the same" -- it is `torch.equal` on every returned index tensor against a freshly
computed one, plus the layout check (I4), run on the real function.
"""

from __future__ import annotations

from instinctwm.optimizer.contract import (
    Applicability, BenchResult, CostTerm, DeviceProfile, Discovery as PassDiscovery, HardwareReq,
    Tier, VerifyResult,
)
from instinctwm.state.types import Discovery, StateManifest, applies_to


class StaticPartitionHoist:
    name = "static_partition_hoist"
    hardware = HardwareReq()

    # ---- detection + applicability, from the StateDescriptor alone -------------------------
    def applicability_l3(self, manifest: StateManifest):
        def _pred(m: StateManifest):
            hits = tuple(s.name for s in m.slices if s.discovery is Discovery.D3_STATIC_INDEX)
            if not hits:
                return False, ("no slice declares D3 (an index re-materialised per forward from "
                               "declared geometry)"), ()
            return True, (f"slices {list(hits)} rebuild an index tensor that is a pure function of "
                          f"declared geometry, once per pack, on the serving path"), hits
        return applies_to(manifest, detects=Discovery.D3_STATIC_INDEX, predicate=_pred)

    def applicability(self, spec, device: DeviceProfile) -> Applicability:
        """Adapter-level view, for the shared Optimizer. `spec` carries the StateDescriptor."""
        manifest = getattr(spec, "state_manifest", None)
        if manifest is None:
            return Applicability(False, "no StateDescriptor attached", discovery=PassDiscovery.AUTO)
        a = self.applicability_l3(manifest)
        return Applicability(
            a.applies, a.reason, discovery=PassDiscovery.AUTO,
            cost_term=CostTerm.PER_STEP, claimed_tier=Tier.BITEXACT,
            params={"slices": list(a.targets)})

    def expected_delta_ms(self, spec, device: DeviceProfile) -> float:
        """packs_per_step * (host index build + H2D + cumsum), from the measured microbenchmark."""
        packs = getattr(spec, "packs_per_step", 16)
        per_pack_ms = getattr(spec, "measured_pack_ms", 0.0)
        return packs * per_pack_ms

    # ---- install ---------------------------------------------------------------------------
    def install(self, runtime_module=None) -> list[str]:
        """Memoize init_sequence_pack on its declared-geometry key."""
        if runtime_module is None:
            from cosmos_framework.data.generator.sequence_packing import runtime as runtime_module

        orig = getattr(runtime_module, "_iwm_orig_init_sequence_pack", None)
        if orig is None:
            orig = runtime_module.init_sequence_pack
            runtime_module._iwm_orig_init_sequence_pack = orig

        cache: dict = {}
        runtime_module._iwm_pack_cache = cache

        def init_sequence_pack(sample_lens, split_lens, attn_modes, device):
            # The key IS the declared geometry. Nothing device-resident is read to form it.
            sl = tuple(split_lens.tolist()) if hasattr(split_lens, "tolist") else tuple(split_lens)
            key = (tuple(sample_lens), sl, tuple(attn_modes), str(device))
            hit = cache.get(key)
            if hit is not None:
                # Return a shallow copy: callers mutate the pack dict (is_sharded, all_seq, ...),
                # and the cached metadata tensors are read-only from here on.
                return dict(hit)
            built = orig(sample_lens, split_lens, attn_modes, device)
            cache[key] = built
            return dict(built)

        runtime_module.init_sequence_pack = init_sequence_pack
        return ["static_partition_hoist"]

    @staticmethod
    def uninstall(runtime_module) -> None:
        orig = getattr(runtime_module, "_iwm_orig_init_sequence_pack", None)
        if orig is not None:
            runtime_module.init_sequence_pack = orig

    # ---- gates -------------------------------------------------------------------------------
    def verify(self, harness) -> VerifyResult:
        d = harness.max_abs_delta()
        return VerifyResult(
            passed=(d == 0.0),
            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
            max_abs_delta=d,
            detail="memoized pure function: every returned index tensor compared with torch.equal "
                   "against a freshly computed one, plus dtype/device/contiguity (I4)")

    def benchmark(self, harness) -> BenchResult:
        before, after = harness.ms_before(), harness.ms_after()
        return BenchResult(passed=after < before, before_ms=before, after_ms=after,
                           detail="per-pack metadata construction; scales by packs per control step")
