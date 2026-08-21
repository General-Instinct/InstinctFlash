"""Declared envelopes for the backends we intend to support. NO IMPLEMENTATIONS.

Every class here answers `capabilities()` honestly and raises from `measure()` and `bind()`. That is
the deliverable: the envelopes are what the planner reasons over, and declaring them is what proves
the abstraction discriminates. If two backends with genuinely different constraints produced
identical capability objects, the vocabulary would be too coarse to be worth having.

Read the table in ATTENTION.md alongside this file. The interesting rows are the refusals:

  * FlashAttention cannot take a data-dependent dense mask, which is exactly the mask LingBot-VA's
    stock KV path builds via `mask.nonzero()` per layer per forward. Ring addressing (P003) is what
    makes the site describable in a way a flash kernel could accept at all.
  * FlashInfer carries a per-forward `plan()` cost, so it is the backend most likely to invert
    between operating points -- fast at Quality's 75 forwards, a loss at Fast's 6.
  * RingAttention is illegal at world_size 1. Not slow: illegal.
  * SanaHybrid implements DIFFERENT SEMANTICS. It is refused for a `SOFTMAX_FULL` site no matter how
    fast it is, and that refusal is the single most important behaviour in this layer.
"""

from __future__ import annotations

from instinctflash.backends.attention.backend import AttentionBinding, AttentionMeasurement
from instinctflash.backends.attention.capabilities import AttentionCapabilities
from instinctflash.backends.attention.semantics import (
    AttentionSemantics as S,
)
from instinctflash.backends.attention.semantics import (
    AttentionShape,
    Distribution,
    MaskKind as M,
    QKVLayout as L,
)
from instinctflash.passes.contract import DeviceProfile, HardwareReq
from instinctflash.runtime.state.types import Addressing as A

_UNBUILT = ("declaration only: this is the architecture step. capabilities() is complete and is what "
            "the planner reasons over; measure() and bind() need an idle fleet and would change "
            "runtime behaviour. See ATTENTION.md.")


class _Declared:
    """Shared refusals, so each backend below is nothing but its envelope."""

    name = "unnamed"
    version = "0.0.0"

    def capabilities(self) -> AttentionCapabilities:  # pragma: no cover - overridden
        raise NotImplementedError

    def expected_delta_ms(self, shape: AttentionShape, forwards_per_cycle: int,
                          device: DeviceProfile) -> float:
        raise NotImplementedError(f"{self.name}: {_UNBUILT}")

    def measure(self, shape: AttentionShape, device: DeviceProfile) -> AttentionMeasurement:
        raise NotImplementedError(f"{self.name}: {_UNBUILT}")

    def bind(self, shape: AttentionShape, **site_attrs) -> AttentionBinding:
        raise NotImplementedError(f"{self.name}: {_UNBUILT}")


class AdapterNativeAttention(_Declared):
    """Whatever the model already does. Always legal, always bit-exact, never a regression.

    This is the incumbent and the baseline. Its presence is why `candidates()` is never empty and why
    "we measured four and kept the original" is an expressible result. `bind()` returning the model's
    own callable is the one implementation that could be written today without changing behaviour --
    it is left unbuilt only because nothing consumes it yet.
    """

    name = "adapter_native"
    version = "1.0.0"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            # Claims every semantics: it IS the model's own function, whatever that is.
            semantics=frozenset(S),
            mask_kinds=frozenset(M),
            layouts=frozenset(L),
            kv_addressing=frozenset(A),
            dtypes=frozenset({"bfloat16", "float16", "float32"}),
            supports_gqa=True,
            supports_varlen=True,
            supports_bias=True,
            is_identity_substitution=True,   # -> Tier.BITEXACT, by construction
            preserves_reduction_order=True,
            capture_safe=True,               # true of LingBot-VA under P003 ring addressing
        )


class TorchSDPA(_Declared):
    """`torch.nn.functional.scaled_dot_product_attention`, backend-agnostic dispatch.

    Accepts a materialised mask, which is what makes it the widest-envelope real backend and the
    natural first port. It is NOT automatically a win: on pi-0's real shapes, swapping eager attention
    for SDPA while keeping the mask measures 133.5 -> 144-184 us. That measurement is the reason
    `measure()` is mandatory in this interface.
    """

    name = "torch_sdpa"
    version = "1.0.0"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            semantics=frozenset({S.SOFTMAX_FULL, S.SOFTMAX_CAUSAL}),
            mask_kinds=frozenset({M.NONE, M.CAUSAL, M.BLOCK_STATIC, M.DENSE_DATA_DEPENDENT}),
            layouts=frozenset({L.BHSD}),
            kv_addressing=frozenset({A.DENSE, A.RING_INTERVAL}),
            supports_gqa=True,
            supports_bias=True,
            # Dispatch may land on a flash or mem-efficient path whose reduction order differs from
            # the eager reference, so BITEXACT is not claimable. -> Tier.NUMERIC
            preserves_reduction_order=False,
            capture_safe=True,
        )


class FlashAttention(_Declared):
    """FlashAttention-2/3. Fastest on long sequences, narrowest mask envelope.

    The mask restriction is the whole story for us: a flash kernel expresses masking as causal or
    windowed flags, not as an arbitrary tensor. LingBot-VA's stock path builds a dense mask with
    `mask.nonzero()` per layer per forward, so this backend is illegal at that site -- and legal only
    because P003 replaced that addressing with a ring interval.
    """

    name = "flash_attn"
    version = "0.0.0"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            semantics=frozenset({S.SOFTMAX_FULL, S.SOFTMAX_CAUSAL, S.SOFTMAX_SLIDING_WINDOW}),
            mask_kinds=frozenset({M.NONE, M.CAUSAL, M.SLIDING}),   # NOT dense, NOT data-dependent
            layouts=frozenset({L.BSHD, L.PACKED_VARLEN}),
            kv_addressing=frozenset({A.DENSE, A.RING_INTERVAL}),
            dtypes=frozenset({"bfloat16", "float16"}),
            head_dim_max=256, head_dim_multiple_of=8,
            supports_gqa=True, supports_varlen=True,
            hardware=HardwareReq(min_capability=(8, 0)),
            preserves_reduction_order=False,   # online softmax -> Tier.NUMERIC
            capture_safe=True,
        )


class FlashInfer(_Declared):
    """FlashInfer. Paged KV and a planned layout, with a per-forward host `plan()` step.

    `host_setup_us` is the field that matters here, and the reason this backend is the likeliest in
    the set to invert between operating points: a fixed per-forward host cost is amortised over
    Quality's 75 forwards and is not amortised over Fast's 6. Exactly the shape of the graph-capture
    inversion, which is why the profitability model charges it explicitly.
    """

    name = "flashinfer"
    version = "0.0.0"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            semantics=frozenset({S.SOFTMAX_FULL, S.SOFTMAX_CAUSAL}),
            mask_kinds=frozenset({M.NONE, M.CAUSAL, M.BLOCK_STATIC}),
            layouts=frozenset({L.BSHD, L.PACKED_VARLEN}),
            kv_addressing=frozenset({A.PAGED, A.DENSE}),
            head_dim_max=256, head_dim_multiple_of=8,
            supports_gqa=True, supports_varlen=True,
            hardware=HardwareReq(min_capability=(8, 0)),
            preserves_reduction_order=False,
            host_setup_us=40.0,      # placeholder; must be MEASURED before it is trusted
            capture_safe=False,      # the plan step is host work inside the region
        )


class CuDNNSDPA(_Declared):
    """cuDNN fused attention. Narrow envelope, strong on the shapes it accepts."""

    name = "cudnn_sdpa"
    version = "0.0.0"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            semantics=frozenset({S.SOFTMAX_FULL, S.SOFTMAX_CAUSAL}),
            mask_kinds=frozenset({M.NONE, M.CAUSAL}),
            layouts=frozenset({L.BHSD}),
            kv_addressing=frozenset({A.DENSE}),
            head_dim_max=128, head_dim_multiple_of=8,
            supports_gqa=True,
            hardware=HardwareReq(min_capability=(8, 0), requires=frozenset({"cudnn"})),
            preserves_reduction_order=False,
            capture_safe=True,
        )


class RingAttention(_Declared):
    """Sequence-sharded attention with rotated KV. Same function, different distribution.

    Illegal at world_size 1 -- not merely unprofitable. This is the case that justifies
    `Distribution` being its own axis: the constraint comes from `DeploymentSpec`, which is a
    site-scoped fact the checkpoint author cannot know.
    """

    name = "ring_attn"
    version = "0.0.0"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            semantics=frozenset({S.SOFTMAX_FULL, S.SOFTMAX_CAUSAL}),
            mask_kinds=frozenset({M.NONE, M.CAUSAL}),
            layouts=frozenset({L.BSHD}),
            kv_addressing=frozenset({A.DENSE, A.RING_INTERVAL}),
            supports_gqa=True,
            distribution=frozenset({Distribution.RING}),
            min_world_size=2,
            preserves_reduction_order=False,
            capture_safe=False,      # collective communication inside the region
        )


class SanaHybrid(_Declared):
    """Sana-Video style per-layer hybrid: some layers full, some windowed.

    THE IMPORTANT ONE. This implements `SOFTMAX_HYBRID`, a DIFFERENT FUNCTION from `SOFTMAX_FULL`.
    A checkpoint trained with full attention at every layer does not compute the same thing when some
    layers are windowed, so this backend is refused for a full-attention site regardless of speed.
    It becomes legal only for a checkpoint that DECLARES hybrid semantics -- which means it was
    trained that way, which makes it a Layer 1 artifact that this layer merely hosts.
    """

    name = "sana_hybrid"
    version = "0.0.0"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            semantics=frozenset({S.SOFTMAX_HYBRID}),
            mask_kinds=frozenset({M.NONE, M.CAUSAL, M.SLIDING}),
            layouts=frozenset({L.BSHD}),
            kv_addressing=frozenset({A.DENSE, A.RING_INTERVAL}),
            supports_gqa=True,
            preserves_reduction_order=False,
            capture_safe=True,
        )


#: The intended set, declaration-only. Registering these is safe precisely because selection raises:
#: nothing can install a backend whose `bind()` does not exist.
DECLARED_BACKENDS = (
    AdapterNativeAttention(), TorchSDPA(), FlashAttention(),
    FlashInfer(), CuDNNSDPA(), RingAttention(), SanaHybrid(),
)


def register_declared(registry) -> tuple[str, ...]:
    """Populate a registry with the declared envelopes. Used by tests and by `explain()` demos."""
    for b in DECLARED_BACKENDS:
        if b.name not in registry.names():
            registry.register(b)
    return registry.names()
