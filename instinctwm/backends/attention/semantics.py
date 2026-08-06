"""The attention vocabulary: what function is being computed, over what layout.

THE DISTINCTION THIS MODULE EXISTS TO ENFORCE

The list of things people call "attention backends" contains two different kinds of thing, and
conflating them is the one mistake that would make this layer unsafe:

    SDPA, FlashAttention, FlashInfer, cuDNN SDPA        same function, different implementation
    Sliding-window, Sana Hybrid, LongSana, linear
    attention, Mamba, DeltaNet                          DIFFERENT FUNCTION

The runtime may substitute freely within the first group: softmax attention computed two ways gives
the same answer to within reduction order, so the choice is a legality-plus-profitability question
exactly like any other pass. The runtime may NEVER substitute across the second group. A checkpoint
trained with full softmax attention does not compute the same thing under a sliding window, and no
amount of measured speedup makes that a valid swap -- it is a different model that happens to load.

So `AttentionSemantics` is a property of the CHECKPOINT, declared by the adapter, and a backend
declares which semantics it *implements*. Selection intersects the two. A backend that implements
only `SOFTMAX_SLIDING_WINDOW` is simply not a candidate for a site declaring `SOFTMAX_FULL`, and the
refusal is structural rather than a review comment.

Ring Attention is a third case and deserves naming: same function, different *distribution*. It is
legal only when the deployment has ranks to distribute over, which is a `DeploymentSpec` fact, not a
checkpoint fact. See `Distribution`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class AttentionSemantics(enum.Enum):
    """WHAT is computed. A checkpoint-scoped fact; the runtime never changes it.

    Two backends may be exchanged only if they implement the same member. Members are deliberately
    coarse: they describe the mathematical function, not the kernel.
    """

    #: softmax(QK^T/sqrt(d))V over all keys. The default for video diffusion trunks.
    SOFTMAX_FULL = "softmax_full"
    #: as above, restricted to j <= i.
    SOFTMAX_CAUSAL = "softmax_causal"
    #: causal within a fixed window w. NOT interchangeable with SOFTMAX_CAUSAL -- see module docstring.
    SOFTMAX_SLIDING_WINDOW = "softmax_sliding_window"
    #: some layers full, some windowed, per a declared per-layer schedule (Sana-Video style).
    SOFTMAX_HYBRID = "softmax_hybrid"
    #: kernel-linearized attention: O(n) with a feature map, no softmax normalisation over all keys.
    LINEAR = "linear"
    #: state-space / gated-delta recurrences (Mamba, DeltaNet). Attention-shaped interface, not
    #: attention. Listed so an adapter can declare it and a backend can claim it; nothing implements
    #: it here.
    STATE_SPACE = "state_space"

    def is_softmax(self) -> bool:
        return self in (self.SOFTMAX_FULL, self.SOFTMAX_CAUSAL,
                        self.SOFTMAX_SLIDING_WINDOW, self.SOFTMAX_HYBRID)


class Distribution(enum.Enum):
    """HOW the computation is spread across ranks. A deployment fact, not a checkpoint fact."""

    LOCAL = "local"        # one rank owns the whole sequence
    RING = "ring"          # sequence sharded across ranks, KV rotated (Ring Attention)
    ULYSSES = "ulysses"    # head-parallel sequence parallelism


class MaskKind(enum.Enum):
    """The mask's STRUCTURE, which is what decides whether a fast path is legal.

    `DENSE_DATA_DEPENDENT` is the one that costs us. LingBot-VA's stock KV path calls
    `mask.nonzero()` per layer per forward: a data-dependent shape, which is why graph capture raised
    `cudaErrorStreamCaptureInvalidated` until ring addressing replaced it. Any backend whose fast
    path needs a static shape must declare that it cannot take this mask, and the planner must be
    able to see that before anything is installed.
    """

    NONE = "none"
    CAUSAL = "causal"
    SLIDING = "sliding"
    #: arbitrary but shape-static across forwards (e.g. a fixed block pattern)
    BLOCK_STATIC = "block_static"
    #: arbitrary and recomputed per forward, with a shape that depends on values
    DENSE_DATA_DEPENDENT = "dense_data_dependent"


class QKVLayout(enum.Enum):
    """Memory layout of Q/K/V as the adapter hands them over."""

    BSHD = "bshd"          # [batch, seq, heads, dim]
    BHSD = "bhsd"          # [batch, heads, seq, dim]  (torch SDPA's native order)
    PACKED_VARLEN = "packed_varlen"   # [total_tokens, heads, dim] + cu_seqlens


@dataclass(frozen=True)
class MaskSpec:
    """What the adapter can honestly say about the mask at this site."""

    kind: MaskKind
    window: int | None = None          # for SLIDING
    #: True when the mask must be materialised as a tensor rather than expressed as a flag. A
    #: backend that only accepts flag-form masks is illegal here even if `kind` looks compatible.
    materialised: bool = False

    def is_shape_static(self) -> bool:
        return self.kind is not MaskKind.DENSE_DATA_DEPENDENT


@dataclass(frozen=True)
class AttentionShape:
    """The shapes a backend is asked to be fast at.

    Sequence lengths are RANGES, not scalars, because they are not known at planning time: LingBot-VA's
    KV grows monotonically through an episode until the ring wraps. A backend is selected on the range
    it will actually see, and `seq_kv_max` is what memory legality is checked against.

    `n_kv_heads < n_heads` is grouped-query attention; a backend that does not support GQA must
    declare so rather than silently broadcasting.
    """

    n_heads: int
    head_dim: int
    n_kv_heads: int | None = None      # None means == n_heads (no GQA)
    seq_q_min: int = 1
    seq_q_max: int = 1
    seq_kv_min: int = 1
    seq_kv_max: int = 1
    batch: int = 1
    dtype: str = "bfloat16"

    def is_gqa(self) -> bool:
        return self.n_kv_heads is not None and self.n_kv_heads != self.n_heads

    def is_decode_like(self) -> bool:
        """One query against many keys -- the shape where kernel choice matters most and where a
        prefill-tuned kernel is usually the wrong pick."""
        return self.seq_q_max == 1 and self.seq_kv_max > 1
