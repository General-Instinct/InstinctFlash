"""LingBot-VA's fusible regions — declarations only, read from source.

`modules/model.py:524-544`. Note the shape of the eager code, because it is what decides the tier:

    norm_hidden_states = (self.norm1(hidden_states.float())      # fp32 compute
                          * (1. + scale_msa) + shift_msa
                         ).type_as(hidden_states)                # <- MATERIALISES in bf16
    ...
    hidden_states = (hidden_states.float()
                     + attn_output * gate_msa
                    ).type_as(hidden_states)                     # <- MATERIALISES in bf16

Each `.type_as(hidden_states)` is a rounding to bf16 that a fused kernel must reproduce to be
bit-exact. Inside the parentheses the arithmetic is already fp32, so the intermediate `norm * scale`
is NOT materialised and carries no rounding of its own — which is why `norm1` is marked
`materializes_as=None` while the modulation output is `bf16`.

The `.float()` / `.type_as()` pair is also why these two regions account for so many launches: each
one is a separate elementwise kernel over [2, N, 3072], and the profile attributes 16,932 copies of
[2,32,3072] and 8,632 of [2,240,3072] per `_infer` to exactly this pattern.
"""

from __future__ import annotations

from instinctflash.backends.regions import FusibleRegion, FusionDescriptor, OpKind, OpSpec

_LAYERS = 30

PRE_ATTENTION = FusibleRegion(
    name="pre_attention_modulated_norm",
    ops=(
        OpSpec("upcast", OpKind.ELEMENTWISE, materializes_as=None, computes_in="fp32"),
        # LayerNorm is a REDUCTION. Its presence is what keeps a naive fused kernel out of the
        # BITEXACT tier unless it also preserves the reduction tree.
        OpSpec("norm1", OpKind.REDUCTION, materializes_as=None, computes_in="fp32"),
        OpSpec("scale", OpKind.ELEMENTWISE, materializes_as=None, computes_in="fp32"),
        OpSpec("shift", OpKind.ELEMENTWISE, materializes_as="bf16", computes_in="fp32"),
    ),
    boundary_in=("hidden_states", "scale_msa", "shift_msa"),
    boundary_out=("norm_hidden_states",),
    phases=("kv_refresh", "video", "action"),
    occurrences_per_forward=_LAYERS,
    note="model.py:533-535",
)

POST_ATTENTION = FusibleRegion(
    name="post_attention_gated_residual",
    ops=(
        OpSpec("upcast", OpKind.ELEMENTWISE, materializes_as=None, computes_in="fp32"),
        # CORRECTED. I first declared this materializes_as=None. It is wrong: `attn_out * gate`
        # is bf16 x bf16 and lands in bf16 BEFORE the fp32 add, so it carries a rounding of its
        # own. The framework's tier/measurement consistency check caught it -- a kernel derived
        # BITEXACT measured max|d| = 6.25e-02.
        OpSpec("gate", OpKind.ELEMENTWISE, materializes_as="bf16", computes_in="bf16"),
        OpSpec("residual_add", OpKind.ELEMENTWISE, materializes_as="bf16", computes_in="fp32"),
    ),
    boundary_in=("hidden_states", "attn_output", "gate_msa"),
    boundary_out=("hidden_states",),
    phases=("kv_refresh", "video", "action"),
    occurrences_per_forward=_LAYERS,
    note="model.py:543-544 -- pure elementwise, NO reduction. This is the region where a "
         "rounding-preserving kernel can legitimately reach BITEXACT.",
)


def lingbot_fusion_descriptor() -> FusionDescriptor:
    return FusionDescriptor(
        model_id="lingbot-va-posttrain-robotwin",
        regions=(PRE_ATTENTION, POST_ATTENTION),
        # measured from the post-ring-KV profile: elementwise/norm 160,225 launches and
        # gather/copy 163,596 launches per cycle, dominated by these two regions
        launches_per_region={
            "pre_attention_modulated_norm": 4,     # upcast, norm, scale+shift, type_as
            "post_attention_gated_residual": 3,    # upcast, gate-mul, add+type_as
        },
    )
