"""Native causal UND prefill for the admitted, unpadded single-sample path.

Keep the original module forwards: conditioning-cache fill/validation wrappers
must observe every UND intermediate. GEN computation is owned by its region.
"""
from __future__ import annotations


class UnderstandingPrefill:
    def __init__(self, layer, *, attention=None):
        self.layer = layer
        if attention is None:
            from cosmos_framework.model.generator.mot import attention as native

            def attention(q, k, v):
                return native.attention(q, k, v, is_causal=True,
                                        causal_type=native.CausalType.DontCare)
        self.attention = attention

    def __call__(self, hidden, cos, sin):
        layer = self.layer
        a = layer.self_attn
        x = layer.input_layernorm(hidden)
        q = a.q_norm(a.q_proj(x).view(-1, a.num_attention_heads, a.head_dim))
        k = a.k_norm(a.k_proj(x).view(-1, a.num_key_value_heads, a.head_dim))
        v = a.v_proj(x).view(-1, a.num_key_value_heads, a.head_dim)
        q_rope, k_rope = a._apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        key_for_gen = k_rope
        if a.k_norm_und_for_gen is not None:
            _, key_for_gen = a._apply_rotary_pos_emb(
                q_rope, a.k_norm_und_for_gen(k), cos, sin, unsqueeze_dim=1)
        attended = self.attention(q_rope.unsqueeze(0), k_rope.unsqueeze(0), v.unsqueeze(0))
        residual = hidden + a.o_proj(attended.squeeze(0).flatten(-2, -1))
        output = residual + layer.mlp(layer.post_attention_layernorm(residual))
        return output, key_for_gen, v
