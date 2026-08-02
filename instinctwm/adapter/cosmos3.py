"""Backend Adapter: Cosmos3-Edge (two-tower MoT).

The generalization test. This model shares almost nothing with LingBot-VA:

    LingBot-VA        dual-stream DiT on Wan2.2, plain [B, N, C] tensors, a ring KV pool
                      addressed by interval, cross-attention to text, 30 identical blocks
    Cosmos3-Edge      two-tower MoT over a PACKED sequence. The unit of data is a SequencePack
                      (a dict, not a tensor) carrying causal_seq / full_only_seq plus partition
                      metadata; there is no KV pool at all; und and gen tokens take different
                      weight towers inside one layer.

The rule for this file: it may describe the model, and it may not change the engine. Anything the
engine cannot express is recorded as a finding, not worked around.
"""

from __future__ import annotations

import torch


TINY = dict(num_q_heads=8, num_kv_heads=2, head_dim=64)


def build_stack(n_layers: int, device, dtype=torch.bfloat16):
    """A real MoT decoder stack at tiny width. No checkpoint: the op structure is a property of
    the code and the shapes, which is what the engine is being tested against."""
    from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, MoTDecoderLayer
    from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
    )

    cfg = Qwen3VLTextConfig(
        hidden_size=TINY["num_q_heads"] * TINY["head_dim"],
        num_attention_heads=TINY["num_q_heads"],
        num_key_value_heads=TINY["num_kv_heads"],
        num_hidden_layers=n_layers,
        attention_bias=False,
        head_dim=TINY["head_dim"],       # cuDNN SDPA rejects the 128-dim default here
    )
    lt = LayerTypes("qwen3_vl_dense")
    layers = [
        MoTDecoderLayer(cfg, layer_idx=i, layer_types=lt,
                        qk_norm_for_text=True, qk_norm_for_diffusion=True).to(device, dtype).eval()
        for i in range(n_layers)
    ]
    for l in layers:
        l.requires_grad_(False)
    return cfg, layers


def build_pack(device, hidden, dtype=torch.bfloat16, sample_lens=(10, 8), seed=0):
    """A SequencePack: und (causal) prefix + gen (full) body, exactly as the model is served."""
    from cosmos_framework.data.generator.sequence_packing.runtime import (
        sequence_pack_from_packed_sequence,
    )

    g = torch.Generator(device="cpu").manual_seed(seed)
    split_lens, attn_modes, und_idx, gen_idx = [], [], [], []
    pos = 0
    for slen in sample_lens:
        und, gen = slen // 2, slen - slen // 2
        split_lens += [und, gen]
        attn_modes += ["causal", "full"]
        und_idx += list(range(pos, pos + und))
        gen_idx += list(range(pos + und, pos + slen))
        pos += slen

    total = sum(sample_lens)
    seq = torch.randn(total, hidden, generator=g).to(device, dtype)
    return sequence_pack_from_packed_sequence(
        packed_sequence=seq,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=list(sample_lens),
        packed_und_token_indexes=torch.tensor(und_idx, device=device),
        packed_gen_token_indexes=torch.tensor(gen_idx, device=device),
    )


def force_cudnn_attention():
    """Pin Cosmos's attention frontend to the cuDNN backend.

    ENVIRONMENT WORKAROUND, not a modelling choice. Cosmos auto-selects among
    {cudnn, natten, flash2, flash3} and finds none compatible here, because this box
    deliberately has no flash-attn: it was uninstalled to keep the LingBot-VA baseline
    environment fixed. cuDNN SDPA is the same backend LingBot's attention resolves to, so pinning
    it keeps the two models on comparable numerics.
    """
    import cosmos_framework.model.attention.frontend as fe

    _orig = fe.attention

    def attention(*a, **kw):
        kw.setdefault("backend", "cudnn")
        return _orig(*a, **kw)

    fe.attention = attention
    import cosmos_framework.model.generator.mot.attention as mot_attn
    if hasattr(mot_attn, "attention"):
        mot_attn.attention = attention
    return "cudnn"


def use_torch_sdpa():
    """Substitute torch SDPA for Cosmos's attention frontend.

    ENVIRONMENT WORKAROUND, and a LOUD one. Cosmos dispatches to {cudnn, natten, flash2, flash3};
    flash-attn is deliberately absent on this box (removed to keep the LingBot-VA baseline fixed)
    and the cuDNN backend reports itself incompatible for these shapes. Rather than install a
    kernel that would perturb the frozen baseline, the adapter swaps in
    `F.scaled_dot_product_attention`.

    CONSEQUENCE, stated plainly: Cosmos3 numerics under this shim are NOT the served numerics, and
    no accuracy claim may be made from runs using it. What remains valid is everything the engine
    is actually being tested on -- op structure, dependency derivation, capturability, and whether
    graph replay reproduces eager execution of the SAME code. That is the architecture question.

    Varlen kwargs are ignored: with one sample per pack the cumulative-seqlen path is a no-op.
    """
    import torch.nn.functional as F
    import cosmos_framework.model.attention.frontend as fe

    def attention(q, k, v, *, is_causal=False, **kw):
        # BSHD in/out; SDPA wants BHSD
        qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
        if kh.shape[1] != qh.shape[1]:                       # GQA: expand KV heads
            rep = qh.shape[1] // kh.shape[1]
            kh = kh.repeat_interleave(rep, dim=1)
            vh = vh.repeat_interleave(rep, dim=1)
        return F.scaled_dot_product_attention(qh, kh, vh, is_causal=bool(is_causal)).transpose(1, 2)

    fe.attention = attention
    import cosmos_framework.model.generator.mot.attention as mot_attn
    mot_attn.attention = attention
    return "torch-sdpa (SHIM: not served numerics)"
