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


def state_roots(layers, pack=None, pos=None) -> dict:
    """Adapter-supplied semantic roots for buffer naming.

    Cosmos keeps no KV pool, so there is nothing analogous to LingBot's `attn_caches`. What it does
    have is the SequencePack itself and the RoPE packs, which are read every layer.
    """
    roots: dict = {}
    if pack is not None:
        roots["pack"] = pack
    if pos is not None:
        roots["rope.cos"], roots["rope.sin"] = pos
    for i, l in enumerate(layers):
        roots[f"layer[{i}]"] = l
    return roots


def build_plan(layers, mask, pos, *, model_id="cosmos3-edge"):
    """A Plan whose unit takes a SequencePack -- a dict of tensors plus host metadata.

    Nothing here is engine-specific beyond naming the unit: the default `TreeBinder` flattens the
    pack, and `mask`/`pos` ride along as captured constants because they do not change per call.
    """
    from instinctwm.planners.plan import CaptureUnit, Plan, PlanBuffer

    def mot_stack(pack):
        x = pack
        for l in layers:
            x = l(x, mask, pos)[0]
        return x

    return Plan(
        model_id=model_id,
        units=(CaptureUnit(name="mot_stack", fn=mot_stack, inputs=("pack",), output="pack_out"),),
        buffers=(),                      # nothing to declare: the binder discovers the leaves
        plan_buffer=PlanBuffer(fields=("actual_len",)),
        notes={"attention": "torch-SDPA SHIM -- plumbing/latency only, NOT served numerics",
               "state": "no KV pool; the SequencePack is the state"},
    )


# =================================================================================================
# AdapterSurface: what this model publishes to passes. Answers WHERE, never WHAT.
# =================================================================================================

class Cosmos3Surface:
    """Site publisher for Cosmos3-Edge.

    Note how little there is. The adapter's job is to point; it holds no optimization policy and
    imports nothing from `instinctwm.passes` except the vocabulary.
    """

    model_id = "cosmos3-edge"

    def __init__(self, layers, mask, pos):
        self.layers, self.mask, self.pos = layers, mask, pos
        self._wrapped = {}

    def sites(self, kind):
        from instinctwm.passes.interface import Site, SiteKind

        if kind is SiteKind.CAPTURE_UNIT:
            # The whole MoT stack is one unit. Cosmos keeps no KV pool and mutates no host state
            # inside the layer, so there is no extent and no deferred commit to arrange.
            yield Site(kind=kind, id="cosmos3.mot_stack",
                       attrs={"capturable": True,
                              "effect_roots": (self.layers,),
                              "arity": 1,
                              "note": "argument is a SequencePack (dict + host metadata)"})
        elif kind is SiteKind.EXECUTION_REGION:
            for i, _l in enumerate(self.layers):
                yield Site(kind=kind, id=f"cosmos3.layer[{i}]", attrs={"index": i})
        # STATE_ADDRESSING: none. Cosmos3 has no KV pool -- the SequencePack IS the state, and it
        # is passed in, not addressed out of a resident buffer. A pass asking for these gets an
        # honest empty answer rather than a missing symbol.
        # INVARIANT_CONDITIONING / ALLOCATION: not yet published.

    def apply(self, rewrite):
        from instinctwm.passes.interface import RewriteKind

        if rewrite.site_id != "cosmos3.mot_stack" or rewrite.kind is not RewriteKind.WRAP:
            raise NotImplementedError(f"cosmos3 surface cannot apply {rewrite}")
        self._wrapped["mot_stack"] = rewrite.payload(self._raw_stack)

    def _raw_stack(self, pack):
        x = pack
        for l in self.layers:
            x = l(x, self.mask, self.pos)[0]
        return x

    def stack(self, pack):
        """The entry point a caller uses; rewritten in place by whatever passes fired."""
        return self._wrapped.get("mot_stack", self._raw_stack)(pack)
