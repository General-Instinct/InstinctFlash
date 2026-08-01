"""Runtime installation of optimizer passes onto the stock LingBot-VA server.

Today the "runtime" is the upstream server plus runtime patches. That is deliberate and
temporary: it keeps every pass verifiable against the existing bit-exactness gate
(`probe_bitexact.py`) before anything is rewritten, and it keeps the vendored upstream tree
clean so `git diff` stays reviewable. When the multi-stream KV core exists these installers get
replaced by real backend methods; the *passes* do not change, which is the point of the layering.

Each installer asserts its structural preconditions at install time and raises if one is false.
A pass that silently no-ops, or worse silently mis-caches, is much more expensive than one that
refuses to load.
"""

from __future__ import annotations

import torch


def install_conditioning_prefill(server_module, va_server_cls) -> list[str]:
    """Cache the episode-constant cross-attention K/V for all layers.

    Patches three things:
      1. `WanAttention.forward` — take a fast path when `cross_kv` is populated.
      2. `WanTransformer3DModel` — gain populate/clear/query methods.
      3. `VA_Server._reset` — release, then repopulate once the prompt embeds exist.
    """
    import modules.model as M

    Attn = M.WanAttention if hasattr(M, "WanAttention") else None
    if Attn is None:
        # Find the attention class structurally rather than by name.
        for obj in vars(M).values():
            if isinstance(obj, type) and hasattr(obj, "attn_caches") is False \
               and getattr(obj, "__module__", "") == M.__name__ \
               and "Attention" in obj.__name__:
                Attn = obj
                break
    if Attn is None:
        raise RuntimeError("could not locate the attention class in modules.model")

    Model = M.WanTransformer3DModel

    # ---- 1. attention fast path ----------------------------------------------------------
    _orig_forward = Attn.forward

    def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
        cross_kv = getattr(self, "_iwm_cross_kv", None)
        if cross_kv is None:
            return _orig_forward(self, q, k, v, rotary_emb, update_cache, cache_name)

        # Preconditions that make the cache correct. Both hold for cross-attention
        # (model.py:552-553 pass rotary_emb=None, update_cache=0). Assert rather than assume:
        # if a future edit routes a rotary or a cache write through here, fail loudly.
        if rotary_emb is not None or update_cache != 0:
            raise RuntimeError(
                "conditioning_prefill: cached cross-attention received "
                f"rotary_emb={rotary_emb is not None}, update_cache={update_cache}. "
                "The cache is only valid when neither is used; refusing to serve a wrong value."
            )

        key, value = cross_kv
        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        hidden_states = self.attn_op(query, key, value)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return self.to_out[1](self.to_out[0](hidden_states))

    Attn.forward = forward

    @torch.no_grad()
    def _project_cross_kv(self, encoder_hidden_states):
        # Byte-for-byte the k/v half of the stock forward (model.py:426-431).
        key = self.norm_k(self.to_k(encoder_hidden_states)).unflatten(2, (self.heads, -1))
        value = self.to_v(encoder_hidden_states).unflatten(2, (self.heads, -1))
        return key, value

    Attn._iwm_project_cross_kv = _project_cross_kv

    # ---- 2. model-level populate / clear -------------------------------------------------
    @torch.no_grad()
    def populate_cross_cache(self, text_emb):
        t = self.condition_embedder.text_embedder(text_emb)
        # Build every layer BEFORE publishing any, so a failure mid-way cannot leave the model
        # half-cached (the same transactional discipline vLLM-Omni uses in
        # manager.populate_cross_attention).
        built = [b.attn2._iwm_project_cross_kv(t) for b in self.blocks]
        if len(built) != len(self.blocks):
            raise RuntimeError("conditioning_prefill: incomplete projection")
        for b, kv in zip(self.blocks, built):
            b.attn2._iwm_cross_kv = kv
        del t

    def clear_cross_cache(self):
        for b in self.blocks:
            b.attn2._iwm_cross_kv = None

    def cross_cache_populated(self):
        return getattr(self.blocks[0].attn2, "_iwm_cross_kv", None) is not None

    Model.populate_cross_cache = populate_cross_cache
    Model.clear_cross_cache = clear_cross_cache
    Model.cross_cache_populated = cross_cache_populated

    # ---- 3. skip the text_embedder on the warm path --------------------------------------
    _orig_model_forward = Model.forward

    def model_forward(self, input_dict, *a, **kw):
        if self.cross_cache_populated():
            # attn2 never dereferences encoder_hidden_states on the cached path, but the stock
            # forward still projects text_emb -> text_hidden_states (model.py:843). Swap in a
            # zero-cost stand-in so that projection is skipped too.
            emb = self.condition_embedder.text_embedder
            self.condition_embedder.text_embedder = _NoopTextEmbedder()
            try:
                return _orig_model_forward(self, input_dict, *a, **kw)
            finally:
                self.condition_embedder.text_embedder = emb
        return _orig_model_forward(self, input_dict, *a, **kw)

    Model.forward = model_forward

    # ---- 4. lifecycle: release on reset, repopulate once prompts exist -------------------
    _orig_reset = va_server_cls._reset

    def _reset(self, prompt=None):
        if hasattr(self, "transformer"):
            self.transformer.clear_cross_cache()
        _orig_reset(self, prompt=prompt)
        # _reset is the ONLY writer of prompt_embeds (wan_va_server.py:424,:426) and also
        # recomputes use_cfg (:379), which is the only thing that can change the batch dim.
        if getattr(self, "prompt_embeds", None) is not None:
            text_emb = self._iwm_text_emb()
            self.transformer.populate_cross_cache(text_emb)

    def _iwm_text_emb(self):
        """Reproduce exactly what _repeat_input_for_cfg puts in input_dict['text_emb']."""
        pos = self.prompt_embeds.to(self.dtype)
        if self.use_cfg:
            return torch.cat([pos, self.negative_prompt_embeds.to(self.dtype)], dim=0)
        return pos

    va_server_cls._reset = _reset
    va_server_cls._iwm_text_emb = _iwm_text_emb

    return ["conditioning_prefill"]


class _NoopTextEmbedder(torch.nn.Module):
    """Stands in for the text embedder while the cross-attention cache is warm.

    Returns None: attn2 ignores its k/v operands on the cached path, and nothing else in the
    forward reads `text_hidden_states`. If that ever stops being true this returns None into an
    op and fails loudly, which is the behaviour we want.
    """

    def forward(self, x):
        return None
