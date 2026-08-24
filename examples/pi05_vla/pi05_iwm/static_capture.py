"""Replay-safe CUDA graph capture for pi05's denoise loop, on static max-extent KV buffers.

THE FIX FOR THE REJECTION documented in `surface.py`. The region was rejected because
`denoise_step` clones the prefix `DynamicCache` and the forward APPENDS 50 suffix entries to the
clone — the region allocates and mutates a Python container per call, so a replay computes with
the K/V of whatever step the capture was taken from (measured: max |d| 2.116e-01 on a new x_t).

The serving engine's answer (serving/, and every CUDA-graph serving stack): make the memory the
graph touches STATIC. Here that is one K/V buffer per layer of extent prefix+suffix, allocated
once:

    slots [0, prefix)                written OUTSIDE the graph, once per chunk (the prefill copy)
    slots [prefix, prefix+suffix)    overwritten INSIDE the graph, every step, same addresses

`_StaticKV.update` replaces the DynamicCache append with an in-place copy into the suffix slots
and returns the full-extent buffers, so attention always runs over a constant length with the 4D
mask pi05 already builds for exactly that width. Nothing in the region allocates, nothing mutates
host state, and a replayed step reads the K/V the CURRENT chunk wrote — which is what makes replay
legal on inputs the capture never saw.

The captured callable's inputs land in static buffers (`x_buf`, `t_buf`) by `copy_`; its output is
cloned out of the pool before it is returned, so the next replay cannot overwrite a tensor a
caller still holds. Chunk boundaries are detected by cache-object identity: `sample_actions`
builds one prefill cache per chunk and passes the same object to all 10 steps, so a new object
means a new chunk and triggers the (out-of-graph) prefix refill.

Opt-in: `install_static_capture(policy.model)`, or IFL_PI05_STATIC_CAPTURE=1 through the adapter.
The graph is captured on the first step of the SECOND chunk: chunk 1 runs the static path eagerly,
which both warms kernels and proves the path before anything is frozen.
"""

from __future__ import annotations

import torch

#: eager steps on the static path before capture; also serves as kernel warmup. The count is
#: chunk-agnostic — what matters is that capture happens after at least one full prefix refill.
WARMUP_STEPS = 12


class _StaticKV:
    """The Cache interface `GemmaAttention.update` needs, over fixed-extent buffers."""

    def __init__(self, prefix_kv, suffix_len: int):
        first_k = next(iter(prefix_kv))[0]
        self.prefix_len = int(first_k.shape[2])
        self.suffix_len = int(suffix_len)
        total = self.prefix_len + self.suffix_len
        self.k, self.v = [], []
        for keys, values, *_ in prefix_kv:
            K = torch.empty(keys.shape[0], keys.shape[1], total, keys.shape[3],
                            dtype=keys.dtype, device=keys.device)
            V = torch.empty_like(K)
            K[:, :, : self.prefix_len].copy_(keys)
            V[:, :, : self.prefix_len].copy_(values)
            self.k.append(K)
            self.v.append(V)

    def refill(self, prefix_kv) -> None:
        """Per-chunk prefix rewrite. Runs OUTSIDE the graph; addresses never change."""
        for K, V, (keys, values, *_) in zip(self.k, self.v, iter(prefix_kv)):
            K[:, :, : self.prefix_len].copy_(keys)
            V[:, :, : self.prefix_len].copy_(values)

    # -- the Cache protocol, as much of it as this model touches -------------------------------
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        K, V = self.k[layer_idx], self.v[layer_idx]
        K[:, :, self.prefix_len:].copy_(key_states)
        V[:, :, self.prefix_len:].copy_(value_states)
        return K, V

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.prefix_len

    def get_max_cache_shape(self) -> int:
        return self.prefix_len + self.suffix_len

    @property
    def is_compileable(self) -> bool:
        return True

    def __len__(self):
        return len(self.k)


class StaticDenoiser:
    """pi05's denoise step over static buffers: eager until warm, then captured and replayed."""

    def __init__(self, model):
        self._m = model
        self._kv: "_StaticKV | None" = None
        self._last_cache_obj = None
        self._const = None            # (mask4d, position_ids, cache_position)
        self._x_buf = None
        self._t_buf = None
        self._graph = None
        self._out = None
        self._steps = 0
        self.replays = 0

    # -- per-chunk, outside the graph ------------------------------------------------------------
    def _begin_chunk(self, prefix_pad_masks, past_key_values) -> None:
        m = self._m
        suffix_len = m.config.chunk_size
        if self._kv is None:
            self._kv = _StaticKV(past_key_values, suffix_len)
        else:
            self._kv.refill(past_key_values)

        if self._const is None:
            from lerobot.policies.pi05.modeling_pi05 import (
                make_att_2d_masks,
                prepare_attention_masks_4d,
            )
            bsize, prefix_len = prefix_pad_masks.shape
            device = prefix_pad_masks.device
            # identical construction to upstream denoise_step, hoisted to chunk scope: for a fixed
            # prompt budget every quantity here is constant across steps AND chunks
            suffix_pad = torch.ones(bsize, suffix_len, dtype=torch.bool, device=device)
            suffix_att = torch.zeros(bsize, suffix_len, device=device)
            suffix_att[:, 0] = 1
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            mask4d = prepare_attention_masks_4d(full_2d)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
            cache_position = torch.arange(prefix_len, prefix_len + suffix_len, device=device)
            self._const = (mask4d, position_ids, cache_position)

    # -- the region itself: static in, static out ------------------------------------------------
    def _forward_static(self):
        m = self._m
        mask4d, position_ids, cache_position = self._const
        suffix_embs, _, _, adarms = m.embed_suffix(self._x_buf, self._t_buf)
        out = m.paligemma_with_expert.gemma_expert.model.forward(
            inputs_embeds=suffix_embs,
            attention_mask=mask4d,
            position_ids=position_ids,
            past_key_values=self._kv,
            use_cache=False,
            cache_position=cache_position,
            adarms_cond=adarms,
        )
        suffix_out = out.last_hidden_state[:, -m.config.chunk_size:]
        return m.action_out_proj(suffix_out.to(dtype=torch.float32))

    def __call__(self, prefix_pad_masks, past_key_values, x_t, timestep):
        if past_key_values is not self._last_cache_obj:
            self._begin_chunk(prefix_pad_masks, past_key_values)
            self._last_cache_obj = past_key_values

        if self._x_buf is None:
            self._x_buf = x_t.clone()
            self._t_buf = timestep.clone()
        else:
            self._x_buf.copy_(x_t)
            self._t_buf.copy_(timestep)

        if self._graph is not None:
            self._graph.replay()
            self.replays += 1
            return self._out.clone()

        self._steps += 1
        if self._steps <= WARMUP_STEPS:
            return self._forward_static()

        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._out = self._forward_static()
        # capture runs the region once on a side stream but its output tensor content is not
        # trustworthy on all driver versions; replay once so _out holds this call's real answer
        self._graph.replay()
        self.replays += 1
        return self._out.clone()


def install_static_capture(model) -> StaticDenoiser:
    """Route `model.denoise_step` through a StaticDenoiser. Returns it (for its counters)."""
    d = StaticDenoiser(model)

    def denoise_step(self_m, prefix_pad_masks, past_key_values, x_t, timestep):
        return d(prefix_pad_masks, past_key_values, x_t, timestep)

    type(model).denoise_step = denoise_step
    return d
