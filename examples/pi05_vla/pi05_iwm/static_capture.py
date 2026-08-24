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
#: chunk-agnostic — what matters is that capture happens after at least one full prefix refill,
#: and (with step tables on) that every timestep of the fixed Euler schedule has been seen once.
WARMUP_STEPS = 12


class _TableDense(torch.nn.Module):
    """Stands in for an AdaRMS `dense` projection inside the captured region.

    The projection's input is the time conditioning, which for a fixed Euler schedule takes ten
    values ever — so its output is a per-step CONSTANT. The serving engine precomputes exactly
    these (step, layer) modulation tables offline and slices them by baked pointer offset
    (serving/flash_rt/frontends/torch/pi05_rtx.py:371-430, models/pi05/pipeline_rtx.py:422-443);
    this is the torch-level equivalent: the module returns a static buffer that the step loop
    fills OUTSIDE the graph from a table of outputs the REAL projection produced during warmup on
    the same conditioning bytes. The swap moves WHEN the GEMV runs (once per timestep per model
    lifetime, not once per step), never WHAT it produces, so bitexactness is preserved by
    construction and re-proven by the verify gates. Measured: the 37 projections plus the time
    MLP cost 0.302 ms inside a 4.57 ms replay (6.6%).
    """

    def __init__(self, real: torch.nn.Module, buf: torch.Tensor, owner):
        super().__init__()
        self.real = real
        self.buf = buf
        self._owner = [owner]                      # plain list: keep the denoiser out of state_dict

    def forward(self, cond):
        # Inside the captured region the owner flag is True and the graph bakes the buffer read.
        # Any OTHER caller of this module (the in-process stock comparator, training) sees the
        # flag False and gets the real projection — the swap must never leak outside the graph.
        if self._owner[0]._tabled_active:
            return self.buf
        return self.real(cond)


class _StaticKV:
    """The Cache interface `GemmaAttention.update` needs, over fixed-extent buffers.

    This is the same memory discipline the serving engine uses for its enc+dec cache: one
    fixed-extent K and V per layer, the decoder's fresh entries written AT AN OFFSET into the
    shared buffer, attention reading the full extent with zero concat and zero copy per step
    (serving/flash_rt/hardware/rtx/attn_backend.py:249-270 allocates K/V each
    (layers, enc_seq_max+chunk, 1, head_dim); models/pi05/pipeline_rtx.py:1011-1029 writes the
    chunk K/V at token offset enc_seq and attends over enc_seq+dec_seq). Here `update` is that
    offset write — suffix slots at [prefix, prefix+suffix) — and returning the whole buffer is
    the zero-materialization read. The `torch.cat` upstream keeps in its Q/K/V head-stacking is
    untouched: capture-pool allocations are replay-stable; it is the CACHE join that must not
    reallocate, and here it never does."""

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

    def __init__(self, model, step_tables: bool = True):
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
        # -- per-step constant tables (the serving engine's style-table absorb) ---------------------------
        self._step_tables = step_tables
        self._table: dict[float, tuple] = {}       # t -> (adarms_cond, [dense outs])
        self._adarms_buf = None
        self._dense_bufs: "list[torch.Tensor] | None" = None
        self._denses: "list | None" = None          # (norm, real dense) pairs, in swap order
        self._tabled_active = False

    # -- per-step-constant compute, done OUTSIDE the graph ----------------------------------------
    def _adarms_denses(self):
        """(norm, real dense) pairs, in a fixed traversal order the buffers share."""
        expert = self._m.paligemma_with_expert.gemma_expert.model
        pairs = []
        for layer in expert.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                if getattr(norm, "dense", None) is not None:
                    pairs.append((norm, norm.dense))
        final = getattr(expert, "norm", None)
        if final is not None and getattr(final, "dense", None) is not None:
            pairs.append((final, final.dense))
        return pairs

    def _time_cond(self, timestep):
        """The embed_suffix time path, byte-identical: same modules, same op order, same dtypes."""
        import torch.nn.functional as F

        from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
        m = self._m
        e = create_sinusoidal_pos_embedding(
            timestep, m.action_in_proj.out_features,
            min_period=m.config.min_period, max_period=m.config.max_period,
            device=timestep.device).type(dtype=timestep.dtype)
        return F.silu(m.time_mlp_out(F.silu(m.time_mlp_in(e))))

    def _record_step(self, timestep) -> None:
        key = round(float(timestep[0]), 9)
        if key in self._table:
            return
        with torch.no_grad():
            cond = self._time_cond(timestep)
            outs = [real(cond).clone() for _, real in self._denses]
        self._table[key] = (cond.clone(), outs)

    def _load_step(self, timestep) -> None:
        key = round(float(timestep[0]), 9)
        if key not in self._table:
            # a timestep the warmup never saw (schedule change): compute the entry now, outside
            # the graph — correctness is preserved, only this step pays the 0.7 ms eager cost
            self._record_step(timestep)
        cond, outs = self._table[key]
        self._adarms_buf.copy_(cond)
        for buf, o in zip(self._dense_bufs, outs):
            buf.copy_(o)

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
        if self._step_tables and self._adarms_buf is not None:
            # tabled path: the time MLP and every AdaRMS projection were computed outside the
            # graph (`_load_step`); the region does only the step-VARYING work
            suffix_embs = m.action_in_proj(self._x_buf)
            adarms = self._adarms_buf
        else:
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
            if self._step_tables:
                self._load_step(timestep)
            self._graph.replay()
            self.replays += 1
            return self._out.clone()

        self._steps += 1
        if self._steps <= WARMUP_STEPS:
            if self._step_tables:
                if self._denses is None:
                    self._denses = self._adarms_denses()
                self._record_step(timestep)
            return self._forward_static()

        if self._step_tables and self._adarms_buf is None:
            # swap point: from here the projections live in tables and the graph never sees them
            cond0, outs0 = next(iter(self._table.values()))
            self._adarms_buf = cond0.clone()
            self._dense_bufs = [o.clone() for o in outs0]
            for (norm, real), buf in zip(self._denses, self._dense_bufs):
                norm.dense = _TableDense(real, buf, self)
            self._load_step(timestep)

        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        self._tabled_active = True
        try:
            with torch.cuda.graph(self._graph):
                self._out = self._forward_static()
        finally:
            self._tabled_active = False
        # capture runs the region once on a side stream but its output tensor content is not
        # trustworthy on all driver versions; replay once so _out holds this call's real answer
        self._graph.replay()
        self.replays += 1
        return self._out.clone()


def install_static_capture(model, step_tables: "bool | None" = None) -> StaticDenoiser:
    """Route `model.denoise_step` through a StaticDenoiser. Returns it (for its counters).

    `step_tables` (default on; IFL_PI05_STEP_TABLES=0 disables) additionally hoists the time MLP
    and the 37 AdaRMS modulation projections out of the captured region into per-timestep tables.
    """
    if step_tables is None:
        import os
        step_tables = os.environ.get("IFL_PI05_STEP_TABLES", "1") != "0"
    d = StaticDenoiser(model, step_tables=step_tables)

    def denoise_step(self_m, prefix_pad_masks, past_key_values, x_t, timestep):
        return d(prefix_pad_masks, past_key_values, x_t, timestep)

    type(model).denoise_step = denoise_step
    return d
