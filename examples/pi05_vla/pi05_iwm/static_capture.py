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

DEFAULT, GATED BY A SELF-CHECK. `Pi05Adapter.install` routes every pi05-class checkpoint here on
capture-capable devices — fresh fine-tunes included (IFL_PI05_NO_CAPTURE=1 is the kill-switch).
The default captures one denoise step after WARMUP_STEPS eager steps. Set
IFL_PI05_FULL_CHUNK_GRAPH=1 to capture the fixed Euler loop and IFL_PI05_PREFIX_GRAPH=1 to capture
the vision/language prefix as a second graph; each graph gets one eager sample for kernel warmup.

No capture is trusted by construction. The per-step path replays against upstream eager on staged
x_t/timestep inputs and a synthetically refilled prefix. The full/prefix path replays complete
sample_actions calls on fresh noise, then on changed prompt/mask/image bytes. Exact equality is
required. PASS serves the graph; FAIL releases every graph, restores upstream denoise_step and
sample_actions, and announces the eager fallback loudly. Dedicated generators keep the model's
own RNG stream unchanged.

IFL_PI05_SELFCHECK_FAULT=1 is the drill switch: it rebinds a graph input between capture and check
so the loud-fallback path stays demonstrable on demand.
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

    @staticmethod
    def _entries(cache):
        """Return ``(keys, values, sliding)`` entries across Cache API revisions.

        Transformers main still exposes ``Cache.__iter__``, but its storage moved from tuple
        lists to ``Cache.layers``. Prefer the public layer objects when present and retain the
        iterable fallback for the older versions covered by the published H100 evidence.
        """
        if hasattr(cache, "layers"):
            return [
                (layer.keys, layer.values,
                 getattr(layer, "_sliding_window_tensor", None))
                for layer in cache.layers
            ]
        return list(iter(cache))

    @staticmethod
    def _validate_entry(keys, values, *, where: str) -> None:
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError(f"{where}: cache keys/values must be tensors")
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                f"{where}: expected rank-4 [batch, heads, seq, dim] K/V, "
                f"got {tuple(keys.shape)} and {tuple(values.shape)}")
        if keys.shape != values.shape:
            raise ValueError(
                f"{where}: K/V shapes differ: {tuple(keys.shape)} vs {tuple(values.shape)}")
        if keys.dtype != values.dtype or keys.device != values.device:
            raise ValueError(
                f"{where}: K/V dtype/device differ: "
                f"{keys.dtype}/{keys.device} vs {values.dtype}/{values.device}")

    def __init__(self, prefix_kv, suffix_len: int):
        if int(suffix_len) <= 0:
            raise ValueError(f"suffix_len must be positive, got {suffix_len}")
        entries = self._entries(prefix_kv)
        if not entries:
            raise ValueError("prefix cache has no initialized layers")
        first_k, first_v, *_ = entries[0]
        self._validate_entry(first_k, first_v, where="prefix layer 0")
        self.prefix_len = int(first_k.shape[2])
        self.suffix_len = int(suffix_len)
        total = self.prefix_len + self.suffix_len
        self.k, self.v = [], []
        for layer_idx, (keys, values, *_) in enumerate(entries):
            self._validate_entry(keys, values, where=f"prefix layer {layer_idx}")
            if keys.shape[:2] != first_k.shape[:2] or keys.shape[2:] != first_k.shape[2:]:
                raise ValueError(
                    f"prefix layer {layer_idx}: shape {tuple(keys.shape)} does not match "
                    f"layer 0 {tuple(first_k.shape)}")
            K = torch.empty(keys.shape[0], keys.shape[1], total, keys.shape[3],
                            dtype=keys.dtype, device=keys.device)
            V = torch.empty_like(K)
            K[:, :, : self.prefix_len].copy_(keys)
            V[:, :, : self.prefix_len].copy_(values)
            self.k.append(K)
            self.v.append(V)

    def refill(self, prefix_kv) -> None:
        """Per-chunk prefix rewrite. Runs OUTSIDE the graph; addresses never change."""
        entries = self._entries(prefix_kv)
        if len(entries) != len(self.k):
            raise ValueError(
                f"prefix cache layer count changed: {len(self.k)} -> {len(entries)}")
        for layer_idx, (K, V, (keys, values, *_)) in enumerate(
                zip(self.k, self.v, entries, strict=True)):
            self._validate_entry(keys, values, where=f"refill layer {layer_idx}")
            expected = K[:, :, : self.prefix_len]
            if (keys.shape != expected.shape or keys.dtype != K.dtype
                    or keys.device != K.device):
                raise ValueError(
                    f"refill layer {layer_idx}: expected shape/dtype/device "
                    f"{tuple(expected.shape)}/{K.dtype}/{K.device}, got "
                    f"{tuple(keys.shape)}/{keys.dtype}/{keys.device}")
            K[:, :, : self.prefix_len].copy_(keys)
            V[:, :, : self.prefix_len].copy_(values)

    # -- the Cache protocol, as much of it as this model touches -------------------------------
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if not 0 <= int(layer_idx) < len(self.k):
            raise IndexError(f"cache layer index out of range: {layer_idx}")
        self._validate_entry(key_states, value_states, where=f"update layer {layer_idx}")
        K, V = self.k[layer_idx], self.v[layer_idx]
        expected = K[:, :, self.prefix_len:]
        if (key_states.shape != expected.shape or key_states.dtype != K.dtype
                or key_states.device != K.device):
            raise ValueError(
                f"update layer {layer_idx}: expected suffix shape/dtype/device "
                f"{tuple(expected.shape)}/{K.dtype}/{K.device}, got "
                f"{tuple(key_states.shape)}/{key_states.dtype}/{key_states.device}")
        K[:, :, self.prefix_len:].copy_(key_states)
        V[:, :, self.prefix_len:].copy_(value_states)
        return K, V

    def __getitem__(self, layer_idx):
        # OpenPI's older patched Gemma uses prefix indexing + its own concat when
        # use_cache=False. Expose only the live prefix, never the reserved suffix.
        return (self.k[layer_idx][:, :, :self.prefix_len],
                self.v[layer_idx][:, :, :self.prefix_len])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.prefix_len

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0):
        """Match upstream ``StaticLayer``: attention always sees full max extent."""
        return self.prefix_len + self.suffix_len, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.prefix_len + self.suffix_len

    @property
    def max_batch_size(self) -> int:
        return int(self.k[0].shape[0])

    @property
    def max_cache_len(self) -> int:
        return self.prefix_len + self.suffix_len

    @property
    def is_initialized(self) -> bool:
        return True

    @property
    def is_sliding(self):
        return [False] * len(self.k)

    @property
    def is_compileable(self) -> bool:
        return True

    def __len__(self):
        return len(self.k)

    def __iter__(self):
        for K, V in zip(self.k, self.v, strict=True):
            yield K, V, None


class StaticDenoiser:
    """pi05's denoise step over static buffers: eager until warm, then captured and replayed."""

    #: staged inputs the post-capture self-check compares on. The second half changes the
    #: prefix/cache as well as the action noise.
    SELF_CHECK_INPUTS = 6

    def __init__(
        self,
        model,
        step_tables: bool = True,
        orig_denoise=None,
        on_self_check=None,
        self_check_inputs: "int | None" = None,
        full_chunk: bool = False,
        prefix_graph: bool = False,
    ):
        if prefix_graph and not full_chunk:
            raise ValueError("prefix_graph requires full_chunk capture")
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
        #: the UPSTREAM denoise_step, bound before install patched it. It is both the reference
        #: arm of the self-check and the landing spot of a rejected capture. None (a caller that
        #: installed by hand without one) disables the check — capture is then trusted the way
        #: it was before the check existed, which standalone measurement scripts rely on.
        self._orig_denoise = orig_denoise
        self._on_self_check = on_self_check
        self._self_check_n = (self.SELF_CHECK_INPUTS if self_check_inputs is None
                              else int(self_check_inputs))
        self._original_sample_actions = None
        self._full_self_checked = False
        #: the verdict, once taken: {n, bitexact, max_abs_delta, seconds, cases}
        self.self_check: "dict | None" = None
        #: True once the self-check failed. Permanent for the process, like the engine pass's
        #: `rejected` set: a capture that replays wrong once will replay wrong again.
        self.rejected = False
        self._last_pad_masks = None
        # -- per-step constant tables (the serving engine's style-table absorb) ---------------------------
        self._step_tables = step_tables
        self._table: dict[float, tuple] = {}       # t -> (adarms_cond, [dense outs])
        self._adarms_buf = None
        self._dense_bufs: "list[torch.Tensor] | None" = None
        self._denses: "list | None" = None          # (norm, real dense) pairs, in swap order
        self._tabled_active = False
        # -- optional fixed-schedule, whole-denoise-loop graph ----------------------------------
        self._full_chunk = bool(full_chunk)
        self._prefix_graph_enabled = bool(prefix_graph)
        self._prefix_graph = None
        self._prefix_inputs = None
        self._prefix_pad_masks = None
        self._prefix_att_masks = None
        self._prefix_kv = None
        self._prefix_samples_completed = 0
        self.prefix_replays = 0
        self._full_samples_completed = 0
        self._chunk_graph = None
        self._chunk_input = None
        self._chunk_final = None
        self._chunk_times = None
        self.chunk_replays = 0

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

    def _enter_baked_full_step_tables(self):
        """Install capture-only constant pointers for every fixed-schedule step.

        The wrappers exist only while the full graph is being captured. Each unrolled invocation
        records a different precomputed tensor address; replay needs no host-side table copy and
        eager/dynamic-NFE callers see the original modules after capture.
        """
        if self._denses is None:
            self._denses = self._adarms_denses()
        for norm, real in self._denses:
            norm.dense = real
        for timestep in self._chunk_times:
            self._record_step(timestep)
        schedule = [
            self._table[round(float(timestep[0]), 9)]
            for timestep in self._chunk_times
        ]
        saved = (self._adarms_buf, self._dense_bufs)
        cond0, outs0 = schedule[0]
        self._adarms_buf = cond0
        for (norm, real), output in zip(self._denses, outs0, strict=True):
            norm.dense = _TableDense(real, output, self)
        self._tabled_active = True
        return schedule, saved

    def _select_baked_full_step(self, constants) -> None:
        cond, outputs = constants
        self._adarms_buf = cond
        for (norm, _), output in zip(self._denses, outputs, strict=True):
            norm.dense.buf = output

    def _exit_baked_full_step_tables(self, saved) -> None:
        self._tabled_active = False
        for norm, real in self._denses:
            norm.dense = real
        self._adarms_buf, self._dense_bufs = saved

    # -- per-chunk, outside the graph ------------------------------------------------------------
    def _begin_chunk(self, prefix_pad_masks, past_key_values) -> None:
        m = self._m
        self._last_pad_masks = prefix_pad_masks
        suffix_len = m.config.chunk_size
        if self._kv is None:
            self._kv = _StaticKV(past_key_values, suffix_len)
        else:
            self._kv.refill(past_key_values)

        from lerobot.policies.pi05 import modeling_pi05
        make_att_2d_masks = modeling_pi05.make_att_2d_masks
        # Older LeRobot keeps the exact upstream conversion on the model instance.
        prepare_attention_masks_4d = getattr(m, "_prepare_attention_masks_4d", None)
        if prepare_attention_masks_4d is None:
            prepare_attention_masks_4d = modeling_pi05.prepare_attention_masks_4d
        bsize, prefix_len = prefix_pad_masks.shape
        device = prefix_pad_masks.device
        # Shapes are fixed by the prompt budget, but values change with the prompt's valid-token
        # mask. Recompute once per CHUNK and copy into the graph-owned addresses in place.
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
        current = (mask4d, position_ids, cache_position)
        if self._const is None:
            self._const = current
        else:
            for static, fresh in zip(self._const, current, strict=True):
                if static.shape != fresh.shape:
                    raise ValueError("pi0.5 prompt shape changed; a distinct graph is required")
                static.copy_(fresh)

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

    @staticmethod
    def _copy_tensor_checked(static, fresh, name: str) -> None:
        if (static.shape != fresh.shape or static.dtype != fresh.dtype
                or static.device != fresh.device):
            raise ValueError(
                f"{name} shape/dtype/device changed: expected "
                f"{tuple(static.shape)}/{static.dtype}/{static.device}, got "
                f"{tuple(fresh.shape)}/{fresh.dtype}/{fresh.device}")
        static.copy_(fresh)

    def _set_prefix_inputs(
        self, images, image_masks, tokens, token_mask, states, state_masks,
    ) -> None:
        if self._prefix_inputs is None:
            self._prefix_inputs = (
                [tensor.clone() for tensor in images],
                [tensor.clone() for tensor in image_masks],
                tokens.clone(),
                token_mask.clone(),
                states.clone() if states is not None else None,
                state_masks.clone() if state_masks is not None else None,
            )
            return

        (
            static_images,
            static_masks,
            static_tokens,
            static_token_mask,
            static_states,
            static_state_masks,
        ) = self._prefix_inputs
        if len(static_images) != len(images) or len(static_masks) != len(image_masks):
            raise ValueError(
                f"prefix view count changed: {len(static_images)} -> {len(images)}")
        for idx, (static, fresh) in enumerate(zip(static_images, images, strict=True)):
            self._copy_tensor_checked(static, fresh, f"prefix image {idx}")
        for idx, (static, fresh) in enumerate(zip(static_masks, image_masks, strict=True)):
            self._copy_tensor_checked(static, fresh, f"prefix image mask {idx}")
        self._copy_tensor_checked(static_tokens, tokens, "prefix tokens")
        self._copy_tensor_checked(static_token_mask, token_mask, "prefix token mask")
        if (static_states is None) != (states is None):
            raise ValueError("prefix states changed between None and Tensor")
        if (static_state_masks is None) != (state_masks is None):
            raise ValueError("prefix state_masks changed between None and Tensor")
        if states is not None:
            self._copy_tensor_checked(static_states, states, "prefix states")
        if state_masks is not None:
            self._copy_tensor_checked(
                static_state_masks, state_masks, "prefix state masks")

    def _embed_prefix_capture_safe(self):
        """Upstream ``embed_prefix`` with its host-created all-zero mask hoisted.

        Upstream constructs ``att_masks`` from a Python list directly on CUDA. That is an
        unpinned CPU-to-GPU copy, which CUDA Graph capture rejects. The mask is statically all
        zero, so retain it at a fixed CUDA address while leaving every learned PyTorch operation
        and concatenation in the graph, in upstream order.
        """
        images, image_masks, tokens, token_mask, states, state_masks = self._prefix_inputs
        m = self._m
        embs = []
        pad_masks = []
        for image, image_mask in zip(images, image_masks, strict=True):
            def image_embed_func(input_image, input_mask):
                if input_image.ndim == 5:
                    return m.paligemma_with_expert.embed_image(
                        input_image,
                        frame_mask=input_mask,
                        temporal_attention_every=m.config.memory_temporal_attention_every,
                    )
                return m.paligemma_with_expert.embed_image(input_image)

            image_emb = m._apply_checkpoint(image_embed_func, image, image_mask)
            bsize, num_image_embs = image_emb.shape[:2]
            embs.append(image_emb)
            current_mask = image_mask[:, -1] if image_mask.ndim == 2 else image_mask
            pad_masks.append(current_mask[:, None].expand(bsize, num_image_embs))

        proprio_history_proj = getattr(m, "proprio_history_proj", None)
        if proprio_history_proj is not None:
            if states is None or state_masks is None:
                raise ValueError("proprioceptive memory requires states and state_masks")
            state_embs = m._apply_checkpoint(proprio_history_proj, states)
            embs.append(state_embs)
            pad_masks.append(state_masks)

        def lang_embed_func(input_tokens):
            return m.paligemma_with_expert.embed_language_tokens(input_tokens)

        embs.append(m._apply_checkpoint(lang_embed_func, tokens))
        pad_masks.append(token_mask)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        if self._prefix_att_masks is None:
            self._prefix_att_masks = torch.zeros_like(pad_masks)
        elif self._prefix_att_masks.shape != pad_masks.shape:
            raise ValueError(
                "prefix attention-mask shape changed: expected "
                f"{tuple(self._prefix_att_masks.shape)}, got {tuple(pad_masks.shape)}")
        return embs, pad_masks, self._prefix_att_masks

    def _forward_prefix(self, upstream_inputs=None):
        m = self._m
        if upstream_inputs is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = (
                self._embed_prefix_capture_safe())
        else:
            # Older LeRobot releases have no proprioceptive-memory arguments.
            # Inspect the declared API rather than swallowing a TypeError from inside it.
            import inspect
            parameters = inspect.signature(m.embed_prefix).parameters
            prefix_args = upstream_inputs if "states" in parameters else upstream_inputs[:4]
            if "states" not in parameters and any(x is not None for x in upstream_inputs[4:]):
                raise ValueError("this pi05 version does not support proprioceptive history")
            prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(*prefix_args)
        from lerobot.policies.pi05.modeling_pi05 import (
            make_att_2d_masks,
            prepare_attention_masks_4d,
        )
        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(
            prefix_att_2d_masks)
        language_config = (
            m.paligemma_with_expert.paligemma.model.language_model.config)
        language_config._attn_implementation = "eager"
        _, past_key_values = m.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_pad_masks, past_key_values

    def run_prefix(
        self, images, image_masks, tokens, token_mask, states=None, state_masks=None,
    ):
        """Run/capture the original PyTorch prefix over fixed input and output addresses."""
        if self.rejected or not self._prefix_graph_enabled:
            # No prefix graph: keep upstream embed_prefix itself, including its host mask build.
            return self._forward_prefix(
                (images, image_masks, tokens, token_mask, states, state_masks))

        self._set_prefix_inputs(
            images, image_masks, tokens, token_mask, states, state_masks)
        if self._prefix_graph is None and self._prefix_samples_completed == 0:
            result = self._forward_prefix()
            self._prefix_samples_completed += 1
            return result

        if self._prefix_graph is None:
            torch.cuda.synchronize()
            self._prefix_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._prefix_graph):
                self._prefix_pad_masks, self._prefix_kv = self._forward_prefix()

        self._prefix_graph.replay()
        self.prefix_replays += 1
        self._prefix_samples_completed += 1
        return self._prefix_pad_masks, self._prefix_kv

    def run_full_chunk(self, prefix_pad_masks, past_key_values, noise, num_steps: int):
        """Run the complete fixed Euler schedule, capturing its original PyTorch kernels.

        The first sample is an eager warmup over static KV. The second captures one graph containing
        all denoise calls and the exact upstream FP32 Euler updates; later samples update prefix/noise
        buffers and replay once. Prefix construction remains outside this graph.
        """
        if not self._full_chunk:
            raise RuntimeError("full-chunk graph was not enabled at installation")
        expected = int(getattr(self._m.config, "num_inference_steps", 0))
        if int(num_steps) != expected or expected <= 0:
            raise ValueError(
                f"full-chunk graph requires fixed num_steps={expected}, got {num_steps}")
        if getattr(self._m, "_rtc_enabled", lambda: False)():
            raise RuntimeError("full-chunk graph refuses RTC schedules")
        if self.rejected:
            if self._orig_denoise is None:
                raise RuntimeError("rejected full-chunk graph has no upstream eager fallback")
            dt = -1.0 / expected
            current = noise
            with torch.no_grad():
                for step in range(expected):
                    timestep = torch.full(
                        (noise.shape[0],), 1.0 + step * dt,
                        dtype=torch.float32, device=noise.device)
                    velocity = self._orig_denoise(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        x_t=current,
                        timestep=timestep,
                    )
                    current = current + dt * velocity
            return current

        # One invocation is exactly one action chunk. Prefix-graph replay intentionally reuses
        # one Cache object while rewriting its tensors, so object identity cannot detect prompt
        # changes here: refresh both K/V and the prompt-dependent mask/positions every sample.
        self._begin_chunk(prefix_pad_masks, past_key_values)
        self._last_cache_obj = past_key_values

        if self._x_buf is None:
            self._x_buf = noise.clone()
            self._t_buf = torch.empty(
                noise.shape[0], dtype=torch.float32, device=noise.device)

        dt = -1.0 / expected
        if self._chunk_input is None:
            self._chunk_input = noise.clone()
            self._chunk_num_steps = expected
            self._chunk_times = [
                torch.tensor(1.0 + step * dt, dtype=torch.float32, device=noise.device)
                .expand(noise.shape[0]).clone()
                for step in range(expected)
            ]
        else:
            if expected != self._chunk_num_steps:
                raise ValueError(
                    f"full-chunk graph num_steps changed: {self._chunk_num_steps} -> {expected}")
            if (noise.shape != self._chunk_input.shape or noise.dtype != self._chunk_input.dtype
                    or noise.device != self._chunk_input.device):
                raise ValueError(
                    "Pi0.5 full-chunk noise shape/dtype/device changed: "
                    f"expected {tuple(self._chunk_input.shape)}/{self._chunk_input.dtype}/"
                    f"{self._chunk_input.device}, got "
                    f"{tuple(noise.shape)}/{noise.dtype}/{noise.device}")
            self._chunk_input.copy_(noise)

        # Warm every exact shape and tactic once before freezing the complete loop.
        if self._chunk_graph is None and self._full_samples_completed == 0:
            current = noise
            for timestep in self._chunk_times:
                self._x_buf.copy_(current)
                self._t_buf.copy_(timestep)
                velocity = self._forward_static()
                current = current + dt * velocity
            self._steps += expected
            self._full_samples_completed += 1
            return current

        if self._chunk_graph is None:
            baked_schedule = saved_table_state = None
            if self._step_tables:
                baked_schedule, saved_table_state = (
                    self._enter_baked_full_step_tables())
            torch.cuda.synchronize()
            self._chunk_graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(self._chunk_graph):
                    current = self._chunk_input
                    for index, timestep in enumerate(self._chunk_times):
                        if baked_schedule is not None:
                            self._select_baked_full_step(baked_schedule[index])
                        self._x_buf.copy_(current)
                        self._t_buf.copy_(timestep)
                        velocity = self._forward_static()
                        # Verbatim upstream ``euler_integrate`` order and dtype promotion.
                        current = current + dt * velocity
                    self._chunk_final = current
            finally:
                if saved_table_state is not None:
                    self._exit_baked_full_step_tables(saved_table_state)

        # Capture execution is not a trustworthy output on every driver; always replay.
        self._chunk_graph.replay()
        self.chunk_replays += 1
        self.replays += expected
        self._full_samples_completed += 1
        return self._chunk_final.clone()

    def __call__(self, prefix_pad_masks, past_key_values, x_t, timestep):
        if self.rejected:
            # a caller still holding the pre-rejection binding: same eager answer, no capture
            return self._orig_denoise(prefix_pad_masks=prefix_pad_masks,
                                      past_key_values=past_key_values, x_t=x_t, timestep=timestep)
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

        # FAULT DRILL, between capture and check on purpose: rebinding the x buffer makes every
        # later copy_ land at an address the graph does not read — the stale-address bug class
        # the self-check exists to catch. Documented in the module docstring; consumed here so
        # the loud-fallback path can be demonstrated on demand without touching the code.
        import os
        import sys
        if os.environ.get("IFL_PI05_SELFCHECK_FAULT") == "1":
            # stderr: a live `serve` defers stdout until the command returns (cli_config.execute),
            # which for a persistent server is never — a drill nobody can watch proves nothing
            print("[pi05 static_capture] FAULT INJECTED (IFL_PI05_SELFCHECK_FAULT=1): x buffer "
                  "rebound between capture and self-check — the check must now fail.",
                  file=sys.stderr, flush=True)
            self._x_buf = self._x_buf.clone()

        # THE GATE. Capturing successfully proves nothing about replaying (measured: the
        # DynamicCache region replayed 1.55x and WRONG by up to 48% of the signal while three
        # separate checks read clean). The graph serves only if replay equals upstream eager
        # EXACTLY on staged inputs it was not captured from.
        if self._orig_denoise is not None and self._self_check_n > 0:
            if not self._self_check(prefix_pad_masks, past_key_values, x_t, timestep):
                return self._orig_denoise(prefix_pad_masks=prefix_pad_masks,
                                          past_key_values=past_key_values,
                                          x_t=x_t, timestep=timestep)

        # capture runs the region once on a side stream but its output tensor content is not
        # trustworthy on all driver versions; replay once so _out holds this call's real answer
        self._graph.replay()
        self.replays += 1
        return self._out.clone()

    # -- the post-capture gate ---------------------------------------------------------------
    def _self_check(self, prefix_pad_masks, past_key_values, x_t, timestep) -> bool:
        """Replay vs upstream eager on staged inputs the capture never saw. Exact equality.

        Startup-only: runs once, at capture time, and restores every buffer it touched — the
        model's own RNG stream never moves (staged draws come from a dedicated generator, the
        eager arm is deterministic, replay consumes no randomness), so a PASSing check leaves
        the served action stream bitwise identical to a build without the check.
        """
        import time

        n = self._self_check_n
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = torch.Generator(device=x_t.device)
        gen.manual_seed(0x51F)

        if self._step_tables and self._table:
            t_values = [float(k) for k in list(self._table)[:3]]
        else:
            t_values = [float(timestep.reshape(-1)[0])]

        worst, cases = 0.0, []
        exact = True
        from instinctflash.runtime.capture_self_check import compare_tensors
        staged_cache = None
        try:
            for i in range(n):
                if i == (n + 1) // 2 and staged_cache is None:
                    # second half: a synthetically REFILLED prefix. A graph that baked the K/V
                    # values it was captured with — instead of reading the live buffers the
                    # per-chunk refill writes — is exactly wrong here and nowhere milder.
                    staged_cache = self._perturbed_prefix(past_key_values, gen)
                    self._kv.refill(staged_cache)
                x = torch.empty_like(x_t)
                x.normal_(generator=gen)
                t = torch.full_like(timestep, t_values[i % len(t_values)])
                cache = staged_cache if staged_cache is not None else past_key_values
                with torch.no_grad():
                    ref = self._orig_denoise(prefix_pad_masks=prefix_pad_masks,
                                             past_key_values=cache, x_t=x, timestep=t).detach().clone()
                self._x_buf.copy_(x)
                self._t_buf.copy_(t)
                if self._step_tables:
                    self._load_step(t)
                self._graph.replay()
                check = compare_tensors(ref, self._out)
                exact = exact and check["valid"] and check["bitexact"]
                d = check["max_abs_delta"]
                worst = max(worst, d)
                cases.append({"input": i + 1, "timestep": t_values[i % len(t_values)],
                              "prefix": "refilled" if staged_cache is not None else "captured-chunk",
                              "max_abs_delta": d})
        finally:
            # restore the real chunk state whatever the verdict: the caller's step must see the
            # answer to ITS inputs, and a later refill must start from the real prefix
            if staged_cache is not None:
                self._kv.refill(past_key_values)
            self._x_buf.copy_(x_t)
            self._t_buf.copy_(timestep)
            if self._step_tables:
                self._load_step(timestep)
            torch.cuda.synchronize()

        passed = bool(cases) and exact
        self.self_check = {"n": n, "bitexact": passed, "max_abs_delta": worst,
                           "seconds": time.perf_counter() - t0, "cases": cases}
        if not passed:
            self._release_and_fall_back(worst)
        if self._on_self_check is not None:
            self._on_self_check(dict(self.self_check))
        return passed

    @staticmethod
    def _changed_prefix_inputs(
        images, image_masks, tokens, token_mask, states, state_masks,
    ):
        """Shape-stable prefix values the whole-sample graph was not captured from."""
        changed_images = [
            torch.flip(image, dims=(-1,)) if image.shape[-1] > 1 else -image
            for image in images
        ]
        changed_masks = [mask.clone() for mask in image_masks]
        changed_tokens = (
            torch.roll(tokens, shifts=1, dims=-1)
            if tokens.shape[-1] > 1 else tokens.clone())
        changed_token_mask = (
            torch.roll(token_mask, shifts=1, dims=-1)
            if token_mask.shape[-1] > 1 else token_mask.clone())
        changed_states = None if states is None else (
            torch.flip(states, dims=(-1,)) if states.shape[-1] > 1 else -states)
        changed_state_masks = None if state_masks is None else (
            torch.roll(state_masks, shifts=1, dims=-1)
            if state_masks.shape[-1] > 1 else state_masks.clone())
        return (
            changed_images,
            changed_masks,
            changed_tokens,
            changed_token_mask,
            changed_states,
            changed_state_masks,
        )

    def _eager_sample_actions(
        self, images, image_masks, tokens, token_mask, states, state_masks,
        noise, num_steps, kwargs,
    ):
        """Call the saved upstream orchestration with the saved upstream denoise binding."""
        if self._original_sample_actions is None or self._orig_denoise is None:
            raise RuntimeError("whole-sample self-check has no upstream reference")
        sentinel = object()
        installed = self._m.__dict__.get("denoise_step", sentinel)
        self._m.denoise_step = self._orig_denoise
        try:
            with torch.no_grad():
                return self._original_sample_actions(
                    images, image_masks, tokens, token_mask,
                    **_history_kwargs(self._original_sample_actions, states, state_masks), noise=noise,
                    num_steps=num_steps, **kwargs)
        finally:
            if not self.rejected:
                if installed is sentinel:
                    self._m.__dict__.pop("denoise_step", None)
                else:
                    self._m.denoise_step = installed

    def _full_self_check(
        self, images, image_masks, tokens, token_mask, states, state_masks,
        noise, num_steps, kwargs,
    ) -> bool:
        """Gate full/prefix graphs on unseen complete sample_actions inputs."""
        import time

        n = self._self_check_n
        torch.cuda.synchronize()
        started = time.perf_counter()
        gen = torch.Generator(device=noise.device)
        gen.manual_seed(0x51F05)
        original = (images, image_masks, tokens, token_mask, states, state_masks)
        changed = self._changed_prefix_inputs(*original)
        split = (n + 1) // 2
        worst, cases = 0.0, []
        for index in range(n):
            inputs = original if index < split else changed
            staged_noise = torch.empty_like(noise)
            staged_noise.normal_(generator=gen)
            reference = self._eager_sample_actions(
                *inputs, staged_noise.clone(), num_steps, kwargs)
            prefix_pad_masks, past_key_values = self.run_prefix(*inputs)
            replay = self.run_full_chunk(
                prefix_pad_masks, past_key_values, staged_noise, num_steps)
            delta = float(
                (reference.detach().float() - replay.detach().float()).abs().max().item())
            worst = max(worst, delta)
            cases.append({
                "input": index + 1,
                "prefix": "captured-chunk" if index < split else "refilled",
                "max_abs_delta": delta,
            })

        torch.cuda.synchronize()
        passed = worst == 0.0
        self._full_self_checked = True
        self.self_check = {
            "n": n,
            "bitexact": passed,
            "max_abs_delta": worst,
            "seconds": time.perf_counter() - started,
            "cases": cases,
            "scope": "sample_actions",
            "prefix_graph": self._prefix_graph_enabled,
        }
        if not passed:
            self._release_and_fall_back(worst)
        if self._on_self_check is not None:
            self._on_self_check(dict(self.self_check))
        return passed

    def _perturbed_prefix(self, past_key_values, gen):
        """A DynamicCache shaped exactly like the real prefix, with values the capture never saw."""
        from pi05_iwm.surface import Pi05CacheBinder

        binder = Pi05CacheBinder()
        leaves, spec = binder.flatten(past_key_values)
        staged = []
        for leaf in leaves:
            noise = torch.empty(leaf.shape, device=leaf.device, dtype=torch.float32)
            noise.normal_(generator=gen)
            scale = leaf.detach().float().std().clamp_min(1e-3) * 0.02
            staged.append((leaf.detach().float() + noise * scale).to(leaf.dtype))
        return binder.unflatten(staged, spec)

    def _release_and_fall_back(self, delta: float) -> None:
        """The FAIL arm: graphs released, upstream rebound, said out loud. Serving continues."""
        self.rejected = True
        self._graph = None
        self._out = None
        self._prefix_graph = None
        self._prefix_pad_masks = None
        self._prefix_kv = None
        self._chunk_graph = None
        self._chunk_final = None
        # undo the table swap: the eager path must run the REAL projections as real modules
        if self._denses is not None and self._adarms_buf is not None:
            for norm, real in self._denses:
                norm.dense = real
            self._adarms_buf = None
            self._dense_bufs = None
        try:
            self._m.denoise_step = self._orig_denoise
            if self._original_sample_actions is not None:
                self._m.sample_actions = self._original_sample_actions
        except Exception:                                          # noqa: BLE001
            pass                       # the __call__ guard still routes every denoise call eager
        import sys
        # stderr, deliberately: the running server's log stream. cli_config.execute defers
        # stdout until the command returns, and a policy server returns never — a fallback
        # printed where nobody can see it until shutdown is not LOUD.
        print(f"[pi05 static_capture] SELF-CHECK FAILED: replay disagrees with eager by "
              f"{delta:.3e} on a staged input it was not captured from. Graphs released; "
              f"denoise_step rebound to upstream; sample_actions restored — serving continues "
              f"on eager arithmetic "
              f"(upstream's, exactly).", file=sys.stderr, flush=True)


def _history_kwargs(original, states, state_masks):
    import inspect
    parameters = inspect.signature(original).parameters
    result = {}
    for name, value in (("states", states), ("state_masks", state_masks)):
        if name in parameters:
            result[name] = value
        elif value is not None:
            raise ValueError(f"this upstream sample_actions does not support {name}")
    return result


def install_static_capture(
    model,
    step_tables: "bool | None" = None,
    on_self_check=None,
    self_check: bool = True,
    full_chunk: "bool | None" = None,
    prefix_graph: "bool | None" = None,
) -> StaticDenoiser:
    """Install replay-safe Static-KV capture and return its driver.

    The default path captures one denoise step and optionally hoists fixed-schedule timestep
    projections. ``full_chunk=True`` (or ``IFL_PI05_FULL_CHUNK_GRAPH=1``) keeps the original
    projections and captures the complete fixed Euler loop. ``prefix_graph=True`` (or
    ``IFL_PI05_PREFIX_GRAPH=1``) also captures the original vision/language prefix. For that fixed
    loop, ``step_tables=True`` (or ``IFL_PI05_FULL_STEP_TABLES=1``) bakes each timestep's exact
    time-MLP/AdaRMS outputs into the unrolled graph and restores the original modules afterward.

    ``self_check`` gates the first captured path on exact replay-vs-upstream equality;
    ``on_self_check`` receives its verdict for the runtime plan.
    """
    import os

    if prefix_graph is None:
        prefix_graph = os.environ.get("IFL_PI05_PREFIX_GRAPH", "0") == "1"
    if full_chunk is None:
        full_env = os.environ.get("IFL_PI05_FULL_CHUNK_GRAPH")
        full_chunk = full_env == "1" or (full_env is None and prefix_graph)
    if prefix_graph and not full_chunk:
        raise ValueError("prefix_graph=True requires full_chunk=True")
    if step_tables is None:
        if full_chunk:
            step_tables = os.environ.get("IFL_PI05_FULL_STEP_TABLES", "0") == "1"
        else:
            step_tables = os.environ.get("IFL_PI05_STEP_TABLES", "1") != "0"
    existing = getattr(model, "_ifl_static_denoiser", None)
    if existing is not None:
        if (bool(existing._full_chunk) != bool(full_chunk)
                or bool(existing._prefix_graph_enabled) != bool(prefix_graph)
                or (bool(full_chunk)
                    and bool(existing._step_tables) != bool(step_tables))):
            raise RuntimeError(
                "Static-KV capture is already installed with a different graph mode")
        return existing
    orig = getattr(model, "denoise_step", None) if self_check else None
    original_sample_actions = getattr(model, "sample_actions", None)
    if full_chunk and original_sample_actions is None:
        raise TypeError("full-chunk capture requires model.sample_actions")
    d = StaticDenoiser(
        model,
        step_tables=bool(step_tables),
        orig_denoise=orig,
        on_self_check=on_self_check,
        full_chunk=bool(full_chunk),
        prefix_graph=bool(prefix_graph),
    )
    d._original_sample_actions = original_sample_actions

    def denoise_step(self_m, prefix_pad_masks, past_key_values, x_t, timestep):
        return d(prefix_pad_masks, past_key_values, x_t, timestep)

    import types
    model.denoise_step = types.MethodType(denoise_step, model)

    if full_chunk:
        def sample_actions(
            self_m,
            images,
            img_masks,
            tokens,
            masks,
            *args,
            **kwargs,
        ):
            # Preserve the installed upstream version's positional-noise API too.
            import inspect
            signature = inspect.signature(original_sample_actions)
            bound = signature.bind(images, img_masks, tokens, masks, *args, **kwargs)
            positional = [name for name, parameter in signature.parameters.items()
                          if parameter.kind in (parameter.POSITIONAL_ONLY,
                                                parameter.POSITIONAL_OR_KEYWORD)]
            extra_key = next((name for name, parameter in signature.parameters.items()
                              if parameter.kind == parameter.VAR_KEYWORD), None)
            extra = bound.arguments.get(extra_key, {}) if extra_key else {}
            supported = set(positional[:4]) | {"states", "state_masks", "noise", "num_steps", extra_key}
            rtc_args = ("inference_delay", "prev_chunk_left_over", "execution_horizon")
            if (set(bound.arguments) - supported
                    or set(extra) - set(rtc_args)):
                # Unknown upstream controls must retain their original meaning.
                return original_sample_actions(images, img_masks, tokens, masks, *args, **kwargs)
            states = bound.arguments.get("states")
            state_masks = bound.arguments.get("state_masks")
            noise = bound.arguments.get("noise")
            num_steps = bound.arguments.get("num_steps")
            kwargs = extra
            if d.rejected:
                return original_sample_actions(
                    images, img_masks, tokens, masks,
                    **_history_kwargs(original_sample_actions, states, state_masks), noise=noise,
                    num_steps=num_steps, **kwargs)
            requested_steps = int(
                self_m.config.num_inference_steps if num_steps is None else num_steps)
            rtc_args = ("inference_delay", "prev_chunk_left_over", "execution_horizon")
            must_fallback = (
                requested_steps != int(self_m.config.num_inference_steps)
                or getattr(self_m, "_rtc_enabled", lambda: False)()
                or any(kwargs.get(name) is not None for name in rtc_args)
            )
            if must_fallback:
                return original_sample_actions(
                    images, img_masks, tokens, masks,
                    **_history_kwargs(original_sample_actions, states, state_masks), noise=noise,
                    num_steps=num_steps, **kwargs)

            bsize = tokens.shape[0]
            device = tokens.device
            if noise is None:
                noise = self_m.sample_noise(
                    (bsize, self_m.config.chunk_size, self_m.config.max_action_dim),
                    device)

            try:
                # Verbatim upstream-main prefix path, optionally replayed as its own graph.
                prefix_pad_masks, past_key_values = d.run_prefix(
                    images, img_masks, tokens, masks, states, state_masks)
                result = d.run_full_chunk(
                    prefix_pad_masks, past_key_values, noise, requested_steps)

                graphs_ready = (
                    d._chunk_graph is not None
                    and (not d._prefix_graph_enabled or d._prefix_graph is not None)
                )
                if (graphs_ready and not d._full_self_checked
                        and d._orig_denoise is not None and d._self_check_n > 0):
                    import os
                    import sys
                    if os.environ.get("IFL_PI05_SELFCHECK_FAULT") == "1":
                        print(
                            "[pi05 static_capture] FAULT INJECTED "
                            "(IFL_PI05_SELFCHECK_FAULT=1): full-chunk input rebound before "
                            "self-check — the check must now fail.",
                            file=sys.stderr,
                            flush=True,
                        )
                        d._chunk_input = d._chunk_input.clone()
                    if not d._full_self_check(
                        images, img_masks, tokens, masks, states, state_masks,
                        noise, requested_steps, kwargs,
                    ):
                        return d._eager_sample_actions(
                            images, img_masks, tokens, masks, states, state_masks,
                            noise, requested_steps, kwargs,
                        )
                return result
            except Exception as error:
                # A capture/shape failure must leave the model on the true eager path,
                # just like a numerical admission failure. Reuse the already drawn noise.
                import warnings
                warnings.warn(f"pi05 full capture failed: {type(error).__name__}: {error}; "
                              "restoring upstream eager", RuntimeWarning)
                d._release_and_fall_back(float("inf"))
                return original_sample_actions(
                    images, img_masks, tokens, masks,
                    **_history_kwargs(original_sample_actions, states, state_masks), noise=noise,
                    num_steps=num_steps, **kwargs)

        model.sample_actions = types.MethodType(sample_actions, model)
        d._original_sample_actions = original_sample_actions

    model._ifl_static_denoiser = d
    return d
