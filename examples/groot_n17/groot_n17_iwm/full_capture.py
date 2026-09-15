"""Offline experiment: full-path CUDA graphs for GR00T N1.7.

This candidate failed qualification. Public Runtime refuses its activation; use the
qualified independent GPU collator and native DiT/eager path. Do not report fallback
latency as full-backbone acceleration.

The existing backend captures one DiT call and replays it four times.  On RTX 5090 the
remaining launch-bound regions are the Qwen3-VL vision/language backbone and the Python Euler
loop around those four calls.  This module captures both regions without changing arithmetic:

* Qwen branches that read CUDA scalars only to select a kernel are resolved from immutable
  prompt/grid metadata before capture.  The same FA2 kernels, masks, RoPE and deep-stack writes
  still run in the same order.
* The complete fixed four-step flow loop receives an explicit upstream ``torch.randn`` sample,
  so replay consumes exactly one noise draw per request and preserves the caller's RNG stream.

Every signature proves patched eager == upstream eager and replay == patched eager exactly.
Any failed proof leaves that region on upstream eager.
"""

from __future__ import annotations

from typing import Any

import torch

from instinctflash.runtime.capture_self_check import run_capture_self_check, compare_tensors

FAMILY = "GR00T N1.7 full graph"


def _storage_signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.untyped_storage().data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
    )


def _tensor_signature(values: tuple[torch.Tensor, ...]) -> tuple:
    return tuple((tuple(value.shape), value.dtype) for value in values)


def _max_delta(left: Any, right: Any) -> float:
    if torch.is_tensor(left):
        verdict = compare_tensors(left, right)
        return 0.0 if verdict["valid"] and verdict["bitexact"] else float("inf")
    if isinstance(left, dict) or hasattr(left, "items"):
        if not hasattr(right, "keys") or set(left.keys()) != set(right.keys()):
            return float("inf")
        return max((_max_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (tuple, list)):
        if not isinstance(right, (tuple, list)) or len(left) != len(right):
            return float("inf")
        return max((_max_delta(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0 if left == right else float("inf")


class CaptureSafeBackbonePatches:
    """Remove host-scalar branches from one Qwen backbone, preserving its kernels exactly."""

    def __init__(self, metadata) -> None:
        self.metadata = metadata
        self.installed = False
        self._restore: list[tuple[Any, str, Any]] = []
        self._qmod = None
        self._original_create_causal_mask = None

    def _replace(self, owner: Any, name: str, value: Any) -> None:
        self._restore.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def install(self) -> None:
        if self.installed:
            return
        from transformers.modeling_flash_attention_utils import (
            _is_packed_sequence,
            prepare_fa_kwargs_from_position_ids,
        )
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qmod

        self._qmod = qmod
        metadata = self.metadata

        # The static graph buffers retain object identity.  Cache content signatures once on
        # warmup instead of synchronizing their values to CPU from inside capture.
        original_content_signature = metadata._content_signature
        content_cache: dict[int, tuple[torch.Tensor, tuple]] = {}

        def content_signature(tensor):
            if tensor is None:
                return None
            key = id(tensor)
            cached = content_cache.get(key)
            if cached is not None and cached[0] is tensor:
                return cached[1]
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Qwen content signature missed during CUDA capture")
            signature = original_content_signature(tensor)
            content_cache[key] = (tensor, signature)
            return signature

        self._replace(metadata, "_content_signature", content_signature)

        # Vision FA2 computes max(cu_seqlens.diff()) as a CUDA scalar in every layer.  Its
        # value is geometry-only; cache the Python int during warmup and call the same FA2 op.
        for block in metadata._visual.blocks:
            attention = block.attn
            max_cache: dict[tuple, int] = {}

            def vision_forward(
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=None,
                position_embeddings=None,
                _attention=attention,
                _cache=max_cache,
                **kwargs,
            ):
                seq_length = hidden_states.shape[0]
                query, key, value = (
                    _attention.qkv(hidden_states)
                    .reshape(seq_length, 3, _attention.num_heads, -1)
                    .permute(1, 0, 2, 3)
                    .unbind(0)
                )
                cos, sin = position_embeddings
                query, key = qmod.apply_rotary_pos_emb_vision(query, key, cos, sin)
                query = query.transpose(0, 1).unsqueeze(0)
                key = key.transpose(0, 1).unsqueeze(0)
                value = value.transpose(0, 1).unsqueeze(0)
                signature = _storage_signature(cu_seqlens)
                max_seqlen = _cache.get(signature)
                if max_seqlen is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "vision max_seqlen missed during CUDA capture"
                        )
                    max_seqlen = int(
                        (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu()
                    )
                    _cache[signature] = max_seqlen
                interface = qmod.ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
                output, _ = interface(
                    _attention,
                    query,
                    key,
                    value,
                    attention_mask=None,
                    scaling=_attention.scaling,
                    dropout=0.0,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=max_seqlen,
                    max_length_k=max_seqlen,
                    is_causal=False,
                    **kwargs,
                )
                return _attention.proj(output.reshape(seq_length, -1).contiguous())

            self._replace(attention, "forward", vision_forward)

        base = metadata._base
        placeholder_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        original_placeholder = base.get_placeholder_mask

        def placeholder(
            input_ids, inputs_embeds, image_features=None, video_features=None
        ):
            if input_ids is None:
                return original_placeholder(
                    input_ids,
                    inputs_embeds,
                    image_features=image_features,
                    video_features=video_features,
                )
            signature = (*_storage_signature(input_ids), tuple(inputs_embeds.shape))
            cached = placeholder_cache.get(signature)
            if cached is not None:
                return cached
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("placeholder mask missed during CUDA capture")
            image = input_ids == base.config.image_token_id
            video = input_ids == base.config.video_token_id
            image = (
                image.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            video = (
                video.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            placeholder_cache[signature] = (image, video)
            return image, video

        self._replace(base, "get_placeholder_mask", placeholder)

        language_model = base.language_model
        index_cache: dict[tuple, torch.Tensor] = {}

        def deepstack(hidden_states, visual_pos_masks, visual_embeds):
            visual_pos_masks = visual_pos_masks.to(hidden_states.device)
            visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
            signature = _storage_signature(visual_pos_masks)
            indices = index_cache.get(signature)
            if indices is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("deepstack mask missed during CUDA capture")
                indices = visual_pos_masks.reshape(-1).nonzero().flatten()
                index_cache[signature] = indices
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            local = flat.index_select(0, indices).clone() + visual_embeds
            flat.index_copy_(0, indices, local)
            return hidden_states

        self._replace(language_model, "_deepstack_process", deepstack)

        # GR00T's public contract is unbatched: its one tokenizer sequence cannot contain
        # left-padding.  Upstream attention_mask.all() therefore always returns True and the
        # FA2 causal mask is None.  Resolve that validation branch without reading a CUDA bool.
        original_create_causal_mask = qmod.create_causal_mask
        target_config = language_model.config

        def create_causal_mask(
            *args, config=None, input_embeds=None, attention_mask=None, **kwargs
        ):
            if (
                config is target_config
                and attention_mask is not None
                and attention_mask.ndim == 2
                and attention_mask.shape[0] == 1
            ):
                return None
            return original_create_causal_mask(
                *args,
                config=config,
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )

        self._original_create_causal_mask = original_create_causal_mask
        qmod.create_causal_mask = create_causal_mask

        # Multimodal text position ids look packed to Transformers.  Compute its exact varlen
        # FA2 arguments once outside capture, then pass them to the same integration kernel.
        position_cache: dict[tuple, tuple[bool, Any]] = {}
        interface = qmod.ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        for layer in language_model.layers:
            attention = layer.self_attn

            def language_attention(
                module,
                query,
                key,
                value,
                attention_mask,
                _interface=interface,
                **kwargs,
            ):
                position_ids = kwargs.get("position_ids")
                if position_ids is not None and query.shape[0] == 1:
                    signature = _storage_signature(position_ids)
                    entry = position_cache.get(signature)
                    if entry is None:
                        if torch.cuda.is_current_stream_capturing():
                            raise RuntimeError(
                                "language varlen signature missed during capture"
                            )
                        packed = bool(
                            _is_packed_sequence(position_ids, 1).detach().cpu()
                        )
                        prepared = (
                            prepare_fa_kwargs_from_position_ids(position_ids)
                            if packed
                            else None
                        )
                        entry = (packed, prepared)
                        position_cache[signature] = entry
                    packed, prepared = entry
                    kwargs = dict(kwargs)
                    kwargs.pop("position_ids", None)
                    if packed:
                        (cu_q, cu_k), (max_q, max_k) = prepared
                        kwargs.update(
                            cu_seq_lens_q=cu_q,
                            cu_seq_lens_k=cu_k,
                            max_length_q=max_q,
                            max_length_k=max_k,
                        )
                return _interface(module, query, key, value, attention_mask, **kwargs)

            # Patch the integration callable selected by this one attention instance without
            # changing global Transformers registries.
            def text_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                cache_position=None,
                _attention=attention,
                _language_attention=language_attention,
                **kwargs,
            ):
                shape = hidden_states.shape[:-1]
                hidden_shape = (*shape, -1, _attention.head_dim)
                query = _attention.q_norm(
                    _attention.q_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                key = _attention.k_norm(
                    _attention.k_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                value = (
                    _attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                )
                cos, sin = position_embeddings
                query, key = qmod.apply_rotary_pos_emb(query, key, cos, sin)
                if past_key_values is not None:
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    key, value = past_key_values.update(
                        key, value, _attention.layer_idx, cache_kwargs
                    )
                output, weights = _language_attention(
                    _attention,
                    query,
                    key,
                    value,
                    attention_mask,
                    dropout=0.0,
                    scaling=_attention.scaling,
                    **kwargs,
                )
                output = output.reshape(*shape, -1).contiguous()
                return _attention.o_proj(output), weights

            self._replace(attention, "forward", text_forward)

        self.installed = True

    def uninstall(self) -> None:
        if self._qmod is not None and self._original_create_causal_mask is not None:
            self._qmod.create_causal_mask = self._original_create_causal_mask
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)
        self._restore.clear()
        self._qmod = None
        self._original_create_causal_mask = None
        self.installed = False


class StaticBackbone:
    """Capture the complete Qwen backbone as dependency-ordered vision/language graphs.

    A single monolithic graph is not replay-safe in Transformers 4.57: its nested
    check_model_inputs recorders retain a stale last-layer tensor after the first
    replay. Qwen's natural vision-to-language boundary avoids that recorder state while
    keeping every input-dependent backbone kernel in CUDA graphs. Static token/grid/RoPE
    metadata is computed once per signature, exactly as in upstream eager.
    """

    SELF_CHECK_INPUTS = 6

    def __init__(self, backbone, metadata, on_self_check=None) -> None:
        self.backbone = backbone
        self.original = backbone.forward
        self.metadata = metadata
        self.base = metadata._base
        self.visual = metadata._visual
        self.language = self.base.language_model
        self.patches = CaptureSafeBackbonePatches(metadata)
        self.entries: dict[tuple, tuple] = {}
        self.vision_entries: dict[tuple, tuple] = {}
        self.captures = 0
        self.replays = 0
        self.rejected = False
        self.self_check = None
        self._on_self_check = on_self_check
        self._original_norm_forward = self.language.norm.forward
        self._pre_norm: torch.Tensor | None = None
        self._norm_installed = False

    @property
    def captured(self) -> bool:
        return bool(self.entries) and not self.rejected

    @staticmethod
    def _signature(values: dict[str, torch.Tensor]) -> tuple:
        pieces = []
        for key, value in sorted(values.items()):
            item: list[Any] = [key, tuple(value.shape), value.dtype]
            if not value.is_floating_point():
                item.append(tuple(value.detach().cpu().reshape(-1).tolist()))
            pieces.append(tuple(item))
        return tuple(pieces)

    @staticmethod
    def _feature(values):
        from transformers.feature_extraction_utils import BatchFeature

        return BatchFeature(data=values)

    def _install_norm_recorder(self) -> None:
        if self._norm_installed:
            return

        def record_pre_norm(hidden_states):
            # The outer Qwen conditional-generation recorder exposes this exact tensor as
            # hidden_states[-1]. Its base-language output replaces it with post-norm, so
            # retaining the norm input is required for GR00T bitexactness.
            self._pre_norm = hidden_states
            return self._original_norm_forward(hidden_states)

        self.language.norm.forward = record_pre_norm
        self._norm_installed = True

    def _restore_norm(self) -> None:
        if self._norm_installed:
            self.language.norm.forward = self._original_norm_forward
            self._norm_installed = False
            self._pre_norm = None

    def _reject(self, reason: str) -> None:
        import sys

        self.rejected = True
        self.entries.clear()
        self.vision_entries.clear()
        self._restore_norm()
        self.patches.uninstall()
        print(
            f"[{FAMILY}] backbone graphs released: {reason}; upstream eager continues.",
            file=sys.stderr,
            flush=True,
        )

    def _vision(self, pixel_values, image_grid_thw):
        image_parts, deepstack = self.base.get_image_features(
            pixel_values, image_grid_thw
        )
        return torch.cat(image_parts, dim=0), tuple(deepstack)

    def _language(
        self,
        input_ids,
        position_ids,
        attention_mask,
        image_mask,
        image_embeds,
        *deepstack,
    ):
        inputs_embeds = self.base.get_input_embeddings()(input_ids)
        expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            expanded_image_mask, image_embeds.to(inputs_embeds.dtype)
        )
        self.language(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            cache_position=None,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=list(deepstack),
            output_hidden_states=False,
        )
        if self._pre_norm is None:
            raise RuntimeError("Qwen pre-norm recorder did not observe the last layer")
        return self._pre_norm

    @staticmethod
    def _vision_signature(values: dict[str, torch.Tensor]) -> tuple:
        pixel = values["pixel_values"]
        grid = values["image_grid_thw"]
        return (
            tuple(pixel.shape),
            pixel.dtype,
            tuple(grid.shape),
            grid.dtype,
            tuple(grid.detach().cpu().reshape(-1).tolist()),
        )

    def _capture_vision(self, values):
        vision_signature = self._vision_signature(values)
        entry = self.vision_entries.get(vision_signature)
        if entry is not None:
            return entry

        pixel_buffer = values["pixel_values"].clone()
        grid_buffer = values["image_grid_thw"].clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._vision(pixel_buffer, grid_buffer)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=pool):
            image_output, deepstack_output = self._vision(pixel_buffer, grid_buffer)
        entry = (
            graph,
            pool,
            pixel_buffer,
            grid_buffer,
            image_output,
            tuple(deepstack_output),
        )
        self.vision_entries[vision_signature] = entry
        self.captures += 1
        return entry

    def _capture_language(self, values, vision_entry):
        (
            _vision_graph,
            _vision_pool,
            _pixel_buffer,
            _grid_buffer,
            image_output,
            deepstack_output,
        ) = vision_entry

        input_ids = values["input_ids"].clone()
        attention_mask = values["attention_mask"].clone()
        image_grid = values["image_grid_thw"].clone()
        image_mask = input_ids == self.base.config.image_token_id
        position_ids, _rope_deltas = self.base.get_rope_index(
            input_ids,
            image_grid,
            None,
            attention_mask=attention_mask,
        )

        language_values = (
            input_ids,
            position_ids,
            attention_mask,
            image_mask,
            image_output,
            *deepstack_output,
        )
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._language(*language_values)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=pool):
            output = self._language(*language_values)
        self.captures += 1
        return (
            graph,
            pool,
            language_values,
            output,
            attention_mask == 1,
            image_mask,
        )

    def _replay_entry(self, entry, values):
        vision_entry, language_entry = entry
        (
            vision_graph,
            _vision_pool,
            pixel_buffer,
            grid_buffer,
            _image_output,
            _deepstack_output,
        ) = vision_entry
        (
            language_graph,
            _language_pool,
            _language_values,
            output,
            backbone_attention_mask,
            image_mask,
        ) = language_entry

        pixel_buffer.copy_(values["pixel_values"])
        grid_buffer.copy_(values["image_grid_thw"])
        vision_graph.replay()
        language_graph.replay()
        return self._feature(
            {
                "backbone_features": output,
                "backbone_attention_mask": backbone_attention_mask,
                "image_mask": image_mask,
            }
        )

    def _upstream_features(self, values):
        """Use unpatched operators for every signature and every staged admission case."""
        installed = self.patches.installed
        self._restore_norm()
        self.patches.uninstall()
        try:
            return self.original(self._feature(values))
        finally:
            if installed:
                self.patches.install()
                self._install_norm_recorder()

    def __call__(self, vl_input):
        values = {
            key: vl_input[key]
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        }
        if self.rejected:
            return self.original(self._feature(values))

        self.backbone.set_frozen_modules_to_eval_mode()
        signature = self._signature(values)
        entry = self.entries.get(signature)
        if entry is None:
            static_values = {key: value.clone() for key, value in values.items()}
            static_input = self._feature(static_values)

            if len(self.entries) >= 4:
                self._reject("bounded graph cache reached four prompt signatures")
                return self.original(static_input)
            try:
                reference = self._upstream_features(static_values)
                self.patches.install()
                self._install_norm_recorder()
                patched = self.original(static_input)
                if _max_delta(reference, patched) != 0.0:
                    self._reject("capture-safe eager patches differ from true upstream")
                    return reference
            except Exception as error:
                self._reject(f"patch admission failed ({type(error).__name__}: {error})")
                return self.original(static_input)

            try:
                vision_entry = self._capture_vision(static_values)
                # The language graph directly reads the retained vision-graph outputs,
                # eliminating four D2D handoff copies.
                vision_entry[2].copy_(static_values["pixel_values"])
                vision_entry[3].copy_(static_values["image_grid_thw"])
                vision_entry[0].replay()
                language_entry = self._capture_language(static_values, vision_entry)
                entry = (vision_entry, language_entry)
            except Exception as error:  # noqa: BLE001 - eager fallback is the contract
                self._reject(f"capture failed ({type(error).__name__}: {error})")
                return reference

            anchor = static_values["pixel_values"]
            generator = torch.Generator(device=anchor.device)
            generator.manual_seed(0x5090)

            def one_case(_index):
                staged = dict(static_values)
                fresh = torch.empty(
                    anchor.shape,
                    device=anchor.device,
                    dtype=torch.float32,
                ).normal_(generator=generator)
                staged["pixel_values"] = fresh.to(anchor.dtype)

                def run_eager():
                    return self._upstream_features(staged)["backbone_features"].clone()

                def run_replay():
                    return self._replay_entry(entry, staged)[
                        "backbone_features"
                    ].clone()

                return "fresh pixel_values", run_eager, run_replay

            verdict = run_capture_self_check(
                family=FAMILY,
                cases=(one_case(index) for index in range(self.SELF_CHECK_INPUTS)),
                tolerance=0.0,
            )
            self.self_check = verdict
            if self._on_self_check is not None:
                self._on_self_check(dict(verdict))
            if not verdict["passed"]:
                self._reject(f"self-check delta {verdict['max_abs_delta']:.3e}")
                return reference

            self.entries[signature] = entry

        result = self._replay_entry(entry, values)
        self.replays += 2
        return result

    def close(self) -> None:
        self.entries.clear()
        self.vision_entries.clear()
        self._restore_norm()
        self.patches.uninstall()
        if self.backbone.forward is self:
            self.backbone.forward = self.original


class StaticFullFlow:
    """Capture action encode + four DiTs + action decode + Euler integration as one graph."""

    SELF_CHECK_INPUTS = 6

    def __init__(self, head, on_self_check=None) -> None:
        self.head = head
        self.original = head.get_action_with_features
        self.entries: dict[tuple, tuple] = {}
        self.captures = 0
        self.replays = 0
        self.rejected = False
        self.self_check = None
        self._upstream_proved = set()
        self._on_self_check = on_self_check
        self._pool = torch.cuda.graph_pool_handle()

    @property
    def captured(self) -> bool:
        return bool(self.entries) and not self.rejected

    def _flow(
        self,
        noise,
        backbone_features,
        state_features,
        embodiment_id,
        image_mask,
        backbone_attention_mask,
    ):
        head = self.head
        actions = noise
        dt = 1.0 / head.num_inference_timesteps
        vel_strength = torch.ones_like(actions)
        for step in range(head.num_inference_timesteps):
            discretized = int(
                (step / float(head.num_inference_timesteps)) * head.num_timestep_buckets
            )
            timesteps = torch.full(
                size=(actions.shape[0],),
                fill_value=discretized,
                device=actions.device,
            )
            action_features = head.action_encoder(actions, timesteps, embodiment_id)
            if head.config.add_pos_embed:
                positions = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=actions.device
                )
                action_features = action_features + head.position_embedding(
                    positions
                ).unsqueeze(0)
            state_action = torch.cat((state_features, action_features), dim=1)
            model_output = head.model(
                hidden_states=state_action,
                encoder_hidden_states=backbone_features,
                timestep=timesteps,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
            prediction = head.action_decoder(model_output, embodiment_id)
            velocity = prediction[:, -head.action_horizon :]
            actions = actions + dt * velocity * vel_strength
        return actions

    def _fallback(
        self,
        backbone_features,
        state_features,
        embodiment_id,
        backbone_output,
        action_input,
        options,
    ):
        return self.original(
            backbone_features,
            state_features,
            embodiment_id,
            backbone_output,
            action_input,
            options,
        )

    def __call__(
        self,
        backbone_features,
        state_features,
        embodiment_id,
        backbone_output,
        action_input,
        options=None,
    ):
        from transformers.feature_extraction_utils import BatchFeature

        if (
            self.rejected
            or options is not None
            or "action" in action_input
            or self.head.num_inference_timesteps != 4
        ):
            return self._fallback(
                backbone_features,
                state_features,
                embodiment_id,
                backbone_output,
                action_input,
                options,
            )
        values = (
            backbone_features,
            state_features,
            embodiment_id,
            backbone_output.image_mask,
            backbone_output.backbone_attention_mask,
        )
        rng_before = torch.cuda.get_rng_state()
        noise = torch.randn(
            size=(
                backbone_features.shape[0],
                self.head.config.action_horizon,
                self.head.action_dim,
            ),
            dtype=backbone_features.dtype,
            device=backbone_features.device,
        )
        rng_after = torch.cuda.get_rng_state()

        signature = _tensor_signature((*values, noise))
        if signature not in self.entries and len(self.entries) >= 4:
            self.rejected = True
            self.entries.clear()
            torch.cuda.set_rng_state(rng_before)
            return self._fallback(backbone_features, state_features, embodiment_id,
                                  backbone_output, action_input, options)
        if signature not in self._upstream_proved:
            torch.cuda.set_rng_state(rng_before)
            reference = self._fallback(
                backbone_features,
                state_features,
                embodiment_id,
                backbone_output,
                action_input,
                options,
            )["action_pred"]
            torch.cuda.set_rng_state(rng_after)
            candidate = self._flow(noise, *values)
            delta = _max_delta(reference, candidate)
            if delta != 0.0:
                self.rejected = True
                return BatchFeature(
                    data={
                        "action_pred": reference,
                        "backbone_features": backbone_features,
                        "state_features": state_features,
                    }
                )
            self._upstream_proved.add(signature)

        signature = _tensor_signature((*values, noise))
        entry = self.entries.get(signature)
        if entry is None:
            buffers = [value.clone() for value in (*values, noise)]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._flow(buffers[-1], *buffers[:-1])
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=self._pool):
                    output = self._flow(buffers[-1], *buffers[:-1])
            except Exception:  # noqa: BLE001 - upstream eager remains available
                self.rejected = True
                self.entries.clear()
                torch.cuda.set_rng_state(rng_before)
                return self._fallback(
                    backbone_features,
                    state_features,
                    embodiment_id,
                    backbone_output,
                    action_input,
                    options,
                )

            generator = torch.Generator(device=noise.device)
            generator.manual_seed(0xF10)

            def one_case(index):
                staged = []
                for value in buffers[:-1]:
                    if value.is_floating_point():
                        fresh = torch.empty(
                            value.shape, device=value.device, dtype=torch.float32
                        ).normal_(generator=generator)
                        staged.append(fresh.to(value.dtype))
                    else:
                        staged.append(value.clone())
                # Seed a temporary RNG stream so upstream draws exactly the same noise
                # as replay, without consuming the caller's random sequence.
                with torch.random.fork_rng(devices=[noise.device]):
                    torch.cuda.manual_seed(0xF10 + index)
                    staged_rng = torch.cuda.get_rng_state()
                    staged_noise = torch.randn(noise.shape, dtype=noise.dtype,
                                               device=noise.device)
                staged_output = BatchFeature(data={
                    "image_mask": staged[3], "backbone_attention_mask": staged[4]})

                def run_eager():
                    with torch.random.fork_rng(devices=[noise.device]):
                        torch.cuda.set_rng_state(staged_rng)
                        return self._fallback(*staged[:3], staged_output,
                                              action_input, options)["action_pred"]

                def run_replay():
                    for buffer, value in zip(buffers, (*staged, staged_noise)):
                        buffer.copy_(value)
                    graph.replay()
                    return output

                return "fresh flow inputs vs upstream", run_eager, run_replay

            verdict = run_capture_self_check(
                family=FAMILY,
                cases=(one_case(index) for index in range(self.SELF_CHECK_INPUTS)),
                tolerance=0.0,
            )
            self.self_check = verdict
            if self._on_self_check is not None:
                self._on_self_check(dict(verdict))
            if not verdict["passed"]:
                self.rejected = True
                self.entries.clear()
                torch.cuda.set_rng_state(rng_before)
                return self._fallback(
                    backbone_features,
                    state_features,
                    embodiment_id,
                    backbone_output,
                    action_input,
                    options,
                )
            entry = (graph, buffers, output)
            self.entries[signature] = entry
            self.captures += 1

        graph, buffers, output = entry
        for buffer, value in zip(buffers, (*values, noise)):
            buffer.copy_(value)
        graph.replay()
        self.replays += 1
        return BatchFeature(
            data={
                "action_pred": output.clone(),
                "backbone_features": backbone_features,
                "state_features": state_features,
            }
        )

    def close(self) -> None:
        self.entries.clear()
        if self.head.get_action_with_features is self:
            self.head.get_action_with_features = self.original


class FullGraphDriver:
    def __init__(self, model, metadata, on_self_check=None) -> None:
        self.model = model
        self.backbone = StaticBackbone(
            model.backbone, metadata, on_self_check=on_self_check
        )
        self.flow = StaticFullFlow(model.action_head, on_self_check=on_self_check)
        model.backbone.forward = self.backbone
        model.action_head.get_action_with_features = self.flow

    @property
    def captured(self) -> bool:
        return self.backbone.captured and self.flow.captured

    @property
    def captures(self) -> int:
        return self.backbone.captures + self.flow.captures

    @property
    def replays(self) -> int:
        return self.backbone.replays + self.flow.replays

    def close(self) -> None:
        self.backbone.close()
        self.flow.close()
        if getattr(self.model, "_instinctflash_full_graph", None) is self:
            delattr(self.model, "_instinctflash_full_graph")


def install_full_capture(model, metadata, on_self_check=None) -> FullGraphDriver:
    current = getattr(model, "_instinctflash_full_graph", None)
    if current is not None:
        return current
    driver = FullGraphDriver(model, metadata, on_self_check=on_self_check)
    model._instinctflash_full_graph = driver
    return driver


__all__ = ["FullGraphDriver", "install_full_capture"]
