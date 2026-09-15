"""SM120 bitexact full-path CUDA graphs for LingBot-VLA-4B.

The family default captures one suffix velocity call. On RTX 5090, the uncaptured Qwen2.5-VL
vision/prefix path is about 39 ms and ten individually replayed velocity graphs still incur host
dispatch between Euler steps. This module adds two dependency-ordered graphs while retaining the
same PyTorch kernels and BF16 arithmetic:

* one graph for image embedding plus the prefix KV prefill;
* one graph for the complete fixed ten-step Euler schedule over StaticVelocity's fixed KV arena.

Graph-incompatible host metadata is hoisted once: the fixed image grid, the vision window index,
and FlashAttention max sequence lengths. Six complete sample_actions inputs, including changed
image/token/state bytes, must replay exactly against the true upstream path before graphs serve.
A failure restores the existing per-step static-KV backend.
"""

from __future__ import annotations

import contextlib
import types
from typing import Any

import torch

from instinctflash.runtime.capture_self_check import run_capture_self_check

FAMILY = "LingBot-VLA-4B full graph"
SELF_CHECK_FAULT_ENV = "IFL_VLA4B_SELFCHECK_FAULT"


def _max_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def _copy_same(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
        raise RuntimeError(
            f"{name} signature changed from "
            f"{(tuple(dst.shape), dst.dtype, dst.device)} to "
            f"{(tuple(src.shape), src.dtype, src.device)}"
        )
    dst.copy_(src)


class _StaticLinear(torch.nn.Module):
    """Return one precomputed projection while the full loop is captured."""

    def __init__(self, real: torch.nn.Module, output: torch.Tensor) -> None:
        super().__init__()
        self.real = real
        self.output = output

    def forward(self, _conditioning):
        return self.output


def _eager_vision_with_host_boundaries(original):
    boundaries = {}

    def forward(module, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None):
        signature = (cu_seqlens.untyped_storage().data_ptr(), tuple(cu_seqlens.shape),
                     cu_seqlens.storage_offset())
        host = boundaries.get(signature)
        if host is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("vision sequence boundaries were not warmed before capture")
            host = cu_seqlens.detach().cpu().clone()
            boundaries[signature] = host
        return original(hidden_states, host, rotary_pos_emb=rotary_pos_emb,
                        position_embeddings=position_embeddings)

    return forward


class CaptureSafePrefixPatches:
    """Hoist fixed Qwen vision metadata without changing learned operations."""

    def __init__(self, fm) -> None:
        self.fm = fm
        self.expert = fm.qwenvl_with_expert
        self.vision = self.expert.qwenvl.visual
        self.original_embed_image = self.expert.embed_image
        self.original_attention = [
            (block.attn, block.attn.forward) for block in self.vision.blocks
        ]
        self.patched_attention: list[tuple[Any, Any]] = []
        self.patched_embed_image = None
        self.image_signature = None
        self.static_grid = None
        self.static_window_index = None
        self.installed = False

    def _initialize(self, image: torch.Tensor) -> None:
        if self.image_signature is not None:
            return
        required = (
            self.expert.rotary_pos_emb,
            self.expert.window_index,
            self.expert.cu_window_seqlens,
            self.expert.cu_seqlens,
        )
        if any(value is None for value in required):
            raise RuntimeError(
                "Qwen vision grid metadata was not warmed before capture"
            )
        h = w = int(image.shape[1] ** 0.5)
        self.image_signature = (
            tuple(image.shape),
            image.dtype,
            image.device,
            h,
            w,
        )
        self.static_grid = torch.tensor(
            [[1, h, w]] * image.shape[0],
            device=image.device,
        )
        self.static_window_index = self.expert.window_index.to(device=image.device)

        from lingbotvla.models.vla.pi0 import qwenvl_in_vla as qwen


        outer = self

        def embed_image(_expert, current, patch_size=14, temporal_patch_size=2):
            signature = (
                tuple(current.shape),
                current.dtype,
                current.device,
                int(current.shape[1] ** 0.5),
                int(current.shape[1] ** 0.5),
            )
            if signature != outer.image_signature:
                raise RuntimeError(
                    f"prefix image signature changed: {outer.image_signature} -> {signature}"
                )
            embeds = outer.vision(
                current,
                grid_thw=outer.static_grid,
                rotary_pos_emb=_expert.rotary_pos_emb,
                window_index=outer.static_window_index,
                cu_window_seqlens=_expert.cu_window_seqlens,
                cu_seqlens=_expert.cu_seqlens,
            )
            split_size = (
                outer.image_signature[3]
                * outer.image_signature[4]
                // outer.vision.spatial_merge_size**2
            )
            return embeds.view(current.shape[0], split_size, -1)

        self.patched_embed_image = types.MethodType(embed_image, self.expert)

        for attention, _original in self.original_attention:
            if type(attention).__name__ == "Qwen2_5_VLVisionAttention":
                # The eager implementation uses cu_seqlens solely as Python slice
                # boundaries. Keep those immutable indices on the host; execute the
                # original attention operations, including its mask fill, unchanged.
                self.patched_attention.append((attention, types.MethodType(
                    _eager_vision_with_host_boundaries(_original), attention)))
                continue
            if type(attention).__name__ != "Qwen2_5_VLVisionFlashAttention2":
                raise RuntimeError(f"unsupported vision attention: {type(attention).__name__}")
            apply_rotary_pos_emb_flashatt = qwen.apply_rotary_pos_emb_flashatt
            flash_attn_varlen_func = qwen.flash_attn_varlen_func
            max_cache: dict[tuple, int] = {}

            def vision_attention(
                module,
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=None,
                position_embeddings=None,
                _cache=max_cache,
            ):
                seq_length = hidden_states.shape[0]
                query, key, value = (
                    module.qkv(hidden_states)
                    .reshape(seq_length, 3, module.num_heads, -1)
                    .permute(1, 0, 2, 3)
                    .unbind(0)
                )
                cos, sin = position_embeddings
                query, key = apply_rotary_pos_emb_flashatt(
                    query.unsqueeze(0), key.unsqueeze(0), cos, sin
                )
                query, key = query.squeeze(0), key.squeeze(0)
                cache_key = (
                    cu_seqlens.untyped_storage().data_ptr(),
                    tuple(cu_seqlens.shape),
                    cu_seqlens.storage_offset(),
                )
                max_seqlen = _cache.get(cache_key)
                if max_seqlen is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "vision max_seqlen cache missed during CUDA capture"
                        )
                    max_seqlen = int(
                        (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu()
                    )
                    _cache[cache_key] = max_seqlen
                output_fp32 = key.dtype == torch.float32
                if output_fp32:
                    query, key, value = (
                        query.to(torch.bfloat16),
                        key.to(torch.bfloat16),
                        value.to(torch.bfloat16),
                    )
                output = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                ).reshape(seq_length, -1)
                if output_fp32:
                    output = output.to(torch.float32)
                return module.proj(output)

            self.patched_attention.append(
                (attention, types.MethodType(vision_attention, attention))
            )

    def install(self, image: torch.Tensor) -> None:
        self._initialize(image)
        if self.installed:
            return
        self.expert.embed_image = self.patched_embed_image
        for attention, patched in self.patched_attention:
            attention.forward = patched
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        self.expert.embed_image = self.original_embed_image
        for (attention, original), (_same, _patched) in zip(
            self.original_attention, self.patched_attention, strict=True
        ):
            attention.forward = original
        self.installed = False


class StaticFullPath:
    """Capture fixed-shape prefix and complete ten-step flow graphs."""

    SELF_CHECK_INPUTS = 6

    def __init__(self, fm, velocity, on_self_check=None) -> None:
        self.fm = fm
        self.velocity = velocity
        self.original_sample_actions = fm.sample_actions
        self.patches = CaptureSafePrefixPatches(fm)
        self._on_self_check = on_self_check
        self.patch_proved = False
        self.self_checked = False
        self.self_check = None
        self.rejected = False

        self.prefix_graph = None
        self.prefix_inputs = None
        self.prefix_pad_output = None
        self.prefix_kv_output = None
        self.prefix_samples = 0
        self.prefix_replays = 0
        self.prefix_pool = torch.cuda.graph_pool_handle()

        self.chunk_graph = None
        self.chunk_input = None
        self.chunk_final = None
        self.chunk_times = None
        self.chunk_dt = None
        self.chunk_steps = None
        self.chunk_samples = 0
        self.chunk_replays = 0
        self.chunk_pool = torch.cuda.graph_pool_handle()

        self.ada_pairs = None
        self.ada_tables = None
        self.ada_wrappers = None

        outer = self

        def sample_actions(
            _fm,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            vlm_causal=False,
            noise=None,
            num_steps=None,
        ):
            return outer(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                vlm_causal=vlm_causal,
                noise=noise,
                num_steps=num_steps,
            )

        self.installed_sample_actions = types.MethodType(sample_actions, fm)
        fm.sample_actions = self.installed_sample_actions

    @property
    def captured(self) -> bool:
        return (
            self.prefix_graph is not None
            and self.chunk_graph is not None
            and self.self_checked
            and not self.rejected
        )

    @contextlib.contextmanager
    def _true_upstream(self, image: torch.Tensor):
        patches_were_installed = self.patches.installed
        installed_predict = type(self.fm).predict_velocity
        self.patches.uninstall()
        self.velocity._bypass = True
        type(self.fm).predict_velocity = self.velocity._orig_predict
        try:
            yield
        finally:
            type(self.fm).predict_velocity = installed_predict
            self.velocity._bypass = False
            if patches_were_installed:
                self.patches.install(image)

    def _call_original(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        *,
        vlm_causal,
        noise,
        num_steps,
        patched_prefix: bool,
    ):
        desired = self.patches.installed
        if patched_prefix:
            self.patches.install(images.reshape(-1, *images.shape[-2:]))
            context = contextlib.nullcontext()
        else:
            context = self._true_upstream(images.reshape(-1, *images.shape[-2:]))
        installed_predict = type(self.fm).predict_velocity
        if patched_prefix:
            self.velocity._bypass = True
            type(self.fm).predict_velocity = self.velocity._orig_predict
        try:
            with context, torch.no_grad():
                return self.original_sample_actions(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    vlm_causal=vlm_causal,
                    noise=noise.clone(),
                    num_steps=num_steps,
                )
        finally:
            if patched_prefix:
                type(self.fm).predict_velocity = installed_predict
                self.velocity._bypass = False
            if desired and not self.patches.installed:
                self.patches.install(images.reshape(-1, *images.shape[-2:]))
            elif not desired and self.patches.installed:
                self.patches.uninstall()

    def _set_prefix_inputs(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
    ) -> None:
        values = (images, img_masks, lang_tokens, lang_masks)
        if self.prefix_inputs is None:
            self.prefix_inputs = tuple(value.clone() for value in values)
            return
        names = ("images", "image masks", "language tokens", "language masks")
        for dst, src, name in zip(self.prefix_inputs, values, names, strict=True):
            _copy_same(dst, src, f"prefix {name}")

    def _forward_prefix(self, vlm_causal: bool):
        from lingbotvla.models.vla.pi0.modeling_lingbot_vla import (
            make_att_2d_masks,
        )

        images, img_masks, lang_tokens, lang_masks = self.prefix_inputs
        fm = self.fm
        prefix_embs, prefix_pad_masks, prefix_att_masks = fm.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            vlm_causal,
        )
        prefix_attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        if fm.use_depth_align and fm.align_type == "query":
            prefix_attention = fm.make_att_2d_masks_with_query(
                prefix_attention,
                prefix_pad_masks.shape[-1],
                img_masks,
            )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = fm.qwenvl_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=fm.config.use_cache,
            fill_kv_cache=True,
        )
        return prefix_pad_masks, past_key_values

    def run_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        vlm_causal: bool,
    ):
        self._set_prefix_inputs(images, img_masks, lang_tokens, lang_masks)
        if self.prefix_graph is None and self.prefix_samples == 0:
            result = self._forward_prefix(vlm_causal)
            self.prefix_samples += 1
            return result
        if self.prefix_graph is None:
            self.prefix_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.prefix_graph, pool=self.prefix_pool):
                self.prefix_pad_output, self.prefix_kv_output = self._forward_prefix(
                    vlm_causal
                )
        self.prefix_graph.replay()
        self.prefix_replays += 1
        self.prefix_samples += 1
        return self.prefix_pad_output, self.prefix_kv_output

    @staticmethod
    def _schedule(noise: torch.Tensor, num_steps: int):
        dt = torch.tensor(
            -1.0 / num_steps,
            dtype=noise.dtype,
            device=noise.device,
        )
        current = torch.tensor(
            1.0,
            dtype=noise.dtype,
            device=noise.device,
        )
        times = []
        while bool((current >= -dt / 2).detach().cpu()):
            times.append(current.expand(noise.shape[0]).clone())
            current += dt
        if len(times) != num_steps:
            raise RuntimeError(
                f"upstream Euler schedule produced {len(times)} steps, expected {num_steps}"
            )
        return dt, times

    def _ada_norm_pairs(self):
        if self.ada_pairs is not None:
            return self.ada_pairs
        expert_model = self.fm.qwenvl_with_expert.qwen_expert.model
        pairs = []
        for module in expert_model.modules():
            if module.__class__.__name__ == "AdaRMSNorm":
                pairs.append((module, module.gamma, module.beta))
        self.ada_pairs = pairs
        return pairs

    def _enter_ada_tables(self):
        pairs = self._ada_norm_pairs()
        if not pairs or not bool(getattr(self.fm.config, "adanorm_time", False)):
            return None

        tables = []
        with torch.no_grad():
            for timestep in self.chunk_times:
                conditioning, _embs, _pad, _attention = self.fm.embed_suffix(
                    self.velocity._state_buf,
                    self.chunk_input,
                    timestep,
                )
                tables.append(
                    [
                        (
                            gamma(conditioning).clone(),
                            beta(conditioning).clone(),
                        )
                        for _module, gamma, beta in pairs
                    ]
                )

        wrappers = []
        for (module, gamma, beta), (gamma_out, beta_out) in zip(
            pairs,
            tables[0],
            strict=True,
        ):
            gamma_wrapper = _StaticLinear(gamma, gamma_out)
            beta_wrapper = _StaticLinear(beta, beta_out)
            module.gamma = gamma_wrapper
            module.beta = beta_wrapper
            wrappers.append((gamma_wrapper, beta_wrapper))
        self.ada_tables = tables
        self.ada_wrappers = wrappers
        return tables

    def _select_ada_table(self, table) -> None:
        for (gamma_wrapper, beta_wrapper), (gamma, beta) in zip(
            self.ada_wrappers,
            table,
            strict=True,
        ):
            gamma_wrapper.output = gamma
            beta_wrapper.output = beta

    def _exit_ada_tables(self) -> None:
        if self.ada_pairs is None:
            return
        for module, gamma, beta in self.ada_pairs:
            module.gamma = gamma
            module.beta = beta

    def run_full(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        noise,
        num_steps: int,
    ):
        velocity = self.velocity
        velocity._begin_chunk(state, prefix_pad_masks, past_key_values)
        velocity._last_pkv_obj = past_key_values
        if velocity._x_buf is None:
            velocity._x_buf = noise.clone()
            velocity._t_buf = torch.empty(
                noise.shape[0],
                dtype=noise.dtype,
                device=noise.device,
            )
        if self.chunk_input is None:
            self.chunk_input = noise.clone()
            self.chunk_dt, self.chunk_times = self._schedule(noise, num_steps)
            self.chunk_steps = int(num_steps)
        else:
            if int(num_steps) != self.chunk_steps:
                raise RuntimeError(
                    f"full graph step count changed: {self.chunk_steps} -> {num_steps}"
                )
            _copy_same(self.chunk_input, noise, "flow noise")

        if self.chunk_graph is None and self.chunk_samples == 0:
            current = self.chunk_input
            for timestep in self.chunk_times:
                velocity._x_buf.copy_(current)
                velocity._t_buf.copy_(timestep)
                current_velocity = velocity._forward_static()
                current += self.chunk_dt * current_velocity
            self.chunk_samples += 1
            return current.clone()

        if self.chunk_graph is None:
            tables = self._enter_ada_tables()
            self.chunk_graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(self.chunk_graph, pool=self.chunk_pool):
                    current = self.chunk_input
                    for index, timestep in enumerate(self.chunk_times):
                        if tables is not None:
                            self._select_ada_table(tables[index])
                        velocity._x_buf.copy_(current)
                        velocity._t_buf.copy_(timestep)
                        current_velocity = velocity._forward_static()
                        current += self.chunk_dt * current_velocity
                    self.chunk_final = current
            finally:
                self._exit_ada_tables()

        self.chunk_input.copy_(noise)
        self.chunk_graph.replay()
        self.chunk_replays += 1
        self.chunk_samples += 1
        velocity.replays += int(num_steps)
        return self.chunk_final.clone()

    def _candidate(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        *,
        vlm_causal,
        noise,
        num_steps,
    ):
        prefix_pad_masks, past_key_values = self.run_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            vlm_causal,
        )
        return self.run_full(
            state,
            prefix_pad_masks,
            past_key_values,
            noise,
            num_steps,
        )

    @staticmethod
    def _changed_inputs(images, img_masks, lang_tokens, lang_masks, state):
        return (
            torch.flip(images, dims=(-1,)),
            img_masks.clone(),
            torch.roll(lang_tokens, shifts=1, dims=-1),
            torch.roll(lang_masks, shifts=1, dims=-1),
            torch.flip(state, dims=(-1,)),
        )

    def _full_self_check(
        self,
        inputs,
        *,
        vlm_causal,
        num_steps,
    ) -> bool:
        generator = torch.Generator(device=inputs[0].device)
        generator.manual_seed(0x4B5090)
        changed = self._changed_inputs(*inputs)
        split = (self.SELF_CHECK_INPUTS + 1) // 2

        def one_case(index):
            staged = inputs if index < split else changed
            noise = torch.empty(
                (
                    staged[4].shape[0],
                    self.fm.config.n_action_steps,
                    self.fm.config.max_action_dim,
                ),
                dtype=staged[4].dtype,
                device=staged[4].device,
            ).normal_(generator=generator)

            def run_eager():
                return self._call_original(
                    *staged,
                    vlm_causal=vlm_causal,
                    noise=noise,
                    num_steps=num_steps,
                    patched_prefix=False,
                )

            def run_replay():
                return self._candidate(
                    *staged,
                    vlm_causal=vlm_causal,
                    noise=noise,
                    num_steps=num_steps,
                )

            label = "captured-chunk" if index < split else "refilled"
            return label, run_eager, run_replay

        verdict = run_capture_self_check(
            family=FAMILY,
            cases=(one_case(index) for index in range(self.SELF_CHECK_INPUTS)),
            tolerance=0.0,
        )
        self.self_check = verdict
        self.self_checked = bool(verdict["passed"])
        if self._on_self_check is not None:
            self._on_self_check(dict(verdict))
        if not verdict["passed"]:
            self._reject(f"full self-check delta {verdict['max_abs_delta']:.3e}")
        return bool(verdict["passed"])

    def _reject(self, reason: str) -> None:
        import sys

        self.rejected = True
        self.self_checked = False
        self.prefix_graph = None
        self.prefix_pad_output = None
        self.prefix_kv_output = None
        self.chunk_graph = None
        self.chunk_final = None
        self.patches.uninstall()
        if self.fm.__dict__.get("sample_actions") is self.installed_sample_actions:
            self.fm.sample_actions = self.original_sample_actions
        print(
            f"[{FAMILY}] graphs released ({reason}); per-step static-KV continues.",
            file=sys.stderr,
            flush=True,
        )

    def __call__(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        *,
        vlm_causal=False,
        noise=None,
        num_steps=None,
    ):
        requested_steps = int(
            self.fm.config.num_steps if num_steps is None else num_steps
        )
        fixed_steps = int(self.fm.config.num_steps)
        if self.rejected or bool(vlm_causal) or requested_steps != fixed_steps:
            return self.original_sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                vlm_causal=vlm_causal,
                noise=noise,
                num_steps=num_steps,
            )

        if noise is None:
            noise = torch.randn(
                (
                    state.shape[0],
                    self.fm.config.n_action_steps,
                    self.fm.config.max_action_dim,
                ),
                device=state.device,
                dtype=state.dtype,
            )

        input_tuple = (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        )
        if not self.patch_proved:
            try:
                reference = self._call_original(
                    *input_tuple,
                    vlm_causal=vlm_causal,
                    noise=noise,
                    num_steps=requested_steps,
                    patched_prefix=False,
                )
                self.patches.install(images.reshape(-1, *images.shape[-2:]))
                patched = self._call_original(
                    *input_tuple,
                    vlm_causal=vlm_causal,
                    noise=noise,
                    num_steps=requested_steps,
                    patched_prefix=True,
                )
                patch_delta = _max_delta(reference, patched)
                if patch_delta != 0.0:
                    self._reject(f"capture-safe eager patches differ by {patch_delta:.3e}")
                    return reference
                candidate = self._candidate(
                    *input_tuple,
                    vlm_causal=vlm_causal,
                    noise=noise,
                    num_steps=requested_steps,
                )
                candidate_delta = _max_delta(reference, candidate)
                if candidate_delta != 0.0:
                    self._reject(
                        f"static full-path warmup differs by {candidate_delta:.3e}"
                    )
                    return reference
                self.patch_proved = True
                return candidate
            except Exception as error:
                self._reject(f"initial full-path admission failed ({type(error).__name__}: {error})")
                return self.original_sample_actions(
                    *input_tuple, vlm_causal=vlm_causal,
                    noise=noise, num_steps=num_steps)

        try:
            candidate = self._candidate(
                *input_tuple,
                vlm_causal=vlm_causal,
                noise=noise,
                num_steps=requested_steps,
            )
        except Exception as error:  # noqa: BLE001 - per-step fallback is the contract
            self._reject(f"capture failed ({type(error).__name__}: {error})")
            return self.original_sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                vlm_causal=vlm_causal,
                noise=noise,
                num_steps=num_steps,
            )

        graphs_ready = self.prefix_graph is not None and self.chunk_graph is not None
        if graphs_ready and not self.self_checked:
            import os
            import sys

            if os.environ.get(SELF_CHECK_FAULT_ENV) == "1":
                print(
                    f"[{FAMILY}] FAULT INJECTED ({SELF_CHECK_FAULT_ENV}=1): "
                    "chunk input rebound before full self-check.",
                    file=sys.stderr,
                    flush=True,
                )
                self.chunk_input = self.chunk_input.clone()
            if not self._full_self_check(
                input_tuple,
                vlm_causal=vlm_causal,
                num_steps=requested_steps,
            ):
                return self._call_original(
                    *input_tuple,
                    vlm_causal=vlm_causal,
                    noise=noise,
                    num_steps=requested_steps,
                    patched_prefix=False,
                )
        return candidate

    def close(self) -> None:
        self.prefix_graph = None
        self.chunk_graph = None
        self.prefix_pad_output = None
        self.prefix_kv_output = None
        self.chunk_final = None
        self.patches.uninstall()
        if self.fm.__dict__.get("sample_actions") is self.installed_sample_actions:
            self.fm.sample_actions = self.original_sample_actions


class FullCaptureDriver:
    def __init__(self, fm, velocity, full) -> None:
        self.fm = fm
        self.velocity = velocity
        self.full = full

    @property
    def captured(self) -> bool:
        return self.full.captured

    @property
    def _graph(self):
        return self.full.chunk_graph

    @property
    def replays(self) -> int:
        return self.velocity.replays

    @property
    def prefix_replays(self) -> int:
        return self.full.prefix_replays

    @property
    def chunk_replays(self) -> int:
        return self.full.chunk_replays

    def close(self) -> None:
        self.full.close()
        installed = type(self.fm).predict_velocity
        if getattr(installed, "__driver__", None) is self.velocity:
            type(self.fm).predict_velocity = self.velocity._orig_predict
        self.velocity.close()
        if getattr(self.fm, "_instinctflash_full_capture", None) is self:
            delattr(self.fm, "_instinctflash_full_capture")


def install_full_capture(fm, on_self_check=None) -> FullCaptureDriver:
    current = getattr(fm, "_instinctflash_full_capture", None)
    if current is not None:
        return current

    from .static_capture import install_static_capture

    velocity = install_static_capture(
        fm,
        on_self_check=on_self_check,
        self_check=True,
    )
    full = StaticFullPath(
        fm,
        velocity,
        on_self_check=on_self_check,
    )
    driver = FullCaptureDriver(fm, velocity, full)
    fm._instinctflash_full_capture = driver
    return driver


__all__ = ["FullCaptureDriver", "install_full_capture"]
