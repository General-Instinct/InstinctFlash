"""Exact Qwen3-VL inference fast path for GR00T N1.7.

Three moves, all gated bitexact under the 6-case H100 protocol (../verify_fastpaths.py):
prompt/grid metadata (positional embeddings, vision RoPE, cu_seqlens, text RoPE index) is
cached per exact content signature; the visual forward is re-expressed without the generic
dispatch; and the lm_head is skipped -- GR00T reads only hidden_states[-1], so its 151k-vocab
logits are dead work (the same dead-lm_head structure our pi05 study found).

DEPLOYMENT LIMIT -- keep this OFF any engine-side CUDA-graph capture path: the content
signatures do a GPU->CPU sync (.detach().cpu()) inside every backbone forward, which is a
whole-backbone capture blocker on its own, beyond FA2 varlen. The engine's prompt-scope
backbone cache subsumes this cache. This is a torch-chain optimization.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


class StaticQwenMetadata:
    """Cache exact prompt/grid metadata and skip the unused language-model head."""

    def __init__(self, backbone, *, max_signatures: int = 8) -> None:
        self._backbone = backbone
        self._conditional = backbone.model
        self._base = self._conditional.model
        self._visual = self._conditional.visual
        self._max_signatures = int(max_signatures)
        self._vision_pos: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._vision_rope: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._vision_cu: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._text_rope: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )
        self._last_grid_tensor = None
        self._last_grid_version = None
        self._last_grid_signature = None
        self.hits = 0
        self.misses = 0

        self._original_pos_embed = self._visual.fast_pos_embed_interpolate
        self._original_vision_rope = self._visual.rot_pos_emb
        self._original_visual_forward = self._visual.forward
        self._original_text_rope = self._base.get_rope_index
        self._original_image_features = self._base.get_image_features
        self._original_lm_head = self._conditional.lm_head.forward

        self._visual.fast_pos_embed_interpolate = self._fast_pos_embed_interpolate
        self._visual.rot_pos_emb = self._rot_pos_emb
        self._visual.forward = self._visual_forward
        self._base.get_rope_index = self._get_rope_index
        self._base.get_image_features = self._get_image_features
        self._conditional.lm_head.forward = self._skip_lm_head

    def _remember(self, cache: OrderedDict, key: tuple, value: Any):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._max_signatures:
            cache.popitem(last=False)
        return value

    def _grid_signature(self, grid_thw: torch.Tensor) -> tuple:
        try:
            version = grid_thw._version
        except RuntimeError:
            # Inference tensors are immutable in this serving path and do not
            # expose version counters. Holding a strong reference prevents an
            # allocator-reused pointer from being mistaken for the same input.
            version = None
        if grid_thw is self._last_grid_tensor and version == self._last_grid_version:
            return self._last_grid_signature
        values = tuple(int(value) for value in grid_thw.detach().cpu().reshape(-1).tolist())
        # Device is part of the key: the cached tensors are device-resident, and two devices
        # in one process (or a backbone moved between them) must never share an entry.
        signature = (tuple(grid_thw.shape), values, str(grid_thw.device))
        self._last_grid_tensor = grid_thw
        self._last_grid_version = version
        self._last_grid_signature = signature
        return signature

    @staticmethod
    def _content_signature(tensor: torch.Tensor | None):
        if tensor is None:
            return None
        values = tuple(tensor.detach().cpu().reshape(-1).tolist())
        return tuple(tensor.shape), str(tensor.dtype), values, str(tensor.device)

    def _fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        key = self._grid_signature(grid_thw)
        cached = self._vision_pos.get(key)
        if cached is not None:
            self.hits += 1
            self._vision_pos.move_to_end(key)
            return cached
        self.misses += 1
        return self._remember(
            self._vision_pos, key, self._original_pos_embed(grid_thw)
        )

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        key = self._grid_signature(grid_thw)
        cached = self._vision_rope.get(key)
        if cached is not None:
            self.hits += 1
            self._vision_rope.move_to_end(key)
            return cached
        self.misses += 1
        return self._remember(
            self._vision_rope, key, self._original_vision_rope(grid_thw)
        )

    def _get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        key = (
            self._content_signature(input_ids),
            self._grid_signature(image_grid_thw) if image_grid_thw is not None else None,
            self._grid_signature(video_grid_thw) if video_grid_thw is not None else None,
            self._content_signature(attention_mask),
        )
        cached = self._text_rope.get(key)
        if cached is not None:
            self.hits += 1
            self._text_rope.move_to_end(key)
            return cached
        self.misses += 1
        return self._remember(
            self._text_rope,
            key,
            self._original_text_rope(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            ),
        )

    def _get_image_features(self, pixel_values, image_grid_thw=None):
        pixel_values = pixel_values.type(self._visual.dtype)
        image_embeds, deepstack_image_embeds = self._visual(
            pixel_values, grid_thw=image_grid_thw
        )
        grid_key = self._grid_signature(image_grid_thw)
        values = grid_key[1]
        merge_area = self._visual.spatial_merge_size**2
        split_sizes = [
            values[index] * values[index + 1] * values[index + 2] // merge_area
            for index in range(0, len(values), 3)
        ]
        return torch.split(image_embeds, split_sizes), deepstack_image_embeds

    def _visual_forward(self, hidden_states, grid_thw, **kwargs):
        hidden_states = self._visual.patch_embed(hidden_states)
        hidden_states = hidden_states + self._visual.fast_pos_embed_interpolate(grid_thw)

        rotary_pos_emb = self._visual.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        grid_key = self._grid_signature(grid_thw)
        cu_seqlens = self._vision_cu.get(grid_key)
        if cu_seqlens is None:
            grid_values = grid_key[1]
            cumulative = [0]
            total = 0
            for index in range(0, len(grid_values), 3):
                frames, height, width = grid_values[index : index + 3]
                for _ in range(frames):
                    total += height * width
                    cumulative.append(total)
            cu_device = (
                grid_thw.device
                if self._visual.config._attn_implementation == "flash_attention_2"
                else torch.device("cpu")
            )
            cu_seqlens = self._remember(
                self._vision_cu,
                grid_key,
                torch.tensor(cumulative, dtype=torch.int32, device=cu_device),
            )
        else:
            self._vision_cu.move_to_end(grid_key)

        deepstack_features = []
        for layer_num, block in enumerate(self._visual.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self._visual.deepstack_visual_indexes:
                merger_index = self._visual.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(
                    self._visual.deepstack_merger_list[merger_index](hidden_states)
                )
        return self._visual.merger(hidden_states), deepstack_features

    @staticmethod
    def _skip_lm_head(hidden_states: torch.Tensor) -> torch.Tensor:
        # GR00T reads only ``outputs.hidden_states[-1]``. Keep the top-level
        # Transformers wrapper (which records the pre-final-norm layer output)
        # and return a zero-work view for its otherwise discarded logits.
        return hidden_states[..., :0]

    def close(self) -> None:
        self._visual.fast_pos_embed_interpolate = self._original_pos_embed
        self._visual.rot_pos_emb = self._original_vision_rope
        self._visual.forward = self._original_visual_forward
        self._base.get_rope_index = self._original_text_rope
        self._base.get_image_features = self._original_image_features
        self._conditional.lm_head.forward = self._original_lm_head
        self._vision_pos.clear()
        self._vision_rope.clear()
        self._vision_cu.clear()
        self._text_rope.clear()
        self._last_grid_tensor = None
        self._last_grid_version = None
        self._last_grid_signature = None
        if getattr(self._backbone, "_instinctflash_static_metadata", None) is self:
            delattr(self._backbone, "_instinctflash_static_metadata")


def install_backbone_fastpath(model) -> StaticQwenMetadata:
    """Install the exact inference-only Qwen3-VL metadata cache."""

    current = getattr(model.backbone, "_instinctflash_static_metadata", None)
    if current is not None:
        return current
    handle = StaticQwenMetadata(model.backbone)
    model.backbone._instinctflash_static_metadata = handle
    return handle


__all__ = ["StaticQwenMetadata", "install_backbone_fastpath"]
