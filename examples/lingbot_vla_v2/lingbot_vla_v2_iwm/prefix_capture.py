"""Static-input CUDA Graph replay for LingBot-VLA-V2 vision and prefix prefill.

The policy's RobotWin contract fixes the three image grids, prompt budget, and prefix extent. This
module uses that invariant at two expensive boundaries which are outside the denoise graph:

* Qwen3-VL vision encoding receives stable image/grid buffers;
* the 36-layer prefix forward receives stable embeddings, masks, positions, and deep-stack inputs.

The prefix graph returns a fresh Python KV dictionary on every replay. Its tensors still belong to
the graph, but the new container identity tells :class:`StaticVelocity` that a new observation has
arrived and its independent ``[prefix | suffix]`` KV allocation must be refilled.
"""

from __future__ import annotations

import torch

VISION_WARMUP_STEPS = 3
PREFILL_WARMUP_STEPS = 3


def _copy_same(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
        raise RuntimeError(
            f"{name} signature changed from {(dst.shape, dst.dtype, dst.device)} to "
            f"{(src.shape, src.dtype, src.device)}; CUDA Graph replay is not valid"
        )
    dst.copy_(src)


class StaticVision:
    """Capture the fixed-grid Qwen3-VL vision encoder."""

    def __init__(self, expert) -> None:
        self._expert = expert
        self._original = expert.embed_image
        self._original_grid_metadata = (
            expert.pos_embeds,
            expert.position_embeddings,
            expert.cu_seqlens,
            expert.visual_split_sizes,
            expert.visual_max_seqlen,
        )
        self._image = None
        self._grid = None
        self.graph = None
        self._output = None
        self._steps = 0
        self.replays = 0

        outer = self

        def embed_image(image, image_grid_thw):
            return outer(image, image_grid_thw)

        object.__setattr__(expert, "embed_image", embed_image)

    def _forward_static(self):
        return self._original(self._image, self._grid)

    def _prepare_grid_metadata(self) -> None:
        """Hoist Qwen's fixed-grid CPU synchronizations out of graph capture."""
        expert = self._expert
        visual = expert.qwenvl.visual
        (
            pos_embeds,
            position_embeddings,
            cu_seqlens,
            split_sizes,
            max_seqlen,
        ) = visual.preprcess_grid_thw(grid_thw=self._grid)
        if pos_embeds is None:
            pos_embeds = visual.fast_pos_embed_interpolate(self._grid)
        expert.pos_embeds = pos_embeds
        expert.position_embeddings = position_embeddings
        expert.cu_seqlens = cu_seqlens
        expert.visual_split_sizes = split_sizes
        expert.visual_max_seqlen = max_seqlen

    def __call__(self, image, image_grid_thw):
        if self._image is None:
            self._image = image.clone()
            self._grid = image_grid_thw.clone()
            self._prepare_grid_metadata()
        else:
            _copy_same(self._image, image, "vision pixels")
            _copy_same(self._grid, image_grid_thw, "vision grid")

        if self.graph is not None:
            self.graph.replay()
            self.replays += 1
            return self._output

        self._steps += 1
        if self._steps <= VISION_WARMUP_STEPS:
            return self._forward_static()

        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._output = self._forward_static()
        self.graph.replay()
        self.replays += 1
        return self._output

    def close(self) -> None:
        expert = self._expert
        if expert is not None:
            object.__setattr__(expert, "embed_image", self._original)
            (
                expert.pos_embeds,
                expert.position_embeddings,
                expert.cu_seqlens,
                expert.visual_split_sizes,
                expert.visual_max_seqlen,
            ) = self._original_grid_metadata
        self.graph = None
        self._output = None
        self._expert = None


class StaticPrefill:
    """Capture the fixed 286-token VLM-only prefix forward."""

    def __init__(self, expert) -> None:
        self._expert = expert
        self._original = expert.forward
        self._language_model = expert.qwenvl.model.language_model
        self._original_deepstack = self._language_model._deepstack_process
        self._visual_indices = None
        self._attention_mask = None
        self._position_ids = None
        self._vlm_position_ids = None
        self._prefix_embs = None
        self._visual_pos_masks = None
        self._deepstack = None
        self._use_cache = None
        self.graph = None
        self._output = None
        self._steps = 0
        self.replays = 0

        outer = self

        def forward(*args, **kwargs):
            # Training and denoise forwards keep their original behavior. The inference prefill
            # call is keyword-only, has no action-expert input, and explicitly fills the KV cache.
            inputs = kwargs.get("inputs_embeds")
            is_prefill = (
                not args
                and bool(kwargs.get("fill_kv_cache"))
                and isinstance(inputs, (list, tuple))
                and len(inputs) == 2
                and inputs[0] is not None
                and inputs[1] is None
                and kwargs.get("past_key_values") is None
            )
            if not is_prefill:
                return outer._original(*args, **kwargs)
            return outer(**kwargs)

        def deepstack_process(hidden_states, visual_pos_masks, visual_embeds):
            # Hugging Face's boolean indexing performs a dynamic nonzero, which cannot run while
            # a stream is capturing. RobotWin's visual token positions are invariant, so hoist
            # those row indices during warmup and use fixed-size gather/scatter kernels.
            if outer._visual_indices is None:
                outer._visual_indices = torch.nonzero(
                    visual_pos_masks.reshape(-1), as_tuple=False
                ).reshape(-1)
            visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            local = flat.index_select(0, outer._visual_indices).clone() + visual_embeds
            flat.index_copy_(0, outer._visual_indices, local)
            return hidden_states

        object.__setattr__(expert, "forward", forward)
        object.__setattr__(self._language_model, "_deepstack_process", deepstack_process)

    def _forward_static(self):
        return self._original(
            attention_mask=self._attention_mask,
            position_ids=self._position_ids,
            vlm_position_ids=self._vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[self._prefix_embs, None],
            use_cache=self._use_cache,
            fill_kv_cache=True,
            ada_cond=None,
            visual_pos_masks=self._visual_pos_masks,
            deepstack_visual_embeds=self._deepstack,
        )

    @staticmethod
    def _fresh_output(output):
        outputs_embeds, past_key_values, router_logits = output
        fresh_kv = {
            index: {
                "key_states": values["key_states"],
                "value_states": values["value_states"],
            }
            for index, values in past_key_values.items()
        }
        return list(outputs_embeds), fresh_kv, router_logits

    def __call__(
        self,
        *,
        attention_mask,
        position_ids,
        vlm_position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        fill_kv_cache,
        ada_cond=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **_unused,
    ):
        if ada_cond is not None:
            raise RuntimeError("LingBot prefix graph does not support action conditioning")
        prefix_embs = inputs_embeds[0]
        deepstack = list(deepstack_visual_embeds or [])
        if self._attention_mask is None:
            self._attention_mask = attention_mask.clone()
            self._position_ids = position_ids.clone()
            self._vlm_position_ids = vlm_position_ids.clone()
            self._prefix_embs = prefix_embs.clone()
            self._visual_pos_masks = visual_pos_masks.clone()
            self._deepstack = [value.clone() for value in deepstack]
            self._use_cache = bool(use_cache)
        else:
            if bool(use_cache) != self._use_cache:
                raise RuntimeError("LingBot prefix use_cache flag changed after graph warmup")
            _copy_same(self._attention_mask, attention_mask, "prefix attention mask")
            _copy_same(self._position_ids, position_ids, "prefix position ids")
            _copy_same(self._vlm_position_ids, vlm_position_ids, "VLM position ids")
            _copy_same(self._prefix_embs, prefix_embs, "prefix embeddings")
            _copy_same(self._visual_pos_masks, visual_pos_masks, "visual position mask")
            if len(deepstack) != len(self._deepstack):
                raise RuntimeError("LingBot deep-stack input count changed after graph warmup")
            for index, (dst, src) in enumerate(zip(self._deepstack, deepstack)):
                _copy_same(dst, src, f"deep-stack input {index}")

        if self.graph is not None:
            self.graph.replay()
            self.replays += 1
            return self._fresh_output(self._output)

        self._steps += 1
        if self._steps <= PREFILL_WARMUP_STEPS:
            return self._forward_static()

        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._output = self._forward_static()
        self.graph.replay()
        self.replays += 1
        return self._fresh_output(self._output)

    def close(self) -> None:
        expert = self._expert
        if expert is not None:
            object.__setattr__(expert, "forward", self._original)
            object.__setattr__(
                self._language_model, "_deepstack_process", self._original_deepstack
            )
        self.graph = None
        self._output = None
        self._expert = None
        self._language_model = None


class StaticPrefixCapture:
    """Combined installation report and cleanup handle."""

    def __init__(self, model) -> None:
        self.model = model
        self.vision = StaticVision(model.qwenvl_with_expert)
        try:
            self.prefill = StaticPrefill(model.qwenvl_with_expert)
        except Exception:
            self.vision.close()
            raise

    @property
    def captured(self) -> bool:
        return self.vision.graph is not None and self.prefill.graph is not None

    def close(self) -> None:
        if self.model is not None:
            self.prefill.close()
            self.vision.close()
            if getattr(self.model, "_instinctflash_static_prefix", None) is self:
                object.__delattr__(self.model, "_instinctflash_static_prefix")
        self.model = None


def install_prefix_capture(model) -> StaticPrefixCapture:
    """Install vision/prefill graph replay on one LingBot policy instance."""
    current = getattr(model, "_instinctflash_static_prefix", None)
    if current is not None:
        return current
    capture = StaticPrefixCapture(model)
    object.__setattr__(model, "_instinctflash_static_prefix", capture)
    return capture


__all__ = [
    "PREFILL_WARMUP_STEPS",
    "VISION_WARMUP_STEPS",
    "StaticPrefixCapture",
    "install_prefix_capture",
]
