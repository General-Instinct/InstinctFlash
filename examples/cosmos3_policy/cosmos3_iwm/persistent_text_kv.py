"""Exact same-prompt text K/V reuse across Cosmos3 action-policy requests.

Cosmos already proves that text K/V are independent of the changing action/vision generation
tokens and reuses them across diffusion steps. The upstream cache is request-local, however, so
the first step of every control cycle recomputes the identical prompt prefix. This installer
extends only that already-approved cache's lifetime across requests with the same transformed
caption.
"""

from __future__ import annotations

import types
from collections import Counter, OrderedDict


class PersistentTextKV:
    """Bounded, transactional prompt cache owned by one model instance."""

    def __init__(self, model, max_prompts: int = 4):
        if max_prompts < 1:
            raise ValueError("persistent text-KV max_prompts must be positive")
        for name in (
            "generate_samples_from_batch",
            "_make_inference_text_kv_cache",
            "_can_reuse_inference_text_kv",
            "input_caption_key",
        ):
            if not hasattr(model, name):
                raise RuntimeError(
                    f"Cosmos persistent text-KV requires model.{name}; upstream changed"
                )
        config = getattr(model, "config", None)
        parallel = getattr(model, "parallel_dims", None)
        if config is None or config.joint_attn_implementation != "two_way":
            raise RuntimeError("persistent text-KV requires two_way attention")
        if config.video_temporal_causal or config.sound_gen:
            raise RuntimeError(
                "persistent text-KV is certified without temporal-causal video or sound"
            )
        if parallel is not None and (
            parallel.cp_enabled or parallel.cfgp_enabled or parallel.dp_shard_enabled
        ):
            raise RuntimeError(
                "persistent text-KV is certified only for unsharded inference"
            )

        self.model = model
        self.max_prompts = int(max_prompts)
        self.caches = OrderedDict()
        self.stats = Counter()
        self._active_key = None
        self._active_cache = None
        self._active_hit = False
        self._generate = model.generate_samples_from_batch
        self._make_cache = model._make_inference_text_kv_cache

    def _key(self, data_batch, kwargs):
        captions = data_batch.get(self.model.input_caption_key)
        if captions is None:
            return None
        if kwargs.get("upsample_task") is not None:
            return None
        guidance = float(kwargs.get("guidance", 1.5))
        if guidance != 1.0 or kwargs.get("velocity_postprocess_builder") is not None:
            return None
        if kwargs.get("has_negative_prompt", False):
            return None
        return tuple(str(caption) for caption in captions)

    def generate(self, data_batch, *args, **kwargs):
        if self._active_key is not None:
            raise RuntimeError(
                "persistent text-KV does not support re-entrant generation"
            )
        key = self._key(data_batch, kwargs)
        if key is None:
            self.stats["bypass"] += 1
            return self._generate(data_batch, *args, **kwargs)

        cached = self.caches.get(key)
        self._active_key = key
        self._active_cache = cached
        self._active_hit = cached is not None
        if cached is not None:
            self.caches.move_to_end(key)
        try:
            result = self._generate(data_batch, *args, **kwargs)
            candidate = self._active_cache
            if candidate is None:
                # The upstream eligibility predicate has the final say (pack geometry, batch,
                # optional modalities). An unanticipated but valid request stays eager rather
                # than turning this optimization into a serving failure.
                self.stats["predicate_bypass"] += 1
                return result
            if not candidate:
                raise RuntimeError("Cosmos produced an empty text K/V cache")
            if not all(cache.is_initialized for cache in candidate):
                raise RuntimeError(
                    "Cosmos text K/V cache was only partially initialized"
                )
            if not self._active_hit:
                # Publish only after a complete successful request. An exception or OOM cannot
                # leak a partially initialized per-layer cache into the next control cycle.
                self.caches[key] = candidate
                while len(self.caches) > self.max_prompts:
                    self.caches.popitem(last=False)
                    self.stats["evictions"] += 1
                self.stats["misses"] += 1
            else:
                self.stats["hits"] += 1
            return result
        finally:
            self._active_key = None
            self._active_cache = None
            self._active_hit = False

    def make_cache(self, net=None):
        if self._active_key is None:
            return self._make_cache(net)
        if self._active_cache is None:
            self._active_cache = self._make_cache(net)
        return self._active_cache


def install_persistent_text_kv(service, *, max_prompts: int = 4) -> PersistentTextKV:
    model = getattr(service, "model", None)
    if model is None:
        raise RuntimeError("Cosmos service exposes no model")
    if getattr(model, "_ifl_persistent_text_kv", None) is not None:
        raise RuntimeError("Cosmos persistent text-KV is already installed")
    if float(getattr(service.cfg, "guidance", -1.0)) != 1.0:
        raise RuntimeError("persistent text-KV is certified only for guidance=1.0")
    cache = PersistentTextKV(model, max_prompts=max_prompts)

    def generate(self, data_batch, *args, **kwargs):
        return cache.generate(data_batch, *args, **kwargs)

    def make_cache(self, net=None):
        return cache.make_cache(net)

    model.generate_samples_from_batch = types.MethodType(generate, model)
    model._make_inference_text_kv_cache = types.MethodType(make_cache, model)
    model._ifl_persistent_text_kv = cache
    return cache


__all__ = ["PersistentTextKV", "install_persistent_text_kv"]
