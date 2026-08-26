"""Replay-safe static-KV CUDA Graph executor for LingBot-VLA-V2.

The upstream denoise step concatenates a new suffix K/V tensor onto a Python dictionary on every
call.  CUDA Graph replay requires stable addresses, so this module gives every layer one fixed
``[prefix | suffix]`` allocation. Prefix slots are refilled once per observation; suffix slots and
all step inputs are overwritten in place before replay. Qwen3-VL position ids are chunk constants
and therefore receive their own static input buffer.
"""

from __future__ import annotations

import torch

WARMUP_STEPS = 12


class _StaticKV:
    def __init__(self, prefill: dict):
        self._pending_prefill = prefill
        self.prefix_len = int(prefill[0]["key_states"].shape[1])
        self.suffix_len: int | None = None
        self.k: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []

    def _allocate(self, suffix_len: int) -> None:
        self.suffix_len = int(suffix_len)
        total = self.prefix_len + self.suffix_len
        prefill = self._pending_prefill
        for idx in range(len(prefill)):
            ks, vs = prefill[idx]["key_states"], prefill[idx]["value_states"]
            key = torch.empty(
                ks.shape[0], total, ks.shape[2], ks.shape[3], dtype=ks.dtype, device=ks.device
            )
            value = torch.empty_like(key)
            key[:, : self.prefix_len].copy_(ks)
            value[:, : self.prefix_len].copy_(vs)
            self.k.append(key)
            self.v.append(value)
        self._pending_prefill = None

    def refill(self, prefill: dict) -> None:
        for idx in range(len(self.k)):
            ks, vs = prefill[idx]["key_states"], prefill[idx]["value_states"]
            if int(ks.shape[1]) != self.prefix_len:
                raise RuntimeError(
                    f"prefix length changed ({ks.shape[1]} vs {self.prefix_len}); static capture "
                    "requires the checkpoint's fixed prompt budget"
                )
            self.k[idx][:, : self.prefix_len].copy_(ks)
            self.v[idx][:, : self.prefix_len].copy_(vs)

    def write_suffix(self, key_states, value_states, layer_idx):
        if self.suffix_len is None:
            self._allocate(key_states.shape[1])
        if int(key_states.shape[1]) != self.suffix_len:
            raise RuntimeError(
                f"suffix length changed ({key_states.shape[1]} vs {self.suffix_len}); cannot replay"
            )
        key, value = self.k[layer_idx], self.v[layer_idx]
        key[:, self.prefix_len :].copy_(key_states)
        value[:, self.prefix_len :].copy_(value_states)
        return key, value


class StaticVelocity:
    """Instance-scoped wrapper around ``FlowMatchingV2.predict_velocity``."""

    def __init__(self, fm):
        self._fm = fm
        self._kv: _StaticKV | None = None
        self._last_pkv_obj = None
        self._state_buf = None
        self._pad_buf = None
        self._pos_buf = None
        self._x_buf = None
        self._t_buf = None
        self.graph = None
        self._out = None
        self._steps = 0
        self.replays = 0
        self._orig_predict = fm.predict_velocity
        self._expert = fm.qwenvl_with_expert
        self._orig_handle = self._expert.handle_kv_cache
        self._install()

    def _install(self) -> None:
        outer = self

        def handle_kv_cache(key_states, value_states, layer_idx, past_key_values=None,
                            use_cache=None, fill_kv_cache=None):
            if fill_kv_cache or outer._kv is None:
                return outer._orig_handle(
                    key_states, value_states, layer_idx, past_key_values=past_key_values,
                    use_cache=use_cache, fill_kv_cache=fill_kv_cache,
                )
            key, value = outer._kv.write_suffix(key_states, value_states, layer_idx)
            return key, value, past_key_values

        def predict_velocity(state, prefix_pad_masks, past_key_values, x_t, timestep,
                             prefix_position_ids=None):
            return outer(
                state, prefix_pad_masks, past_key_values, x_t, timestep,
                prefix_position_ids=prefix_position_ids,
            )

        object.__setattr__(self._expert, "handle_kv_cache", handle_kv_cache)
        object.__setattr__(self._fm, "predict_velocity", predict_velocity)

    def _begin_chunk(self, state, prefix_pad_masks, past_key_values, prefix_position_ids) -> None:
        if prefix_position_ids is None:
            raise ValueError("LingBot-VLA-V2 static capture requires prefix_position_ids")
        if self._kv is None:
            self._kv = _StaticKV(past_key_values)
            self._state_buf = state.clone()
            self._pad_buf = prefix_pad_masks.clone()
            self._pos_buf = prefix_position_ids.clone()
            return
        self._copy_same(self._state_buf, state, "state")
        self._copy_same(self._pad_buf, prefix_pad_masks, "prefix pad mask")
        self._copy_same(self._pos_buf, prefix_position_ids, "prefix position ids")
        self._kv.refill(past_key_values)

    @staticmethod
    def _copy_same(dst, src, name: str) -> None:
        if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
            raise RuntimeError(
                f"{name} signature changed from {(dst.shape, dst.dtype, dst.device)} to "
                f"{(src.shape, src.dtype, src.device)}; CUDA Graph replay is not valid"
            )
        dst.copy_(src)

    def _forward_static(self):
        return self._orig_predict(
            self._state_buf, self._pad_buf, None, self._x_buf, self._t_buf,
            prefix_position_ids=self._pos_buf,
        )

    def __call__(self, state, prefix_pad_masks, past_key_values, x_t, timestep,
                 prefix_position_ids=None):
        if past_key_values is not self._last_pkv_obj:
            self._begin_chunk(state, prefix_pad_masks, past_key_values, prefix_position_ids)
            self._last_pkv_obj = past_key_values

        if self._x_buf is None:
            self._x_buf = x_t.clone()
            self._t_buf = timestep.clone()
        else:
            self._copy_same(self._x_buf, x_t, "noisy action")
            self._copy_same(self._t_buf, timestep, "timestep")

        if self.graph is not None:
            self.graph.replay()
            self.replays += 1
            return self._out.clone()

        self._steps += 1
        if self._steps <= WARMUP_STEPS:
            return self._forward_static()

        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._out = self._forward_static()
        self.graph.replay()
        self.replays += 1
        return self._out.clone()

    def close(self) -> None:
        if self._fm is not None:
            object.__setattr__(self._fm, "predict_velocity", self._orig_predict)
            object.__setattr__(self._expert, "handle_kv_cache", self._orig_handle)
            if getattr(self._fm, "_instinctflash_static_velocity", None) is self:
                object.__delattr__(self._fm, "_instinctflash_static_velocity")
        self.graph = None
        self._out = None
        self._fm = None


def install_static_capture(fm) -> StaticVelocity:
    """Install the executor on one model instance and return its counters/cleanup handle."""
    current = getattr(fm, "_instinctflash_static_velocity", None)
    if current is not None:
        return current
    driver = StaticVelocity(fm)
    object.__setattr__(fm, "_instinctflash_static_velocity", driver)
    return driver


__all__ = ["StaticVelocity", "WARMUP_STEPS", "install_static_capture"]
