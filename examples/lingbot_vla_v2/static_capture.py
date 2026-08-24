"""Replay-safe CUDA graph capture for LingBot-VLA-V2's denoise loop — the MoE generality test.

Third port of the static max-extent KV pattern (pi05 -> LingBot-VLA v1 -> here). Measured before
building (profile_infer.py, H100, in-process): one infer is 840.5 ms eager, of which the 10-step
Euler loop is 751.1 ms (89%) at 75.1 ms/step; upstream's own shipped default (torch.compile, on
by default in deploy) reaches 270.5 ms in-process. The loop is the target either way.

V2-specific facts this port rests on:

  * `handle_kv_cache` is byte-identical in shape to v1's (per-layer dict, torch.cat on dim=1) —
    the same replay-illegal address dependence, the same fixed-slot fix.
  * `predict_velocity` takes one extra chunk-constant tensor, `prefix_position_ids` (Qwen3-VL
    3D-RoPE positions). It lands in a static buffer refilled per chunk, like the pad mask.
  * The sparse-MoE action expert routes tokens with on-device topk/gather/one_hot into a fused
    group-GEMM (`qwen2_action_expert.py` — "all-to-all computation for torch.compile
    compatibility"; `ops/fused_moe.py` uses one_hot+cumsum, no .item()/.nonzero()/host sync).
    Routing is DATA-dependent but SHAPE-static and device-resident, so a captured graph replays
    the router on every new input — routing is recomputed correctly inside the graph, not baked.
    The robby_moe workspace is allocated on first use, which the warmup steps exercise before
    capture; its key is shape-only, so no allocation happens inside the graph.

Everything else is the v1 mechanism verbatim: prefix slots rewritten outside the graph at chunk
boundaries (detected by prefill-dict identity), suffix slots overwritten in place inside it,
warmup runs chunk 1 eagerly on the static path, capture on chunk 2.

Opt-in: `install_static_capture(server.vla.model)` (the FlowMatchingV2 module), or
IFL_VLA2_STATIC_CAPTURE=1 through the verify/measure harnesses.
"""

from __future__ import annotations

import torch

#: one full chunk (10 steps) plus a step of the next, so a prefix refill happens pre-capture
WARMUP_STEPS = 12


class _StaticKV:
    """Fixed-extent K/V per layer, sequence dim 1; suffix length read off the first write."""

    def __init__(self, prefill: dict):
        self._pending_prefill = prefill
        self.prefix_len = int(prefill[0]["key_states"].shape[1])
        self.suffix_len = None
        self.k, self.v = [], []

    def _allocate(self, suffix_len: int) -> None:
        self.suffix_len = int(suffix_len)
        total = self.prefix_len + self.suffix_len
        prefill = self._pending_prefill
        for idx in range(len(prefill)):
            ks, vs = prefill[idx]["key_states"], prefill[idx]["value_states"]
            K = torch.empty(ks.shape[0], total, ks.shape[2], ks.shape[3],
                            dtype=ks.dtype, device=ks.device)
            V = torch.empty_like(K)
            K[:, : self.prefix_len].copy_(ks)
            V[:, : self.prefix_len].copy_(vs)
            self.k.append(K)
            self.v.append(V)
        self._pending_prefill = None

    def refill(self, prefill: dict) -> None:
        for idx in range(len(self.k)):
            ks = prefill[idx]["key_states"]
            if int(ks.shape[1]) != self.prefix_len:
                raise RuntimeError(
                    f"prefix length changed ({ks.shape[1]} vs {self.prefix_len}); static capture "
                    f"assumes a fixed prompt budget")
            self.k[idx][:, : self.prefix_len].copy_(ks)
            self.v[idx][:, : self.prefix_len].copy_(prefill[idx]["value_states"])

    def write_suffix(self, key_states, value_states, layer_idx):
        if self.suffix_len is None:
            self._allocate(key_states.shape[1])
        K, V = self.k[layer_idx], self.v[layer_idx]
        K[:, self.prefix_len:].copy_(key_states)
        V[:, self.prefix_len:].copy_(value_states)
        return K, V


class StaticVelocity:
    """`FlowMatchingV2.predict_velocity` over static buffers: eager until warm, then replayed."""

    def __init__(self, fm):
        self._fm = fm
        self._kv: "_StaticKV | None" = None
        self._last_pkv_obj = None
        self._state_buf = None
        self._pad_buf = None
        self._pos_buf = None
        self._x_buf = None
        self._t_buf = None
        self._graph = None
        self._out = None
        self._steps = 0
        self.replays = 0
        self._orig_handle = type(fm.qwenvl_with_expert).handle_kv_cache
        self._install_handler()

    def _install_handler(self):
        outer = self

        def handle_kv_cache(m_self, key_states, value_states, layer_idx,
                            past_key_values=None, use_cache=None, fill_kv_cache=None):
            if fill_kv_cache or outer._kv is None:
                return outer._orig_handle(m_self, key_states, value_states, layer_idx,
                                          past_key_values=past_key_values, use_cache=use_cache,
                                          fill_kv_cache=fill_kv_cache)
            K, V = outer._kv.write_suffix(key_states, value_states, layer_idx)
            return K, V, past_key_values

        self._fm.qwenvl_with_expert.handle_kv_cache = handle_kv_cache.__get__(
            self._fm.qwenvl_with_expert)

    def _begin_chunk(self, state, prefix_pad_masks, past_key_values, prefix_position_ids) -> None:
        if self._kv is None:
            self._kv = _StaticKV(past_key_values)
            self._state_buf = state.clone()
            self._pad_buf = prefix_pad_masks.clone()
            self._pos_buf = prefix_position_ids.clone()
        else:
            self._kv.refill(past_key_values)
            self._state_buf.copy_(state)
            self._pad_buf.copy_(prefix_pad_masks)
            self._pos_buf.copy_(prefix_position_ids)

    def _forward_static(self):
        fm = self._fm
        return type(fm).predict_velocity.__wrapped__(
            fm, self._state_buf, self._pad_buf, None, self._x_buf, self._t_buf,
            prefix_position_ids=self._pos_buf)

    def __call__(self, state, prefix_pad_masks, past_key_values, x_t, timestep,
                 prefix_position_ids):
        if past_key_values is not self._last_pkv_obj:
            self._begin_chunk(state, prefix_pad_masks, past_key_values, prefix_position_ids)
            self._last_pkv_obj = past_key_values

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
        self._graph.replay()
        self.replays += 1
        return self._out.clone()


def install_static_capture(fm) -> StaticVelocity:
    """Route `FlowMatchingV2.predict_velocity` through a StaticVelocity. Returns it (counters)."""
    orig = type(fm).predict_velocity
    if not hasattr(orig, "__wrapped__"):
        d = StaticVelocity(fm)

        def predict_velocity(self_fm, state, prefix_pad_masks, past_key_values, x_t, timestep,
                             prefix_position_ids=None):
            return d(state, prefix_pad_masks, past_key_values, x_t, timestep,
                     prefix_position_ids)

        predict_velocity.__wrapped__ = orig
        type(fm).predict_velocity = predict_velocity
        return d
    raise RuntimeError("static capture already installed")
