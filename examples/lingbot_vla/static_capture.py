"""Replay-safe CUDA graph capture for LingBot-VLA's denoise loop, on static max-extent KV buffers.

The pi05 pattern (examples/pi05_vla/pi05_iwm/static_capture.py), ported to LingbotVlaPolicy.
Measured before building (profile_infer.py, H100, in-process): one infer is 659.8 ms of which the
10-step denoise loop is 547.1 ms (83%) at 54.7 ms/step, prefill 44.7 ms. The loop is the target.

Why the stock loop cannot be replayed: `QwenvlWithExpertModel.handle_kv_cache` concatenates the
chunk's prefill K/V with the step's suffix K/V per layer per step. A captured graph bakes the
prefill tensors' ADDRESSES; the next chunk prefill allocates new tensors, so a replay would read
the old chunk's K/V. The fix is the serving-engine move: one K/V buffer per layer of extent
prefix+suffix, allocated once —

    slots [0, prefix)               rewritten OUTSIDE the graph, once per chunk (copy_ from prefill)
    slots [prefix, prefix+suffix)   overwritten INSIDE the graph, every step, same addresses

`handle_kv_cache` is patched (on the instance) so the non-fill path writes suffix K/V into the
fixed slots and returns the full-extent buffers — the same math the cat produced, over constant
addresses. Chunk-varying tensors the region reads (state, prefix pad mask) land in static buffers
by copy_ at chunk boundaries; step-varying inputs (x_t, timestep) by copy_ every call. Chunk
boundaries are detected by the prefill dict's object identity, exactly like pi05.

Opt-in: `install_static_capture(server.vla.model)` on the FlowMatching module, or
IFL_VLA_STATIC_CAPTURE=1 through serve_static.py. Chunk 1 runs the static path eagerly (warmup +
proof), the graph is captured on the second chunk.
"""

from __future__ import annotations

import torch

#: eager steps on the static path before capture — at least one full chunk plus one step, so the
#: capture happens only after a prefix refill has been exercised.
WARMUP_STEPS = 12


class _StaticKV:
    """Fixed-extent K/V per layer; sequence dim is 1 (B, L, H, D) in this model.

    Buffers are allocated on the FIRST suffix write: the suffix is not `n_action_steps` — the
    model prepends a state token (measured: 51 for chunk 50) — so the true length is read off the
    first suffix tensor rather than assumed from config.
    """

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
                    f"assumes a fixed prompt budget — falling back is the caller's job")
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
    """`FlowMatching.predict_velocity` over static buffers: eager until warm, then replayed."""

    def __init__(self, fm):
        self._fm = fm
        self._kv: "_StaticKV | None" = None
        self._last_pkv_obj = None
        self._state_buf = None
        self._pad_buf = None
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

        # instance-level: other models in the process keep the stock behaviour
        self._fm.qwenvl_with_expert.handle_kv_cache = handle_kv_cache.__get__(
            self._fm.qwenvl_with_expert)

    # -- per-chunk, outside the graph --------------------------------------------------------------
    def _begin_chunk(self, state, prefix_pad_masks, past_key_values) -> None:
        if self._kv is None:
            self._kv = _StaticKV(past_key_values)
            self._state_buf = state.clone()
            self._pad_buf = prefix_pad_masks.clone()
        else:
            self._kv.refill(past_key_values)
            self._state_buf.copy_(state)
            self._pad_buf.copy_(prefix_pad_masks)

    def _forward_static(self):
        fm = self._fm
        # the ORIGINAL predict_velocity body, on static tensors: every op inside is capture-legal
        # because the patched handle_kv_cache replaces the only cross-chunk address dependence
        return type(fm).predict_velocity.__wrapped__(
            fm, self._state_buf, self._pad_buf, None, self._x_buf, self._t_buf)

    def __call__(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        if past_key_values is not self._last_pkv_obj:
            self._begin_chunk(state, prefix_pad_masks, past_key_values)
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
        # replay once so _out carries THIS call's answer, not capture-pass artifacts
        self._graph.replay()
        self.replays += 1
        return self._out.clone()


def install_static_capture(fm) -> StaticVelocity:
    """Route `FlowMatching.predict_velocity` through a StaticVelocity. Returns it (counters)."""
    orig = type(fm).predict_velocity
    if not hasattr(orig, "__wrapped__"):
        d = StaticVelocity(fm)

        def predict_velocity(self_fm, state, prefix_pad_masks, past_key_values, x_t, timestep):
            return d(state, prefix_pad_masks, past_key_values, x_t, timestep)

        predict_velocity.__wrapped__ = orig
        type(fm).predict_velocity = predict_velocity
        return d
    raise RuntimeError("static capture already installed")
