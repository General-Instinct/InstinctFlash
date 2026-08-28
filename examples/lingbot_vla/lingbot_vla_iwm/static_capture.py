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

DEFAULT, GATED BY A SELF-CHECK. `LingBotVLA4BAdapter.install` routes every 4B-class checkpoint
here when the plan applies graph_capture on a CUDA build — fresh fine-tunes included
(IFL_VLA4B_NO_CAPTURE=1 is the kill-switch). Chunk 1 runs the static path eagerly (warmup +
proof), the graph is captured on the second chunk — and the capture is not trusted by
construction: immediately after it is taken, `_self_check` replays it against the UPSTREAM eager
`predict_velocity` (bound before install patched it, run through the stock DynamicCache-style
concat path) on staged inputs the capture never saw — fresh `x_t` draws from a dedicated
generator (the model's own RNG stream must not move), up to three distinct schedule timesteps,
and a synthetically REFILLED prefix so a graph that baked K/V values instead of reading the live
buffers cannot pass. Exact equality is required (this family's capture tier is BITEXACT: the
six-case gate in verify_static_capture.py measured 0.0 everywhere). PASS → replay serves; FAIL →
the graph is released, `predict_velocity` is rebound to upstream, and the fallback is announced
loudly — serving continues on eager arithmetic. The check costs a few seconds, once per process,
at first capture.

IFL_VLA4B_SELFCHECK_FAULT=1 is the drill switch: it rebinds the x buffer between capture and
check — exactly the stale-address bug class the check exists for — so the loud-fallback path
stays demonstrable on demand.
"""

from __future__ import annotations

import torch

from instinctflash.runtime.capture_self_check import run_capture_self_check

FAMILY = "LingBot-VLA-4B"
#: the drill switch — see the module docstring.
SELF_CHECK_FAULT_ENV = "IFL_VLA4B_SELFCHECK_FAULT"

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

    #: staged inputs the post-capture self-check compares on. Chosen to cover the protocol's
    #: axes compactly: fresh x_t on every input, up to three distinct schedule timesteps, and
    #: the second half runs against a synthetically refilled prefix.
    SELF_CHECK_INPUTS = 6

    def __init__(self, fm, orig_predict=None, on_self_check=None,
                 self_check_inputs: "int | None" = None):
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
        #: the UPSTREAM predict_velocity (the class attribute before install patched it). It is
        #: both the reference arm of the self-check and the landing spot of a rejected capture.
        #: None (a caller that installed by hand without one) disables the check — capture is
        #: then trusted the way it was before the check existed, which standalone measurement
        #: scripts rely on.
        self._orig_predict = orig_predict
        self._on_self_check = on_self_check
        self._self_check_n = (self.SELF_CHECK_INPUTS if self_check_inputs is None
                              else int(self_check_inputs))
        #: the verdict, once taken — see instinctflash.runtime.capture_self_check
        self.self_check: "dict | None" = None
        #: True once the self-check failed. Permanent for the process: a capture that replays
        #: wrong once will replay wrong again.
        self.rejected = False
        #: True while the self-check computes its eager reference: routes handle_kv_cache to
        #: the stock concat path so the reference is upstream's arithmetic, not the static KV.
        self._bypass = False
        #: distinct schedule timesteps seen during warmup, for the staged cases
        self._seen_ts: list[float] = []
        self._orig_handle = type(fm.qwenvl_with_expert).handle_kv_cache
        self._install_handler()

    def _install_handler(self):
        outer = self

        def handle_kv_cache(m_self, key_states, value_states, layer_idx,
                            past_key_values=None, use_cache=None, fill_kv_cache=None):
            if fill_kv_cache or outer._kv is None or outer._bypass:
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
        if self.rejected:
            # a caller still holding the pre-rejection binding: same eager answer, no capture
            return self._forward_upstream(state, prefix_pad_masks, past_key_values,
                                          x_t, timestep)
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
            key = round(float(timestep.reshape(-1)[0]), 9)
            if key not in self._seen_ts:
                self._seen_ts.append(key)
            return self._forward_static()

        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._out = self._forward_static()

        # FAULT DRILL, between capture and check on purpose: rebinding the x buffer makes every
        # later copy_ land at an address the graph does not read — the stale-address bug class
        # the self-check exists for. Consumed here so the loud-fallback path stays demonstrable.
        import os
        import sys
        if os.environ.get(SELF_CHECK_FAULT_ENV) == "1":
            print(f"[{FAMILY} static_capture] FAULT INJECTED ({SELF_CHECK_FAULT_ENV}=1): x "
                  f"buffer rebound between capture and self-check — the check must now fail.",
                  file=sys.stderr, flush=True)
            self._x_buf = self._x_buf.clone()

        # THE GATE. Capturing successfully proves nothing about replaying; the graph serves
        # only if replay equals upstream eager EXACTLY on staged inputs it was not captured
        # from (this family's capture tier is BITEXACT).
        if self._orig_predict is not None and self._self_check_n > 0:
            if not self._self_check(state, prefix_pad_masks, past_key_values, x_t, timestep):
                return self._forward_upstream(state, prefix_pad_masks, past_key_values,
                                              x_t, timestep)

        # replay once so _out carries THIS call's answer, not capture-pass artifacts
        self._graph.replay()
        self.replays += 1
        return self._out.clone()

    # -- the post-capture gate ---------------------------------------------------------------
    def _forward_upstream(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """Upstream's arithmetic, exactly: the original predict_velocity over the stock
        concat-per-step KV path (the bypass keeps the static buffers out of it)."""
        self._bypass = True
        try:
            return self._orig_predict(self._fm, state, prefix_pad_masks, past_key_values,
                                      x_t, timestep)
        finally:
            self._bypass = False

    def _perturbed_prefill(self, prefill, gen):
        """A prefill dict shaped exactly like the real one, with values the capture never saw."""
        staged = {}
        for idx in range(len(self._kv.k)):
            entry = dict(prefill[idx])
            for name in ("key_states", "value_states"):
                leaf = entry[name]
                noise = torch.empty(leaf.shape, device=leaf.device, dtype=torch.float32)
                noise.normal_(generator=gen)
                scale = leaf.detach().float().std().clamp_min(1e-3) * 0.02
                entry[name] = (leaf.detach().float() + noise * scale).to(leaf.dtype)
            staged[idx] = entry
        return staged

    def _self_check(self, state, prefix_pad_masks, past_key_values, x_t, timestep) -> bool:
        """Replay vs upstream eager on staged inputs the capture never saw. Exact equality.

        Startup-only: runs once, at capture time, and restores every buffer it touched — the
        model's own RNG stream never moves (staged draws come from a dedicated generator, the
        eager arm is deterministic, replay consumes no randomness), so a PASSing check leaves
        the served action stream bitwise identical to a build without the check.
        """
        n = self._self_check_n
        gen = torch.Generator(device=x_t.device)
        gen.manual_seed(0x51F)
        t_values = self._seen_ts[:3] or [round(float(timestep.reshape(-1)[0]), 9)]

        staged_prefill = None

        def one_case(i):
            nonlocal staged_prefill
            if i == (n + 1) // 2 and staged_prefill is None:
                # second half: a synthetically REFILLED prefix. A graph that baked the K/V
                # values it was captured with — instead of reading the live buffers the
                # per-chunk refill writes — is exactly wrong here and nowhere milder.
                staged_prefill = self._perturbed_prefill(past_key_values, gen)
                self._kv.refill(staged_prefill)
            x = torch.empty_like(x_t)
            x.normal_(generator=gen)
            t = torch.full_like(timestep, t_values[i % len(t_values)])
            cache = staged_prefill if staged_prefill is not None else past_key_values
            label = "refilled" if staged_prefill is not None else "captured-chunk"

            def run_eager():
                return self._forward_upstream(state, prefix_pad_masks, cache, x, t)

            def run_replay():
                self._x_buf.copy_(x)
                self._t_buf.copy_(t)
                self._graph.replay()
                return self._out
            return label, run_eager, run_replay

        try:
            # a GENERATOR, deliberately: the refill side effect in one_case must land between
            # case i-1's runs and case i's, not while the case list is being built
            verdict = run_capture_self_check(
                family=FAMILY, cases=(one_case(i) for i in range(n)), tolerance=0.0)
        finally:
            # restore the real chunk state whatever the verdict: the caller's step must see
            # the answer to ITS inputs, and a later refill must start from the real prefix
            if staged_prefill is not None:
                self._kv.refill(past_key_values)
            self._x_buf.copy_(x_t)
            self._t_buf.copy_(timestep)
            torch.cuda.synchronize()

        self.self_check = verdict
        if not verdict["passed"]:
            self._release_and_fall_back()
        if self._on_self_check is not None:
            self._on_self_check(dict(verdict))
        return verdict["passed"]

    def _release_and_fall_back(self) -> None:
        """The FAIL arm: graph released, upstream rebound, said out loud. Serving continues."""
        import sys
        self.rejected = True
        self._graph = None
        self._out = None
        try:
            type(self._fm).predict_velocity = self._orig_predict
        except Exception:                                          # noqa: BLE001
            pass                       # the __call__ guard still routes every call to eager
        try:
            del self._fm.qwenvl_with_expert.handle_kv_cache        # uncovers the class method
        except AttributeError:
            pass
        print(f"[{FAMILY} static_capture] Graph released; predict_velocity rebound to "
              f"upstream — serving continues on eager arithmetic (upstream's, exactly).",
              file=sys.stderr, flush=True)

    def close(self) -> None:
        """Restore the instance-level KV handler and drop the graph. Compute is untouched:
        close() exists so a Runtime that unloads the model releases the captured pool."""
        try:
            del self._fm.qwenvl_with_expert.handle_kv_cache      # uncovers the class method
        except AttributeError:
            pass
        self._graph = None
        self._out = None
        self._kv = None


def install_static_capture(fm, on_self_check=None, self_check: bool = True) -> StaticVelocity:
    """Route `FlowMatching.predict_velocity` through a StaticVelocity. Returns it (counters).

    Idempotent: a second install on the same class returns the first driver rather than
    stacking wrappers (two drivers would each own static buffers and disagree on addresses).

    `self_check` (default on) gates the first capture on the bit-exact replay-vs-eager check —
    see the module docstring. `on_self_check` receives the verdict dict so an installer can put
    it on the plan.
    """
    orig = type(fm).predict_velocity
    if hasattr(orig, "__wrapped__"):
        return orig.__driver__
    d = StaticVelocity(fm, orig_predict=(orig if self_check else None),
                       on_self_check=on_self_check)

    def predict_velocity(self_fm, state, prefix_pad_masks, past_key_values, x_t, timestep):
        return d(state, prefix_pad_masks, past_key_values, x_t, timestep)

    predict_velocity.__wrapped__ = orig
    predict_velocity.__driver__ = d
    type(fm).predict_velocity = predict_velocity
    return d
