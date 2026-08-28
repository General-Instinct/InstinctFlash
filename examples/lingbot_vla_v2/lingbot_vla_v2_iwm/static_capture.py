"""Replay-safe static-KV CUDA Graph executor for LingBot-VLA-V2.

The upstream denoise step concatenates a new suffix K/V tensor onto a Python dictionary on every
call.  CUDA Graph replay requires stable addresses, so this module gives every layer one fixed
``[prefix | suffix]`` allocation. Prefix slots are refilled once per observation; suffix slots and
all step inputs are overwritten in place before replay. Qwen3-VL position ids are chunk constants
and therefore receive their own static input buffer.

DEFAULT, GATED BY A SELF-CHECK. ``LingBotVLAV2Adapter.install`` routes every V2-class checkpoint
here when the plan applies graph_capture on a CUDA build — fresh fine-tunes included
(IFL_VLA2_NO_CAPTURE=1 is the kill-switch) — and the capture is not trusted by construction:
immediately after the first capture, ``_self_check`` replays it against the UPSTREAM eager
``predict_velocity`` (bound before install patched it, run through the stock concat-per-step KV
path) on staged inputs the capture never saw — fresh ``x_t`` draws from a dedicated generator,
up to three distinct schedule timesteps, and a synthetically REFILLED prefix so a graph that
baked K/V values instead of reading the live buffers cannot pass.

THE GATE IS NOT atol=0 FOR THIS FAMILY, and that is a statement about upstream, not about the
graph: V2's fused-MoE kernel is nondeterministic — two STOCK runs on identical seeds disagree by
up to 5.08e-02 (``moe_kernel_results.json`` null_control_deltas, H100 6-case protocol) — so exact
equality is unattainable for ANY serving of this model, including upstream's own. The self-check
therefore gates on that recorded stock-vs-stock envelope (``NULL_ENVELOPE``), the same standard
the family's published row was verified under. PASS → replay serves; FAIL → the graph is
released (the vision/prefill graphs with it, via the adapter's recorder), ``predict_velocity`` is
rebound to upstream, and the fallback is announced loudly — serving continues on eager
arithmetic.

IFL_VLA2_SELFCHECK_FAULT=1 is the drill switch: it rebinds the x buffer between capture and
check — the stale-address bug class the check exists for — so the loud-fallback path stays
demonstrable on demand.
"""

from __future__ import annotations

import torch

from instinctflash.runtime.capture_self_check import run_capture_self_check

WARMUP_STEPS = 12

FAMILY = "LingBot-VLA-V2"
#: the drill switch — see the module docstring.
SELF_CHECK_FAULT_ENV = "IFL_VLA2_SELFCHECK_FAULT"
#: The family's recorded nondeterminism envelope: the largest stock-vs-stock action delta on
#: identical seeds (3 paired stock runs, H100 6-case protocol —
#: examples/lingbot_vla_v2/moe_kernel_results.json null_control_deltas, max 5.08e-02). The
#: self-check gates on it because upstream's fused-MoE kernel makes atol=0 unattainable even
#: for upstream against itself.
NULL_ENVELOPE = 5.083918571472168e-02
NULL_ENVELOPE_PROVENANCE = ("the stock-vs-stock null control on identical seeds, "
                            "moe_kernel_results.json (H100, 6-case protocol)")


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

    #: staged inputs the post-capture self-check compares on: fresh x_t on every input, up to
    #: three distinct schedule timesteps, second half against a synthetically refilled prefix.
    SELF_CHECK_INPUTS = 6

    def __init__(self, fm, self_check: bool = True, on_self_check=None,
                 self_check_inputs: "int | None" = None):
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
        #: the UPSTREAM predict_velocity, bound before _install patches it: the reference arm
        #: of the self-check and the landing spot of a rejected capture.
        self._orig_predict = fm.predict_velocity
        self._expert = fm.qwenvl_with_expert
        self._orig_handle = self._expert.handle_kv_cache
        self._self_check_enabled = bool(self_check)
        self._on_self_check = on_self_check
        self._self_check_n = (self.SELF_CHECK_INPUTS if self_check_inputs is None
                              else int(self_check_inputs))
        #: the verdict, once taken — see instinctflash.runtime.capture_self_check
        self.self_check: "dict | None" = None
        #: True once the self-check failed. Permanent for the process.
        self.rejected = False
        #: True while the self-check computes its eager reference: routes handle_kv_cache to
        #: the stock concat path so the reference is upstream's arithmetic, not the static KV.
        self._bypass = False
        #: distinct schedule timesteps seen during warmup, for the staged cases
        self._seen_ts: list[float] = []
        self._install()

    def _install(self) -> None:
        outer = self

        def handle_kv_cache(key_states, value_states, layer_idx, past_key_values=None,
                            use_cache=None, fill_kv_cache=None):
            if fill_kv_cache or outer._kv is None or outer._bypass:
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
        if self.rejected:
            # a caller still holding the pre-rejection binding: same eager answer, no capture
            return self._forward_upstream(state, prefix_pad_masks, past_key_values,
                                          x_t, timestep, prefix_position_ids)
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
            key = round(float(timestep.reshape(-1)[0]), 9)
            if key not in self._seen_ts:
                self._seen_ts.append(key)
            return self._forward_static()

        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
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

        # THE GATE. The graph serves only if replay agrees with upstream eager on staged
        # inputs it was not captured from, within the family's recorded stock-vs-stock
        # envelope (this family's capture tier is NUMERIC — see the module docstring).
        if self._self_check_enabled and self._self_check_n > 0:
            if not self._self_check(state, prefix_pad_masks, past_key_values,
                                    x_t, timestep, prefix_position_ids):
                return self._forward_upstream(state, prefix_pad_masks, past_key_values,
                                              x_t, timestep, prefix_position_ids)

        self.graph.replay()
        self.replays += 1
        return self._out.clone()

    # -- the post-capture gate ---------------------------------------------------------------
    def _forward_upstream(self, state, prefix_pad_masks, past_key_values, x_t, timestep,
                          prefix_position_ids):
        """Upstream's arithmetic, exactly: the original predict_velocity over the stock
        concat-per-step KV path (the bypass keeps the static buffers out of it)."""
        self._bypass = True
        try:
            return self._orig_predict(state, prefix_pad_masks, past_key_values, x_t, timestep,
                                      prefix_position_ids=prefix_position_ids)
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

    def _self_check(self, state, prefix_pad_masks, past_key_values, x_t, timestep,
                    prefix_position_ids) -> bool:
        """Replay vs upstream eager on staged inputs, gated by the recorded null envelope.

        Startup-only: runs once, at capture time, and restores every buffer it touched — the
        model's own RNG stream never moves (staged draws come from a dedicated generator,
        replay consumes no randomness), so the served action stream after a PASS differs from
        a build without the check only by upstream's own fused-MoE nondeterminism.
        """
        n = self._self_check_n
        gen = torch.Generator(device=x_t.device)
        gen.manual_seed(0x51F)
        t_values = self._seen_ts[:3] or [round(float(timestep.reshape(-1)[0]), 9)]

        staged_prefill = None

        def one_case(i):
            nonlocal staged_prefill
            if i == (n + 1) // 2 and staged_prefill is None:
                # second half: a synthetically REFILLED prefix — a graph that baked K/V values
                # instead of reading the live buffers is exactly wrong here and nowhere milder
                staged_prefill = self._perturbed_prefill(past_key_values, gen)
                self._kv.refill(staged_prefill)
            x = torch.empty_like(x_t)
            x.normal_(generator=gen)
            t = torch.full_like(timestep, t_values[i % len(t_values)])
            cache = staged_prefill if staged_prefill is not None else past_key_values
            label = "refilled" if staged_prefill is not None else "captured-chunk"

            def run_eager():
                return self._forward_upstream(state, prefix_pad_masks, cache, x, t,
                                              prefix_position_ids)

            def run_replay():
                self._x_buf.copy_(x)
                self._t_buf.copy_(t)
                self.graph.replay()
                return self._out
            return label, run_eager, run_replay

        try:
            # a GENERATOR, deliberately: the refill side effect in one_case must land between
            # case i-1's runs and case i's, not while a case list is being built
            verdict = run_capture_self_check(
                family=FAMILY, cases=(one_case(i) for i in range(n)),
                tolerance=NULL_ENVELOPE, tolerance_provenance=NULL_ENVELOPE_PROVENANCE)
        finally:
            # restore the real chunk state whatever the verdict
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
        self.graph = None
        self._out = None
        try:
            object.__setattr__(self._fm, "predict_velocity", self._orig_predict)
            object.__setattr__(self._expert, "handle_kv_cache", self._orig_handle)
        except Exception:                                          # noqa: BLE001
            pass                       # the __call__ guard still routes every call to eager
        print(f"[{FAMILY} static_capture] Graph released; predict_velocity rebound to "
              f"upstream — serving continues on eager arithmetic (upstream's, exactly).",
              file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._fm is not None:
            object.__setattr__(self._fm, "predict_velocity", self._orig_predict)
            object.__setattr__(self._expert, "handle_kv_cache", self._orig_handle)
            if getattr(self._fm, "_instinctflash_static_velocity", None) is self:
                object.__delattr__(self._fm, "_instinctflash_static_velocity")
        self.graph = None
        self._out = None
        self._fm = None


def install_static_capture(fm, on_self_check=None, self_check: bool = True) -> StaticVelocity:
    """Install the executor on one model instance and return its counters/cleanup handle.

    ``self_check`` (default on) gates the first capture on the envelope-gated replay-vs-eager
    check — see the module docstring. ``on_self_check`` receives the verdict dict so an
    installer can put it on the plan.
    """
    current = getattr(fm, "_instinctflash_static_velocity", None)
    if current is not None:
        return current
    driver = StaticVelocity(fm, self_check=self_check, on_self_check=on_self_check)
    object.__setattr__(fm, "_instinctflash_static_velocity", driver)
    return driver


__all__ = ["StaticVelocity", "WARMUP_STEPS", "NULL_ENVELOPE", "install_static_capture"]
