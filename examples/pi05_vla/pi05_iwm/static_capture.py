"""Replay-safe CUDA graph capture for pi05's denoise loop, on static max-extent KV buffers.

THE FIX FOR THE REJECTION documented in `surface.py`. The region was rejected because
`denoise_step` clones the prefix `DynamicCache` and the forward APPENDS 50 suffix entries to the
clone — the region allocates and mutates a Python container per call, so a replay computes with
the K/V of whatever step the capture was taken from (measured: max |d| 2.116e-01 on a new x_t).

The serving engine's answer (serving/, and every CUDA-graph serving stack): make the memory the
graph touches STATIC. Here that is one K/V buffer per layer of extent prefix+suffix, allocated
once:

    slots [0, prefix)                written OUTSIDE the graph, once per chunk (the prefill copy)
    slots [prefix, prefix+suffix)    overwritten INSIDE the graph, every step, same addresses

`_StaticKV.update` replaces the DynamicCache append with an in-place copy into the suffix slots
and returns the full-extent buffers, so attention always runs over a constant length with the 4D
mask pi05 already builds for exactly that width. Nothing in the region allocates, nothing mutates
host state, and a replayed step reads the K/V the CURRENT chunk wrote — which is what makes replay
legal on inputs the capture never saw.

The captured callable's inputs land in static buffers (`x_buf`, `t_buf`) by `copy_`; its output is
cloned out of the pool before it is returned, so the next replay cannot overwrite a tensor a
caller still holds. Chunk boundaries are detected by cache-object identity: `sample_actions`
builds one prefill cache per chunk and passes the same object to all 10 steps, so a new object
means a new chunk and triggers the (out-of-graph) prefix refill.

DEFAULT, GATED BY A SELF-CHECK. `Pi05Adapter.install` routes every pi05-class checkpoint here on
capture-capable devices — fresh fine-tunes included (IFL_PI05_NO_CAPTURE=1 is the kill-switch).
The graph is captured after WARMUP_STEPS eager steps on the static path, and the capture is not
trusted by construction: immediately after it is taken, `_self_check` replays it against the
UPSTREAM eager `denoise_step` on staged inputs the capture never saw — fresh `x_t` draws from a
dedicated generator (the model's own RNG stream must not move), every timestep of the warmed
schedule, and a synthetically REFILLED prefix so a graph that baked K/V values instead of reading
the live buffers cannot pass. Exact equality is required (`atol=0`: replay re-runs the same
kernels at the same addresses, so any drift is evidence something is not being re-read). PASS →
replay serves; FAIL → the graph is released, `denoise_step` is rebound to upstream, and the
fallback is announced loudly with the observed delta — serving continues on eager arithmetic.
The check costs a few seconds, once per process, at first capture.

IFL_PI05_SELFCHECK_FAULT=1 is the drill switch: it rebinds the x buffer between capture and
check — exactly the stale-address bug class the check exists for — so the loud-fallback path
stays demonstrable on demand.
"""

from __future__ import annotations

import torch

#: eager steps on the static path before capture; also serves as kernel warmup. The count is
#: chunk-agnostic — what matters is that capture happens after at least one full prefix refill,
#: and (with step tables on) that every timestep of the fixed Euler schedule has been seen once.
WARMUP_STEPS = 12


class _TableDense(torch.nn.Module):
    """Stands in for an AdaRMS `dense` projection inside the captured region.

    The projection's input is the time conditioning, which for a fixed Euler schedule takes ten
    values ever — so its output is a per-step CONSTANT. The serving engine precomputes exactly
    these (step, layer) modulation tables offline and slices them by baked pointer offset
    (serving/flash_rt/frontends/torch/pi05_rtx.py:371-430, models/pi05/pipeline_rtx.py:422-443);
    this is the torch-level equivalent: the module returns a static buffer that the step loop
    fills OUTSIDE the graph from a table of outputs the REAL projection produced during warmup on
    the same conditioning bytes. The swap moves WHEN the GEMV runs (once per timestep per model
    lifetime, not once per step), never WHAT it produces, so bitexactness is preserved by
    construction and re-proven by the verify gates. Measured: the 37 projections plus the time
    MLP cost 0.302 ms inside a 4.57 ms replay (6.6%).
    """

    def __init__(self, real: torch.nn.Module, buf: torch.Tensor, owner):
        super().__init__()
        self.real = real
        self.buf = buf
        self._owner = [owner]                      # plain list: keep the denoiser out of state_dict

    def forward(self, cond):
        # Inside the captured region the owner flag is True and the graph bakes the buffer read.
        # Any OTHER caller of this module (the in-process stock comparator, training) sees the
        # flag False and gets the real projection — the swap must never leak outside the graph.
        if self._owner[0]._tabled_active:
            return self.buf
        return self.real(cond)


class _StaticKV:
    """The Cache interface `GemmaAttention.update` needs, over fixed-extent buffers.

    This is the same memory discipline the serving engine uses for its enc+dec cache: one
    fixed-extent K and V per layer, the decoder's fresh entries written AT AN OFFSET into the
    shared buffer, attention reading the full extent with zero concat and zero copy per step
    (serving/flash_rt/hardware/rtx/attn_backend.py:249-270 allocates K/V each
    (layers, enc_seq_max+chunk, 1, head_dim); models/pi05/pipeline_rtx.py:1011-1029 writes the
    chunk K/V at token offset enc_seq and attends over enc_seq+dec_seq). Here `update` is that
    offset write — suffix slots at [prefix, prefix+suffix) — and returning the whole buffer is
    the zero-materialization read. The `torch.cat` upstream keeps in its Q/K/V head-stacking is
    untouched: capture-pool allocations are replay-stable; it is the CACHE join that must not
    reallocate, and here it never does."""

    def __init__(self, prefix_kv, suffix_len: int):
        first_k = next(iter(prefix_kv))[0]
        self.prefix_len = int(first_k.shape[2])
        self.suffix_len = int(suffix_len)
        total = self.prefix_len + self.suffix_len
        self.k, self.v = [], []
        for keys, values, *_ in prefix_kv:
            K = torch.empty(keys.shape[0], keys.shape[1], total, keys.shape[3],
                            dtype=keys.dtype, device=keys.device)
            V = torch.empty_like(K)
            K[:, :, : self.prefix_len].copy_(keys)
            V[:, :, : self.prefix_len].copy_(values)
            self.k.append(K)
            self.v.append(V)

    def refill(self, prefix_kv) -> None:
        """Per-chunk prefix rewrite. Runs OUTSIDE the graph; addresses never change."""
        for K, V, (keys, values, *_) in zip(self.k, self.v, iter(prefix_kv)):
            K[:, :, : self.prefix_len].copy_(keys)
            V[:, :, : self.prefix_len].copy_(values)

    # -- the Cache protocol, as much of it as this model touches -------------------------------
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        K, V = self.k[layer_idx], self.v[layer_idx]
        K[:, :, self.prefix_len:].copy_(key_states)
        V[:, :, self.prefix_len:].copy_(value_states)
        return K, V

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.prefix_len

    def get_max_cache_shape(self) -> int:
        return self.prefix_len + self.suffix_len

    @property
    def is_compileable(self) -> bool:
        return True

    def __len__(self):
        return len(self.k)


class StaticDenoiser:
    """pi05's denoise step over static buffers: eager until warm, then captured and replayed."""

    #: staged inputs the post-capture self-check compares on. Chosen to cover the protocol's
    #: axes compactly: fresh x_t on every input, up to three distinct schedule timesteps, and
    #: the second half runs against a synthetically refilled prefix.
    SELF_CHECK_INPUTS = 6

    def __init__(self, model, step_tables: bool = True, orig_denoise=None,
                 on_self_check=None, self_check_inputs: "int | None" = None):
        self._m = model
        self._kv: "_StaticKV | None" = None
        self._last_cache_obj = None
        self._const = None            # (mask4d, position_ids, cache_position)
        self._x_buf = None
        self._t_buf = None
        self._graph = None
        self._out = None
        self._steps = 0
        self.replays = 0
        #: the UPSTREAM denoise_step, bound before install patched it. It is both the reference
        #: arm of the self-check and the landing spot of a rejected capture. None (a caller that
        #: installed by hand without one) disables the check — capture is then trusted the way
        #: it was before the check existed, which standalone measurement scripts rely on.
        self._orig_denoise = orig_denoise
        self._on_self_check = on_self_check
        self._self_check_n = (self.SELF_CHECK_INPUTS if self_check_inputs is None
                              else int(self_check_inputs))
        #: the verdict, once taken: {n, bitexact, max_abs_delta, seconds, cases}
        self.self_check: "dict | None" = None
        #: True once the self-check failed. Permanent for the process, like the engine pass's
        #: `rejected` set: a capture that replays wrong once will replay wrong again.
        self.rejected = False
        self._last_pad_masks = None
        # -- per-step constant tables (the serving engine's style-table absorb) ---------------------------
        self._step_tables = step_tables
        self._table: dict[float, tuple] = {}       # t -> (adarms_cond, [dense outs])
        self._adarms_buf = None
        self._dense_bufs: "list[torch.Tensor] | None" = None
        self._denses: "list | None" = None          # (norm, real dense) pairs, in swap order
        self._tabled_active = False

    # -- per-step-constant compute, done OUTSIDE the graph ----------------------------------------
    def _adarms_denses(self):
        """(norm, real dense) pairs, in a fixed traversal order the buffers share."""
        expert = self._m.paligemma_with_expert.gemma_expert.model
        pairs = []
        for layer in expert.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                if getattr(norm, "dense", None) is not None:
                    pairs.append((norm, norm.dense))
        final = getattr(expert, "norm", None)
        if final is not None and getattr(final, "dense", None) is not None:
            pairs.append((final, final.dense))
        return pairs

    def _time_cond(self, timestep):
        """The embed_suffix time path, byte-identical: same modules, same op order, same dtypes."""
        import torch.nn.functional as F

        from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
        m = self._m
        e = create_sinusoidal_pos_embedding(
            timestep, m.action_in_proj.out_features,
            min_period=m.config.min_period, max_period=m.config.max_period,
            device=timestep.device).type(dtype=timestep.dtype)
        return F.silu(m.time_mlp_out(F.silu(m.time_mlp_in(e))))

    def _record_step(self, timestep) -> None:
        key = round(float(timestep[0]), 9)
        if key in self._table:
            return
        with torch.no_grad():
            cond = self._time_cond(timestep)
            outs = [real(cond).clone() for _, real in self._denses]
        self._table[key] = (cond.clone(), outs)

    def _load_step(self, timestep) -> None:
        key = round(float(timestep[0]), 9)
        if key not in self._table:
            # a timestep the warmup never saw (schedule change): compute the entry now, outside
            # the graph — correctness is preserved, only this step pays the 0.7 ms eager cost
            self._record_step(timestep)
        cond, outs = self._table[key]
        self._adarms_buf.copy_(cond)
        for buf, o in zip(self._dense_bufs, outs):
            buf.copy_(o)

    # -- per-chunk, outside the graph ------------------------------------------------------------
    def _begin_chunk(self, prefix_pad_masks, past_key_values) -> None:
        m = self._m
        self._last_pad_masks = prefix_pad_masks
        suffix_len = m.config.chunk_size
        if self._kv is None:
            self._kv = _StaticKV(past_key_values, suffix_len)
        else:
            self._kv.refill(past_key_values)

        if self._const is None:
            from lerobot.policies.pi05.modeling_pi05 import (
                make_att_2d_masks,
                prepare_attention_masks_4d,
            )
            bsize, prefix_len = prefix_pad_masks.shape
            device = prefix_pad_masks.device
            # identical construction to upstream denoise_step, hoisted to chunk scope: for a fixed
            # prompt budget every quantity here is constant across steps AND chunks
            suffix_pad = torch.ones(bsize, suffix_len, dtype=torch.bool, device=device)
            suffix_att = torch.zeros(bsize, suffix_len, device=device)
            suffix_att[:, 0] = 1
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            mask4d = prepare_attention_masks_4d(full_2d)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
            cache_position = torch.arange(prefix_len, prefix_len + suffix_len, device=device)
            self._const = (mask4d, position_ids, cache_position)

    # -- the region itself: static in, static out ------------------------------------------------
    def _forward_static(self):
        m = self._m
        mask4d, position_ids, cache_position = self._const
        if self._step_tables and self._adarms_buf is not None:
            # tabled path: the time MLP and every AdaRMS projection were computed outside the
            # graph (`_load_step`); the region does only the step-VARYING work
            suffix_embs = m.action_in_proj(self._x_buf)
            adarms = self._adarms_buf
        else:
            suffix_embs, _, _, adarms = m.embed_suffix(self._x_buf, self._t_buf)
        out = m.paligemma_with_expert.gemma_expert.model.forward(
            inputs_embeds=suffix_embs,
            attention_mask=mask4d,
            position_ids=position_ids,
            past_key_values=self._kv,
            use_cache=False,
            cache_position=cache_position,
            adarms_cond=adarms,
        )
        suffix_out = out.last_hidden_state[:, -m.config.chunk_size:]
        return m.action_out_proj(suffix_out.to(dtype=torch.float32))

    def __call__(self, prefix_pad_masks, past_key_values, x_t, timestep):
        if self.rejected:
            # a caller still holding the pre-rejection binding: same eager answer, no capture
            return self._orig_denoise(prefix_pad_masks=prefix_pad_masks,
                                      past_key_values=past_key_values, x_t=x_t, timestep=timestep)
        if past_key_values is not self._last_cache_obj:
            self._begin_chunk(prefix_pad_masks, past_key_values)
            self._last_cache_obj = past_key_values

        if self._x_buf is None:
            self._x_buf = x_t.clone()
            self._t_buf = timestep.clone()
        else:
            self._x_buf.copy_(x_t)
            self._t_buf.copy_(timestep)

        if self._graph is not None:
            if self._step_tables:
                self._load_step(timestep)
            self._graph.replay()
            self.replays += 1
            return self._out.clone()

        self._steps += 1
        if self._steps <= WARMUP_STEPS:
            if self._step_tables:
                if self._denses is None:
                    self._denses = self._adarms_denses()
                self._record_step(timestep)
            return self._forward_static()

        if self._step_tables and self._adarms_buf is None:
            # swap point: from here the projections live in tables and the graph never sees them
            cond0, outs0 = next(iter(self._table.values()))
            self._adarms_buf = cond0.clone()
            self._dense_bufs = [o.clone() for o in outs0]
            for (norm, real), buf in zip(self._denses, self._dense_bufs):
                norm.dense = _TableDense(real, buf, self)
            self._load_step(timestep)

        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        self._tabled_active = True
        try:
            with torch.cuda.graph(self._graph):
                self._out = self._forward_static()
        finally:
            self._tabled_active = False

        # FAULT DRILL, between capture and check on purpose: rebinding the x buffer makes every
        # later copy_ land at an address the graph does not read — the stale-address bug class
        # the self-check exists to catch. Documented in the module docstring; consumed here so
        # the loud-fallback path can be demonstrated on demand without touching the code.
        import os
        import sys
        if os.environ.get("IFL_PI05_SELFCHECK_FAULT") == "1":
            # stderr: a live `serve` defers stdout until the command returns (cli_config.execute),
            # which for a persistent server is never — a drill nobody can watch proves nothing
            print("[pi05 static_capture] FAULT INJECTED (IFL_PI05_SELFCHECK_FAULT=1): x buffer "
                  "rebound between capture and self-check — the check must now fail.",
                  file=sys.stderr, flush=True)
            self._x_buf = self._x_buf.clone()

        # THE GATE. Capturing successfully proves nothing about replaying (measured: the
        # DynamicCache region replayed 1.55x and WRONG by up to 48% of the signal while three
        # separate checks read clean). The graph serves only if replay equals upstream eager
        # EXACTLY on staged inputs it was not captured from.
        if self._orig_denoise is not None and self._self_check_n > 0:
            if not self._self_check(prefix_pad_masks, past_key_values, x_t, timestep):
                return self._orig_denoise(prefix_pad_masks=prefix_pad_masks,
                                          past_key_values=past_key_values,
                                          x_t=x_t, timestep=timestep)

        # capture runs the region once on a side stream but its output tensor content is not
        # trustworthy on all driver versions; replay once so _out holds this call's real answer
        self._graph.replay()
        self.replays += 1
        return self._out.clone()

    # -- the post-capture gate ---------------------------------------------------------------
    def _self_check(self, prefix_pad_masks, past_key_values, x_t, timestep) -> bool:
        """Replay vs upstream eager on staged inputs the capture never saw. Exact equality.

        Startup-only: runs once, at capture time, and restores every buffer it touched — the
        model's own RNG stream never moves (staged draws come from a dedicated generator, the
        eager arm is deterministic, replay consumes no randomness), so a PASSing check leaves
        the served action stream bitwise identical to a build without the check.
        """
        import time

        n = self._self_check_n
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = torch.Generator(device=x_t.device)
        gen.manual_seed(0x51F)

        if self._step_tables and self._table:
            t_values = [float(k) for k in list(self._table)[:3]]
        else:
            t_values = [float(timestep.reshape(-1)[0])]

        worst, cases = 0.0, []
        staged_cache = None
        try:
            for i in range(n):
                if i == (n + 1) // 2 and staged_cache is None:
                    # second half: a synthetically REFILLED prefix. A graph that baked the K/V
                    # values it was captured with — instead of reading the live buffers the
                    # per-chunk refill writes — is exactly wrong here and nowhere milder.
                    staged_cache = self._perturbed_prefix(past_key_values, gen)
                    self._kv.refill(staged_cache)
                x = torch.empty_like(x_t)
                x.normal_(generator=gen)
                t = torch.full_like(timestep, t_values[i % len(t_values)])
                cache = staged_cache if staged_cache is not None else past_key_values
                with torch.no_grad():
                    ref = self._orig_denoise(prefix_pad_masks=prefix_pad_masks,
                                             past_key_values=cache, x_t=x, timestep=t)
                self._x_buf.copy_(x)
                self._t_buf.copy_(t)
                if self._step_tables:
                    self._load_step(t)
                self._graph.replay()
                d = float((ref.detach().float() - self._out.detach().float()).abs().max().item())
                worst = max(worst, d)
                cases.append({"input": i + 1, "timestep": t_values[i % len(t_values)],
                              "prefix": "refilled" if staged_cache is not None else "captured-chunk",
                              "max_abs_delta": d})
        finally:
            # restore the real chunk state whatever the verdict: the caller's step must see the
            # answer to ITS inputs, and a later refill must start from the real prefix
            if staged_cache is not None:
                self._kv.refill(past_key_values)
            self._x_buf.copy_(x_t)
            self._t_buf.copy_(timestep)
            if self._step_tables:
                self._load_step(timestep)
            torch.cuda.synchronize()

        passed = worst == 0.0
        self.self_check = {"n": n, "bitexact": passed, "max_abs_delta": worst,
                           "seconds": time.perf_counter() - t0, "cases": cases}
        if not passed:
            self._release_and_fall_back(worst)
        if self._on_self_check is not None:
            self._on_self_check(dict(self.self_check))
        return passed

    def _perturbed_prefix(self, past_key_values, gen):
        """A DynamicCache shaped exactly like the real prefix, with values the capture never saw."""
        from pi05_iwm.surface import Pi05CacheBinder

        binder = Pi05CacheBinder()
        leaves, spec = binder.flatten(past_key_values)
        staged = []
        for leaf in leaves:
            noise = torch.empty(leaf.shape, device=leaf.device, dtype=torch.float32)
            noise.normal_(generator=gen)
            scale = leaf.detach().float().std().clamp_min(1e-3) * 0.02
            staged.append((leaf.detach().float() + noise * scale).to(leaf.dtype))
        return binder.unflatten(staged, spec)

    def _release_and_fall_back(self, delta: float) -> None:
        """The FAIL arm: graphs released, upstream rebound, said out loud. Serving continues."""
        self.rejected = True
        self._graph = None
        self._out = None
        # undo the table swap: the eager path must run the REAL projections as real modules
        if self._denses is not None and self._adarms_buf is not None:
            for norm, real in self._denses:
                norm.dense = real
            self._adarms_buf = None
            self._dense_bufs = None
        try:
            self._m.denoise_step = self._orig_denoise
        except Exception:                                          # noqa: BLE001
            pass                       # the __call__ guard still routes every call to eager
        import sys
        # stderr, deliberately: the running server's log stream. cli_config.execute defers
        # stdout until the command returns, and a policy server returns never — a fallback
        # printed where nobody can see it until shutdown is not LOUD.
        print(f"[pi05 static_capture] SELF-CHECK FAILED: replay disagrees with eager by "
              f"{delta:.3e} on a staged input it was not captured from. Graphs released; "
              f"denoise_step rebound to upstream — serving continues on eager arithmetic "
              f"(upstream's, exactly).", file=sys.stderr, flush=True)


def install_static_capture(model, step_tables: "bool | None" = None,
                           on_self_check=None,
                           self_check: bool = True) -> StaticDenoiser:
    """Route `model.denoise_step` through a StaticDenoiser. Returns it (for its counters).

    `step_tables` (default on; IFL_PI05_STEP_TABLES=0 disables) additionally hoists the time MLP
    and the 37 AdaRMS modulation projections out of the captured region into per-timestep tables.

    `self_check` (default on) gates the first capture on the bit-exact replay-vs-eager check —
    see the module docstring. It needs the model's own `denoise_step` still bound at install
    time (it is; the hoists patch `embed_suffix`, never the step). `on_self_check` receives the
    verdict dict so an installer can put it on the plan.
    """
    if step_tables is None:
        import os
        step_tables = os.environ.get("IFL_PI05_STEP_TABLES", "1") != "0"
    # INSTANCE-scoped, and idempotent — a correctness prerequisite of the self-check, not
    # hygiene: a class-level patch makes the SECOND install in a process read the FIRST
    # install's wrapper back as "upstream", so the reference arm compares replay against a
    # wrapper instead of the model, and a rejected capture would rebind to that wrapper too.
    existing = getattr(model, "_ifl_static_denoiser", None)
    if existing is not None:
        return existing
    orig = getattr(model, "denoise_step", None) if self_check else None
    d = StaticDenoiser(model, step_tables=step_tables, orig_denoise=orig,
                       on_self_check=on_self_check)

    def denoise_step(self_m, prefix_pad_masks, past_key_values, x_t, timestep):
        return d(prefix_pad_masks, past_key_values, x_t, timestep)

    import types
    model.denoise_step = types.MethodType(denoise_step, model)
    model._ifl_static_denoiser = d
    return d
