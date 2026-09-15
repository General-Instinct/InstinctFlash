"""Runtime installation of optimizer passes onto the stock LingBot-VA server.

Today the "runtime" is the upstream server plus runtime patches. That is deliberate and
temporary: it keeps every pass verifiable against the existing bit-exactness gate
(`probe_bitexact.py`) before anything is rewritten, and it keeps the vendored upstream tree
clean so `git diff` stays reviewable. When the multi-stream KV core exists these installers get
replaced by real backend methods; the *passes* do not change, which is the point of the layering.

Each installer asserts its structural preconditions at install time and raises if one is false.
A pass that silently no-ops, or worse silently mis-caches, is much more expensive than one that
refuses to load.
"""

from __future__ import annotations

import inspect
import os
import sys
import threading
from typing import Callable, Sequence

import torch

from instinctflash.runtime.sm120_install import install_sm120_gated_residual
from instinctflash.runtime.sm120_stage2_install import install_sm120_wan_stage2
from instinctflash.runtime.sm120_stage3_install import install_sm120_wan_stage3


# --- substrate passes -------------------------------------------------------------------
# These substrate passes used to live as inline patches in eval/lingbot_va_robotwin/serve_variant.py.
# They are here so the A/B harness and `plan.serve()` apply the SAME code: a measured
# speedup that came from a different patch than the one production installs is not a
# measurement of anything.


def install_fsdp_elision(server_module, va_server_cls=None) -> list[str]:
    """Do not shard across one GPU. See `passes/substrate.py:FSDPElision` for the cost.

    `wan_va_server` binds `_configure_model` at import time, so the BOUND name is what has to
    be replaced — patching `distributed.util` would be too late.
    """

    def _configure_model_nofsdp(model, shard_fn, param_dtype, device, eval_mode=True):
        if eval_mode:
            model.eval().requires_grad_(False)
        model.to(param_dtype)
        model.to(device)
        return model

    if not hasattr(server_module, "_configure_model"):
        raise RuntimeError(
            "fsdp_elision: wan_va_server has no _configure_model to replace. The upstream "
            "server changed shape; refusing to report an optimization that was not applied."
        )
    server_module._configure_model = _configure_model_nofsdp
    return ["fsdp_elision"]


def install_allocator_churn_elision(server_module, va_server_cls=None) -> list[str]:
    """Stop handing the caching allocator back to the driver between control steps.

    Patched on `torch.cuda` rather than on the server module because the server calls through
    to `torch.cuda.empty_cache` from two different sites (`wan_va_server.py:569`, `:603`).
    """
    torch.cuda.empty_cache = lambda *a, **k: None
    return ["allocator_churn_elision"]


def install_debug_dump_elision(server_module, va_server_cls=None) -> list[str]:
    """Take the blocking device->host telemetry copy off the critical path."""
    if not hasattr(server_module, "save_async"):
        raise RuntimeError(
            "debug_dump_elision: wan_va_server has no save_async to neuter. The upstream "
            "server changed shape; refusing to report an optimization that was not applied."
        )
    server_module.save_async = lambda obj, path: None
    return ["debug_dump_elision"]


class _ElidedObservationDecoder(torch.nn.Module):
    """Parameter-free sentinel left where the unused predicted-pixel decoder used to be.

    Keeping a module-shaped sentinel makes an accidental decode fail at the exact boundary that
    was elided. Setting ``decoder = None`` would instead fail later with an opaque call error.
    """

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "obs_decode_elision removed the VAE decoder because this deployment requested "
            "actions only. Recompile the plan with want_pixels=True before requesting decoded "
            "video."
        )


def install_obs_decode_elision(server_module, va_server_cls) -> list[str]:
    """Remove VAE decoder residency after the upstream server constructs its encoder stack.

    LingBot-VA's action-serving path returns predicted latents and never calls ``vae.decode``.
    Nevertheless the upstream constructor loads two complete Wan VAEs for RoboTwin -- one for the
    head camera and one for the half-resolution wrist composite -- including two 1.03 GiB decoder
    copies. On a 32 GiB RTX 5090 those dead weights are the difference between a successful action
    forward and an OOM in the following observation/KV commit.

    Installation happens before the model is built, so wrap ``VA_Server.__init__`` and strip every
    unique VAE only after ALL discovered decoder targets have been validated. That transactional
    ordering prevents an upstream shape change from leaving a half-elided server. The decoder is
    replaced rather than the encoder being reconstructed, preserving every encoder parameter,
    buffer, dtype and streaming-cache object byte-for-byte.

    The class wrapper persists because plan installation precedes construction, but its ACTIVATION
    is one-shot and thread-local, exactly like install_conv_layout_autotune and the SM120
    installers. This pass physically strips both VAE decoders, so a permanent rebind would leak an
    actions-only plan into every later ``VA_Server`` in the interpreter -- including a want_pixels
    Runtime whose plan DECLINED the pass, which would then raise at the exact boundary it paid to
    keep.
    """
    if not hasattr(va_server_cls, "__init__"):
        raise RuntimeError(
            "obs_decode_elision: VA_Server has no constructor to wrap. The upstream server "
            "changed shape; refusing to report an optimization that was not applied."
        )

    state = getattr(va_server_cls, "_iwm_obs_decode_elision_state", None)
    if state is not None:
        state.pending = True
        return ["obs_decode_elision"]
    state = threading.local()

    original_init = va_server_cls.__init__

    def _init_without_observation_decoders(self, *args, **kwargs):
        pending = getattr(state, "pending", False)
        if pending:
            # Consume before construction: an upstream exception cannot leak this actions-only
            # elision into the next, possibly want_pixels, Runtime in this interpreter.
            del state.pending
        original_init(self, *args, **kwargs)
        if not pending:
            return

        targets = []
        seen = set()
        for attr in ("streaming_vae", "streaming_vae_half"):
            wrapper = getattr(self, attr, None)
            if wrapper is None:
                continue
            vae = getattr(wrapper, "vae", None)
            if vae is None:
                raise RuntimeError(
                    f"obs_decode_elision: {attr} has no .vae module. The upstream server "
                    "changed shape; no decoder was removed."
                )
            if id(vae) in seen:
                continue
            seen.add(id(vae))
            decoder = getattr(vae, "decoder", None)
            if not isinstance(decoder, torch.nn.Module):
                raise RuntimeError(
                    f"obs_decode_elision: {attr}.vae.decoder is not a torch module. The upstream "
                    "server changed shape; no decoder was removed."
                )
            targets.append((vae, decoder))

        if not targets:
            raise RuntimeError(
                "obs_decode_elision: VA_Server exposes neither streaming_vae nor "
                "streaming_vae_half. The upstream server changed shape; no decoder was removed."
            )

        def tensor_bytes(module):
            tensors = list(module.parameters()) + list(module.buffers())
            return sum(t.numel() * t.element_size() for t in tensors)

        freed = sum(tensor_bytes(decoder) for _, decoder in targets)
        for vae, _decoder in targets:
            vae.decoder = _ElidedObservationDecoder()

        self._iwm_obs_decode_elided_bytes = freed
        self._iwm_obs_decode_elided_count = len(targets)
        print(
            f"InstinctFlash observation decode residency: removed {len(targets)} decoder(s), "
            f"released {freed / 2**30:.2f} GiB (actions-only deployment).",
            flush=True,
        )

    va_server_cls.__init__ = _init_without_observation_decoders
    va_server_cls._iwm_obs_decode_elision_state = state
    va_server_cls._iwm_obs_decode_elision_installed = True
    state.pending = True
    return ["obs_decode_elision"]


#: One-shot, thread-local arming for prompt_encoder_staging, the same shape as the constructor
#: tokens in install_conv_layout_autotune and the SM120 installers. `pending` unset means NO PLAN
#: SPOKE (the serve_variant flag path), where the measured device class decides as it always has.
_PROMPT_ENCODER_STAGING = threading.local()


def install_prompt_encoder_staging(server_module, va_server_cls) -> list[str]:
    """Arm the low-memory-SM120 T5 staging for exactly the next server built by this thread.

    The mechanism itself lives in `install_conditioning_prefill._reset` -- the only site that
    knows when the prompt encode has finished -- and `install_plan` enforces that pairing. This
    installer only arms it, so the plan line and the runtime behaviour cannot diverge: the
    hardware predicate consulted at reset time is identical to the one
    `PromptEncoderStaging.evaluate()` planned against.
    """
    _PROMPT_ENCODER_STAGING.pending = True
    return ["prompt_encoder_staging"]


def install_conditioning_prefill(server_module, va_server_cls) -> list[str]:
    """Cache the episode-constant cross-attention K/V for all layers.

    Patches three things:
      1. `WanAttention.forward` — take a fast path when `cross_kv` is populated.
      2. `WanTransformer3DModel` — gain populate/clear/query methods.
      3. `VA_Server._reset` — release, then repopulate once the prompt embeds exist.
    """
    import modules.model as M

    Attn = M.WanAttention if hasattr(M, "WanAttention") else None
    if Attn is None:
        # Find the attention class structurally rather than by name.
        for obj in vars(M).values():
            if isinstance(obj, type) and hasattr(obj, "attn_caches") is False \
               and getattr(obj, "__module__", "") == M.__name__ \
               and "Attention" in obj.__name__:
                Attn = obj
                break
    if Attn is None:
        raise RuntimeError("could not locate the attention class in modules.model")

    Model = M.WanTransformer3DModel

    # ---- 1. attention fast path ----------------------------------------------------------
    _orig_forward = Attn.forward

    def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
        cross_kv = getattr(self, "_iwm_cross_kv", None)
        if cross_kv is None:
            return _orig_forward(self, q, k, v, rotary_emb, update_cache, cache_name)

        # Preconditions that make the cache correct. Both hold for cross-attention
        # (model.py:552-553 pass rotary_emb=None, update_cache=0). Assert rather than assume:
        # if a future edit routes a rotary or a cache write through here, fail loudly.
        if rotary_emb is not None or update_cache != 0:
            raise RuntimeError(
                "conditioning_prefill: cached cross-attention received "
                f"rotary_emb={rotary_emb is not None}, update_cache={update_cache}. "
                "The cache is only valid when neither is used; refusing to serve a wrong value."
            )

        key, value = cross_kv
        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        hidden_states = self.attn_op(query, key, value)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return self.to_out[1](self.to_out[0](hidden_states))

    Attn.forward = forward

    @torch.no_grad()
    def _project_cross_kv(self, encoder_hidden_states):
        # Byte-for-byte the k/v half of the stock forward (model.py:426-431).
        key = self.norm_k(self.to_k(encoder_hidden_states)).unflatten(2, (self.heads, -1))
        value = self.to_v(encoder_hidden_states).unflatten(2, (self.heads, -1))
        return key, value

    Attn._iwm_project_cross_kv = _project_cross_kv

    # ---- 2. model-level populate / clear -------------------------------------------------
    @torch.no_grad()
    def populate_cross_cache(self, text_emb):
        t = self.condition_embedder.text_embedder(text_emb)
        # Build every layer BEFORE publishing any, so a failure mid-way cannot leave the model
        # half-cached (the same transactional discipline vLLM-Omni uses in
        # manager.populate_cross_attention).
        built = [b.attn2._iwm_project_cross_kv(t) for b in self.blocks]
        if len(built) != len(self.blocks):
            raise RuntimeError("conditioning_prefill: incomplete projection")
        for b, kv in zip(self.blocks, built):
            b.attn2._iwm_cross_kv = kv
        del t

    def clear_cross_cache(self):
        for b in self.blocks:
            b.attn2._iwm_cross_kv = None

    def cross_cache_populated(self):
        return getattr(self.blocks[0].attn2, "_iwm_cross_kv", None) is not None

    Model.populate_cross_cache = populate_cross_cache
    Model.clear_cross_cache = clear_cross_cache
    Model.cross_cache_populated = cross_cache_populated

    # ---- 3. skip the text_embedder on the warm path --------------------------------------
    _orig_model_forward = Model.forward

    def model_forward(self, input_dict, *a, **kw):
        if self.cross_cache_populated():
            # attn2 never dereferences encoder_hidden_states on the cached path, but the stock
            # forward still projects text_emb -> text_hidden_states (model.py:843). Swap in a
            # zero-cost stand-in so that projection is skipped too.
            emb = self.condition_embedder.text_embedder
            self.condition_embedder.text_embedder = _NoopTextEmbedder()
            try:
                return _orig_model_forward(self, input_dict, *a, **kw)
            finally:
                self.condition_embedder.text_embedder = emb
        return _orig_model_forward(self, input_dict, *a, **kw)

    Model.forward = model_forward

    # ---- 4. lifecycle: release on reset, repopulate once prompts exist -------------------
    _orig_reset = va_server_cls._reset

    def _staging_decision(self):
        # Bound per server at first reset, then sticky. The plan's verdict -- armed one-shot on
        # this thread by install_plan / install_prompt_encoder_staging -- wins; with no plan in
        # play (the serve_variant flag path) the measured device class decides, as it always has.
        # Either way the hardware predicate below is the same one PromptEncoderStaging.evaluate()
        # plans against, so a plan line and the runtime behaviour cannot disagree on a probed
        # device.
        decided = getattr(self, "_iwm_prompt_encoder_staged_decision", None)
        if decided is None:
            planned = getattr(_PROMPT_ENCODER_STAGING, "pending", None)
            if planned is not None:
                del _PROMPT_ENCODER_STAGING.pending
            hardware = False
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.device)
                hardware = ((props.major, props.minor) == (12, 0)
                            and props.total_memory <= 40 << 30)
            decided = hardware if planned is None else (planned and hardware)
            self._iwm_prompt_encoder_staged_decision = decided
        return decided

    def _reset(self, prompt=None):
        prompt_encoder_staged = _staging_decision(self)
        if prompt_encoder_staged:
            # Keep prompt numerics on the same CUDA kernels as the reference. Between episodes the
            # 5.7B T5 is dead weight; move it back only for reset, then stage it out again below.
            self.text_encoder.to(self.device)
        if hasattr(self, "transformer"):
            self.transformer.clear_cross_cache()
        _orig_reset(self, prompt=prompt)
        # _reset is the ONLY writer of prompt_embeds (wan_va_server.py:424,:426) and also
        # recomputes use_cfg (:379), which is the only thing that can change the batch dim.
        if getattr(self, "prompt_embeds", None) is not None:
            text_emb = self._iwm_text_emb()
            self.transformer.populate_cross_cache(text_emb)
        if prompt_encoder_staged:
            tensors = list(self.text_encoder.parameters()) + list(self.text_encoder.buffers())
            staged_bytes = sum(t.numel() * t.element_size() for t in tensors)
            self.text_encoder.to("cpu")
            # allocator_churn_elision deliberately declines on this measured low-memory SM120
            # configuration, so this is the real CUDA allocator call. Returning the now-dead T5
            # blocks gives the following 8-frame VAE commit contiguous headroom.
            torch.cuda.empty_cache()
            self._iwm_prompt_encoder_staged_bytes = staged_bytes
            if not getattr(self, "_iwm_prompt_encoder_staging_reported", False):
                print(
                    f"InstinctFlash prompt encoder residency: staged {staged_bytes / 2**30:.2f} "
                    "GiB to CPU between episode resets (low-memory SM120).",
                    flush=True,
                )
                self._iwm_prompt_encoder_staging_reported = True

    def _iwm_text_emb(self):
        """Reproduce exactly what _repeat_input_for_cfg puts in input_dict['text_emb']."""
        pos = self.prompt_embeds.to(self.dtype)
        if self.use_cfg:
            return torch.cat([pos, self.negative_prompt_embeds.to(self.dtype)], dim=0)
        return pos

    va_server_cls._reset = _reset
    va_server_cls._iwm_text_emb = _iwm_text_emb

    return ["conditioning_prefill"]


def install_ring_kv_addressing(server_module, va_server_cls) -> list[str]:
    """Install the released P003 allocator once on the upstream attention class."""
    import modules.model as model_module

    attention_cls = model_module.WanAttention
    if getattr(attention_cls, "_iwm_ring_kv_addressing_installed", False):
        return ["ring_kv_addressing"]

    from instinctflash.passes.lingbot.ring_kv import RingKVAddressing

    RingKVAddressing().install(server_module, va_server_cls)
    attention_cls._iwm_ring_kv_addressing_installed = True
    return ["ring_kv_addressing"]


def install_action_terminal_forward_elision(server_module, va_server_cls) -> list[str]:
    """Install the action pred-commit elision once on the upstream transformer + server classes.

    Opt-in (serve_variant `--action-terminal-elision`) until its wrap-crossing A/B is recorded in
    verify/released.py. Works under both allocators: the pass detects RingKVAddressing per cache at
    the first reservation, so it may be installed before or after `--ring-kv`.
    """
    import modules.model as model_module

    model_cls = model_module.WanTransformer3DModel
    if getattr(model_cls, "_iwm_ate_installed", False):
        return ["action_terminal_forward_elision"]

    from instinctflash.passes.lingbot.action_terminal_elision import ActionTerminalForwardElision

    ActionTerminalForwardElision().install(server_module, va_server_cls)
    return ["action_terminal_forward_elision"]


def install_conv_layout_autotune(server_module, va_server_cls, plan=None) -> list[str]:
    """Arm P007 for exactly the next server built by this thread's plan.

    The class wrapper persists because plan installation precedes construction, but its activation
    must not: a later Runtime in the same interpreter may have a BITEXACT ceiling or want pixels.

    `plan=` is optional and carries the compiled plan whose conv_layout_ndhwc PassResult should
    receive the autotune Decision: the backend's chosen layout/kernel per signature is decided at
    construction, and before this it reached only this process's stdout while `plan.explain()` --
    the thing every measurement is labelled with -- said nothing.
    """
    state = getattr(va_server_cls, "_iwm_conv_layout_autotune_state", None)
    if state is None:
        state = threading.local()
        original_init = va_server_cls.__init__

        def _init_with_plan_scoped_conv_layout(self, *args, **kwargs):
            pending = getattr(state, "pending", False)
            plan_result = getattr(state, "plan_result", None)
            if pending:
                # Consume before construction: an upstream exception cannot leak this NUMERIC arm
                # into the next, possibly BITEXACT or want_pixels Runtime in this interpreter.
                del state.pending
                state.plan_result = None
            original_init(self, *args, **kwargs)
            if pending:
                from instinctflash.backends.conv.apply import install_conv_layout

                lines = install_conv_layout(self, prefer_bitexact=False, model_id="lingbot-va")
                self._iwm_conv_layout_decision = tuple(lines)
                if plan_result is not None:
                    # PassResult is frozen but params is the runtime-facing dict by contract;
                    # explain() surfaces this as 'decision:' lines.
                    plan_result.params["decision"] = tuple(lines)
                for line in lines:
                    print(f"InstinctFlash conv layout: {line}", flush=True)

        va_server_cls.__init__ = _init_with_plan_scoped_conv_layout
        va_server_cls._iwm_conv_layout_autotune_state = state
        va_server_cls._iwm_conv_layout_autotune_installed = True

    state.pending = True
    state.plan_result = None if plan is None else next(
        (r for r in plan.applied if r.name == "conv_layout_ndhwc"), None)
    return ["conv_layout_ndhwc"]


class _NoopTextEmbedder(torch.nn.Module):
    """Stands in for the text embedder while the cross-attention cache is warm.

    Returns None: attn2 ignores its k/v operands on the cached path, and nothing else in the
    forward reads `text_hidden_states`. If that ever stops being true this returns None into an
    op and fails loudly, which is the behaviour we want.
    """

    def forward(self, x):
        return None


# --- plan installation ------------------------------------------------------------------

#: pass name -> installer. Every entry here changes the running server.
INSTALLERS: dict[str, Callable[..., list[str]]] = {
    "fsdp_elision": install_fsdp_elision,
    "allocator_churn_elision": install_allocator_churn_elision,
    "debug_dump_elision": install_debug_dump_elision,
    "obs_decode_elision": install_obs_decode_elision,
    "conditioning_prefill": install_conditioning_prefill,
    "prompt_encoder_staging": install_prompt_encoder_staging,
    "ring_kv_addressing": install_ring_kv_addressing,
    "action_terminal_forward_elision": install_action_terminal_forward_elision,
    "sm120_gated_residual": install_sm120_gated_residual,
    "sm120_wan_stage2": install_sm120_wan_stage2,
    "sm120_wan_stage3": install_sm120_wan_stage3,
    "conv_layout_ndhwc": install_conv_layout_autotune,
}

#: pass name -> why this backend needs no runtime action for it. Separate from INSTALLERS on
#: purpose: "we did nothing, here is why" and "we did something" are different claims, and a
#: pass that quietly fell into neither bucket is the failure this split exists to prevent.
NO_RUNTIME_ACTION: dict[str, str] = {}


def install_plan(server_module, va_server_cls, plan) -> list[str]:
    """Apply every applied pass in `plan` to the imported upstream server.

    Raises on any applied pass this backend cannot install. That is the point: a server whose
    `plan.explain()` claims a pass fired, while the pass was silently skipped, invalidates
    every number measured against it. The failure mode this framework sells against is
    precisely a plausible wrong number.

    The whole plan is checked BEFORE anything is patched, for the same reason
    `populate_cross_cache` builds every layer before publishing any: a partial install leaves
    the server module mutated in a state no plan describes, and the patches are monkeypatches
    on an imported module, so there is nothing to roll back to.
    """
    applied_names = {result.name for result in plan.applied}
    if "sm120_wan_stage2" in applied_names and "sm120_gated_residual" not in applied_names:
        raise RuntimeError(
            "sm120_wan_stage2 requires sm120_gated_residual in the same plan. P009-A2 reuses "
            "P009-A1 for the final FFN residual, so installing it alone would produce a plan "
            "whose runtime dependency is absent."
        )
    if "sm120_wan_stage3" in applied_names and "sm120_wan_stage2" not in applied_names:
        raise RuntimeError(
            "sm120_wan_stage3 requires sm120_wan_stage2 in the same plan. P009-A3 reuses "
            "A2's two middle block kernels and cannot install independently."
        )
    if "prompt_encoder_staging" in applied_names and "conditioning_prefill" not in applied_names:
        raise RuntimeError(
            "prompt_encoder_staging requires conditioning_prefill in the same plan: the staging "
            "mechanism lives in conditioning_prefill's reset wrapper, which is the only site "
            "that knows when the prompt encode has finished."
        )
    if "conditioning_prefill" in applied_names and "prompt_encoder_staging" not in applied_names:
        # The plan spoke and did not apply staging (declined, ceiling, or Plan.without). The
        # runtime device sniff must not re-enable what explain() reports as skipped, so disarm
        # the one-shot token for the next server built by this thread.
        _PROMPT_ENCODER_STAGING.pending = False

    unsupported = [
        r.name for r in plan.applied
        if r.name not in INSTALLERS and r.name not in NO_RUNTIME_ACTION
    ]
    if unsupported:
        raise NotImplementedError(
            f"the lingbot-va backend has no installer for {unsupported}. Either implement one "
            f"in instinctflash/runtime/lingbot_install.py, or drop the pass from the plan with "
            f"plan.without({', '.join(repr(n) for n in unsupported)}) — but do not serve a "
            f"plan whose explain() output claims work that was never applied."
        )

    applied: list[str] = []
    for result in plan.applied:
        installer = INSTALLERS.get(result.name)
        if installer is not None:
            if "plan" in inspect.signature(installer).parameters:
                # installers that record a construction-time Decision get the plan, so the
                # decision lands in the PassResult that explain() prints.
                applied.extend(installer(server_module, va_server_cls, plan=plan))
            else:
                applied.extend(installer(server_module, va_server_cls))
        else:
            applied.append(f"{result.name} (no runtime action: {NO_RUNTIME_ACTION[result.name]})")
    return applied


def install_deterministic_seed(server_module, seed: int) -> list[str]:
    """Seed the noise draw so two servers can be compared at all.

    `_infer` draws `torch.randn` for the initial video latents and action tokens
    (`wan_va_server.py:449-462`) with no seeding, so two *stock* servers already disagree and
    any A/B on output values measures the noise draw rather than the variant.

    Seeded as a function of `frame_st_id`, not a constant: a constant would start every chunk
    in an episode from the SAME noise, which is not the stock distribution and would itself be
    a behaviour change.
    """
    _orig_infer = server_module.VA_Server._infer

    def _seeded_infer(self, obs, frame_st_id=0):
        torch.manual_seed(seed + frame_st_id)
        torch.cuda.manual_seed_all(seed + frame_st_id)
        return _orig_infer(self, obs, frame_st_id=frame_st_id)

    server_module.VA_Server._infer = _seeded_infer
    return [f"deterministic_seed={seed}"]


def resolve_lingbot_root(explicit: str | None = None) -> str:
    """Locate the upstream lingbot-va tree, or say exactly what is missing.

    Order: explicit argument, `LINGBOT_ROOT`, a cache directory, then this machine's historical
    default. Raises rather than letting `import wan_va_server` fail with a bare ModuleNotFoundError
    several frames later.

    The message matters more than the lookup. An external user following the README had no way to
    learn this dependency existed: the old error named `LINGBOT_ROOT` without saying what to put in
    it or where to get it, and the fallback path was one developer's home directory. Serving a
    backbone must not require guessing.
    """
    cache = os.path.join(os.environ.get("XDG_CACHE_HOME") or
                         os.path.join(os.path.expanduser("~"), ".cache"), "instinctflash", "lingbot-va")
    if explicit is not None:
        # An explicit path is a claim, not a hint: silently falling back to a cache when it is
        # wrong would serve a different checkout than the one the caller named.
        if os.path.isdir(explicit):
            return explicit
        raise FileNotFoundError(
            f"LINGBOT_ROOT candidate {explicit!r} does not exist. Point it at a checkout of "
            f"https://github.com/robbyant/lingbot-va, or unset it to use the default cache.")
    candidates = [c for c in (os.environ.get("LINGBOT_ROOT"), cache,
                              "/home/ubuntu/lingbot-va") if c]
    for root in candidates:
        if os.path.isdir(root):
            return root
    raise FileNotFoundError(
        f"The wan_va backbone needs the upstream LingBot-VA serving code, which is not vendored "
        f"here. Get it once:\n\n"
        f"    git clone https://github.com/robbyant/lingbot-va {cache}\n\n"
        f"or point LINGBOT_ROOT at an existing checkout:\n\n"
        f"    export LINGBOT_ROOT=/path/to/lingbot-va\n\n"
        f"It provides wan_va/wan_va_server.py, which InstinctFlash patches at runtime instead of "
        f"copying, so that the optimizations stay verifiable against the upstream implementation.\n"
        f"Looked in: {', '.join(candidates)}")


def _ensure_flash_attn_importable() -> bool:
    """Make `import flash_attn` succeed when the real wheel is absent. Returns True if stubbed.

    `wan_va/modules/model.py` imports `flash_attn_func` at module scope, so the server cannot be
    imported at all without *some* `flash_attn` -- even though the shipped path never calls it. Both
    the checkpoint (`attn_mode: torch`) and the server select `custom_sdpa`, i.e.
    `scaled_dot_product_attention`; `flash_attn_func` is bound only when `attn_mode == 'flashattn'`.

    This used to live in a directory the caller had to put on PYTHONPATH, which meant serving
    LingBot-VA required knowing about a file no document mentioned. It is in the package now.

    The stub RAISES if it is ever called, which is the point: it cannot change any numbers, and if
    some path does reach flash attention the run dies loudly instead of quietly producing a result
    under a different attention kernel.
    """
    import importlib.util
    if importlib.util.find_spec("flash_attn") is not None:
        return False                                             # a real wheel always wins

    # A directory on sys.path, not a synthesised module: `transformers` calls
    # `importlib.util.find_spec("flash_attn")`, which raises `ValueError: flash_attn.__spec__ is
    # None` for a hand-built ModuleType. Shipping the shim as a real package keeps the import
    # machinery happy and reproduces the configuration this project has always run under.
    shims = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shims")
    if shims not in sys.path:
        sys.path.insert(0, shims)
    importlib.invalidate_caches()
    return True


def import_lingbot_server(lingbot_root: str | None = None, extra_paths: Sequence[str] = ()):
    """Put the upstream tree on `sys.path` and import its server module."""
    root = resolve_lingbot_root(lingbot_root)
    for entry in (os.path.join(root, "wan_va"), root, *extra_paths):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    if _ensure_flash_attn_importable():
        print("InstinctFlash: flash-attn not installed; using the import-only stub "
              "(this serving path runs attn_mode='torch' and never calls it).", flush=True)

    import wan_va_server  # noqa: E402  (importable only after the sys.path insert above)

    return wan_va_server
