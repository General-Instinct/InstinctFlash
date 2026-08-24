"""CFG-batched diffusion forwards for DreamZero's Wan2.2 action head.

Measured on the official server (H100, canonical request): a call is 18 B=1 DiT forwards —
8 computed denoise steps x {cond, uncond} plus one KV-commit pair — and the diffusion term is
87% of the 3.12 s wall. The cond/uncond pair differs only in text context and KV cache, so the
pair batches into one B=2 forward: 18 forwards become 9. This is the same lever vLLM-Omni's
DreamZero pipeline pulls ("CFG-parallel"); here it is a runtime install against the unmodified
upstream server, in the serve_variant.py tradition.

Caches: the upstream keeps kv_cache1/kv_cache_neg as separate per-layer lists and _run_diffusion_steps
loops over them. The batched path maintains ONE stacked store per layer ([2, 2B, L, H, D]) built
whenever the caller hands us freshly (re)created caches (detected by object identity), so the
per-step forward consumes it directly with no per-call stacking.

Tier: to be established by the verify script — the upstream scheduler is torch.compile'd and the
model may carry batch-order kernel effects, so the null control runs first and the claim follows
the measurement, not the mechanism.
"""

from __future__ import annotations

import types

import torch


def install_cfg_batch(action_head) -> None:
    if getattr(action_head, "_ifl_cfg_batch_installed", False):
        return
    orig = action_head._run_diffusion_steps

    def batched(self, noisy_input, timestep, action, timestep_action, state, embodiment_id,
                context, seq_len, y, clip_feature, kv_caches, crossattn_caches,
                kv_cache_metadata):
        if len(context) != 2 or len(kv_caches) != 2:
            return orig(
                noisy_input=noisy_input, timestep=timestep, action=action,
                timestep_action=timestep_action, state=state, embodiment_id=embodiment_id,
                context=context, seq_len=seq_len, y=y, clip_feature=clip_feature,
                kv_caches=kv_caches, crossattn_caches=crossattn_caches,
                kv_cache_metadata=kv_cache_metadata)

        B = noisy_input.shape[0]

        # (Re)build the stacked stores when the caller handed us fresh caches (new session/chunk 0).
        marker = id(kv_caches[0])
        if getattr(self, "_ifl_kv_marker", None) != marker:
            self._ifl_kv_marker = marker
            self._ifl_kv2 = [torch.cat([a, b], dim=1)
                             for a, b in zip(kv_caches[0], kv_caches[1])]
            self._ifl_ca2 = [torch.cat([a, b], dim=1)
                             for a, b in zip(crossattn_caches[0], crossattn_caches[1])]

        def rep(t):
            return torch.cat([t, t], dim=0) if isinstance(t, torch.Tensor) else t

        out = self.model(
            rep(noisy_input), rep(timestep),
            action=rep(action), timestep_action=rep(timestep_action), state=rep(state),
            embodiment_id=rep(embodiment_id),
            context=torch.cat([context[0], context[1]], dim=0),
            seq_len=seq_len, y=rep(y), clip_feature=rep(clip_feature),
            kv_cache=self._ifl_kv2, crossattn_cache=self._ifl_ca2,
            current_start_frame=kv_cache_metadata["start_frame"],
        )
        obs_pred, act_pred, updated = out
        if kv_cache_metadata["update_kv_cache"]:
            self._ifl_kv2 = [u.clone() for u in updated]
        a_act = act_pred[:B] if act_pred is not None else None
        b_act = act_pred[B:] if act_pred is not None else None
        return [(obs_pred[:B], a_act), (obs_pred[B:], b_act)]

    action_head._run_diffusion_steps = types.MethodType(batched, action_head)
    action_head._ifl_cfg_batch_installed = True
