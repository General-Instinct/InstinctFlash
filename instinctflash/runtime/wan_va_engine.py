"""Native VA conditioning and action processing around the FP8 DiT frontend.

This bridge owns a native server instance; it replaces that instance's transformer.
The public backend must qualify the bridge before registering the family.
"""
from __future__ import annotations

import torch


def build_native_conditioning(server_module, config):
    """Construct native T5/VAE without loading a discarded native DiT.

    Use a separate server module namespace so default-path class patches and
    constructor changes cannot leak into this engine instance, or vice versa.
    The caller must install WanVaEngineServer before invoking the server.
    """
    import copy
    import importlib.util
    import uuid

    spec = importlib.util.spec_from_file_location(
        "_ifl_va_conditioning_" + uuid.uuid4().hex, server_module.__file__)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the native VA conditioning source")
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    for name in ("load_transformer", "_configure_model", "VA_Server"):
        if not hasattr(isolated, name):
            raise RuntimeError(f"Native VA source lacks required constructor hook {name}")

    class UnbuiltTransformer:
        def __getattr__(self, name):
            raise RuntimeError("Install the FP8 VA bridge before using the native server")

    isolated.load_transformer = lambda *args, **kwargs: UnbuiltTransformer()
    isolated._configure_model = lambda model, **kwargs: model
    return isolated.VA_Server(copy.deepcopy(config))


class WanVaEngineLoop:
    """The EngineBackend call contract, including automatic deferred commit."""

    def __init__(self, server):
        from instinctflash.adapters.lingbot_va import _ControlLoop
        cfg = server.native.job_config
        self._loop = _ControlLoop(server, tuple(cfg.obs_cam_keys),
                                  frame_chunk_size=cfg.frame_chunk_size)

    def _require_open(self):
        if self._loop is None:
            raise RuntimeError("The VA engine loop is closed")
        return self._loop

    @property
    def backend_stats(self):
        server = self._require_open()._server
        return dict(backend="native_t5_vae_fp8_dit", precision="fp8",
                    declaration=server.frontend.declaration(),
                    calibrated=server.calibrated,
                    graphs_captured=len(server.frontend._graphs),
                    frame_st_id=server.frontend.frame_st_id)

    def declaration(self):
        return self._require_open()._server.frontend.declaration()

    def reset(self, **conditioning):
        self._require_open().reset(**conditioning)

    def predict(self, observation, *, executed_action=None):
        loop = self._require_open()
        out = loop.predict(dict(observation))
        loop.commit(dict(observation),
                    out["action"] if executed_action is None else executed_action)
        return out

    def close(self):
        if self._loop is not None:
            self._loop.close()
            self._loop = None


class _EngineCache:
    """Honor the cache lifecycle called by the native server's reset method."""

    def __init__(self, frontend):
        self.frontend = frontend

    def clear_cache(self, cache_name):
        self.frontend.reset_episode()

    def create_empty_cache(self, cache_name, attn_window, video_tokens, action_tokens,
                           device, dtype, batch_size):
        e = self.frontend
        g = e.geometry
        if ((video_tokens, action_tokens, batch_size) !=
                (g.video_tokens, g.action_tokens, e.B)
                or g.pool_slots(attn_window) != e.pool_slots):
            raise ValueError("Native VA cache geometry differs from the FP8 build")


class WanVaEngineServer:
    """Preserve the native reset/predict/history wire contract.

Calibration uses the first real conditioning sample and the same sampled noise
as the returned prediction. It never encodes an observation twice or commits
predicted history in place of frames observed during execution.
"""

    def __init__(self, native_server, frontend, *, calibrated=False):
        from flash_rt.models.wan_va.geometry import WanVaGeometry
        from flash_rt.models.wan_va.operating_point import WanVaOperatingPoint

        cfg = native_server.job_config
        frontend.assert_point(WanVaOperatingPoint.from_job_config(cfg))
        if frontend.geometry != WanVaGeometry.from_job_config(cfg):
            raise ValueError("Native VA geometry differs from the FP8 build")
        if cfg.video_exec_step != -1:
            raise ValueError("VA engine requires the complete declared video schedule")
        if cfg.action_guidance_scale > 1:
            raise ValueError("VA frontend currently implements positive-only action guidance")
        if frontend.precision != "fp8":
            raise ValueError("The FP8 bridge requires an actual FP8 frontend")
        families = {"qkv_w", "o_w", "cq_w", "co_w", "ff1_w", "ff2_w"}
        if families.issubset(getattr(frontend, "fp16_families", ())):
            raise ValueError("The FP8 bridge refuses a recipe with every GEMM in FP16")
        mask = torch.ones(30, dtype=torch.bool)
        mask[list(cfg.used_action_channel_ids)] = False
        if not torch.equal(frontend._action_unused.cpu(), mask):
            raise ValueError("Native VA action channels differ from the FP8 build")
        if frontend.pool_slots != frontend.geometry.pool_slots(cfg.attn_window):
            raise ValueError("Native VA attention window differs from the FP8 build")
        self.native = native_server
        self.frontend = frontend
        self.calibrated = bool(calibrated)
        self.calibration_receipt = None
        self._ready = False
        # Only this owned instance is changed; no module/class monkey-patching.
        native_server.transformer = _EngineCache(frontend)

    @torch.no_grad()
    def reset(self, prompt):
        self._ready = False
        s, e = self.native, self.frontend
        # Native reset owns VAE caches, normalization statistics, and live T5.
        s._reset(prompt=prompt)
        if s.prompt_embeds is None:
            return
        text = s.prompt_embeds
        if e.B == 2:
            if s.negative_prompt_embeds is None:
                raise ValueError("The CFG build requires native negative prompt embeddings")
            text = torch.cat((text, s.negative_prompt_embeds), dim=0)
        e.set_prompt(text)
        self._ready = True

    def _check_ready(self):
        if not self._ready:
            raise RuntimeError("Reset the VA engine with a prompt before inference")
        if self.native.frame_st_id != self.frontend.frame_st_id:
            raise RuntimeError("Native and FP8 VA history positions have diverged")

    @torch.no_grad()
    def predict(self, observation):
        self._check_ready()
        s, e = self.native, self.frontend
        g = e.geometry
        first = s.frame_st_id == 0
        if first:
            s.init_latent = s._encode_obs(observation)
        # Keep upstream draw order and dtype: video, then action.
        v = torch.randn(1, 48, g.frame_chunk, g.latent_h, g.latent_w,
                        device=s.device, dtype=s.dtype)
        a = torch.randn(1, 30, g.frame_chunk, g.action_per_frame, 1,
                        device=s.device, dtype=s.dtype)
        init = s.init_latent[:, :, :1] if first else None
        if not self.calibrated:
            if not first:
                raise RuntimeError("VA activation calibration requires an episode start")
            e.begin_calibration()
            e.infer_cycle(v, a, init)
            self.calibration_receipt = e.end_calibration()
            e.reset_episode()
            self.calibrated = True
        actions, latents = e.infer_cycle(v, a, init)
        if not torch.isfinite(actions).all() or not torch.isfinite(latents).all():
            raise RuntimeError("VA FP8 prediction contains non-finite values")
        return s.postprocess_action(actions)

    @torch.no_grad()
    def commit(self, observation):
        self._check_ready()
        s, e = self.native, self.frontend
        latents = s._encode_obs(observation)
        if s.frame_st_id == 0:
            latents = (torch.cat((s.init_latent, latents), dim=2)
                       if latents is not None else s.init_latent)
        actions = s.preprocess_action(observation["state"]).to(latents)
        e.commit_chunk(latents, actions)
        s.frame_st_id += latents.shape[2]
        self._check_ready()

    def infer(self, observation):
        if observation.get("reset", False):
            self.reset(observation.get("prompt"))
            return {}
        if observation.get("compute_kv_cache", False):
            self.commit(observation)
            return {}
        return {"action": self.predict(observation)}
