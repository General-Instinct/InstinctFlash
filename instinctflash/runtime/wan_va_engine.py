"""Native VA server with the GI FP8 transformer installed in place of the upstream DiT.

The native ``VA_Server`` keeps its complete control loop -- T5 prompt encoding, VAE
encode, latent/action normalization, schedules, CFG combine, Euler updates, noise and
frame ids. Only its ``transformer`` attribute is replaced; the upstream DiT is never
loaded.
"""
from __future__ import annotations


def build_native_conditioning(server_module, config, *, actions_only=True):
    """Construct native T5/VAE without loading a discarded native DiT.

    Use a separate server module namespace so class patches cannot leak into other
    server instances. ``actions_only`` removes the unused VAE decoders and the
    per-cycle debug dumps, exactly as the native actions-only plan does.
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
            raise RuntimeError("Install the GI VA transformer before using the native server")

    isolated.load_transformer = lambda *args, **kwargs: UnbuiltTransformer()
    isolated._configure_model = lambda model, **kwargs: model
    if actions_only:
        from instinctflash.runtime.lingbot_install import install_debug_dump_elision, install_obs_decode_elision
        install_debug_dump_elision(isolated)
        install_obs_decode_elision(isolated, isolated.VA_Server)
    return isolated.VA_Server(copy.deepcopy(config))


def served_declaration(cfg):
    """The operating point this build serves (the native loop honors the configured schedule)."""
    video_cfg = float(cfg.guidance_scale)
    return {
        "steps": {"video": int(cfg.num_inference_steps), "action": int(cfg.action_num_inference_steps)},
        "guidance": {"video": ("cfg", video_cfg) if video_cfg > 1 else ("none", 1.0),
                     "action": ("none", 1.0)},
        "frontend": "instinctflash/engines/wan_va/engine.py",
        "evidence": "the native VA_Server schedule with the GI transformer installed at build time",
    }


class WanVaEngineLoop:
    """The EngineBackend call contract, including automatic deferred commit."""

    def __init__(self, server, transformer=None, declaration=None):
        from instinctflash.adapters.lingbot_va import _ControlLoop
        cfg = server.job_config
        self._server, self._transformer, self._declaration = server, transformer, declaration
        self._loop = _ControlLoop(server, tuple(cfg.obs_cam_keys), frame_chunk_size=cfg.frame_chunk_size)

    def _require_open(self):
        if self._loop is None:
            raise RuntimeError("The VA engine loop is closed")
        return self._loop

    @property
    def backend_stats(self):
        self._require_open()
        stats = dict(backend="native_t5_vae_gi_fp8_dit", precision="fp8",
                     declaration=dict(self._declaration or {}), frame_st_id=self._server.frame_st_id)
        t = self._transformer
        if t is not None:
            stats.update(graphs_captured=len(t._graphs.graphs), forwards=t.forwards, skipped_forwards=t.skipped)
        return stats

    def declaration(self):
        self._require_open()
        return dict(self._declaration or {})

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
        self._server = self._transformer = None
