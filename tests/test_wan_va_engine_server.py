"""The GI LingBot-VA bridge: native server ownership, deferred commit, pool semantics."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from instinctflash.engines.wan_va.pool import SlotPool
from instinctflash.runtime import engine_backend as eb
from instinctflash.runtime.wan_va_engine import WanVaEngineLoop, build_native_conditioning, served_declaration
from instinctflash.runtime.wan_va_engine_build import _token_geometry, resolve_native_config


def config(**kw):
    base = dict(frame_chunk_size=4, height=128, width=128, action_per_frame=4, action_dim=30,
                patch_size=(1, 2, 2), env_type="none", obs_cam_keys=["camera"], attn_window=30,
                num_inference_steps=20, action_num_inference_steps=50, guidance_scale=5,
                action_guidance_scale=1, snr_shift=5.0, action_snr_shift=0.05, video_exec_step=-1,
                used_action_channel_ids=list(range(7)))
    base.update(kw)
    return SimpleNamespace(**base)


class _UpstreamPool:
    """Line-by-line transcription of upstream WanAttention cache bookkeeping (torch)."""

    def __init__(self, n):
        import torch
        self.torch = torch
        self.mask = torch.zeros(n, dtype=torch.bool)
        self.ids = torch.full((n,), -1, dtype=torch.int64)
        self.is_pred = torch.zeros(n, dtype=torch.bool)

    def append(self, k, is_pred):
        torch, mask, ids = self.torch, self.mask, self.ids
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)
        if free.numel() < k:
            used = mask.nonzero(as_tuple=False).squeeze(-1)
            order = torch.argsort(ids[used], stable=True)
            to_free = used[order[:k - free.numel()]]
            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)
        slots = free[:k]
        new_id = ids[mask].max() + 1 if mask.any() else torch.tensor(0)
        mask[slots] = True
        ids[slots] = new_id
        self.is_pred[slots] = is_pred
        return slots.numpy()


def test_slot_pool_matches_upstream_bookkeeping_across_wraparound():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    size, video, action = 8 * 24 + 8 * 4, 24, 4
    ours, ref = SlotPool(size), _UpstreamPool(size)
    for cycle in range(40):
        for kind, n, steps in (("v", video, 3), ("a", action, 5)):
            for i in range(steps):
                uc = 1 if i == steps - 1 else 0
                a, b = ours.append(n, is_pred=uc == 1), ref.append(n, uc == 1)
                assert np.array_equal(a, b)
                assert np.array_equal(ours.live(), ref.mask.nonzero().squeeze(-1).numpy())
                if uc == 0:
                    ours.restore(a)
                    ref.mask[b] = False
        ours.clear_pred()
        ref.mask[ref.is_pred] = False
        for n in (video, action):
            a, b = ours.append(n, is_pred=False), ref.append(n, False)
            assert np.array_equal(a, b)
        if rng.random() < 0.1:
            ours.reset(); ref.__init__(size)
    assert np.array_equal(ours.ids, ref.ids.numpy())


def test_conditioning_constructor_is_isolated_and_skips_dit(tmp_path):
    source = tmp_path / "native_server.py"
    source.write_text('''
def load_transformer(*args, **kwargs):
    raise AssertionError("native DiT must not be loaded")
def _configure_model(*args, **kwargs):
    raise AssertionError("native DiT configuration must not run")
class VA_Server:
    def __init__(self, cfg):
        cfg.value += 1
        self.job_config = cfg
        self.transformer = _configure_model(model=load_transformer("weights"), device="cuda")
    def original(self):
        return True
''')
    module = SimpleNamespace(__file__=str(source), VA_Server=object(),
                             load_transformer=object(), _configure_model=object())
    original = dict(vars(module))
    cfg = SimpleNamespace(value=1)
    server = build_native_conditioning(module, cfg, actions_only=False)
    assert server.original() and server.job_config.value == 2
    assert cfg.value == 1 and vars(module) == original
    with pytest.raises(RuntimeError, match="Install"):
        server.transformer.clear_cache("cache")


def test_actions_only_conditioning_drops_decoders_and_debug_dumps(tmp_path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "native_server.py"
    source.write_text('''
import torch
def save_async(obj, path):
    raise AssertionError("debug dump must be elided")
def load_transformer(*args, **kwargs):
    raise AssertionError("native DiT must not be loaded")
def _configure_model(*args, **kwargs):
    return None
class _Vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(4, 4)
class VA_Server:
    def __init__(self, cfg):
        self.job_config = cfg
        self.streaming_vae = type("S", (), {"vae": _Vae()})()
        self.transformer = _configure_model(model=load_transformer("w"))
    def dump(self):
        save_async(1, "x")
''')
    module = SimpleNamespace(__file__=str(source))
    server = build_native_conditioning(module, SimpleNamespace())
    server.dump()
    assert not isinstance(server.streaming_vae.vae.decoder, torch.nn.Linear)
    assert isinstance(server.streaming_vae.vae.encoder, torch.nn.Linear)


class _NativeServer:
    def __init__(self):
        self.job_config = config()
        self.frame_st_id = 0
        self.calls = []

    def infer(self, obs):
        self.calls.append(dict(obs))
        if obs.get("reset"):
            self.frame_st_id = 0
            return {}
        if obs.get("compute_kv_cache"):
            self.frame_st_id += 4
            return {}
        return {"action": np.zeros((7, 4, 4), np.float32)}


def test_engine_loop_defers_commit_uses_override_and_closes():
    server = _NativeServer()
    transformer = SimpleNamespace(_graphs=SimpleNamespace(graphs={}), forwards=0, skipped=0)
    loop = WanVaEngineLoop(server, transformer, served_declaration(server.job_config))
    loop.reset(prompt="task")
    override = np.full((7, 4, 4), 0.25, np.float32)
    loop.predict({"obs": [{"camera": 0}]}, executed_action=override)
    assert not any(c.get("compute_kv_cache") for c in server.calls)
    loop.predict({"obs": [{"camera": i} for i in range(12)]})
    commit = next(c for c in server.calls if c.get("compute_kv_cache"))
    assert np.all(commit["state"] == 0.25) and len(commit["obs"]) == 12
    assert loop.backend_stats["backend"] == "native_t5_vae_gi_fp8_dit"
    loop.close()
    loop.close()
    with pytest.raises(RuntimeError, match="closed"):
        loop.predict({})


def test_declaration_is_the_built_schedule_and_the_registry_accepts_it():
    cfg = config(num_inference_steps=2, action_num_inference_steps=4)
    decl = served_declaration(cfg)
    assert decl["steps"] == {"video": 2, "action": 4}
    assert decl["guidance"] == {"video": ("cfg", 5.0), "action": ("none", 1.0)}
    baked = eb.engine_baked_operating_point("wan_va", decl)
    assert baked.steps == {"video": 2, "action": 4} and baked.frontend.endswith("engines/wan_va/engine.py")
    assert eb.engine_baked_operating_point("wan_va") is None
    assert eb.engine_operating_point_problem(baked, {"video": 2, "action": 4},
                                             {"video": ("cfg", 5.0), "action": ("none", 1.0)}) is None
    assert "requests video=25" in eb.engine_operating_point_problem(
        baked, {"video": 25, "action": 4}, {"video": ("cfg", 5.0), "action": ("none", 1.0)})


def test_token_geometry_matches_native_reset():
    assert _token_geometry(config()) == (4 * 8 * 8 // 4, 16)
    t = config(env_type="robotwin_tshape", height=256, width=320, frame_chunk_size=2, action_per_frame=16)
    assert _token_geometry(t) == (240, 32)


def test_factory_resolves_checkpoint_schedule_without_mutating_native_config():
    base = config()
    source = SimpleNamespace(VA_CONFIGS={"libero": base})
    execution = SimpleNamespace(extra={"va_config": "libero", "frame_chunk_size": 4},
                                nfe={"video": 20, "action": 50},
                                guidance={"video": "positive_only", "action": "positive_only"})
    checkpoint = SimpleNamespace(execution=execution)
    adapter = SimpleNamespace(materialize=lambda ckpt: "/pinned/checkpoint")
    cfg = resolve_native_config(adapter, checkpoint, source, nfe={"video": 10})
    assert (cfg.num_inference_steps, cfg.action_num_inference_steps) == (10, 50)
    assert cfg.action_snr_shift == 0.05 and cfg.frame_chunk_size == 4
    assert cfg.guidance_scale == 1 and base.guidance_scale == 5
    assert base.num_inference_steps == 20
    assert cfg.wan22_pretrained_model_name_or_path == "/pinned/checkpoint"
    for override in ({"action": 0}, {"action": True}, {"kv_refresh": 1}, {"unknown": 2}):
        with pytest.raises(ValueError):
            resolve_native_config(adapter, checkpoint, source, nfe=override)
