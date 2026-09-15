"""Verify native conditioning ownership through the real deferred-commit loop."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "serving")]
os.environ.setdefault("WAN_VA_TORCH_REF_ONLY", "1")
from flash_rt.models.wan_va.geometry import WanVaGeometry
from flash_rt.models.wan_va.operating_point import WanVaOperatingPoint
from instinctflash.adapters.lingbot_va import _ControlLoop
from instinctflash.runtime.wan_va_engine import WanVaEngineServer
from instinctflash.runtime.wan_va_engine import WanVaEngineLoop, build_native_conditioning
from instinctflash.runtime.wan_va_engine_build import resolve_native_config


def build(frame_chunk):
    cfg = SimpleNamespace(
        frame_chunk_size=frame_chunk, height=128, width=128,
        action_per_frame=4, action_dim=30, patch_size=(1, 2, 2),
        env_type="none", obs_cam_keys=["camera"], attn_window=30,
        num_inference_steps=20, action_num_inference_steps=50,
        guidance_scale=5, action_guidance_scale=1,
        snr_shift=5.0, action_snr_shift=0.05, video_exec_step=-1,
        used_action_channel_ids=list(range(7)),
    )
    g = WanVaGeometry.from_job_config(cfg)
    events, samples = [], []

    class Frontend:
        geometry, B, precision = g, 2, "fp8"
        pool_slots = g.pool_slots(cfg.attn_window)
        _action_unused = torch.arange(30) >= 7
        frame_st_id = 0

        def assert_point(self, point):
            WanVaOperatingPoint.from_job_config(cfg).assert_serves(point)

        def reset_episode(self):
            self.frame_st_id = 0

        def set_prompt(self, text):
            assert text.shape == (2, 512, 4096)
            assert text[0, 0, 0] == 1 and text[1, 0, 0] == -1
            self.reset_episode()

        def begin_calibration(self):
            events.append("calibrate")

        def end_calibration(self):
            return {"scope": "test"}

        def infer_cycle(self, v, a, init):
            samples.append((v.clone(), a.clone()))
            events.append("dit")
            return a.clone(), v.clone()

        def commit_chunk(self, latents, action):
            assert latents.shape[2] == frame_chunk
            assert torch.equal(action, native.last_processed)
            self.frame_st_id += latents.shape[2]
            events.append("commit")

    class Native:
        job_config, device, dtype = cfg, "cpu", torch.bfloat16
        transformer = object()

        def _reset(self, prompt):
            self.frame_st_id = 0
            self.init_latent = None
            self.transformer.clear_cache("cache")
            self.transformer.create_empty_cache(
                "cache", cfg.attn_window, g.video_tokens, g.action_tokens,
                self.device, self.dtype, 2)
            self.prompt_embeds = torch.ones(1, 512, 4096)
            self.negative_prompt_embeds = -self.prompt_embeds
            events.append("native-reset")

        def _encode_obs(self, observation):
            n = len(observation["obs"])
            events.append(("encode", n))
            return torch.ones(1, 48, 1 if n == 1 else n // 4, g.latent_h, g.latent_w)

        def postprocess_action(self, a):
            events.append("postprocess")
            return a[0, :7, ..., 0].float().numpy()

        def preprocess_action(self, a):
            events.append("preprocess")
            self.last_processed = torch.zeros(1, 30, frame_chunk, 4, 1)
            self.last_processed[:, :7, ..., 0] = torch.from_numpy(a)
            return self.last_processed

    native, frontend = Native(), Frontend()
    return native, frontend, events, samples


@pytest.mark.parametrize("frame_chunk", [2, 4])
def test_calibration_and_observed_history_are_not_replayed(frame_chunk):
    s, e, events, samples = build(frame_chunk)
    bridge = WanVaEngineServer(s, e)
    loop = _ControlLoop(bridge, ("camera",), frame_chunk_size=frame_chunk)
    loop.reset(prompt="move")
    first = loop.predict({"obs": [{"camera": 0}]})["action"]
    assert events == ["native-reset", ("encode", 1), "calibrate", "dit", "dit", "postprocess"]
    assert all(torch.equal(x, y) for x, y in zip(samples[0], samples[1]))
    assert s.frame_st_id == e.frame_st_id == 0
    loop.commit({}, first)
    observed = [{"camera": i} for i in range((frame_chunk - 1) * 4)]
    second = loop.predict({"obs": observed})["action"]
    assert events[-5:] == [("encode", len(observed)), "preprocess", "commit", "dit", "postprocess"]
    assert s.frame_st_id == e.frame_st_id == frame_chunk
    loop.commit({}, second)
    loop.predict({"obs": [{"camera": i} for i in range(frame_chunk * 4)]})
    assert s.frame_st_id == e.frame_st_id == frame_chunk * 2
    loop.reset(prompt="new episode")
    loop.predict({"obs": [{"camera": 0}]})
    assert events.count("calibrate") == 1
    assert s.frame_st_id == e.frame_st_id == 0


def test_bad_build_is_refused_before_replacing_native_transformer():
    s, e, _, _ = build(4)
    original = s.transformer
    e.pool_slots += 1
    with pytest.raises(ValueError, match="attention window"):
        WanVaEngineServer(s, e)
    assert s.transformer is original


def test_reset_and_history_drift_are_enforced():
    s, e, _, _ = build(4)
    bridge = WanVaEngineServer(s, e)
    with pytest.raises(RuntimeError, match="Reset"):
        bridge.infer({"obs": [{"camera": 0}]})
    bridge.reset("task")
    s.frame_st_id = 1
    with pytest.raises(RuntimeError, match="diverged"):
        bridge.infer({"obs": [{"camera": 0}]})


def test_engine_loop_commits_override_and_closes():
    s, e, events, _ = build(4)
    loop = WanVaEngineLoop(WanVaEngineServer(s, e))
    loop.reset(prompt="task")
    override = torch.full((7, 4, 4), 0.25).numpy()
    loop.predict({"obs": [{"camera": 0}]}, executed_action=override)
    assert "commit" not in events
    loop.predict({"obs": [{"camera": i} for i in range(12)]})
    assert torch.all(s.last_processed[:, :7] == 0.25)
    loop.reset(prompt="new task")
    loop.predict({"obs": [{"camera": 0}]})
    assert s.frame_st_id == 0
    loop.close()
    loop.close()
    with pytest.raises(RuntimeError, match="closed"):
        loop.predict({})


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
    # A default-path class patch must not be inherited by the engine server.
    module = SimpleNamespace(__file__=str(source), VA_Server=object(),
                             load_transformer=object(), _configure_model=object())
    original = dict(vars(module))
    cfg = SimpleNamespace(value=1)
    server = build_native_conditioning(module, cfg)
    assert server.original() and server.job_config.value == 2
    assert cfg.value == 1 and vars(module) == original
    with pytest.raises(RuntimeError, match="Install"):
        server.transformer.clear_cache("cache")


def test_factory_resolves_checkpoint_schedule_without_mutating_native_config():
    s, _, _, _ = build(4)
    source = SimpleNamespace(VA_CONFIGS={"libero": s.job_config})
    execution = SimpleNamespace(extra={"va_config": "libero", "frame_chunk_size": 4},
                                nfe={"video": 20, "action": 50},
                                guidance={"video": "positive_only", "action": "positive_only"})
    checkpoint = SimpleNamespace(execution=execution)
    adapter = SimpleNamespace(materialize=lambda ckpt: "/pinned/checkpoint")
    cfg = resolve_native_config(adapter, checkpoint, source, nfe={"video": 10})
    assert (cfg.num_inference_steps, cfg.action_num_inference_steps) == (10, 50)
    assert cfg.action_snr_shift == 0.05 and cfg.frame_chunk_size == 4
    assert cfg.guidance_scale == 1 and s.job_config.guidance_scale == 5
    assert s.job_config.num_inference_steps == 20
    assert cfg.wan22_pretrained_model_name_or_path == "/pinned/checkpoint"
    for override in ({"action": 0}, {"action": True}, {"kv_refresh": 1}, {"unknown": 2}):
        with pytest.raises(ValueError):
            resolve_native_config(adapter, checkpoint, source, nfe=override)
