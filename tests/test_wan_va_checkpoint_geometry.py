"""Exercise checkpoint layouts and full scheduler loops without loading DiT weights.

The stub DiT supplies constant velocity, so expected Euler outputs are independent
of the engine, while all token layout, pinning, masks and history code runs live.
This is a CPU contract test, not GPU or task-quality certification.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))
os.environ.setdefault("WAN_VA_TORCH_REF_ONLY", "1")

from flash_rt.frontends.torch.wan_va_thor import WanVaTorchFrontendThor, unpatchify
from flash_rt.models.wan_va.geometry import WanVaGeometry
from flash_rt.models.wan_va.operating_point import WanVaOperatingPoint, OperatingPointMismatch
from flash_rt.models.wan_va.step_tables import flow_match_sigmas


def config(libero):
    return SimpleNamespace(
        patch_size=(1, 2, 2), action_dim=30,
        frame_chunk_size=4 if libero else 2,
        height=128 if libero else 256, width=128 if libero else 320,
        obs_cam_keys=["a", "b"] if libero else ["a", "b", "c"],
        env_type="none" if libero else "robotwin_tshape",
        action_per_frame=4 if libero else 16,
        num_inference_steps=20 if libero else 25,
        action_num_inference_steps=50, guidance_scale=5, action_guidance_scale=1,
        snr_shift=5.0, action_snr_shift=0.05 if libero else 1.0,
    )


@pytest.mark.parametrize("libero", [False, True])
def test_checkpoint_geometry_and_output_layout(libero):
    g = WanVaGeometry.from_job_config(config(libero))
    assert (g.latent_h, g.latent_w) == ((8, 16) if libero else (24, 20))
    assert (g.video_tokens, g.action_tokens) == ((128, 16) if libero else (240, 32))
    assert g.pool_slots(30 if libero else 72) == (2160 if libero else 9792)
    # Native output packs patch position before channel, unlike input patchify.
    x = torch.arange(g.video_tokens * 192).reshape(1, g.video_tokens, 192)
    y = unpatchify(x, g.frame_chunk, g.latent_h, g.latent_w)
    for f, h, w, c in [(0, 0, 1, 7), (g.frame_chunk-1, g.latent_h-1, g.latent_w-1, 47)]:
        token = (f * (g.latent_h // 2) + h // 2) * (g.latent_w // 2) + w // 2
        assert y[0, c, f, h, w] == x[0, token, ((h % 2) * 2 + w % 2) * 48 + c]


def test_schedule_shift_is_part_of_build_identity():
    p = WanVaOperatingPoint.from_job_config(config(True))
    wrong = WanVaOperatingPoint(20, 50)
    assert p.key() != wrong.key()
    assert p.declaration()["shifts"]["action"] == 0.05
    assert p.t_values("action") != wrong.t_values("action")
    with pytest.raises(OperatingPointMismatch, match="shifts"):
        p.assert_serves(wrong)
    for shift in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            WanVaOperatingPoint(20, 50, action_shift=shift)


@pytest.mark.parametrize("libero", [False, True])
def test_cycle_shapes_pinning_and_history(libero):
    g = WanVaGeometry.from_job_config(config(libero))
    e = object.__new__(WanVaTorchFrontendThor)
    e.geometry, e.point = g, WanVaOperatingPoint.from_job_config(config(libero))
    e.device, e.B, e._prompt_set, e.profile = "cpu", 2, True, False
    e.pool_slots, e.slab_rows = g.pool_slots(30 if libero else 72), 15360
    e.action_terminal_elision = True
    e._action_unused = torch.arange(30) >= 7
    e._t_values = {s: e.point.t_values(s) for s in ("video", "action")}
    e._sigmas = {s: flow_match_sigmas(e.point.steps[s], e.point.shifts[s]) for s in e._t_values}
    e._rope_cache = {}
    e.reset_episode()
    calls = []

    def forward(stream, tokens, num_frames, spans, kind):
        count = num_frames * (g.tokens_per_frame if stream == "video" else g.action_per_frame)
        assert tokens.shape == (count, 192 if stream == "video" else 30)
        assert sum(span[1] for span in spans) == count
        assert e._rope_tables(stream, e.frame_st_id, num_frames)[0].shape == (count, 128)
        calls.append((stream, num_frames, kind))
        return torch.ones(2, count, 192 if stream == "video" else 30, dtype=torch.bfloat16)

    e._forward = forward
    v = torch.zeros(1, 48, g.frame_chunk, g.latent_h, g.latent_w, dtype=torch.bfloat16)
    a = torch.zeros(1, 30, g.frame_chunk, g.action_per_frame, 1, dtype=torch.bfloat16)
    init = torch.full_like(v[:, :, :1], 3)
    out, lat = e.infer_cycle(v, a, init)
    assert torch.equal(lat[:, :, :1], init)
    assert torch.count_nonzero(out[:, :, :1]) == 0
    assert torch.count_nonzero(out[:, 7:]) == 0
    # Literal upstream FlowMatchScheduler operation order, including BF16 rounding.
    expected = torch.zeros(1, dtype=torch.bfloat16)
    sigma = flow_match_sigmas(50, 0.05 if libero else 1.0)
    for i in range(50):
        expected = expected + torch.ones(1, dtype=torch.bfloat16) * ((sigma[i+1] if i+1 < 50 else 0) - sigma[i])
    assert torch.all(out[:, :7, 1:] == expected)
    assert len(calls) == e.point.video_steps + 1 + e.point.action_steps
    for fv in (g.frame_chunk + 1, g.frame_chunk):
        e.commit_chunk(torch.zeros(1, 48, fv, g.latent_h, g.latent_w), a)
        assert calls[-2:] == [("video", fv, "commit"), ("action", g.frame_chunk, "commit")]
        e.infer_cycle(v, a)
    frame = e.frame_st_id
    with pytest.raises(ValueError, match="video commit"):
        e.commit_chunk(v[:, :, :, :1], a)
    assert e.frame_st_id == frame
    e.reset_episode()
    assert e.frame_st_id == 0
