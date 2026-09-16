"""The wan_va (LingBot-VA) Thor engine is built FOR a declared operating point and declines any other.

Stage 2 of the VA engine (iwm_distill/thor_va_engine/stage2_report.md; h2_thor_realtime_design.md
§5): the build takes a ``WanVaOperatingPoint`` — per-stream schedule grid, per-stream guidance
(mode, scale), hence CFG batching — and derives its step tables, KV slab stream count and forward
count from it. The same fail-closed rule as 9bf2337's ``engine_operating_point_problem`` is
mirrored on the serving side (serving/ cannot import instinctflash); these tests pin that the
two rules agree on every request, that the two Stage-2 points derive DIFFERENT tables and
forward counts (the proof the registry asks for before a build-parameterized frontend may be
declared), and that the runtime registry bridge produces the ordinary record for a build.

No GPU, no torch weights: the frontend module is imported in torch_ref_only mode only where a
checkpoint is present (skipped otherwise).
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "serving")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("WAN_VA_TORCH_REF_ONLY", "1")

from flash_rt.models.wan_va import operating_point as op  # noqa: E402
from flash_rt.models.wan_va.pipeline_thor import expand_spans, launches_per_forward  # noqa: E402
from instinctflash.runtime import engine_backend as eb  # noqa: E402

P5, P1 = op.POINT_2V4A_W5, op.POINT_2V2A_W1
CKPT = Path(os.environ.get(
    "LINGBOT_CKPT", "/home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin")) / "transformer"


def test_the_two_stage2_points_derive_different_tables_and_forward_counts():
    assert P5.cfg_batch == 2 and P1.cfg_batch == 1
    assert P5.forwards_per_cycle == 10 and P1.forwards_per_cycle == 8
    # P010 (d7e6103): the action pred-commit forward is dead compute — served counts 9 / 7
    assert P5.forwards_per_cycle_served() == 9 and P1.forwards_per_cycle_served() == 7
    assert P5.forwards_per_cycle_served(False) == 10
    assert P5.t_values("action") == [1000.0, 750.0, 500.0, 250.0, 0.0]
    assert P1.t_values("action") == [1000.0, 500.0, 0.0]
    assert P5.t_values("video") == P1.t_values("video") == [1000.0, 833.3333129882812, 0.0]
    assert P1.euler("action") == [-0.5, -0.5] and P5.euler("action") == [-0.25] * 4
    assert P5.combine_scale("video") == 5.0 and P1.combine_scale("video") == 1.0
    # the H2 student's tuple (2V/1A@w1) is expressible: 7 forwards, action grid {1000} + commit row
    s1 = op.WanVaOperatingPoint(2, 1, ("positive_only", 1.0), ("positive_only", 1.0))
    assert s1.forwards_per_cycle == 7 and s1.forwards_per_cycle_served() == 6
    assert s1.t_values("action") == [1000.0, 0.0]
    assert P5.key() != P1.key() != s1.key()
    assert op.self_check()


def test_serving_side_rule_and_runtime_rule_agree_on_every_request():
    """One rule, two surfaces: point.problem(...) (serving) vs engine_operating_point_problem
    (runtime, on the record the bridge builds) must accept/decline identically."""
    step_grid = [1, 2, 4]
    guid_grid = [("cfg", 5.0), ("cfg", 3.0), ("cfg", 1.0), ("positive_only", 1.0), ("none", 1.0)]
    for point in (P5, P1):
        rec = eb.wan_va_baked_operating_point(point.declaration())
        assert rec.steps_parameterized is False and rec.backbone == "wan_va"
        for v, a, gv, ga in itertools.product(step_grid, step_grid, guid_grid, guid_grid):
            req_steps = {"video": v, "action": a}
            req_guid = {"video": gv, "action": ga}
            serving = point.problem(req_steps, req_guid)
            runtime = eb.engine_operating_point_problem(rec, req_steps, req_guid)
            assert (serving is None) == (runtime is None), (point.key(), req_steps, req_guid,
                                                             serving, runtime)
        # the build's own point is accepted; cfg@1 == positive_only == none is accepted too
        assert point.problem(point.steps, point.guidance) is None
        assert eb.engine_operating_point_problem(rec, point.steps, point.guidance) is None
        # a stream that cannot be read declines on both surfaces
        assert point.problem({"video": point.video_steps}, point.guidance) is not None
        assert eb.engine_operating_point_problem(rec, {"video": point.video_steps},
                                                 point.guidance) is not None
    # and the two points decline each other
    assert P5.problem(P1.steps, P1.guidance) and P1.problem(P5.steps, P5.guidance)
    with pytest.raises(op.OperatingPointMismatch):
        P5.assert_serves(P1)


def test_worker_flags_and_job_config_resolve_to_the_same_points():
    assert op.WanVaOperatingPoint.from_worker_flags("2,4") == P5
    assert op.WanVaOperatingPoint.from_worker_flags("2,4", "video=5") == P5
    assert op.WanVaOperatingPoint.from_worker_flags("2,2", "video=positive_only") == P1
    assert op.WanVaOperatingPoint.from_worker_flags(
        "2,2", "video=positive_only,action=positive_only") == P1
    assert op.WanVaOperatingPoint.from_worker_flags("2,2", "video=1") == P1

    class Cfg:  # the resolved stock config after serve_variant applied its flags
        num_inference_steps, action_num_inference_steps = 2, 4
        guidance_scale, action_guidance_scale = 5, 1
        snr_shift, action_snr_shift = 5.0, 1.0
    assert op.WanVaOperatingPoint.from_job_config(Cfg) == P5
    Cfg.action_num_inference_steps, Cfg.guidance_scale = 2, 1.0
    assert op.WanVaOperatingPoint.from_job_config(Cfg) == P1


def test_registry_bridge_gives_a_built_wan_va_engine_the_ordinary_record_and_none_unbuilt():
    assert "wan_va" in eb.BUILD_DECLARED_BACKBONES
    assert eb.engine_baked_operating_point("wan_va") is None, "an unbuilt wan_va engine has no point"
    assert all(cap.backbone != "wan_va" for cap in eb.ENGINE_BAKED_OPERATING_POINTS), \
        "wan_va is build-declared, never a static literal entry"
    rec = eb.engine_baked_operating_point("wan_va", build=P1.declaration())
    assert rec.steps == {"video": 2, "action": 2}
    assert rec.guidance == {"video": ("positive_only", 1.0), "action": ("positive_only", 1.0)}
    assert "2 steps" in rec.served_point() and "positive_only@1" in rec.served_point()
    assert (ROOT / rec.frontend).is_file()
    # the other engines are untouched by the bridge
    assert eb.engine_baked_operating_point("pi05").steps == {"action": eb.PI05_ENGINE_ACTION_STEPS}
    assert eb.engine_baked_operating_point("wam") is None


def test_span_expansion_and_launch_budget():
    # one span covers all B streams in one call; cycle-0 two-span forwards expand per stream
    assert expand_spans(((0, 240, 1),), 240, 2) == ((0, 480, 1),)
    assert expand_spans(((0, 120, 2), (120, 120, 0)), 240, 2) == (
        (0, 120, 2), (120, 120, 0), (240, 120, 2), (360, 120, 0))
    # the eager launch count per forward (M2c input) is ~1k; the fallback gate costs 60 more
    assert 800 < launches_per_forward(1, 1, False, True) < launches_per_forward(2, 1, True, False) < 1400
    assert launches_per_forward(2, 1, True, False) - launches_per_forward(2, 1, True, True) == 60


@pytest.mark.skipif(not (CKPT / "diffusion_pytorch_model.safetensors.index.json").exists(),
                    reason="LingBot-VA checkpoint not present on this host")
def test_frontend_builds_for_both_points_and_declines_the_other_torch_ref_only():
    from flash_rt.frontends.torch import wan_va_thor as W
    assert W.fvk is None, "this test must never touch a GPU"
    for point, other in ((P5, P1), (P1, P5)):
        fr = W.WanVaTorchFrontendThor(str(CKPT), point=point, precision="fp16")
        d = fr.declaration()
        assert d["steps"] == point.steps and d["cfg_batch"] == point.cfg_batch
        assert tuple(fr._tables["action"]["mod"].shape) == (len(point.t_values("action")), 30, 6, 3072)
        fr.assert_point(point)
        with pytest.raises(op.OperatingPointMismatch):
            fr.assert_point(other)
        rec = eb.engine_baked_operating_point("wan_va", build=d)
        assert rec.steps == point.steps
        del fr
