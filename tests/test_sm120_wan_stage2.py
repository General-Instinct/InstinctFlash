#!/usr/bin/env python3
"""Offline and lifecycle gates for P009-A2; GPU numerics have a separate locked certificate."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from instinctflash.adapters.lingbot_va import lingbot_va_spec
from instinctflash.backends import sm120_wan_stage2
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import DeviceProfile, KNOWN_FEATURES
from instinctflash.passes.lingbot import default_passes
from instinctflash.passes.lingbot.sm120_wan_stage2 import SM120WanStage2
from instinctflash.planners.planner import Optimizer, PassResult, Plan, Tier
from instinctflash.runtime.sm120_stage2_install import install_sm120_wan_stage2


def _device(capability=(12, 0), a1=True, stage2=True):
    features = {"cuda"}
    if a1:
        features.add("sm120_kernels")
    if stage2:
        features.add("sm120_stage2_kernels")
    return DeviceProfile(
        name="synthetic", capability=capability, total_memory=32 << 30,
        features=frozenset(features),
    )


def _result(device):
    return Optimizer(passes=[SM120WanStage2()]).compile(
        lingbot_va_spec(), DeploymentSpec(device=device),
        capabilities=frozenset({"backbone:wan_va"}),
    ).results[0]


def test_plan_requires_a1_stage2_abi_and_exact_sm120():
    assert "sm120_stage2_kernels" in KNOWN_FEATURES
    assert _result(_device()).applies
    no_a1 = _result(_device(a1=False))
    assert not no_a1.applies and "sm120_kernels" in no_a1.reason
    no_stage2 = _result(_device(stage2=False))
    assert not no_stage2.applies and "sm120_stage2_kernels" in no_stage2.reason
    wrong_arch = _result(_device((11, 0)))
    assert not wrong_arch.applies and "sm_120" in wrong_arch.reason


def test_default_order_keeps_stage2_after_a1():
    names = [item.name for item in default_passes()]
    assert names.index("sm120_gated_residual") + 1 == names.index("sm120_wan_stage2")


def test_library_override_requires_the_independent_abi():
    old = os.environ.get(sm120_wan_stage2.LIBRARY_ENV)
    old_probe = sm120_wan_stage2._library_abi
    try:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / sm120_wan_stage2.LIBRARY_NAME
            os.environ[sm120_wan_stage2.LIBRARY_ENV] = str(library)
            assert not sm120_wan_stage2.available()
            library.touch()
            assert not sm120_wan_stage2.available()
            sm120_wan_stage2._library_abi = lambda path: sm120_wan_stage2.ABI_VERSION
            assert sm120_wan_stage2.available()
            assert sm120_wan_stage2.resolve_library() == library
    finally:
        sm120_wan_stage2._library_abi = old_probe
        if old is None:
            os.environ.pop(sm120_wan_stage2.LIBRARY_ENV, None)
        else:
            os.environ[sm120_wan_stage2.LIBRARY_ENV] = old


def test_installer_activation_is_one_shot_and_thread_scoped():
    calls = []
    kernels = iter((object(), object()))
    old_kernel = sm120_wan_stage2.SM120WanStage2Kernels
    old_install = sm120_wan_stage2.install_wan_stage2
    try:
        sm120_wan_stage2.SM120WanStage2Kernels = lambda: next(kernels)
        sm120_wan_stage2.install_wan_stage2 = (
            lambda transformer, kernel: calls.append((transformer, kernel))
        )

        class VA:
            def __init__(self):
                self.transformer = object()

        assert install_sm120_wan_stage2(object(), VA) == ["sm120_wan_stage2"]
        enabled = VA()
        assert calls == [(enabled.transformer, VA._ifl_sm120_wan_stage2_kernels)]

        excluded = VA()
        assert calls == [(enabled.transformer, VA._ifl_sm120_wan_stage2_kernels)]

        assert install_sm120_wan_stage2(object(), VA) == ["sm120_wan_stage2"]
        reenabled = VA()
        assert len(calls) == 2
        assert calls[1] == (reenabled.transformer, VA._ifl_sm120_wan_stage2_kernels)
    finally:
        sm120_wan_stage2.SM120WanStage2Kernels = old_kernel
        sm120_wan_stage2.install_wan_stage2 = old_install


def test_install_plan_refuses_stage2_without_a1_before_mutating():
    try:
        from instinctflash.runtime.lingbot_install import install_plan
    except ImportError:
        return

    result = PassResult("sm120_wan_stage2", True, Tier.BITEXACT, "synthetic")
    try:
        install_plan(object(), type("VA", (), {}), Plan("synthetic", [result]))
    except RuntimeError as error:
        assert "requires sm120_gated_residual" in str(error)
    else:
        raise AssertionError("P009-A2 must not install without its P009-A1 dependency")


def test_native_source_and_worker_surface_are_explicit():
    root = Path(sm120_wan_stage2.__file__).resolve().parents[2]
    source = (root / "instinctflash" / "native" / "wan_stage2_sm120.cu").read_text()
    for needle in (
        "instinctflash_sm120_wan_stage2_abi_version", "welford_online", "welford_combine",
        "__fmul_rn", "__fadd_rn", "__float2bfloat16_rn", "dim3(32, 4, 1)",
    ):
        assert needle in source
    cmake = (root / "instinctflash" / "native" / "CMakeLists.txt").read_text()
    assert "instinctflash_sm120_wan_stage2" in cmake
    worker = (root / "instinctflash" / "runtime" / "lingbot_worker.py").read_text()
    assert '"--sm120-wan-stage2"' in worker
    assert "--sm120-wan-stage2 requires --sm120-gated-residual" in worker


def test_worker_forwards_both_native_flags_and_libraries():
    from instinctflash.adapters.lingbot_va import LingBotVA

    class Execution:
        nfe = {"video": 2, "action": 4}
        guidance = {}
        extra = {
            "base_weights": "/tmp/synthetic-base",
            "obs_cam_keys": [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ],
            "height": 256,
            "width": 320,
            "env_type": "robotwin_tshape",
        }

    class Checkpoint:
        path = "/tmp/synthetic-package"
        model_id = "synthetic/wan"
        execution = Execution()

    class PlanWithStage2:
        applied = [
            type("Result", (), {"name": name})()
            for name in ("sm120_gated_residual", "sm120_wan_stage2")
        ]

    adapter = LingBotVA()
    adapter.materialize = lambda checkpoint: "/tmp/synthetic-composed"
    old_a1 = os.environ.get("IFL_SM120_KERNEL_LIBRARY")
    old_a2 = os.environ.get("IFL_SM120_STAGE2_LIBRARY")
    try:
        os.environ["IFL_SM120_KERNEL_LIBRARY"] = "/tmp/a1.so"
        os.environ["IFL_SM120_STAGE2_LIBRARY"] = "/tmp/a2.so"
        argv, env = adapter.worker_command(
            Checkpoint(), PlanWithStage2(), port=1234, python="python3", device=None, nfe=None,
        )
    finally:
        if old_a1 is None:
            os.environ.pop("IFL_SM120_KERNEL_LIBRARY", None)
        else:
            os.environ["IFL_SM120_KERNEL_LIBRARY"] = old_a1
        if old_a2 is None:
            os.environ.pop("IFL_SM120_STAGE2_LIBRARY", None)
        else:
            os.environ["IFL_SM120_STAGE2_LIBRARY"] = old_a2

    joined = " ".join(argv)
    assert "--sm120-gated-residual" in joined
    assert "--sm120-wan-stage2" in joined
    assert env["IFL_SM120_KERNEL_LIBRARY"] == "/tmp/a1.so"
    assert env["IFL_SM120_STAGE2_LIBRARY"] == "/tmp/a2.so"


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
