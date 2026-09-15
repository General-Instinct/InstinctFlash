#!/usr/bin/env python3
"""Offline, lifecycle, and worker gates for P009-A3."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from instinctflash.adapters.lingbot_va import lingbot_va_spec
from instinctflash.backends import sm120_wan_stage3
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import DeviceProfile, KNOWN_FEATURES
from instinctflash.passes.lingbot import default_passes
from instinctflash.passes.lingbot.sm120_wan_stage3 import SM120WanStage3
from instinctflash.planners.planner import Optimizer, PassResult, Plan, Tier
from instinctflash.runtime.sm120_stage3_install import install_sm120_wan_stage3


def _device(capability=(12, 0), stage3=True):
    features = {"cuda", "sm120_kernels", "sm120_stage2_kernels"}
    if stage3:
        features.add("sm120_stage3_kernels")
    return DeviceProfile(
        name="synthetic", capability=capability, total_memory=32 << 30,
        features=frozenset(features),
    )


def _result(device):
    return Optimizer(passes=[SM120WanStage3()]).compile(
        lingbot_va_spec(), DeploymentSpec(device=device),
        capabilities=frozenset({"backbone:wan_va"}),
    ).results[0]


def test_plan_requires_full_native_chain_and_exact_sm120():
    assert "sm120_stage3_kernels" in KNOWN_FEATURES
    live = _result(_device())
    assert live.applies and live.tier is Tier.BITEXACT
    missing = _result(_device(stage3=False))
    assert not missing.applies and "sm120_stage3_kernels" in missing.reason
    wrong = _result(_device((11, 0)))
    assert not wrong.applies and "sm_120" in wrong.reason
    unprobed = _result(None)
    assert not unprobed.applies and "unprobed" in unprobed.reason


def test_default_order_keeps_a3_immediately_after_a2():
    names = [item.name for item in default_passes()]
    assert names.index("sm120_wan_stage2") + 1 == names.index("sm120_wan_stage3")


def test_library_override_requires_independent_a3_abi():
    old = os.environ.get(sm120_wan_stage3.LIBRARY_ENV)
    old_probe = sm120_wan_stage3._library_abi
    try:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / sm120_wan_stage3.LIBRARY_NAME
            os.environ[sm120_wan_stage3.LIBRARY_ENV] = str(library)
            assert not sm120_wan_stage3.available()
            library.touch()
            assert not sm120_wan_stage3.available()
            sm120_wan_stage3._library_abi = lambda path: sm120_wan_stage3.ABI_VERSION
            assert sm120_wan_stage3.available()
            assert sm120_wan_stage3.resolve_library() == library
    finally:
        sm120_wan_stage3._library_abi = old_probe
        if old is None:
            os.environ.pop(sm120_wan_stage3.LIBRARY_ENV, None)
        else:
            os.environ[sm120_wan_stage3.LIBRARY_ENV] = old


def test_installer_activation_is_one_shot_and_thread_scoped():
    calls = []
    kernels = iter((object(), object()))
    old_kernel = sm120_wan_stage3.SM120WanStage3Kernels
    old_install = sm120_wan_stage3.install_wan_stage3
    try:
        sm120_wan_stage3.SM120WanStage3Kernels = lambda: next(kernels)
        sm120_wan_stage3.install_wan_stage3 = (
            lambda transformer, kernel: calls.append((transformer, kernel)))

        class VA:
            def __init__(self):
                self.transformer = object()

        assert install_sm120_wan_stage3(object(), VA) == ["sm120_wan_stage3"]
        enabled = VA()
        assert calls == [(enabled.transformer, VA._ifl_sm120_wan_stage3_kernels)]
        VA()
        assert len(calls) == 1
        install_sm120_wan_stage3(object(), VA)
        reenabled = VA()
        assert len(calls) == 2
        assert calls[1] == (reenabled.transformer, VA._ifl_sm120_wan_stage3_kernels)
    finally:
        sm120_wan_stage3.SM120WanStage3Kernels = old_kernel
        sm120_wan_stage3.install_wan_stage3 = old_install


def test_install_plan_refuses_a3_without_a2_before_mutating():
    from instinctflash.runtime.lingbot_install import install_plan

    result = PassResult("sm120_wan_stage3", True, Tier.BITEXACT, "synthetic")
    try:
        install_plan(object(), type("VA", (), {}), Plan("synthetic", [result]))
    except RuntimeError as error:
        assert "requires sm120_wan_stage2" in str(error)
    else:
        raise AssertionError("P009-A3 must not install without P009-A2")


def test_native_source_cmake_and_worker_surface_are_explicit():
    root = Path(sm120_wan_stage3.__file__).resolve().parents[2]
    source = (root / "instinctflash" / "native" / "wan_stage3_sm120.cu").read_text()
    for needle in (
        "instinctflash_sm120_wan_stage3_abi_version",
        "wan_norm1_ada_layer_norm_bf16",
        "welford_online", "welford_combine", "__fsub_rn", "__fmul_rn",
        "__fadd_rn", "__float2bfloat16_rn", "dim3(32, 4, 1)",
    ):
        assert needle in source
    cmake = (root / "instinctflash" / "native" / "CMakeLists.txt").read_text()
    assert "instinctflash_sm120_wan_stage3" in cmake
    worker = (root / "instinctflash" / "runtime" / "lingbot_worker.py").read_text()
    assert '"--sm120-wan-stage3"' in worker
    assert "--sm120-wan-stage3 requires --sm120-wan-stage2" in worker


def test_worker_forwards_full_native_chain_and_libraries():
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
            "height": 256, "width": 320, "env_type": "robotwin_tshape",
        }

    class Checkpoint:
        path = "/tmp/synthetic-package"
        model_id = "synthetic/wan"
        execution = Execution()

    class PlanWithA3:
        applied = [
            type("Result", (), {"name": name})()
            for name in ("sm120_gated_residual", "sm120_wan_stage2", "sm120_wan_stage3")
        ]

    adapter = LingBotVA()
    adapter.materialize = lambda checkpoint: "/tmp/synthetic-composed"
    keys = (
        "IFL_SM120_KERNEL_LIBRARY",
        "IFL_SM120_STAGE2_LIBRARY",
        "IFL_SM120_STAGE3_LIBRARY",
    )
    old = {key: os.environ.get(key) for key in keys}
    try:
        for index, key in enumerate(keys, 1):
            os.environ[key] = f"/tmp/a{index}.so"
        argv, env = adapter.worker_command(
            Checkpoint(), PlanWithA3(), port=1234, python="python3", device=None, nfe=None)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    joined = " ".join(argv)
    for flag in ("--sm120-gated-residual", "--sm120-wan-stage2", "--sm120-wan-stage3"):
        assert flag in joined
    assert env["IFL_SM120_KERNEL_LIBRARY"] == "/tmp/a1.so"
    assert env["IFL_SM120_STAGE2_LIBRARY"] == "/tmp/a2.so"
    assert env["IFL_SM120_STAGE3_LIBRARY"] == "/tmp/a3.so"


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))
