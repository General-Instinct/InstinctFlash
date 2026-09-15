"""Offline, lifecycle, integration, and worker gates for P009-A6."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from instinctflash.adapters.lingbot_va import lingbot_va_spec
from instinctflash.backends import sm120_wan_ring_concat
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import KNOWN_FEATURES, DeviceProfile
from instinctflash.passes.lingbot import default_passes
from instinctflash.passes.lingbot.sm120_wan_ring_concat import SM120WanRingConcat
from instinctflash.planners.planner import Optimizer, PassResult, Plan, Tier
from instinctflash.runtime.sm120_ring_concat_install import (
    install_sm120_wan_ring_concat,
)


def device(cap=(12, 0), live=True):
    features = {
        "cuda",
        "cublas",
        "sm120_kernels",
        "sm120_stage2_kernels",
        "sm120_stage3_kernels",
        "sm120_qk_rope_kernels",
        "sm120_gemm_kernels",
    }
    if live:
        features.add("sm120_ring_concat_kernels")
    return DeviceProfile(
        name="synthetic",
        capability=cap,
        total_memory=32 << 30,
        features=frozenset(features),
    )


def result(dev):
    return (
        Optimizer(passes=[SM120WanRingConcat()])
        .compile(
            lingbot_va_spec(),
            DeploymentSpec(device=dev),
            capabilities=frozenset({"backbone:wan_va"}),
        )
        .results[0]
    )


def test_plan_requires_full_chain_and_sm120():
    assert "sm120_ring_concat_kernels" in KNOWN_FEATURES
    assert result(device()).applies
    missing = result(device(live=False))
    assert not missing.applies and "sm120_ring_concat_kernels" in missing.reason
    assert not result(device((11, 0))).applies
    assert not result(None).applies


def test_default_order_a6_after_a5():
    names = [pass_.name for pass_ in default_passes()]
    assert names.index("sm120_wan_gemm") + 1 == names.index("sm120_wan_ring_concat")


def test_library_override_requires_a6_abi():
    old = os.environ.get(sm120_wan_ring_concat.LIBRARY_ENV)
    probe = sm120_wan_ring_concat._library_abi
    try:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / sm120_wan_ring_concat.LIBRARY_NAME
            os.environ[sm120_wan_ring_concat.LIBRARY_ENV] = str(library)
            assert not sm120_wan_ring_concat.available()
            library.touch()
            assert not sm120_wan_ring_concat.available()
            sm120_wan_ring_concat._library_abi = lambda path: (
                sm120_wan_ring_concat.ABI_VERSION
            )
            assert sm120_wan_ring_concat.available()
            assert sm120_wan_ring_concat.resolve_library() == library
    finally:
        sm120_wan_ring_concat._library_abi = probe
        if old is None:
            os.environ.pop(sm120_wan_ring_concat.LIBRARY_ENV, None)
        else:
            os.environ[sm120_wan_ring_concat.LIBRARY_ENV] = old


def test_backend_install_requires_a5_and_all_a4_sites():
    class FakeKernels:
        def concat(self, *args, **kwargs):
            return args

    transformer = SimpleNamespace(
        blocks=[SimpleNamespace(attn1=SimpleNamespace()) for _ in range(30)]
    )
    try:
        sm120_wan_ring_concat.install_wan_ring_concat(transformer, FakeKernels())
    except RuntimeError as error:
        assert "A1-A5" in str(error)
    else:
        raise AssertionError("A6 installed without A5")

    transformer._ifl_wan_gemm_kernels = object()
    for block in transformer.blocks:
        block.attn1._ifl_wan_qk_rope_installed = True
    kernels = FakeKernels()
    installed = sm120_wan_ring_concat.install_wan_ring_concat(transformer, kernels)
    assert installed is kernels
    assert len(transformer._ifl_wan_ring_concat_sites) == 30
    assert all(
        block.attn1._iwm_ring_concat == kernels.concat for block in transformer.blocks
    )


def test_validate_rejects_non_cuda_and_non_tensor_inputs():
    kernels = object.__new__(sm120_wan_ring_concat.SM120WanRingConcatKernels)
    try:
        kernels._validate(object(), object(), start=9000, count=1000, total=9792)
    except TypeError as error:
        assert "must be Tensor" in str(error)
    else:
        raise AssertionError("A6 accepted non-tensors")

    cpu = torch.empty((1,), dtype=torch.bfloat16)
    try:
        kernels._validate(cpu, cpu.clone(), start=9000, count=1000, total=9792)
    except ValueError as error:
        assert "must be CUDA" in str(error)
    else:
        raise AssertionError("A6 accepted CPU tensors")


def test_installer_one_shot_thread_scoped():
    calls = []
    generated = iter((object(), object()))
    original_kernels = sm120_wan_ring_concat.SM120WanRingConcatKernels
    original_install = sm120_wan_ring_concat.install_wan_ring_concat
    try:
        sm120_wan_ring_concat.SM120WanRingConcatKernels = lambda: next(generated)
        sm120_wan_ring_concat.install_wan_ring_concat = lambda transformer, kernels: (
            calls.append((transformer, kernels))
        )

        class VA:
            def __init__(self):
                self.transformer = object()

        assert install_sm120_wan_ring_concat(object(), VA) == ["sm120_wan_ring_concat"]
        first = VA()
        assert calls == [(first.transformer, VA._ifl_sm120_wan_ring_concat_kernels)]
        VA()
        assert len(calls) == 1
        install_sm120_wan_ring_concat(object(), VA)
        second = VA()
        assert calls[1] == (second.transformer, VA._ifl_sm120_wan_ring_concat_kernels)
    finally:
        sm120_wan_ring_concat.SM120WanRingConcatKernels = original_kernels
        sm120_wan_ring_concat.install_wan_ring_concat = original_install


def test_install_plan_requires_a5():
    from instinctflash.runtime.lingbot_install import install_plan

    applied = PassResult("sm120_wan_ring_concat", True, Tier.BITEXACT, "synthetic")
    try:
        install_plan(object(), type("VA", (), {}), Plan("x", [applied]))
    except RuntimeError as error:
        assert "requires sm120_wan_gemm" in str(error)
    else:
        raise AssertionError("A6 installed without A5")


def test_native_and_worker_surface():
    root = Path(sm120_wan_ring_concat.__file__).resolve().parents[2]
    source = (root / "instinctflash/native/wan_ring_concat_sm120.cu").read_text()
    for name in (
        "instinctflash_sm120_wan_ring_concat_abi_version",
        "ring_concat_bf16_vec8",
        "wan_ring_concat_bf16",
        "uint4",
        "out_key[index] = key[source]",
        "out_value[index] = value[source]",
    ):
        assert name in source
    cmake = (root / "instinctflash/native/CMakeLists.txt").read_text()
    assert "instinctflash_sm120_wan_ring_concat" in cmake
    worker = (root / "instinctflash/runtime/lingbot_worker.py").read_text()
    assert '"--sm120-wan-ring-concat"' in worker
    assert "--sm120-wan-ring-concat requires --sm120-wan-gemm" in worker
    for implementation in (
        root / "instinctflash/passes/lingbot/ring_kv.py",
        root / "instinctflash/backends/sm120_wan_qk_rope.py",
    ):
        assert "_iwm_ring_concat" in implementation.read_text()


def test_worker_forwards_full_chain():
    from instinctflash.adapters.lingbot_va import LingBotVA

    execution = SimpleNamespace(
        nfe={"video": 2, "action": 4},
        guidance={},
        extra={
            "base_weights": "/tmp/base",
            "obs_cam_keys": [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ],
            "height": 256,
            "width": 320,
            "env_type": "robotwin_tshape",
        },
    )
    checkpoint = SimpleNamespace(path="/tmp/pkg", model_id="x", execution=execution)

    names = (
        "sm120_gated_residual",
        "sm120_wan_stage2",
        "sm120_wan_stage3",
        "sm120_wan_qk_rope",
        "sm120_wan_gemm",
        "sm120_wan_ring_concat",
    )

    plan = SimpleNamespace(
        applied=[SimpleNamespace(name=name) for name in names],
    )

    adapter = LingBotVA()
    adapter.materialize = lambda checkpoint: "/tmp/composed"
    keys = (
        "IFL_SM120_KERNEL_LIBRARY",
        "IFL_SM120_STAGE2_LIBRARY",
        "IFL_SM120_STAGE3_LIBRARY",
        "IFL_SM120_QK_ROPE_LIBRARY",
        "IFL_SM120_GEMM_LIBRARY",
        "IFL_SM120_RING_CONCAT_LIBRARY",
    )
    old = {key: os.environ.get(key) for key in keys}
    try:
        for index, key in enumerate(keys, 1):
            os.environ[key] = f"/tmp/a{index}.so"
        argv, env = adapter.worker_command(
            checkpoint, plan, port=1, python="python", device=None, nfe=None
        )
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    for flag in (
        "--sm120-gated-residual",
        "--sm120-wan-stage2",
        "--sm120-wan-stage3",
        "--sm120-wan-qk-rope",
        "--sm120-wan-gemm",
        "--sm120-wan-ring-concat",
    ):
        assert flag in argv
    assert env["IFL_SM120_RING_CONCAT_LIBRARY"] == "/tmp/a6.so"


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
