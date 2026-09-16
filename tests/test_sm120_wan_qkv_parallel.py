"""Offline planning, lifecycle, native-surface, and worker gates for P009-A7."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from instinctflash.adapters.lingbot_va import lingbot_va_spec
from instinctflash.backends import sm120_wan_qkv_parallel
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import KNOWN_FEATURES, DeviceProfile
from instinctflash.passes.lingbot import default_passes
from instinctflash.passes.lingbot.sm120_wan_qkv_parallel import SM120WanParallelQKV
from instinctflash.planners.planner import Optimizer, PassResult, Plan, Tier
from instinctflash.runtime.sm120_qkv_parallel_install import (
    install_sm120_wan_qkv_parallel,
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
        "sm120_ring_concat_kernels",
    }
    if live:
        features.add("sm120_qkv_parallel_kernels")
    return DeviceProfile("synthetic", cap, 32 << 30, frozenset(features))


def result(profile):
    return (
        Optimizer(passes=[SM120WanParallelQKV()])
        .compile(
            lingbot_va_spec(),
            DeploymentSpec(device=profile),
            capabilities=frozenset({"backbone:wan_va"}),
        )
        .results[0]
    )


def test_plan_requires_full_chain_and_sm120():
    assert "sm120_qkv_parallel_kernels" in KNOWN_FEATURES
    assert result(device()).applies
    missing = result(device(live=False))
    assert not missing.applies and "sm120_qkv_parallel_kernels" in missing.reason
    assert not result(device((11, 0))).applies
    assert not result(None).applies


def test_default_order_a7_after_a6():
    names = [item.name for item in default_passes()]
    assert names.index("sm120_wan_ring_concat") + 1 == names.index(
        "sm120_wan_qkv_parallel"
    )


def test_library_override_requires_abi_and_cublaslt():
    old = os.environ.get(sm120_wan_qkv_parallel.LIBRARY_ENV)
    abi = sm120_wan_qkv_parallel._library_abi
    version = sm120_wan_qkv_parallel._library_cublaslt_version
    try:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / sm120_wan_qkv_parallel.LIBRARY_NAME
            os.environ[sm120_wan_qkv_parallel.LIBRARY_ENV] = str(library)
            library.touch()
            assert not sm120_wan_qkv_parallel.available()
            sm120_wan_qkv_parallel._library_abi = lambda path: 1
            sm120_wan_qkv_parallel._library_cublaslt_version = lambda path: 120804
            assert sm120_wan_qkv_parallel.available()
            assert sm120_wan_qkv_parallel.resolve_library() == library
    finally:
        sm120_wan_qkv_parallel._library_abi = abi
        sm120_wan_qkv_parallel._library_cublaslt_version = version
        if old is None:
            os.environ.pop(sm120_wan_qkv_parallel.LIBRARY_ENV, None)
        else:
            os.environ[sm120_wan_qkv_parallel.LIBRARY_ENV] = old


def test_installer_one_shot_thread_scoped():
    calls, generated = [], iter((object(), object()))
    original_kernels = sm120_wan_qkv_parallel.SM120WanParallelQKVKernels
    original_install = sm120_wan_qkv_parallel.install_wan_qkv_parallel
    try:
        sm120_wan_qkv_parallel.SM120WanParallelQKVKernels = lambda: next(generated)
        sm120_wan_qkv_parallel.install_wan_qkv_parallel = lambda transformer, kernels: (
            calls.append((transformer, kernels))
        )

        class VA:
            def __init__(self):
                self.transformer = object()

        assert install_sm120_wan_qkv_parallel(object(), VA) == [
            "sm120_wan_qkv_parallel"
        ]
        first = VA()
        assert calls == [(first.transformer, VA._ifl_sm120_wan_qkv_parallel_kernels)]
        VA()
        assert len(calls) == 1
        install_sm120_wan_qkv_parallel(object(), VA)
        second = VA()
        assert calls[1] == (second.transformer, VA._ifl_sm120_wan_qkv_parallel_kernels)
    finally:
        sm120_wan_qkv_parallel.SM120WanParallelQKVKernels = original_kernels
        sm120_wan_qkv_parallel.install_wan_qkv_parallel = original_install


def test_install_plan_requires_a6():
    from instinctflash.runtime.lingbot_install import install_plan

    applied = PassResult("sm120_wan_qkv_parallel", True, Tier.BITEXACT, "synthetic")
    try:
        install_plan(object(), type("VA", (), {}), Plan("x", [applied]))
    except RuntimeError as error:
        assert "requires sm120_wan_ring_concat" in str(error)
    else:
        raise AssertionError("A7 installed without A6")


def test_native_and_worker_surface():
    root = Path(sm120_wan_qkv_parallel.__file__).resolve().parents[2]
    source = (root / "instinctflash/native/wan_qkv_parallel_sm120.cu").read_text()
    for name in (
        "instinctflash_sm120_wan_qkv_parallel_abi_version",
        "wan_qkv_parallel_plan_register",
        "wan_qkv_parallel_bf16",
        "cudaStreamCreateWithFlags",
        "cudaEventRecord",
        "CUBLASLT_EPILOGUE_BIAS",
        "split_k = 1",
        "checked.workspaceSize != 0",
    ):
        assert name in source
    assert (
        "instinctflash_sm120_wan_qkv_parallel"
        in (root / "instinctflash/native/CMakeLists.txt").read_text()
    )
    worker = (root / "instinctflash/runtime/lingbot_worker.py").read_text()
    assert '"--sm120-wan-qkv-parallel"' in worker
    assert "--sm120-wan-qkv-parallel requires --sm120-wan-ring-concat" in worker


def test_worker_forwards_a7_flag_and_library():
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
        "sm120_wan_qkv_parallel",
    )
    plan = SimpleNamespace(applied=[SimpleNamespace(name=name) for name in names])
    adapter = LingBotVA()
    adapter.materialize = lambda checkpoint: "/tmp/composed"
    key = "IFL_SM120_QKV_PARALLEL_LIBRARY"
    old = os.environ.get(key)
    try:
        os.environ[key] = "/tmp/a7.so"
        argv, env = adapter.worker_command(
            checkpoint, plan, port=1, python="python", device=None, nfe=None
        )
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old
    assert "--sm120-wan-qkv-parallel" in argv
    assert env[key] == "/tmp/a7.so"


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
