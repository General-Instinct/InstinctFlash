#!/usr/bin/env python3
"""Core/offline gates for P009-A1; requires neither torch nor a CUDA library."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from instinctflash.adapters.lingbot_va import lingbot_va_spec
from instinctflash.backends import sm120_residual
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import DeviceProfile
from instinctflash.passes.lingbot.sm120_gated_residual import SM120GatedResidual
from instinctflash.planners.planner import Optimizer, Tier
from instinctflash.runtime.sm120_install import install_sm120_gated_residual


def _device(capability=(12, 0), native=True):
    features = {"cuda"}
    if native:
        features.add("sm120_kernels")
    return DeviceProfile(
        name="synthetic", capability=capability, total_memory=32 << 30,
        features=frozenset(features),
    )


def _result(device):
    return Optimizer(passes=[SM120GatedResidual()]).compile(
        lingbot_va_spec(), DeploymentSpec(device=device),
        capabilities=frozenset({"backbone:wan_va"}),
    ).results[0]


def test_plan_requires_sm120_and_built_library():
    live = _result(_device())
    assert live.applies and live.tier is Tier.BITEXACT
    missing = _result(_device(native=False))
    assert not missing.applies and "sm120_kernels" in missing.reason
    wrong_arch = _result(_device((11, 0)))
    assert not wrong_arch.applies and "sm_120" in wrong_arch.reason


def test_deviceless_plan_declines_rather_than_staying_undecided():
    """contract.py doctrine: an unprobed target must decline, not defer to install time.

    applies=True under the planner's APPLICABILITY UNCHECKED annotation rode into
    install_plan, which then crashed constructing the kernel on every non-5090 machine --
    the branch's own tests/test_runtime.py failed off-5090 until this declined.
    """
    from instinctflash.passes.lingbot.sm120_wan_stage2 import SM120WanStage2

    unprobed = _result(None)
    assert not unprobed.applies and "unprobed" in unprobed.reason
    stage2 = Optimizer(passes=[SM120WanStage2()]).compile(
        lingbot_va_spec(), DeploymentSpec(device=None),
        capabilities=frozenset({"backbone:wan_va"}),
    ).results[0]
    assert not stage2.applies and "unprobed" in stage2.reason

def test_compile_probes_the_explicit_target_device():
    from instinctflash.descriptors.checkpoint import ExecutionDeclaration
    from instinctflash.descriptors.package import Checkpoint
    from instinctflash.runtime.facade import _compile_declaration

    seen = []
    old_probe = DeviceProfile.probe
    DeviceProfile.probe = staticmethod(
        lambda device=None: seen.append(device) or _device()
    )
    try:
        decl = ExecutionDeclaration(
            model_id="synthetic", backbone="wan_va", servable=True,
        )
        _compile_declaration(
            Checkpoint("nowhere", decl), device="cuda:7", probe_device=True,
        )
    finally:
        DeviceProfile.probe = old_probe
    assert seen == ["cuda:7"]



def test_library_override_controls_capability():
    old = os.environ.get(sm120_residual.LIBRARY_ENV)
    old_probe = sm120_residual._library_abi
    try:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / sm120_residual.LIBRARY_NAME
            os.environ[sm120_residual.LIBRARY_ENV] = str(library)
            assert not sm120_residual.available()
            library.touch()
            # Mere existence is not a capability: zero-byte/wrong-ABI files stay unavailable.
            assert not sm120_residual.available()
            sm120_residual._library_abi = lambda path: sm120_residual.ABI_VERSION
            assert sm120_residual.available()
            assert sm120_residual.resolve_library() == library
    finally:
        sm120_residual._library_abi = old_probe
        if old is None:
            os.environ.pop(sm120_residual.LIBRARY_ENV, None)
        else:
            os.environ[sm120_residual.LIBRARY_ENV] = old


def test_installer_patches_after_server_construction():
    calls = []
    fake_kernels = iter((object(), object()))
    old_kernel = sm120_residual.SM120GatedResidualKernel
    old_install = sm120_residual.install_wan_blocks
    try:
        sm120_residual.SM120GatedResidualKernel = lambda: next(fake_kernels)
        sm120_residual.install_wan_blocks = (
            lambda transformer, kernel: calls.append((transformer, kernel))
        )

        class VA:
            def __init__(self):
                self.transformer = object()

        # Enabled plan: the next construction consumes exactly one pending kernel.
        assert install_sm120_gated_residual(object(), VA) == ["sm120_gated_residual"]
        enabled = VA()
        assert calls == [(enabled.transformer, VA._ifl_sm120_gated_residual_kernel)]

        # Excluded later plan: the permanent wrapper has no pending token and installs nothing.
        excluded = VA()
        assert calls == [(enabled.transformer, VA._ifl_sm120_gated_residual_kernel)]

        # A later enabled plan arms a fresh kernel for exactly its next construction.
        assert install_sm120_gated_residual(object(), VA) == ["sm120_gated_residual"]
        reenabled = VA()
        assert len(calls) == 2
        assert calls[1] == (reenabled.transformer, VA._ifl_sm120_gated_residual_kernel)
    finally:
        sm120_residual.SM120GatedResidualKernel = old_kernel
        sm120_residual.install_wan_blocks = old_install


def test_cuda_source_forbids_fma_contraction():
    source = Path(sm120_residual.__file__).resolve().parents[1] / "native" / "wan_residual_sm120.cu"
    text = source.read_text()
    assert "__fmul_rn" in text
    assert "__fadd_rn" in text
    assert "__float2bfloat16_rn" in text


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
