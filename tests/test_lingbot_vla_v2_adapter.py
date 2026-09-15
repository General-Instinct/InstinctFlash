from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "examples" / "lingbot_vla_v2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from instinctflash.adapters.base import KVLifetime
from instinctflash.descriptors.known import lookup
from instinctflash.descriptors.package import _declared_view, validate_package
from lingbot_vla_v2_iwm.adapter import (
    LingBotVLAV2Adapter,
    _LingBotVLAV2Loop,
    _env_flag,
    _resolve_model_path,
    _configure_thor_ptxas,
)


def test_thor_compiler_selection_preserves_user_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        compiler = Path(tmp)/'bin/ptxas'
        compiler.parent.mkdir()
        compiler.touch()
        env = {'CUDA_HOME':tmp,'TRITON_PTXAS_PATH':'/user/ptxas'}
        with patch.dict(os.environ,env,clear=True), patch('subprocess.run') as run:
            run.return_value = SimpleNamespace(returncode=0,stdout='sm_110a',stderr='')
            result = _configure_thor_ptxas((11,0))
            assert result['TRITON_PTXAS_PATH']=='/user/ptxas'
            assert result['TRITON_PTXAS_BLACKWELL_PATH']==str(compiler)
            assert run.call_args.args[0]==[str(compiler),'--help']
        with patch.dict(os.environ,env,clear=True), patch('subprocess.run') as run:
            assert _configure_thor_ptxas((9,0))=={}
            run.assert_not_called()


def test_thor_compiler_does_not_select_unsupported_assembler():
    with patch.dict(os.environ,{},clear=True), patch('pathlib.Path.is_file',return_value=True), \
            patch('subprocess.run') as run:
        run.return_value=SimpleNamespace(returncode=0,stdout='sm_90',stderr='')
        assert all(v is None for v in _configure_thor_ptxas((11,0)).values())


def test_example_surface_stays_product_shaped():
    # The measurement artifacts are LOAD-BEARING: reproduce_h100.{sh,py} +
    # reproduce_h100_results.json back the published README H100 row (671.1 -> 127.5 ms, 5.26x,
    # the Runtime DEFAULT arm, 2026-08-28 re-sweep); static_capture.py +
    # verify_static_capture.py + static_capture_results.json are the module-level 6-case gate
    # for the denoise graph, and the Thor wall record cites static_capture.py as its protocol
    # artifact. This pinned set exists so a port or cleanup can never silently delete or
    # replace them.
    root_files = {path.name for path in PLUGIN_ROOT.iterdir() if path.is_file()}
    assert root_files == {
        "LICENSE",
        "README.md",
        "instinctwm.json",
        "moe_kernel_results.json",
        "profile_infer.py",
        "pyproject.toml",
        "reproduce_h100.py",
        "reproduce_h100.sh",
        "reproduce_h100_results.json",
        "static_capture.py",
        "static_capture_results.json",
        "verify_moe_kernel.py",
        "verify_static_capture.py",
    }
    assert os.access(PLUGIN_ROOT / "reproduce_h100.sh", os.X_OK)
    assert not (PLUGIN_ROOT / "csrc").exists()
    assert not (PLUGIN_ROOT / "lingbot_vla_v2_iwm" / "moe_cutlass.py").exists()


def test_published_h100_artifact_is_the_six_case_protocol():
    # The row artifact must stay the H100 6-case, envelope-judged protocol; a weaker protocol
    # (e.g. an 8-sample A100 run with a 4-case null threshold) must never stand in for it.
    import json

    doc = json.loads((PLUGIN_ROOT / "reproduce_h100_results.json").read_text())
    assert doc["stock_ms_p50"] == 671.1
    assert doc["ours_ms_p50"] == 127.5
    assert len(doc["gates"]) == 6
    assert all(case["max_abs_d"] <= doc["null_envelope"] for case in doc["gates"])

    # The module-level gate stays the 6-case protocol too (same box, denoise graph alone).
    module = json.loads((PLUGIN_ROOT / "static_capture_results.json").read_text())
    assert module["stock_ms_p50_inprocess"] == 667.0
    assert module["ours_ms_p50_inprocess"] == 165.0
    assert len(module["gates"]["cases"]) == 6


def test_v2_spec_declares_the_published_control_cycle():
    spec = LingBotVLAV2Adapter().spec()
    assert spec.model_id == "robbyant/lingbot-vla-v2-6b-robotwin"
    assert spec.streams[0].lifetime is KVLifetime.CHUNK
    assert spec.streams[0].tokens_per_frame == 286
    assert spec.phase("prefix").nfe == 1
    assert spec.phase("action").nfe == 10
    assert spec.notes["capture_tier"] == "NUMERIC"
    assert spec.observation.batched is False
    assert [field.shape for field in spec.observation.fields] == [
        (480, 640, 3), (480, 640, 3), (480, 640, 3), (14,),
    ]


def test_known_release_exposes_nested_config_without_copying_weights():
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        snapshot = tmp_path / "snapshot"
        nested = snapshot / "checkpoints" / "global_step_50000" / "hf_ckpt"
        nested.mkdir(parents=True)
        (nested / "config.json").write_text('{"vlm_family":"qwen3_vl"}')
        (snapshot / "lingbotvla_cli.yaml").write_text("model: {}\n")
        cache = tmp_path / "cache"
        previous = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = str(cache)
        try:
            doc = lookup("robbyant/lingbot-vla-v2-6b-robotwin")
            view = _declared_view(snapshot, "robbyant/lingbot-vla-v2-6b-robotwin", doc)
            assert (view / "config.json").is_symlink()
            assert (view / "config.json").resolve() == (nested / "config.json").resolve()
            report = validate_package(view)
            assert report.ok, report.explain()
            assert report.declaration.extra["checkpoint_subdir"].endswith("hf_ckpt")
        finally:
            if previous is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = previous


def test_model_path_resolves_declared_nested_layout():
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        nested = tmp_path / "checkpoints" / "global_step_50000" / "hf_ckpt"
        nested.mkdir(parents=True)
        (nested / "model.safetensors.index.json").write_text("{}")
        checkpoint = SimpleNamespace(
            path=str(tmp_path),
            model_id="test/v2",
            execution=SimpleNamespace(extra={
                "checkpoint_subdir": "checkpoints/global_step_50000/hf_ckpt",
            }),
        )
        assert _resolve_model_path(checkpoint) == nested


def test_runtime_loop_preserves_upstream_keys_and_maps_prompt():
    class FakeServer:
        def __init__(self):
            self.reset_to = None
            self.observation = None

        def reset(self, robot):
            self.reset_to = robot

        def infer(self, observation):
            self.observation = observation
            return {"action": np.zeros((50, 14), dtype=np.float32)}

    with tempfile.TemporaryDirectory() as td:
        server = FakeServer()
        loop = _LingBotVLAV2Loop(server, Path(td), robot="robotwin")
        loop.reset(prompt="pick the block")
        result = loop.predict({"observation.state": np.zeros(14, dtype=np.float32)})
        assert server.reset_to == "robotwin"
        assert server.observation["prompt"] == server.observation["task"] == "pick the block"
        assert result["action"].shape == (50, 14)
        assert loop.graph_stats["cuda_kernels"] is False


def test_triton_kernels_are_refused_on_sm110():
    # Triton is measured-dead on Thor sm_110a and the vendor fallback path crashes
    # (undefined logger / no try-except); the guard must refuse rather than die later.
    from lingbot_vla_v2_iwm.adapter import _triton_kernels_allowed

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.get_device_capability", return_value=(11, 0)
    ):
        try:
            _triton_kernels_allowed("IFL_VLA2_MOE_KERNEL")
        except RuntimeError as exc:
            assert "SM110" in str(exc) and "IFL_VLA2_MOE_KERNEL" in str(exc)
        else:
            raise AssertionError("Triton kernels must be refused on SM110")
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.get_device_capability", return_value=(9, 0)
    ):
        assert _triton_kernels_allowed("IFL_VLA2_MOE_KERNEL") is True


def test_cuda_kernel_feature_flag_is_strict_and_reversible():
    name = "IFL_TEST_VLA2_KERNEL_FLAG"
    previous = os.environ.get(name)
    try:
        os.environ.pop(name, None)
        assert _env_flag(name, default=True) is True
        os.environ[name] = "off"
        assert _env_flag(name, default=True) is False
        os.environ[name] = "YES"
        assert _env_flag(name, default=False) is True
        os.environ[name] = "sometimes"
        try:
            _env_flag(name, default=False)
        except RuntimeError as exc:
            assert name in str(exc)
        else:
            raise AssertionError("an invalid kernel flag must fail closed")
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))


def test_sdpa_host_is_not_rejected_for_legacy_qwen_dependencies():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root/'deploy').mkdir()
        (root/'deploy/lingbot_vla_v2_policy.py').touch()
        requested = []
        def imports(modules):
            requested.extend(modules)
            if {'flash_attn','qwen_vl_utils'} & set(modules):
                return False, 'optional legacy packages unavailable'
            return True, 'core stack imports'
        with patch.dict(os.environ,{'LINGBOT_VLA_V2_ROOT':str(root)}), \
             patch('instinctflash.runtime.execution.imports_available',side_effect=imports):
            ok, why = LingBotVLAV2Adapter().can_host_in_process()
        assert ok and str(root) in why
        assert {'torch','transformers','lerobot'} <= set(requested)
