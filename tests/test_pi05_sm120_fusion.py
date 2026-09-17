"""CPU-only contracts for the explicitly installed, instance-local CUDA addon."""
import gc
import sys
from types import SimpleNamespace
import weakref

import pytest

from instinctflash.runtime.pi05_sm120_fusion import (
    BASE_SOURCE_TREE, MODEL_SHA256, FusionHandle, _vision_layer, install_pi05_sm120_fusion,
)


class Pipeline:
    num_views = 2
    fp8_layout = "nk"
    fp8_calibrated = False


class Runtime:
    def __init__(self):
        self._scales = {"layer": 0.25}
        self.frontend = SimpleNamespace(pipeline=None, calibrated=False, graph_recorded=False)
        self.profiles = {}
        self.uploads = 0
        self.releases = 0

    def _set_tokens(self, ids):
        self.frontend.pipeline = self.profiles.setdefault(len(ids), Pipeline())
        return self.frontend

    def _install_scales(self, pipeline):
        self.uploads += 1
        pipeline.fp8_calibrated = True

    def _release_profiles(self):
        self.releases += 1
        self.profiles.clear()


def install(runtime, *, mode="all8", hoist=False):
    handle = FusionHandle(runtime, object(), mode, hoist, {})
    runtime._set_tokens = handle
    if hoist:
        runtime._install_scales = handle.install_scales
    return handle


def test_install_is_local_and_close_restores_base_methods():
    runtime, other = Runtime(), Runtime()
    handle = install(runtime, hoist=True)
    pipeline = runtime._set_tokens([1]).pipeline
    runtime._set_tokens([2])
    assert handle.profile_installs == 1
    assert "_vision_layer" in pipeline.__dict__
    assert "_set_tokens" not in other.__dict__
    handle.close()
    assert runtime.releases == 1 and runtime.frontend.pipeline is None
    assert "_set_tokens" not in runtime.__dict__
    assert "_install_scales" not in runtime.__dict__
    assert runtime._set_tokens([3]).pipeline is not pipeline
    handle.close()
    assert runtime.releases == 1
    with pytest.raises(RuntimeError, match="closed"):
        handle([1])


def test_dispatch_does_not_keep_retired_pipeline_alive():
    runtime = Runtime()
    handle = install(runtime)
    pipeline = runtime._set_tokens([1]).pipeline
    reference = weakref.ref(pipeline)
    callback = pipeline._vision_layer
    handle.close()
    del pipeline
    gc.collect()
    assert reference() is None and not handle.prepared
    with pytest.raises(RuntimeError, match="closed"):
        callback(0, 512, True, 0)


def test_context_cleanup_after_runtime_already_closed():
    runtime = Runtime()
    with install(runtime, hoist=True) as handle:
        runtime.frontend = None
    assert handle.closed
    assert "_set_tokens" not in runtime.__dict__ and "_install_scales" not in runtime.__dict__


def test_scale_hoist_is_once_per_live_profile_and_detects_mutation():
    runtime = Runtime()
    handle = install(runtime, mode="none", hoist=True)
    first = runtime._set_tokens([1]).pipeline
    runtime._install_scales(first)
    runtime._install_scales(first)
    assert runtime.uploads == 1 and handle.skipped_scale_installs == 1
    assert "_vision_layer" not in first.__dict__
    second = runtime._set_tokens([1, 2]).pipeline
    runtime._install_scales(second)
    first.fp8_calibrated = False
    runtime._install_scales(first)
    assert runtime.uploads == 3
    runtime._scales["layer"] = 0.5
    with pytest.raises(ValueError, match="mutated"):
        runtime._install_scales(second)


@pytest.mark.parametrize("field,value", [("num_views", 1), ("fp8_layout", "kn")])
def test_unsupported_pipeline_is_rejected(field, value):
    runtime = Runtime()
    pipeline = runtime._set_tokens([1]).pipeline
    setattr(pipeline, field, value)
    handle = install(runtime)
    with pytest.raises(ValueError, match="geometry/layout"):
        handle([1])
    assert "_vision_layer" not in pipeline.__dict__


@pytest.mark.parametrize("mode,expected", [
    ("qkv", (0, 1, 0, 0, 4)),
    ("ffn4", (0, 0, 1, 1, 3)), ("ffn8", (0, 0, 1, 1, 3)),
    ("all4", (0, 1, 1, 1, 3)), ("all8", (0, 1, 1, 1, 3)),
])
def test_ablation_dispatch_keeps_four_gemms_and_attention(mode, expected):
    calls = []

    class Recorder:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                calls.append((name, args, kwargs))
                return 55 if name == "run" else None
            return call

    pipeline = Pipeline()
    pipeline.fp8_calibrated = True
    pipeline.fvk = pipeline.attn = Recorder()
    pipeline.bufs = {name: SimpleNamespace(ptr=SimpleNamespace(value=i + 100))
        for i, name in enumerate(("vision_x", "vision_x_norm", "vision_QKV", "vision_hidden"))}
    pipeline.weights = {name: [i + 200] for i, name in enumerate((
        "vision_pre_attn_norm_w", "vision_pre_attn_norm_b", "vision_pre_ffn_norm_w",
        "vision_pre_ffn_norm_b", "vision_attn_qkv_b", "vision_attn_o_b", "vision_ffn_up_b", "vision_ffn_down_b"))}
    pipeline._attn_ptrs = {name: i + 300 for i, name in enumerate(("vis_Q", "vis_K", "vis_V"))}
    pipeline.fp8_act_scales = {f"vision_{name}_w_0": SimpleNamespace(ptr=SimpleNamespace(value=i + 400))
        for i, name in enumerate(("attn_qkv", "ffn_up", "ffn_down"))}
    pipeline._pick_fp8_scratch = lambda *args: (600, 601)
    pipeline._fp8_gemm = Recorder().gemm
    pipeline._fp8_gemm_fused = Recorder().fused_gemm
    pipeline._bias_add_bf16 = Recorder().add_bias
    _vision_layer(pipeline, 0, 512, True, 909, Recorder(), mode)
    names = [row[0] for row in calls]
    assert tuple(names.count(name) for name in ("layernorm_fp8", "bias_qkv", "bias_gelu_fp8", "fused_gemm", "gemm")) == expected
    assert names.count("run") == 1 and names.count("bias_residual") == 2
    assert names.count("layer_norm") == 2, "normalization must remain on the original path"
    assert next(row for row in calls if row[0] == "run")[2] == {"q_seq": 256, "stream": 909}
    if "bias_gelu_fp8" in names:
        assert next(row for row in calls if row[0] == "bias_gelu_fp8")[1][-2:] == (4 if mode.endswith("4") else 8, 909)


@pytest.mark.parametrize("seq,fp8,calibrated", [(256, True, True), (512, False, True), (512, True, False)])
def test_uncalibrated_or_different_geometry_fails_before_launch(seq, fp8, calibrated):
    pipeline = Pipeline()
    pipeline.fp8_calibrated = calibrated
    with pytest.raises(ValueError, match="restored two-view"):
        _vision_layer(pipeline, 0, seq, fp8, 0, object(), "all8")


@pytest.fixture
def restored_runtime(monkeypatch):
    class Frozen(Runtime):
        _restored_parts = "both"
        _prompt = "task"

        def identity(self):
            return payload["identity"]

    runtime = Frozen()
    runtime._profiles = runtime.profiles
    runtime.frontend.gemm = SimpleNamespace(export_algo_cache=lambda: [(0, 512, 3456, 1152, b"algo")])
    payload = {"identity": {"weights": MODEL_SHA256, "source_tree": BASE_SOURCE_TREE, "contract": {"views": 2}},
               "scale_bits": dict(runtime._scales), "prompt": "task",
               "gemm": {"entries": [[0, 512, 3456, 1152, b"algo".hex()]]}}
    extension = SimpleNamespace(abi_version=1, __file__="extension.so")
    cuda = SimpleNamespace(get_device_capability=lambda: (12, 0), get_device_name=lambda: "NVIDIA GeForce RTX 5090")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "flash_rt", SimpleNamespace(flash_rt_pi05_sm120_fusion=extension))
    monkeypatch.setitem(sys.modules, "flash_rt.core", SimpleNamespace(frozen_state=SimpleNamespace(
        load=lambda *args: payload, scale_bits=dict, file_hash=lambda path: f"hash:{path}")))
    monkeypatch.setitem(sys.modules, "flash_rt.frontends.torch.pi05_frozen", SimpleNamespace(FrozenPi05Frontend=Frozen))
    return runtime, payload, cuda, extension


def test_installer_binds_artifact_and_does_not_enable_scale_hoist_by_default(restored_runtime):
    runtime, _, _, _ = restored_runtime
    handle = install_pi05_sm120_fusion(runtime, "state.json")
    assert handle.signature["base_state_sha256"] == "hash:state.json"
    assert handle.signature["extension_sha256"] == "hash:extension.so"
    assert handle.mode == "all8" and not handle.hoist_scales
    assert "_install_scales" not in runtime.__dict__
    with pytest.raises(ValueError, match="fresh"):
        install_pi05_sm120_fusion(runtime, "state.json")
    handle.close()


@pytest.mark.parametrize("mutation,message", [
    ("source", "source changed"), ("weights", "checkpoint"), ("views", "checkpoint"),
    ("scales", "scales, prompt"), ("prompt", "scales, prompt"), ("algorithms", "GEMM choices"),
    ("partial_state", "strictly restored"), ("captured", "fresh"), ("scale_hook", "fresh"),
    ("gpu", "device/ABI"), ("capability", "device/ABI"), ("abi", "device/ABI"),
])
def test_installer_fails_closed_without_mutating_runtime(restored_runtime, mutation, message):
    runtime, payload, cuda, extension = restored_runtime
    if mutation == "source": payload["identity"]["source_tree"] = "changed"
    elif mutation == "weights": payload["identity"]["weights"] = "changed"
    elif mutation == "views": payload["identity"]["contract"]["views"] = 3
    elif mutation == "scales": runtime._scales["layer"] = .5
    elif mutation == "prompt": runtime._prompt = "changed"
    elif mutation == "algorithms": payload["gemm"]["entries"] = []
    elif mutation == "partial_state": runtime._restored_parts = "scales"
    elif mutation == "captured": runtime.frontend.pipeline = Pipeline()
    elif mutation == "scale_hook": runtime._install_scales = lambda _: None
    elif mutation == "gpu": cuda.get_device_name = lambda: "RTX PRO 4000 Blackwell"
    elif mutation == "capability": cuda.get_device_capability = lambda: (9, 0)
    elif mutation == "abi": extension.abi_version = 2
    with pytest.raises(ValueError, match=message):
        install_pi05_sm120_fusion(runtime, "state.json")
    assert "_set_tokens" not in runtime.__dict__
