"""Explicit, instance-local SM120 vision fusion over a strictly restored pi0.5 state.

The base state still describes its original kernels. This extension is separately
identified and qualified; it never patches global classes or changes GEMM choices.
"""
from __future__ import annotations

from pathlib import Path
import weakref

MODEL_SHA256 = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
BASE_SOURCE_TREE = "529e61ff0471ed4f9af8780f3d906b32990eadfa0aeae6084619a9a8dd57fe5d"
MODES = ("qkv", "ffn4", "ffn8", "all4", "all8")


def _vision_layer(p, i, seq, use_fp8, stream, extension, mode):
    if not use_fp8 or not p.fp8_calibrated or seq != 512:
        raise ValueError("SM120 fusion requires the restored two-view FP8 vision path")
    qkv = mode in ("qkv", "all4", "all8")
    ffn = mode.startswith("ffn") or mode.startswith("all")
    vector = 4 if mode.endswith("4") else 8
    fvk, w, b = p.fvk, p.weights, p.bufs
    pointers = p._attn_ptrs
    seq, dim, hidden = 512, 1152, 4304

    def ptr(name): return b[name].ptr.value

    def normalized_gemm(stage, name, destination, n):
        fvk.layer_norm(ptr("vision_x"), w[f"vision_pre_{stage}_norm_w"][i],
            w[f"vision_pre_{stage}_norm_b"][i], ptr("vision_x_norm"), seq, dim, 1e-5, stream=stream)
        p._fp8_gemm(ptr("vision_x_norm"), seq * dim, name, destination, seq, n, dim, stream)

    normalized_gemm("attn", f"vision_attn_qkv_w_{i}", ptr("vision_QKV"), 3 * dim)
    if qkv:
        extension.bias_qkv(ptr("vision_QKV"), w["vision_attn_qkv_b"][i],
            pointers["vis_Q"], pointers["vis_K"], pointers["vis_V"], seq, dim, stream)
    else:
        p._bias_add_bf16(ptr("vision_QKV"), w["vision_attn_qkv_b"][i], seq, 3 * dim, stream)
        fvk.qkv_split(ptr("vision_QKV"), pointers["vis_Q"], pointers["vis_K"], pointers["vis_V"],
                      seq, dim, dim, dim, stream=stream)
    attention = p.attn.run("siglip", i, q_seq=256, stream=stream)
    p._fp8_gemm(attention, seq * dim, f"vision_attn_o_w_{i}", ptr("vision_x_norm"), seq, dim, dim, stream)
    fvk.bias_residual(ptr("vision_x"), ptr("vision_x_norm"), w["vision_attn_o_b"][i], seq, dim, stream=stream)

    normalized_gemm("ffn", f"vision_ffn_up_w_{i}", ptr("vision_hidden"), hidden)
    name = f"vision_ffn_down_w_{i}"
    if ffn:
        activation, _ = p._pick_fp8_scratch(name, seq * hidden)
        scale = p.fp8_act_scales[name].ptr.value
        extension.bias_gelu_fp8(ptr("vision_hidden"), w["vision_ffn_up_b"][i],
                               activation, scale, seq, hidden, vector, stream)
        p._fp8_gemm_fused(activation, name, ptr("vision_x_norm"), seq, dim, hidden, scale, stream)
    else:
        p._bias_add_bf16(ptr("vision_hidden"), w["vision_ffn_up_b"][i], seq, hidden, stream)
        fvk.gelu_inplace(ptr("vision_hidden"), seq * hidden, stream=stream)
        p._fp8_gemm(ptr("vision_hidden"), seq * hidden, name, ptr("vision_x_norm"), seq, dim, hidden, stream)
    fvk.bias_residual(ptr("vision_x"), ptr("vision_x_norm"), w["vision_ffn_down_b"][i], seq, dim, stream=stream)


class _VisionDispatch:
    def __init__(self, pipeline, extension, mode):
        self.pipeline = weakref.ref(pipeline)
        self.extension, self.mode = extension, mode

    def __call__(self, i, seq, use_fp8, stream):
        pipeline = self.pipeline()
        if pipeline is None: raise RuntimeError("fusion pipeline is closed")
        return _vision_layer(pipeline, i, seq, use_fp8, stream, self.extension, self.mode)


class FusionHandle:
    def __init__(self, runtime, extension, mode, hoist_scales, signature):
        self.runtime = weakref.ref(runtime)
        self.extension, self.mode, self.hoist_scales = extension, mode, hoist_scales
        self.signature = signature
        self.original_tokens = type(runtime)._set_tokens
        self.original_scales = type(runtime)._install_scales
        self.scales = dict(runtime._scales)
        self.prepared = weakref.WeakSet()
        self.uploaded = weakref.WeakSet()
        self.profile_installs = 0
        self.skipped_scale_installs = 0
        self.closed = False

    def __call__(self, ids):
        runtime = self.runtime()
        if self.closed or runtime is None: raise RuntimeError("fusion handle is closed")
        frontend = self.original_tokens(runtime, ids)
        pipeline = frontend.pipeline
        if pipeline not in self.prepared:
            if pipeline.num_views != 2 or pipeline.fp8_layout != "nk":
                raise ValueError("unsupported fusion pipeline geometry/layout")
            if self.mode != "none":
                pipeline._vision_layer = _VisionDispatch(pipeline, self.extension, self.mode)
            self.prepared.add(pipeline)
            self.profile_installs += 1
        return frontend

    def install_scales(self, pipeline):
        runtime = self.runtime()
        if self.closed or runtime is None: raise RuntimeError("fusion handle is closed")
        if runtime._scales != self.scales:
            raise ValueError("frozen scales were mutated after fusion installation")
        if pipeline not in self.uploaded or not pipeline.fp8_calibrated:
            self.original_scales(runtime, pipeline)
            self.uploaded.add(pipeline)
        else:
            self.skipped_scale_installs += 1

    def close(self):
        if self.closed: return
        runtime = self.runtime()
        if runtime is not None and runtime.frontend is not None:
            runtime._release_profiles()
            runtime.frontend.pipeline = None
            runtime.frontend.calibrated = runtime.frontend.graph_recorded = False
        if runtime is not None:
            if runtime.__dict__.get("_set_tokens") is self:
                del runtime.__dict__["_set_tokens"]
            if self.hoist_scales:
                callback = runtime.__dict__.get("_install_scales")
                if getattr(callback, "__self__", None) is self:
                    del runtime.__dict__["_install_scales"]
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("fusion handle is closed")
        return self

    def __exit__(self, *exc):
        self.close()


def install_pi05_sm120_fusion(runtime, state_path, *, mode="all8", hoist_scales=False):
    import torch
    from flash_rt.core import frozen_state as state
    from flash_rt.frontends.torch.pi05_frozen import FrozenPi05Frontend
    from flash_rt import flash_rt_pi05_sm120_fusion as extension
    if mode not in (*MODES, "none"):
        raise ValueError("unknown SM120 fusion mode")
    if type(hoist_scales) is not bool or (mode == "none" and not hoist_scales):
        raise ValueError("select a fusion or explicit scale-upload hoist")
    if type(runtime) is not FrozenPi05Frontend or runtime._restored_parts != "both":
        raise ValueError("fusion requires a strictly restored FrozenPi05Frontend")
    if (runtime._profiles or runtime.frontend.pipeline is not None
            or any(name in runtime.__dict__ for name in ("_set_tokens", "_install_scales"))):
        raise ValueError("install fusion on a fresh restored model before any graph capture")
    if (torch.cuda.get_device_capability() != (12, 0) or "5090" not in torch.cuda.get_device_name()
            or extension.abi_version != 1):
        raise ValueError("incompatible SM120 fusion device/ABI")
    payload = state.load(state_path, runtime.identity())
    if payload["identity"]["source_tree"] != BASE_SOURCE_TREE:
        raise ValueError("base FlashRT source changed; requalify the fusion dispatch first")
    if payload["identity"]["weights"] != MODEL_SHA256 or payload["identity"]["contract"]["views"] != 2:
        raise ValueError("fusion is qualified only for the pinned two-view LIBERO checkpoint")
    if (state.scale_bits(runtime._scales) != payload["scale_bits"] or runtime._prompt != payload["prompt"]
            or [[*r[:4], r[4].hex()] for r in runtime.frontend.gemm.export_algo_cache()] != payload["gemm"]["entries"]):
        raise ValueError("runtime scales, prompt or GEMM choices differ from the frozen artifact")
    signature = {"mode": mode, "hoist_scales": bool(hoist_scales), "base_state_sha256": state.file_hash(state_path),
                 "extension_sha256": state.file_hash(extension.__file__),
                 "integration_sha256": state.file_hash(Path(__file__)), "abi_version": extension.abi_version}
    handle = FusionHandle(runtime, extension, mode, hoist_scales, signature)
    runtime._set_tokens = handle
    if hoist_scales: runtime._install_scales = handle.install_scales
    return handle
