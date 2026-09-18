"""Bounded BF16 residency for the existing native DreamZero policy on SM120.

Keep the Thor full-checkpoint loader, native modules, KV history and scheduler.
Only weight residency changes: immutable CPU masters are uploaded per component
or DiT block. This module never imports an experimental GEMM implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
from types import MethodType


def selected(capability, plan, mode="auto"):
    if mode not in {"auto", "off", "block"}:
        raise ValueError("IFL_DREAMZERO_SM120_RESIDENCY must be auto, block or off")
    if mode == "off" or getattr(plan, "resolved_gemm_backend", None):
        return False
    if tuple(capability) != (12, 0):
        if mode == "block":
            raise ValueError("DreamZero block residency is an SM120 native profile")
        return False
    return True


class Stage:
    """An inference-only module's immutable CPU weights, borrowed on one device."""
    def __init__(self, owner, module, name):
        self.owner, self.module, self.name = owner, module, name
        self.parameters = [(p, p.detach()) for p in module.parameters()]
        self.buffers = [(m, key, value) for m in module.modules()
                        for key, value in m._buffers.items() if value is not None]
        if any(p.requires_grad or p.device.type != "cpu" for p, _ in self.parameters):
            raise ValueError("Native residency requires frozen CPU parameter masters")
        self.bytes = sum(p.numel() * p.element_size() for p, _ in self.parameters)
        self.calls = 0
        self.handles = [module.register_forward_pre_hook(self._enter),
                        module.register_forward_hook(self._leave, always_call=True)]

    def _enter(self, module, args):
        import torch
        if torch.is_grad_enabled() or module.training:
            raise RuntimeError("Native block residency is inference-only")
        if self.owner.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native block residency cannot be captured in a CUDA graph")
        self.owner.lock.acquire()
        if self.owner.active is not None or self.owner.closed:
            self.owner.lock.release()
            raise RuntimeError("Overlapping or closed native weight residency")
        self.owner.active = self
        try:
            for parameter, master in self.parameters:
                parameter.data = master.to(self.owner.device)
            for child, key, master in self.buffers:
                child._buffers[key] = master.to(self.owner.device)
            self.calls += 1
        except BaseException:
            self._release()
            raise

    def _release(self):
        try:
            # No weight D2H copy: computations finish before the GPU copies are
            # dropped, and the untouched CPU masters are rebound verbatim.
            self.owner.synchronize()
            for parameter, master in self.parameters:
                parameter.data = master
            for child, key, master in self.buffers:
                child._buffers[key] = master
        finally:
            self.owner.active = None
            self.owner.lock.release()

    def _leave(self, module, args, output):
        if self.owner.active is self:
            self._release()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.parameters.clear()
        self.buffers.clear()
        self.module = None


class NativeResidency:
    def __init__(self, head):
        import torch
        self.device = torch.device(head._device)
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise ValueError("Native DreamZero block residency requires SM120")
        if head.train_architecture != "full" or head.model.dim != 5120 or head.model.num_layers != 40:
            raise ValueError("Requalify native residency for this DreamZero architecture")
        self.lock = threading.RLock()
        self.active = None
        self.closed = False
        self.stages = []
        self.synchronize = lambda: torch.cuda.synchronize(self.device)
        # The same BF16 cast as native post_initialize, performed on CPU before
        # bounded uploads. No quantization or compiler replacement is introduced.
        head.to(device="cpu", dtype=torch.bfloat16)
        try:
            self.stages.append(Stage(self, head.text_encoder, "text_encoder"))
            self.stages.append(Stage(self, head.image_encoder.model.visual, "image_encoder.visual"))
            for i, block in enumerate(head.model.blocks):
                self.stages.append(Stage(self, block, f"model.blocks.{i}"))
            # VAE and small DiT input/output projections stay resident. Native
            # VAE encode is a direct method, so a forward hook would miss it.
            head.vae.to(self.device)
            for name, child in head.model.named_children():
                if name != "blocks":
                    child.to(self.device)
            for parameter in head.model.parameters(recurse=False):
                parameter.data = parameter.data.to(self.device)
            for name, buffer in head.model._buffers.items():
                if buffer is not None:
                    head.model._buffers[name] = buffer.to(self.device)
            head._vae_device_ready = True
            head.trt_engine = None
            head.cpu_offload = True
        except BaseException:
            self.close()
            raise

    @contextmanager
    def request(self):
        import torch
        with self.lock, torch.inference_mode():
            if self.closed:
                raise RuntimeError("Native DreamZero residency is closed")
            yield

    def report(self):
        return {"profile": "sm120_native_bf16_block_residency", "quantization": False,
                "precision": "bfloat16", "auxiliary_compile": False,
                "weight_copy_back": False,
                "scope": "Original Thor/native modules, BF16 values and scheduler; bounded CPU/GPU weight residency",
                "stages": [{"name": s.name, "weight_bytes": s.bytes, "calls": s.calls} for s in self.stages]}

    def close(self):
        with self.lock:
            if self.active is not None:
                raise RuntimeError("Cannot close active native residency")
            if self.closed:
                return
            self.synchronize()
            for stage in self.stages:
                stage.close()
            self.closed = True


def build_head(config, ifl_dynamic_cache_schedule, ifl_fixed_dit_steps):
    """Owned Hydra factory; leave the original class and global functions alone."""
    import os
    from .schedule import build_head as build_scheduled_head
    if os.environ.get("ENABLE_TENSORRT", "false").lower() == "true" or "LOAD_TRT_ENGINE" in os.environ:
        raise ValueError("Native SM120 residency cannot be used behind TensorRT")
    head = build_scheduled_head(config, ifl_dynamic_cache_schedule, ifl_fixed_dit_steps)

    def post_initialize(instance):
        instance._instinctflash_native_residency = NativeResidency(instance)

    # Native GrootSimPolicy invokes this once, after CPU checkpoint loading.
    head.post_initialize = MethodType(post_initialize, head)
    return head


def anchor_policy_device(policy):
    """HF's default .device reads the first parameter, now intentionally on CPU."""
    model = policy.trained_model
    original = type(model)
    device = policy.device

    class ResidentNativeModel(original):
        @property
        def device(self):
            import torch
            return torch.device(device)

    model.__class__ = ResidentNativeModel


def prepare_streaming_view(root):
    """Modify only our temporary view, never the original checkpoint metadata."""
    import json
    from pathlib import Path
    from omegaconf import OmegaConf
    from instinctflash.runtime.dreamzero_checkpoint import DIT_TARGET
    root = Path(root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    dit = config["action_head_cfg"]["config"]["diffusion_model_cfg"]
    dit.pop("checkpoint_path", None)
    dit["_target_"] = DIT_TARGET
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    experiment = root / "experiment_cfg"
    if not experiment.is_symlink():
        raise ValueError("Streaming load requires an owned symlink checkpoint view")
    original = experiment.resolve(strict=True)
    cfg = OmegaConf.load(original / "conf.yaml")
    if cfg.model._target_ != "groot.vla.model.dreamzero.base_vla.VLA":
        raise ValueError("Requalify streaming load for this DreamZero root architecture")
    cfg.model._target_ = "dreamzero_iwm.sm120_native.StreamingVLA"
    experiment.unlink()  # only our temporary symlink; the original remains intact
    experiment.mkdir()
    for item in original.iterdir():
        if item.name != "conf.yaml":
            (experiment / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    OmegaConf.save(cfg, experiment / "conf.yaml")


def load_streamed_native(root):
    """Construct original modules with empty parameters, then assign exact shards.

    Accelerate leaves buffers/non-parameter tensors on CPU so native constants
    and RoPE arithmetic are retained. No global default-dtype override is used.
    """
    import json
    from pathlib import Path
    import torch
    import warnings
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from groot.vla.model.dreamzero.base_vla import VLA
    from instinctflash.runtime.dreamzero_checkpoint import read_tensor_headers
    root = Path(root)
    headers = read_tensor_headers(root)
    if {v["dtype"] for v in headers.values()} != {"BF16"}:
        raise ValueError("Streaming native loader requires the audited all-BF16 checkpoint")
    config = VLA.config_class.from_dict(json.loads((root / "config.json").read_text()))
    with warnings.catch_warnings(), init_empty_weights(include_buffers=False):
        # Native construction still resolves its original T5/CLIP/VAE assets.
        # These initial copies into meta parameters are deliberately superseded
        # by the complete, strictly validated final checkpoint assignment below.
        warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
        model = VLA(config)
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    if set(expected) != set(headers) or any(expected[k] != headers[k]["shape"] for k in expected):
        raise ValueError("Full native checkpoint coverage/shape mismatch; cannot stream a partial model")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    assigned = set()
    for shard in sorted(set(index.values())):
        if Path(shard).name != shard:
            raise ValueError("Expected root-level checkpoint shards")
        with safe_open(str(root / shard), framework="pt", device="cpu") as file:
            state = {key: file.get_tensor(key) for key in file.keys()}
            if assigned.intersection(state):
                raise ValueError("Duplicate streamed checkpoint tensors")
            model.load_state_dict(state, strict=False, assign=True)
            assigned.update(state)
            del state
    if assigned != set(expected) or any(t.is_meta for t in model.state_dict().values()):
        raise RuntimeError("Streaming load left missing/meta checkpoint state")
    # Do not silently materialize unregistered meta constants with invented data.
    def is_meta(value):
        if isinstance(value, torch.Tensor):
            return value.is_meta
        if isinstance(value, (list, tuple)):
            return any(is_meta(v) for v in value)
        if isinstance(value, dict):
            return any(is_meta(v) for v in value.values())
        return False
    if any(is_meta(v) for module in model.modules() for v in vars(module).values()):
        raise RuntimeError("Native constructor left an unregistered meta tensor")
    model._instinctflash_streamed_tensors = len(assigned)
    return model


def __getattr__(name):
    # GrootSimPolicy resolves the class named by its OWNED experiment config.
    # Keep this lazy so metadata/CPU tests do not import the native model stack.
    if name != "StreamingVLA":
        raise AttributeError(name)
    from groot.vla.model.dreamzero.base_vla import VLA

    class StreamingVLA(VLA):
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            if kwargs:
                raise ValueError("Requalify native streaming for model-config overrides")
            return load_streamed_native(pretrained_model_name_or_path)
    return StreamingVLA
