"""Single-desktop-GPU residency for the original DreamZero policy and complete history.

Weights and committed KV may live on CPU; all model arithmetic still runs on
CUDA. Native processors, both CFG branches, solver updates and cache contents
are unchanged. Transfer time belongs to the prediction latency. This path uses
eager component forwards because moving storage cannot participate in a CUDA
graph. It does not assert action equivalence or task quality without measurement.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import textwrap
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType

from instinctflash.runtime.desktop_fp8 import SM89, target_for_capability

_POST_INIT_HASH = "71358c4fc3ae74a0d944ac3003df39febbec60926095e9aa404c144b351cb978"
_CONSTRUCTION = ContextVar("dreamzero_residency_construction", default=None)


def use_sm89_residency(device="cuda"):
    import torch

    return (torch.cuda.is_available()
            and torch.cuda.get_device_capability(device) == (8, 9)
            and torch.cuda.get_device_properties(device).total_memory <= 40 << 30)


def use_desktop_residency(device="cuda"):
    import torch

    return (torch.cuda.is_available()
            and target_for_capability(torch.cuda.get_device_capability(device)) is not None
            and torch.cuda.get_device_properties(device).total_memory <= 40 << 30)


@contextmanager
def construction_scope():
    """Own a head even if later native policy/processor construction fails."""
    if _CONSTRUCTION.get() is not None:
        raise RuntimeError("Nested DreamZero residency construction is unsupported")
    owners = []
    token = _CONSTRUCTION.set(owners)
    try:
        yield owners
        if len(owners) != 1 or not owners[0].initialized:
            raise RuntimeError("Expected one initialized native DreamZero residency owner")
    except BaseException:
        for owner in reversed(owners):
            owner.close()
        raise
    finally:
        _CONSTRUCTION.reset(token)


def build_head(config, ifl_dynamic_cache_schedule, ifl_fixed_dit_steps,
               ifl_residency_precision):
    """Return the original native head with an instance-owned placement method."""
    import torch

    from .schedule import build_head as native_head

    if ifl_residency_precision not in ("native", "fp8"):
        raise ValueError("DreamZero residency precision must be native or fp8")
    owners = _CONSTRUCTION.get()
    if owners is None or owners:
        raise RuntimeError("DreamZero residency head requires its single owned construction scope")
    if not use_desktop_residency():
        raise ValueError("DreamZero residency requires an SM89 or SM120 device with at most 40 GiB")
    from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
        WANPolicyHead,
    )

    from instinctflash.runtime.precision import require_dreamzero_fp8_environment

    # Native code may load a TRT engine even with its enable flag false.
    require_dreamzero_fp8_environment()
    source = textwrap.dedent(inspect.getsource(WANPolicyHead.post_initialize)).strip()
    if hashlib.sha256(source.encode()).hexdigest() != _POST_INIT_HASH:
        raise ValueError("DreamZero post-initialize changed; re-audit residency construction")
    head = native_head(config, ifl_dynamic_cache_schedule, ifl_fixed_dit_steps)
    owner = DreamZeroResidency(head, precision=ifl_residency_precision,
                               target=target_for_capability(torch.cuda.get_device_capability()))
    owners.append(owner)
    head._ifl_residency = owner
    head.post_initialize = MethodType(_post_initialize, head)
    return head


def _post_initialize(head):
    head._ifl_residency.initialize()


class CPUKVStorage:
    """Transport one complete native layer's KV while its forward executes.

    Native commits still clone the returned KV and own the history lists. No
    token, branch or element is omitted. Moving the returned tensor to CPU also
    bounds the native list-of-updated-caches peak across all forty layers.
    """

    def __init__(self, blocks, *, device="cuda"):
        import torch

        self.device = torch.device(device)
        self.handles = []
        self.active = set()
        self.closed = False
        self.forward_count = 0
        self.input_bytes = 0
        self.output_bytes = 0
        try:
            for block in blocks:
                if "kv_cache" not in inspect.signature(block.forward).parameters:
                    raise ValueError("DreamZero block has no native kv_cache argument")
                self.handles.append(block.register_forward_pre_hook(self._before, with_kwargs=True))
                self.handles.append(block.register_forward_hook(
                    self._after, with_kwargs=True, always_call=True))
        except BaseException:
            self.close()
            raise

    def _before(self, block, args, kwargs):
        import torch

        if self.closed or id(block) in self.active:
            raise RuntimeError("Closed or reentrant DreamZero KV residency")
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DreamZero KV residency requires eager inference without CUDA capture")
        cache = kwargs.get("kv_cache")
        if not isinstance(cache, torch.Tensor) or cache.ndim != 5 or cache.shape[0] != 2:
            raise ValueError("DreamZero residency requires the complete native [2,B,T,H,D] cache")
        if cache.dtype != torch.bfloat16:
            raise ValueError("DreamZero KV residency preserves native BF16 cache values")
        moved = cache.to(self.device)
        if cache.device.type == "cpu":
            self.input_bytes += cache.numel() * cache.element_size()
        self.active.add(id(block))
        return args, {**kwargs, "kv_cache": moved}

    def _after(self, block, args, kwargs, output):
        del args, kwargs
        if id(block) not in self.active:
            return None
        try:
            if output is None:  # preserve an exception from the native forward
                return None
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("DreamZero native block return contract changed")
            value, cache = output
            import torch

            if not isinstance(cache, torch.Tensor) or cache.dtype != torch.bfloat16:
                raise RuntimeError("DreamZero native block did not return its BF16 KV cache")
            stored = cache.to("cpu")
            self.forward_count += 1
            self.output_bytes += cache.numel() * cache.element_size()
            return value, stored
        finally:
            self.active.remove(id(block))

    def report(self):
        return {"storage": "cpu", "compute_device": str(self.device),
                "complete_history": True, "native_commit_clone_preserved": True,
                "forwards": self.forward_count, "host_to_device_bytes": self.input_bytes,
                "device_to_host_bytes": self.output_bytes}

    def close(self):
        if self.active:
            raise RuntimeError("Cannot close DreamZero KV residency during a native forward")
        if self.closed:
            return
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.closed = True


class DreamZeroResidency:
    def __init__(self, head, *, precision, target=SM89):
        self.head = head
        self.precision = precision
        self.target = target
        self.initialized = False
        self.closed = False
        self.weights = None
        self.kv = None
        self.fp8_recipe = None

    def initialize(self):
        import torch

        from instinctflash.runtime.module_residency import install_module_residency

        if self.closed or self.initialized:
            raise RuntimeError("DreamZero residency post-initialize must execute exactly once")
        head = self.head
        if head.training or any(p.requires_grad for p in head.parameters()):
            raise ValueError("DreamZero residency requires the native frozen evaluation model")
        if any(p.device.type != "cpu" for p in head.parameters()):
            raise ValueError("DreamZero residency requires native lazy CPU loading")
        if type(head.model).__name__ != "CausalWanModel":
            raise ValueError("DreamZero residency requires the native merged CausalWanModel")
        dit = list(head.model.blocks)
        text = list(head.text_encoder.blocks)
        if (not dit or any(type(b).__name__ != "CausalWanAttentionBlock" for b in dit)
                or not text or any(type(b).__name__ != "T5SelfAttention" for b in text)):
            raise ValueError("DreamZero residency layer layout changed")
        try:
            # Same final native post_initialize dtype; never cast after FP8 packing.
            head.to(dtype=torch.bfloat16)
            if self.precision == "fp8":
                from instinctflash.runtime.desktop_fp8 import backend_for_capability
                implementation = backend_for_capability(self.target.capability)
                install_fp8 = getattr(implementation, f"install_{self.target.prefix}_fp8")
                self.fp8_recipe = install_fp8(
                    head.model, "dreamzero", device="cuda", include_mlp=True,
                    storage_device="cpu")
            self.weights = install_module_residency(
                head, [*dit, *text], device="cuda", reserve_bytes=8 << 30)
            self.kv = CPUKVStorage(dit, device="cuda")
            # Both are also established by the original native post_initialize.
            head.trt_engine = None
            head._vae_device_ready = True
            if head.device.type != "cuda" or head.dtype != torch.bfloat16:
                raise RuntimeError("Native DreamZero input placement must still resolve to CUDA BF16")
            self.initialized = True
        except BaseException:
            self.close()
            raise

    def report(self):
        return copy.deepcopy({
            "recipe": f"dreamzero_{self.target.prefix}_native_layer_and_kv_residency_v1",
            "precision": self.precision, "component_compilation": False,
            "component_execution": "native eager CUDA; transfer time included",
            "weights": self.weights.report() if self.weights else None,
            "kv": self.kv.report() if self.kv else None,
            "quality_status": "unverified", "schedule_changed_by_residency": False,
        })

    def close(self):
        if self.closed:
            return
        if self.kv is not None:
            self.kv.close()
        if self.weights is not None:
            self.weights.close()
        if self.head is not None:
            self.head.__dict__.pop("post_initialize", None)
            self.head.__dict__.pop("_ifl_residency", None)
        self.head = None
        self.closed = True
