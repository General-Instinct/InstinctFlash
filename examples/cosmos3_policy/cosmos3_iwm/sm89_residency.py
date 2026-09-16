"""Owned Cosmos SM89 CPU loading and decoder-layer residency.

The checkpoint loader, trained weights, native layer classes and forwards are
retained. Only the selected network's first materialization changes device.
Native CUDA initialization of analytical buffers still runs normally. Streaming
layers retain their CPU tensors and borrow CUDA copies for one inference call.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
from types import MethodType

import torch


INFERENCE_RESERVE_BYTES = 8 << 30
_construction_lock = threading.RLock()


class CPUConstruction:
    def __init__(self):
        self.net = None
        self.materializations = 0
        self.residency = None

    def claim(self, net):
        if self.net is not None or "to_empty" in net.__dict__:
            raise RuntimeError("Cosmos CPU construction requires one unmodified native network")
        self.net = net
        original = net.to_empty
        owner = self

        def materialize(instance, *, device, recurse=True):
            if (torch.device(device).type != "cuda" or not recurse
                    or owner.materializations != 0
                    or not list(instance.parameters())
                    or any(p.device.type != "meta" for p in instance.parameters())):
                raise RuntimeError("Cosmos CPU load requires the first meta-to-CUDA materialization")
            owner.materializations += 1
            # Restore the normal method before delegating or raising. CUDA
            # analytical buffer initialization runs afterward, unchanged.
            del instance.to_empty
            return original(device="cpu", recurse=True)

        net.to_empty = MethodType(materialize, net)

    def verify(self, model):
        if self.materializations != 1 or getattr(model, "net", None) is not self.net:
            raise RuntimeError("Cosmos service did not consume its owned CPU construction")
        if any(p.device.type != "cpu" for p in self.net.parameters()):
            raise RuntimeError("Cosmos checkpoint weights must remain on CPU until placement")
        if any(b.device.type == "meta" for b in self.net.buffers()):
            raise RuntimeError("Cosmos native buffer initialization was incomplete")

    def cleanup(self):
        if self.net is not None and "to_empty" in self.net.__dict__:
            del self.net.to_empty


@contextmanager
def cpu_construction(enabled=True, *, network_type=None):
    """A constructor wrapper restricted to one instance on the calling thread."""
    if not enabled:
        yield None
        return
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise RuntimeError("Cosmos CPU loading requires single-rank execution")
    if network_type is None:
        from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
        network_type = Cosmos3VFMNetwork
    owner = CPUConstruction()
    thread_id = threading.get_ident()
    with _construction_lock:
        original = network_type.__init__

        def construct(net, *args, **kwargs):
            original(net, *args, **kwargs)
            if threading.get_ident() == thread_id:
                owner.claim(net)

        network_type.__init__ = construct
        try:
            yield owner
        except BaseException:
            if owner.residency is not None:
                owner.residency.close()
            raise
        finally:
            network_type.__init__ = original
            owner.cleanup()


def finalize_service(service, construction, *, precision, device):
    from cosmos_framework.model.generator.mot.unified_mot import MoTDecoderLayer

    from instinctflash.runtime.desktop_fp8 import (
        backend_for_capability,
        target_for_capability,
    )
    from instinctflash.runtime.module_residency import install_module_residency

    model = service.model
    construction.verify(model)
    capability = torch.cuda.get_device_capability(device)
    if (target_for_capability(capability) is None
            or (torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1)):
        raise RuntimeError("Cosmos desktop residency requires single-rank SM89 or SM120 execution")
    if (model.config.compile.enabled or getattr(model.config.compile, "use_cuda_graphs", False)
            or model.config.lora_enabled):
        raise ValueError("Cosmos desktop residency requires the native eager, non-LoRA network")
    layers = list(model.net.language_model.model.layers)
    if not layers or any(type(layer) is not MoTDecoderLayer for layer in layers):
        raise ValueError("Cosmos desktop residency requires native MoTDecoderLayer modules")
    implementation = backend_for_capability(capability)
    install_fp8 = getattr(implementation, f"install_{target_for_capability(capability).prefix}_fp8")
    fp8 = (install_fp8(model, "cosmos3_policy", device=device,
                            storage_device="cpu", include_mlp=True)
           if precision == "fp8" else None)
    owner = install_module_residency(model.net, layers, device=device,
                                     reserve_bytes=INFERENCE_RESERVE_BYTES)
    owner.receipt.update(checkpoint_materialization="cpu",
                         native_buffer_initialization="unchanged CUDA")
    construction.residency = owner
    service._ifl_sm89_residency = owner
    return fp8


def build_native_service(create_service, *, nano, device="cuda"):
    """Retain a stock benchmark service factory with the same owned residency."""
    from instinctflash.runtime.desktop_fp8 import target_for_capability

    from .nano_action_only import (
        nano_action_only_construction,
        verify_nano_action_only_model,
    )

    if target_for_capability(torch.cuda.get_device_capability(device)) is None:
        raise RuntimeError("Cosmos native residency requires an SM89 or SM120 device")
    with nano_action_only_construction(enabled=nano), cpu_construction() as construction:
        service = create_service()
        elided = verify_nano_action_only_model(service.model) if nano else 0
        finalize_service(service, construction, precision="native", device=device)
        service._ifl_native_optimizations = {"elided_lm_head_bytes": elided}
        return service
