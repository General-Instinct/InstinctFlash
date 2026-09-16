"""Frozen inference with CPU master tensors and transient CUDA module copies.

Call only after checkpoint loading and final dtype selection. Each selected block
must be an independent descendant of root; arithmetic and native module classes
are retained. Persistent and nonpersistent registered buffers keep their dtypes.
The caller owns evaluation mode, serial inference, and closing this scheduler.
"""
from __future__ import annotations

import weakref
import torch


def _slots(module):
    for child in module.modules():
        for mapping in (child._parameters, child._buffers):
            for name, value in mapping.items():
                if value is not None:
                    yield mapping, name, value


def _storage_key(tensor):
    if tensor.device.type == "meta":
        raise ValueError("Cannot place an unmaterialized module tensor")
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        raise ValueError("Module residency requires unsharded native tensors")
    return tensor.device, tensor.untyped_storage()._cdata


def _storage_sizes(module):
    return {_storage_key(t): t.untyped_storage().nbytes() for _, _, t in _slots(module)}


def resident_prefix(layer_bytes, base_bytes, budget_bytes):
    """Reserve the largest streaming layer as well as all persistent tensors."""
    for count in range(len(layer_bytes), -1, -1):
        transient = max(layer_bytes[count:], default=0)
        if base_bytes + sum(layer_bytes[:count]) + transient <= budget_bytes:
            return count
    raise RuntimeError("Base tensors and one streaming block exceed the device weight budget")


def _move_except(module, device, excluded):
    if module in excluded:
        return
    for child in module.children():
        _move_except(child, device, excluded)
    # Move only this module's own tensors; descendants were handled above.
    module._apply(lambda tensor: tensor.to(device=device), recurse=False)


class ModuleResidency:
    def __init__(self, net, layers, *, device, budget_bytes, reserve_bytes, outside_bytes=0):
        self.net, self.device = net, torch.device(device)
        layers = list(layers)
        names = {module: name for name, module in net.named_modules()}
        descendants = set()
        for layer in layers:
            children = set(layer.modules())
            if layer is net or layer not in names or descendants.intersection(children):
                raise ValueError("Residency blocks must be distinct, nonoverlapping descendants of root")
            descendants.update(children)
        self.handles = []
        self.states = {}
        self.active = None
        self.active_copies = {}
        self.active_stream = None
        self.closed = False
        self.calls = 0
        self.transferred_bytes = 0
        sizes = _storage_sizes(net)
        layer_sizes = [_storage_sizes(layer) for layer in layers]
        seen = set()
        for layer, storage in zip(layers, layer_sizes):
            objects = {}
            for _, _, tensor in _slots(layer):
                key = _storage_key(tensor)
                if (storage[key] and key in objects and objects[key] is not tensor):
                    raise ValueError("Residency cannot copy distinct tensor views sharing block storage")
                objects[key] = tensor
            if seen.intersection(storage):
                raise ValueError("Module residency cannot split shared decoder-layer storage")
            seen.update(storage)
        # Shared layer/root storage would otherwise be counted as a resident
        # tensor and moved behind the CPU owner's back.
        layer_modules = {child for layer in layers for child in layer.modules()}
        for child in net.modules():
            if child not in layer_modules:
                for mapping in (child._parameters, child._buffers):
                    if any(value is not None and _storage_key(value) in seen
                           for value in mapping.values()):
                        raise ValueError("Module decoder storage aliases a non-decoder tensor")
        layer_bytes = [sum(storage.values()) for storage in layer_sizes]
        base_bytes = sum(sizes.values()) - sum(layer_bytes)
        count = resident_prefix(layer_bytes, base_bytes, budget_bytes)
        streaming = list(layers[count:])
        self.receipt = {
            "schema": "instinctflash.module_residency.v1",
            "device": str(self.device),
            "transfer_policy": "CPU master tensors with transient CUDA decoder copies",
            "block_paths": [names[layer] for layer in layers],
            "registered_buffer_policy": "preserve dtype; reject mutation, including nonpersistent buffers",
            "weight_bytes": sum(sizes.values()), "base_weight_bytes": base_bytes,
            "resident_blocks": list(range(count)),
            "streamed_blocks": list(range(count, len(layers))),
            "block_bytes": layer_bytes, "weight_budget_bytes": budget_bytes,
            "inference_reserve_bytes": reserve_bytes, "outside_network_bytes": outside_bytes,
            "planned_peak_weight_bytes": base_bytes + sum(layer_bytes[:count])
                + max(layer_bytes[count:], default=0),
            "arithmetic": "unchanged by residency", "device_validation": "pending",
        }
        # Already packed FP8 buffers can be on CUDA; reclaim selected layers
        # before placing any remaining native weights. No dtype cast occurs.
        for layer in streaming:
            layer.to(device="cpu")
            self.states[layer] = list(_slots(layer))
        try:
            _move_except(net, self.device, set(streaming))
            reference = weakref.ref(self)
            for layer in streaming:
                def before(module, args, reference=reference):
                    owner = reference()
                    if owner is None:
                        raise RuntimeError("Module residency owner no longer exists")
                    owner.enter(module)

                def after(module, args, output, reference=reference):
                    owner = reference()
                    if owner is not None:
                        owner.leave(module)

                self.handles.append(layer.register_forward_pre_hook(before, prepend=True))
                self.handles.append(layer.register_forward_hook(after, always_call=True))
        except BaseException:
            self.close()
            raise

    def enter(self, layer):
        if self.closed or self.active is not None:
            raise RuntimeError("Module residency requires an open, serial inference call")
        if torch.is_grad_enabled() or layer.training:
            raise RuntimeError("Module CPU residency supports inference only")
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Module CPU residency cannot execute inside CUDA graph capture")
        self.active = layer
        self.active_stream = (torch.cuda.current_stream(self.device)
                              if self.device.type == "cuda" else None)
        copies = self.active_copies
        try:
            for mapping, name, original in self.states[layer]:
                if mapping[name] is not original:
                    raise RuntimeError("Module streamed weights changed after residency installation")
                if id(original) not in copies:
                    # Ordinary tensors retain version counters even when the
                    # caller uses inference_mode. Frozen registered buffers must
                    # not silently lose in-place updates when masters return.
                    with torch.inference_mode(False), torch.no_grad():
                        value = original.to(device=self.device, copy=True)
                        if isinstance(original, torch.nn.Parameter):
                            value = torch.nn.Parameter(value, requires_grad=original.requires_grad)
                    copies[id(original)] = (value, value._version)
                    self.transferred_bytes += original.numel() * original.element_size()
                mapping[name] = copies[id(original)][0]
            self.calls += 1
        except BaseException:
            self.leave(layer, check_mutation=False)
            raise

    def leave(self, layer, *, check_mutation=True):
        if self.active is layer:
            changed = False
            stream_error = None
            for mapping, name, original in self.states[layer]:
                copied = self.active_copies.get(id(original))
                if copied is not None:
                    value, version = copied
                    changed |= mapping[name] is not value or value._version != version
                    if value.device.type == "cuda":
                        # The allocator may reuse these tensors only after the
                        # forward's CUDA work completes; do not synchronize every
                        # block or retain an entire request's CUDA weight copies.
                        try:
                            value.record_stream(self.active_stream)
                            value.record_stream(torch.cuda.current_stream(self.device))
                        except BaseException as error:
                            # Even a device failure must restore every native
                            # dictionary entry before propagating the error.
                            stream_error = stream_error or error
                mapping[name] = original
            self.active = None
            self.active_copies.clear()
            self.active_stream = None
            if stream_error is not None:
                raise stream_error
            if changed and check_mutation:
                raise RuntimeError("Frozen residency block mutated a registered tensor")

    def report(self):
        return {**self.receipt, "streamed_block_calls": self.calls,
                "host_to_device_bytes": self.transferred_bytes, "closed": self.closed}

    def close(self):
        if self.closed:
            return
        if self.active is not None:
            raise RuntimeError("Cannot close module residency during an active forward")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.states.clear()
        self.net = None
        self.closed = True


def install_module_residency(root, blocks, *, device="cuda", reserve_bytes=8 << 30):
    """Choose a resident prefix using actual storage bytes and available capacity.

    The reserve is a family-owned request-work allowance, not proof of fit.
    Device qualification must still measure the original input and history sizes.
    """
    device = torch.device(device)
    if device.type != "cuda" or reserve_bytes < 0:
        raise ValueError("Module residency requires a CUDA target and nonnegative reserve")
    sizes = _storage_sizes(root)
    placed = sum(size for (location, _), size in sizes.items()
                 if location.type == "cuda")
    free, total = torch.cuda.mem_get_info(device)
    outside = max(0, total - free - placed)
    return ModuleResidency(root, blocks, device=device,
                           budget_bytes=total - outside - reserve_bytes,
                           reserve_bytes=reserve_bytes, outside_bytes=outside)
