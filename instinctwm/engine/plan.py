"""The Plan: what will run, as data rather than as monkeypatches.

P001-P004 patch methods onto the live server at import time. That was fine for four passes and
does not survive forty: patches collide, order matters invisibly, and nothing can be inspected
before it runs. A Plan is the same information as a data structure -- printable, diffable,
hashable, and verifiable before a single kernel launches.

Deliberately small. This is the seam, not the finished abstraction; it exists so graph capture is
written against a Plan instead of against `wan_va_server`. Everything here earns its place by
being required for capture to be *safe*:

  BufferSpec      A captured graph reads whatever sits at the captured addresses. If an input is
                  written to a fresh tensor instead of the bound one, replay silently computes on
                  stale data and returns a plausible wrong answer. Naming buffers and binding
                  through the Plan is what makes that a type error instead of a debugging session.

  PlanBuffer      Anything varying faster than capture time cannot live in a shape, because shapes
                  are frozen at capture. It lives here, device-resident, read by kernels inside the
                  graph, updated by one small H2D per cycle. (FlashInfer's plan/run seam. P003 was
                  already an instance of this -- it replaced nonzero() with two host integers.)

  CaptureUnit     The granularity of capture. One unit -> one graph per shape key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class BufferSpec:
    """A tensor with a stable address for the lifetime of the plan."""
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    def allocate(self, device) -> torch.Tensor:
        return torch.zeros(self.shape, dtype=self.dtype, device=device)


@dataclass(frozen=True)
class CaptureUnit:
    """One capturable region of execution.

    `fn` takes the bound input tensors positionally and returns one tensor. It must not allocate
    anything whose address the caller keeps, must not sync, and must not branch on data.
    """
    name: str
    fn: Callable[..., torch.Tensor]
    inputs: tuple[str, ...]
    output: str
    #: distinguishes graphs that differ only by shape (e.g. "video" vs "action")
    shape_key: str = "default"

    @property
    def key(self) -> str:
        return f"{self.name}/{self.shape_key}"


@dataclass
class PlanBuffer:
    """Device-resident scalars the graph reads. The only legal dynamism inside a capture.

    Fields are int32 slots in one small tensor so updating them is a single H2D. Reading them
    inside a kernel is what lets one captured graph serve a changing KV length without recapture.
    """
    fields: tuple[str, ...] = ()
    _tensor: torch.Tensor | None = None
    _host: torch.Tensor | None = None

    def allocate(self, device) -> None:
        n = max(len(self.fields), 1)
        self._tensor = torch.zeros(n, dtype=torch.int32, device=device)
        self._host = torch.zeros(n, dtype=torch.int32, device="cpu").pin_memory()

    def set(self, **values: int) -> None:
        """Stage values on the host. Call `commit()` once before replay."""
        for k, v in values.items():
            self._host[self.fields.index(k)] = v

    def commit(self) -> None:
        self._tensor.copy_(self._host, non_blocking=True)

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor


@dataclass
class Plan:
    """Everything the executor needs, and nothing about how it will be executed."""
    model_id: str
    units: tuple[CaptureUnit, ...]
    buffers: tuple[BufferSpec, ...]
    plan_buffer: PlanBuffer = field(default_factory=PlanBuffer)
    notes: Mapping[str, str] = field(default_factory=dict)

    def buffer(self, name: str) -> BufferSpec:
        for b in self.buffers:
            if b.name == name:
                return b
        raise KeyError(f"no buffer {name!r} in plan; declared: {[b.name for b in self.buffers]}")

    def describe(self) -> str:
        out = [f"Plan[{self.model_id}]  {len(self.units)} units, {len(self.buffers)} buffers"]
        total = sum(
            torch.empty(0, dtype=b.dtype).element_size() * max(1, int(torch.tensor(b.shape).prod()))
            for b in self.buffers)
        out.append(f"  bound memory: {total/1e6:.1f} MB")
        for u in self.units:
            out.append(f"  unit {u.key:<28s} in={list(u.inputs)} out={u.output}")
        if self.plan_buffer.fields:
            out.append(f"  plan buffer: {list(self.plan_buffer.fields)}")
        for k, v in self.notes.items():
            out.append(f"  note {k}: {v}")
        return "\n".join(out)
