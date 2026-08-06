"""Executors: different ways to realize the same Plan.

    EagerExecutor       interpret the plan unit by unit. The correctness ORACLE.
    GraphExecutor       capture each unit once, replay thereafter.

The point of the seam is that a megakernel executor later is a third entry in this list rather
than a rewrite: a persistent kernel walking a device-side work queue is the same architecture as
a graph replay walking a driver-side one.

Measured motivation (docs/EXECUTION_ENGINE.md): per-op cost is 6.2 us, of which 83.6% is
`cudaLaunchKernel` itself. Graph replay drops the whole per-op cost to ~1.17 us and makes CPU
enqueue for an 8-block stack 0.031 ms instead of 7.5 ms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from instinctwm.executors.binding import leaf_shapes, spec_key
from instinctwm.planners.plan import Plan


class EagerExecutor:
    """Runs the plan directly. No capture, no bound buffers, no assumptions.

    Every other executor is verified against this one.
    """
    name = "eager"

    def __init__(self, plan: Plan, device):
        self.plan, self.device = plan, device

    def prepare(self) -> None:
        pass

    def run(self, unit_key: str, **inputs) -> object:
        u = self._unit(unit_key)
        with torch.no_grad():
            return u.fn(*(inputs[n] for n in u.inputs))

    def _unit(self, key):
        for u in self.plan.units:
            if u.key == key:
                return u
        raise KeyError(f"no unit {key!r}; have {[u.key for u in self.plan.units]}")


class CaptureFailed(RuntimeError):
    """Capture failed. The message names the unit; the cause is almost always a host sync."""


class GraphExecutor:
    """Captures a CUDA graph per (unit, dynamic extent) and replays it.

    Inputs are copied into buffers bound at plan time. This is not a convenience -- it is the
    safety property. A replayed graph reads the addresses it was captured with, so binding by name
    through the plan is what prevents a mis-bound input from silently producing a wrong answer.

    HOW KV GROWTH IS HANDLED, and why not the obvious way.

    The KV live set grows every cycle, so a captured slice shape goes stale. The obvious answer is
    to pad to full capacity and mask the tail, giving one permanent graph. Measured, that is
    **not bit-exact**: masked SDPA over a padded extent differs from unmasked SDPA over the exact
    prefix on 750 of 196,608 elements (max|d| 4.883e-04), because the masked path re-trees the
    reduction. It also costs 1.11-1.19x in attention work.

    Capture, meanwhile, is cheap: 29.4 ms for a 30-block stack, 42 MB of graph pool. So this
    executor **recaptures when the extent changes** and keeps exact-length slices. Bit-exact, no
    padding penalty, and at ~2 graphs per cycle it costs ~2.3% of a 2540 ms cycle.

    `extent` is whatever integer the unit's shapes depend on (here, KV live length). Bucketing it
    trades a little attention work for fewer recaptures; `extent_bucket` exposes that dial without
    changing anything else.
    """
    name = "graph"

    def __init__(self, plan: Plan, device, extent_bucket: int = 1, max_graphs: int = 8):
        self.plan, self.device = plan, device
        self.extent_bucket = max(1, extent_bucket)
        self.max_graphs = max_graphs
        self.bound: dict[str, torch.Tensor] = {}
        self.graphs: dict[tuple[str, int], torch.cuda.CUDAGraph] = {}
        self.outputs: dict[tuple[str, int], torch.Tensor] = {}
        self.blockers: dict[str, str] = {}
        self.stable: dict = {}
        self.n_captures = 0

    def prepare(self, warmup: int = 3, extent: int = 0) -> None:
        """Declared buffers are optional now.

        A unit's real bindable surface is whatever its binder flattens out of the arguments it is
        actually called with, which cannot be known before the first call for a model whose unit
        is a nested structure. So binding is lazy: `run` captures on first sight of a (spec,
        shapes, extent) it has not seen.
        """
        for b in self.plan.buffers:
            self.bound[b.name] = b.allocate(self.device)
        self.plan.plan_buffer.allocate(self.device)

    def _bucket(self, extent: int) -> int:
        return (extent // self.extent_bucket) * self.extent_bucket

    def _capture(self, u, slot, args, warmup: int = 3):
        # Warm up on a side stream: cuBLAS workspace allocation, autotuning and lazy module init
        # all happen on first call and are NOT capturable.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup):
                u.fn(*args)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g), torch.no_grad():
                out = u.fn(*args)
        except Exception as ex:
            self.blockers[u.key] = f"{type(ex).__name__}: {str(ex)[:160]}"
            raise CaptureFailed(
                f"unit {u.key!r} is not capturable: {self.blockers[u.key]}\n"
                f"  The usual cause is a host synchronization or a data-dependent shape "
                f"(nonzero/item/masked_select) somewhere inside the unit.") from ex

        while len(self.graphs) >= self.max_graphs:         # bound the graph pool
            oldest = next(iter(self.graphs))
            del self.graphs[oldest], self.outputs[oldest], self.stable[oldest]
        self.graphs[slot] = g
        self.outputs[slot] = out
        self.n_captures += 1
        return out

    # -- generic binding ------------------------------------------------------------------------

    def _slots(self, u, ext, spec_sig):
        return (u.key, ext, spec_sig)

    def _bind_and_key(self, u, values):
        """Flatten every argument, allocate stable leaves once, and build the graph key.

        The key carries the SPEC of each argument (container shape plus every non-tensor value)
        and the leaf shapes/dtypes. Both are frozen by capture, so both must key it. This is where
        Cosmos3's `SplitInfo`, dict keys and `max_num_tokens` end up -- without the executor
        knowing any of those names.
        """
        flat, specs = [], []
        for v in values:
            leaves, spec = u.binder.flatten(v)
            flat.append((leaves, spec))
            specs.append((spec_key(spec), leaf_shapes(leaves)))
        return flat, tuple(specs)

    def run(self, unit_key: str, extent: int = 0, **inputs) -> object:
        u = self._unit(unit_key)
        ext = self._bucket(extent)
        values = [inputs[n] for n in u.inputs]
        flat, spec_sig = self._bind_and_key(u, values)
        slot = self._slots(u, ext, spec_sig)

        if slot not in self.graphs:
            # Allocate a stable clone of every tensor leaf, rebuild the argument structure around
            # those clones, and capture against it. The structure is rebuilt, not copied, so host
            # metadata (dict keys, SplitInfo, ints) is carried through by the binder untouched.
            stable = []
            for leaves, spec in flat:
                owned = [t.detach().clone() for t in leaves]
                stable.append((owned, spec))
            self.stable[slot] = stable
            args = [u.binder.unflatten(o, sp) for o, sp in stable]
            self._capture(u, slot, args)

        for (leaves, _spec), (owned, _s2) in zip(flat, self.stable[slot]):
            for dst, src in zip(owned, leaves):
                dst.copy_(src)                       # bind by SLOT, never by pointer
        self.graphs[slot].replay()
        return self.outputs[slot]

    def _unit(self, key):
        for u in self.plan.units:
            if u.key == key:
                return u
        raise KeyError(key)


# -------------------------------------------------------------------------------------------
# verification and benchmarking, built into the engine rather than bolted on per pass
# -------------------------------------------------------------------------------------------

@dataclass
class UnitResult:
    key: str
    max_abs_delta: float
    bit_exact: bool
    eager_ms: float
    exec_ms: float
    eager_enqueue_ms: float
    exec_enqueue_ms: float

    @property
    def speedup(self) -> float:
        return self.eager_ms / self.exec_ms if self.exec_ms else float("nan")


def _bench(fn, it=30, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    enq = (time.perf_counter() - t0) / it
    torch.cuda.synchronize()
    return enq * 1e3, (time.perf_counter() - t0) / it * 1e3


def verify_and_bench(plan: Plan, executor, device, inputs_for) -> list[UnitResult]:
    """Run every unit under both the executor and the eager oracle. Bit-exactness is the gate.

    `inputs_for(unit)` returns the dict of input tensors for that unit.
    """
    oracle = EagerExecutor(plan, device)
    results = []
    for u in plan.units:
        ins = inputs_for(u)
        ref = oracle.run(u.key, **ins).clone()
        got = executor.run(u.key, **ins)
        d = (got.float() - ref.float()).abs().max().item()
        e_enq, e_ms = _bench(lambda: oracle.run(u.key, **ins))
        x_enq, x_ms = _bench(lambda: executor.run(u.key, **ins))
        results.append(UnitResult(u.key, d, d == 0.0, e_ms, x_ms, e_enq, x_enq))
    return results


def report(results: list[UnitResult]) -> str:
    out = [f"{'unit':<28} {'eager':>9} {'exec':>9} {'speedup':>8} {'enq eager':>10} "
           f"{'enq exec':>9} {'max|d|':>10} {'gate':>9}"]
    for r in results:
        out.append(f"{r.key:<28} {r.eager_ms:7.3f}ms {r.exec_ms:7.3f}ms {r.speedup:7.2f}x "
                   f"{r.eager_enqueue_ms:8.3f}ms {r.exec_enqueue_ms:7.3f}ms {r.max_abs_delta:10.3e} "
                   f"{'BITEXACT' if r.bit_exact else 'FAIL':>9}")
    return "\n".join(out)
