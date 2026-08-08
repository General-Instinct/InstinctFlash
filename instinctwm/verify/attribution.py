"""Operator attribution: "where does this operator come from?" as a first-class answer.

An operator total is not a target. `aten::copy_ 66.4 ms` was the largest line in the Fast profile for
weeks, and it turned out to be 82% one thing — `slow_conv_dilated3d`'s `vol2col` lowering — which was
fixed by a layout decision two layers away, with no copy kernel written. Ranking by operator total
picked the wrong work; ranking by CALLSITE would have pointed at the convolution immediately.

So this reports, per (operator, callsite): calls, bytes moved, the shapes seen, and measured exclusive
device time.

FOUR EARLIER ATTEMPTS FAILED, AND THE DESIGN IS SHAPED BY HOW:

  1. `profile(with_stack=True)` then reading `ev.stack`      -> empty for these ops
  2. `key_averages(group_by_stack_n=6)`                      -> empty frame lists
  3. monkeypatching `torch.Tensor.copy_` / `.fill_`          -> ZERO calls counted, because these
     copies are emitted INSIDE C++ ops (slice assignment, `.to()`, conv lowering) and never pass
     through a Python `.copy_()`
  4. `TorchDispatchMode` counting                            -> worked, but saw 7,589 calls where the
     profiler counted 64,391. About 12%.

Attempt 4 is the trap this module exists to close. It produced a confident table, its top row read
"47.4% of watched calls", and that 47.4% was 47.4% of an unrepresentative eighth. **A partial sample
that does not announce itself is worse than no sample**, so `report()` computes COVERAGE per operator
against the profiler's independent count and `format_table()` refuses to rank an operator whose
coverage is too low to support a decision.

HOW EXCLUSIVE TIME IS OBTAINED. Dispatch interception alone cannot see device time, and putting a CUDA
sync around 64,000 ops would measure the syncs. Instead each intercepted op is wrapped in a
`record_function` scope named for its callsite, so the PROFILER attributes device time to that scope
itself — measured, not apportioned. Nesting is handled by the profiler's own self-time accounting,
which is what makes the number exclusive rather than inclusive.

    from instinctwm.verify.attribution import attribute

    rep = attribute(run_one_cycle, watch=("copy_", "cat", "addmm"), repeats=3)
    print(rep.format_table())
    print(rep.coverage_warnings())
"""

from __future__ import annotations

import collections
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

#: Below this fraction of an operator's true call count, a callsite breakdown is not rankable.
MIN_COVERAGE = 0.60
#: ABOVE this, it is not rankable either, and the first version of this module missed that entirely.
#: Coverage cannot exceed 100% for a stationary workload -- the attributed pass and the ground-truth
#: pass run the same code. When it does, the two passes disagreed because the operator's call count
#: DEPENDS ON STATE that advanced between them. `cat` measured 121%: the ring-wrap branch in
#: ring_kv.py fires only during the wrap transition, so its count varies cycle to cycle, and a
#: 2-cycle sample cannot characterise it. Over-coverage is a non-stationarity alarm, not a rounding
#: artefact, and an operator that trips it must be re-measured over many cycles before it is ranked.
MAX_COVERAGE = 1.10

_SCOPE = re.compile(r"^iwm::(?P<op>[^@]+)@(?P<site>.+)$")
_SELF = str(Path(__file__).resolve())


@dataclass
class Row:
    """One (operator, callsite) pair."""

    operator: str
    callsite: str
    calls: int = 0
    nbytes: int = 0
    shapes: collections.Counter = field(default_factory=collections.Counter)
    exclusive_us: float = 0.0

    def mib(self) -> float:
        return self.nbytes / 2 ** 20

    def dominant_shape(self) -> tuple:
        return self.shapes.most_common(1)[0][0] if self.shapes else ()

    def n_shapes(self) -> int:
        return len(self.shapes)


@dataclass
class Report:
    rows: list[Row]
    #: operator -> calls the profiler counted, independent of attribution
    true_calls: dict[str, int]
    #: operator -> device us the profiler attributes to the operator as a whole
    true_us: dict[str, float]
    cycles: int = 1

    def by_operator(self) -> dict[str, list[Row]]:
        """Callsites per operator, most significant first.

        Ranked by exclusive device time -- EXCEPT when the operator has none. A metadata operator
        (`view`, `t`, `squeeze`, `as_strided`) launches no kernel, so every one of its callsites scores
        0.00 ms and sorting by time leaves the printed "top N" in insertion order: arbitrary rows
        presented where the dominant ones belong. Those operators are exactly Layer 6's targets, and
        their significance is call count, not time. So fall back to calls when the operator is
        time-free.
        """
        out = collections.defaultdict(list)
        for r in self.rows:
            out[r.operator].append(r)
        ranked = {}
        for k, v in out.items():
            timeless = sum(r.exclusive_us for r in v) <= 0.0
            ranked[k] = sorted(v, key=(lambda r: -r.calls) if timeless
                               else (lambda r: -r.exclusive_us))
        return ranked

    def coverage(self, operator: str) -> float:
        """Attributed calls / calls the profiler saw. The number attempt 4 did not compute."""
        true = self.true_calls.get(operator, 0)
        got = sum(r.calls for r in self.rows if r.operator == operator)
        return (got / true) if true else 0.0

    def rankable(self, operator: str) -> bool:
        return MIN_COVERAGE <= self.coverage(operator) <= MAX_COVERAGE

    def coverage_warnings(self) -> str:
        under = [(op, self.coverage(op)) for op in sorted(self.by_operator())
                 if self.coverage(op) < MIN_COVERAGE]
        over = [(op, self.coverage(op)) for op in sorted(self.by_operator())
                if self.coverage(op) > MAX_COVERAGE]
        if not under and not over:
            return "coverage: every reported operator is rankable."
        lines = []
        if under:
            lines.append(f"PARTIAL -- not rankable (< {MIN_COVERAGE:.0%} of calls attributed):")
            for op, c in under:
                lines.append(f"  {op:<20} {c:6.1%} of {self.true_calls.get(op, 0)} calls. Its "
                             f"callsites are likely inside C++; a different instrument is needed.")
        if over:
            lines.append(f"NON-STATIONARY -- not rankable (> {MAX_COVERAGE:.0%} attributed):")
            for op, c in over:
                lines.append(f"  {op:<20} {c:6.1%}. Coverage cannot exceed 100% for a stationary "
                             f"workload, so this operator's call count depends on state that")
                lines.append(f"  {'':<20} advanced between the two passes. Re-measure over many "
                             f"cycles before ranking it; a short sample describes one ring position.")
        return "\n".join(lines)

    def format_table(self, top: int = 24) -> str:
        out = [f"{'operator':<18}{'ms/cyc':>8}{'calls/cyc':>10}{'MiB/cyc':>9}"
               f"{'shapes':>7}  callsite"]
        out.append("-" * 118)
        # Operators ordered by device time, then by calls -- the second term is what orders the
        # metadata operators, which all score 0.00 ms and would otherwise print in dict order.
        for op, rows in sorted(self.by_operator().items(),
                               key=lambda kv: (-sum(r.exclusive_us for r in kv[1]),
                                               -sum(r.calls for r in kv[1]))):
            cov = self.coverage(op)
            tot = sum(r.exclusive_us for r in rows) / 1000 / self.cycles
            flag = ("" if self.rankable(op) else
                    "   [PARTIAL, NOT RANKABLE]" if cov < MIN_COVERAGE else
                    "   [NON-STATIONARY, NOT RANKABLE]")
            out.append(f"{op:<18}{tot:>8.2f}{sum(r.calls for r in rows) / self.cycles:>10.0f}"
                       f"{sum(r.nbytes for r in rows) / 2**20 / self.cycles:>9.1f}"
                       f"{'':>7}  <all callsites>  coverage {cov:.0%}{flag}")
            for r in rows[:top]:
                out.append(f"{'':<18}{r.exclusive_us / 1000 / self.cycles:>8.2f}"
                           f"{r.calls / self.cycles:>10.0f}{r.mib() / self.cycles:>9.1f}"
                           f"{r.n_shapes():>7}  {r.callsite}")
                if r.shapes:
                    out.append(f"{'':<52}  dominant {r.dominant_shape()}")
        return "\n".join(out)


class _Attributor:
    """TorchDispatchMode that tags each watched op with a `record_function` scope naming its caller."""

    def __init__(self, watch: Iterable[str], skip_prefixes: Iterable[str]):
        self.watch = set(watch)
        self.skip = tuple(skip_prefixes)
        self.rows: dict[tuple[str, str], Row] = {}
        self.enabled = False

    def _callsite(self) -> str:
        # Skip THIS module by exact path. An earlier version used `fn.endswith("attribution.py")`,
        # which also matched `tests/test_attribution.py` and silently skipped the entire caller --
        # every call resolved to "[?] unattributed". A suffix match on a filename is not an identity
        # check, and the synthetic test is what caught it.
        # NO SLICE. An earlier version used `[:-4]` to drop "the dispatch machinery", but
        # extract_stack() returns oldest-first, so [:-4] drops the four INNERMOST frames -- which is
        # precisely where the caller is. It reported the caller's caller instead: `alpha` and `beta`
        # both collapsed onto the test function that invoked them, and two callsites read as one. The
        # filename filter below already excludes this module and torch, so no slice is needed.
        for f in reversed(traceback.extract_stack()):
            fn = f.filename
            if "/torch/" in fn or fn == _SELF or "_python_dispatch" in fn:
                continue
            if any(s in fn for s in self.skip):
                continue
            tag = ("iwm" if "/instinctwm/" in fn else
                   "lingbot" if "/wan_va/" in fn or "/lingbot" in fn else
                   "diffusers" if "diffusers" in fn else "app")
            return f"[{tag}] {Path(fn).name}:{f.lineno} {f.name}"
        return "[?] unattributed"

    @staticmethod
    def _bytes_and_shape(args) -> tuple[int, tuple]:
        n, shp = 0, ()
        for a in args:
            items = a if isinstance(a, (list, tuple)) else (a,)
            for t in items:
                if hasattr(t, "numel") and hasattr(t, "element_size"):
                    try:
                        n += t.numel() * t.element_size()
                        if not shp:
                            shp = tuple(t.shape)
                    except Exception:
                        pass
        return n, shp


def attribute(workload: Callable[[], None], *, watch: Iterable[str],
              repeats: int = 1, skip_prefixes: Iterable[str] = ()) -> Report:
    """Run `workload` under dispatch interception plus the profiler, and attribute by callsite.

    `watch` is a list of bare operator names (`"copy_"`, `"cat"`). Watching everything is possible and
    very slow; the point of naming them is that a target has already been identified by total and the
    question is where it comes from.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from torch.utils._python_dispatch import TorchDispatchMode

    att = _Attributor(watch, skip_prefixes)

    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func).split(".")[-2] if "." in str(func) else str(func)
            if not att.enabled or name not in att.watch:
                return func(*args, **kwargs)
            site = att._callsite()
            key = (name, site)
            row = att.rows.get(key)
            if row is None:
                row = att.rows[key] = Row(operator=name, callsite=site)
            nb, shp = att._bytes_and_shape(args)
            row.calls += 1
            row.nbytes += nb
            if shp:
                row.shapes[shp] += 1
            # The profiler attributes device time to this named scope, so the exclusive time below is
            # MEASURED rather than apportioned from the operator total.
            with record_function(f"iwm::{name}@{site}"):
                return func(*args, **kwargs)

    workload()                                     # warm the path before measuring
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # PASS 1: ground truth, with NO interception. The dispatch mode re-dispatches each watched op
    # inside its record_function scope, so a profiler running alongside it counts every watched call
    # twice -- coverage read 50% on a workload that was in fact fully attributed. True counts have to
    # come from an uninstrumented pass.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as base:
        for _ in range(repeats):
            workload()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # PASS 2: attribution.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with Mode():
            att.enabled = True
            for _ in range(repeats):
                workload()
            att.enabled = False
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Scope device time -> the row it names. Independent of the dispatch-side counting, so the two
    # can disagree and the disagreement is visible.
    for e in prof.key_averages():
        m = _SCOPE.match(e.key)
        if not m:
            continue
        key = (m.group("op"), m.group("site"))
        if key in att.rows:
            att.rows[key].exclusive_us += (getattr(e, "self_device_time_total", 0) or 0)

    true_calls: dict[str, int] = {}
    true_us: dict[str, float] = {}
    for e in base.key_averages():
        if e.key.startswith("aten::"):
            op = e.key[len("aten::"):]
            if op in att.watch:
                true_calls[op] = true_calls.get(op, 0) + e.count
                true_us[op] = true_us.get(op, 0.0) + (getattr(e, "self_device_time_total", 0) or 0)

    return Report(rows=list(att.rows.values()), true_calls=true_calls,
                  true_us=true_us, cycles=repeats)
