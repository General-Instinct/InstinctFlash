"""The per-family adapter surface: what a model family must DECLARE to be distillable here.

The serving side already has this discipline — `adapters/base.py` declares execution facts and
the optimizer derives the rewrites. Distillation needs five more declarations, and each one is
motivated by a way the campaign has already seen a training run go wrong or a paper go unread:

  streams()            Every published method conditions the whole tensor on ONE scalar time;
                       WAMs carry a VECTOR of per-modality times (video shift-5, action
                       shift-1). Any recipe must therefore be instantiated per stream —
                       Flash-WAM's discipline (methods memo §1.1, C4) — so the framework's
                       unit of work is a stream, never a model.
  schedule_grid()      Few-step quality is set by WHICH t-grid points the sampler visits, not
                       how many (memo §2.2: pi05 nfe1 == nfe10 while nfe2 is −10pp — the
                       mid-grid anchor is the weak point). Training must build targets on the
                       grid the DEPLOYED sampler will visit; re-deriving a schedule is how a
                       training grid stops matching the sampler (the PDD recipe learned this,
                       `train/recipes/pdd.py:_synthetic_grid`). So the adapter states the grid
                       and the trainer never invents one. The grid is also a free SEARCH axis
                       (hypothesis H3): `schedule_grid(stream, nfe, shift=...)` accepts an
                       override precisely so a shifted grid is an eval-only arm, not a fork.
  trainable_set()      The asymmetric bet (H1) is expressible only if "which parameters may
                       move" is per-stream: video-stream distillation with the action expert
                       BIT-UNTOUCHED is the property a policy certificate cares about, and
                       heads-vs-trunk capacity was the difference between our PDD +0.0 and the
                       open 1V question (E6 vs E4).
  build_student()      Student init = teacher is load-bearing (Flash-WAM's action
                       parametrization at init IS the untrained control); how to clone/freeze
                       is family-specific.
  consistency_target() The teacher-side target for one step — where H2 (distill from the
                       CERTIFIED reduced-step teacher, not the 79-forward one) plugs in.

Everything here is import-clean without torch: declarations are laptop-checkable, exactly like
the serving adapters, and the capability gate (`train/recipe.py`) runs before any GPU work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from instinctflash.train.recipe import Capabilities


class ExperimentalSeam(NotImplementedError):
    """A declared-but-not-yet-landed training path. Raised loudly, never worked around.

    The skeleton lands adapters whose DECLARATIONS are complete (streams, grids, trainable
    sets — all testable on CPU) while their training halves are seams for campaigns running
    concurrently. Hitting one of these is the framework telling you which experiment owns the
    gap, not a bug.
    """


@dataclass(frozen=True)
class StreamSpec:
    """One denoise stream's schedule and guidance facts. Facts only, no method."""

    name: str
    #: scheduler family; "flow_match_shift" is the SNR-shifted FlowMatchScheduler both LingBot
    #: streams and pi05 use (shift=1.0 degenerates to the linear grid).
    scheduler: str
    #: the stream's SNR shift as DEPLOYED (LingBot video 5.0 / action 1.0, from
    #: va_robotwin_cfg; the same constants as train/recipes/pdd.py DEFAULT_SHIFTS).
    shift: float
    #: the teacher's own step count for this stream (what the model was trained/tuned at).
    teacher_nfe: int
    #: guidance scale as deployed; 1.0 means the negative branch buys nothing.
    guidance_scale: float = 1.0
    #: "cfg" (both branches computed and combined), "positive_only" (negative computed then
    #: DISCARDED — LingBot action), or "none". Mirrors adapters/base.py GuidanceMode.
    guidance_mode: str = "none"
    #: does one forward carry both CFG branches at batch 2 (C4: all LingBot forwards do).
    cfg_batched: bool = False
    #: protocol forwards at t=0 that write KV and are NOT denoise steps — NFE reduction can
    #: never remove them (C5), and cost models that forget them overclaim.
    commit_forwards: int = 0
    #: an NFE for this stream already carried by a paired closed-loop certificate, if any —
    #: the H2 teacher candidate. None means the only certified point is the teacher's own.
    certified_nfe: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class ScheduleGrid:
    """The exact conditioning times a sampler visits at some NFE. THE fact training must match."""

    stream: str
    nfe: int
    #: descending conditioning values in model units (e.g. sigma*1000 for the Wan family).
    times: tuple[float, ...]
    shift: float
    time_scale: float = 1000.0

    def describe(self) -> str:
        ts = ", ".join(f"{t:g}" for t in self.times)
        return f"{self.stream}@{self.nfe} (shift {self.shift:g}): t = {{{ts}}}"


@dataclass(frozen=True)
class TrainableSet:
    """Which parameters a stream's distillation may touch. A hint the trainer enforces.

    `unfreeze` empty means this stream is preserved BY CONSTRUCTION — its weights and schedule
    are not inputs to the run at all, which is a stronger statement than "trained and verified
    unchanged" and is the H1 action-stream position.
    """

    stream: str
    unfreeze: tuple[str, ...] = ()
    keep_frozen: tuple[str, ...] = ()
    note: str = ""

    @property
    def untouched(self) -> bool:
        return not self.unfreeze


def flow_match_times(
    nfe: int, shift: float, *, time_scale: float = 1000.0
) -> tuple[float, ...]:
    """The SNR-shifted flow-match grid, exactly as the deployed scheduler derives it.

    sigma_i = shift * s_i / (1 + (shift - 1) * s_i) over the uniform descending base
    s_i = 1 - i/nfe — the FlowMatchScheduler rule (mapping memo §0.5). Reproduces the deployed
    grids bit-for-bit in the cases we have measured (E8): video shift-5 at 2 steps gives
    t = {1000, 833.33…}; action shift-1 at 4 steps gives the linear {1000, 750, 500, 250}.
    """
    if nfe < 1:
        raise ValueError(f"nfe must be >= 1, got {nfe}")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    times = []
    for i in range(nfe):
        s = 1.0 - i / nfe
        sigma = shift * s / (1.0 + (shift - 1.0) * s)
        times.append(sigma * time_scale)
    return tuple(times)


class FamilyAdapter(ABC):
    """What one model family declares so the family-agnostic machinery can distill it."""

    #: the backbone name, matching the serving adapter / registry vocabulary ("wan_va", "pi05").
    family: str = ""

    # -- declarations (CPU, no weights) ---------------------------------------------------------

    @abstractmethod
    def streams(self) -> tuple[StreamSpec, ...]:
        """The denoise streams, in execution order."""

    def stream(self, name: str) -> StreamSpec:
        for s in self.streams():
            if s.name == name:
                return s
        raise KeyError(
            f"{self.family}: no stream {name!r}; declared: {[s.name for s in self.streams()]}"
        )

    def schedule_grid(self, stream: str, nfe: int, *, shift: float | None = None) -> ScheduleGrid:
        """The exact grid the deployed sampler visits at `nfe` — overridable shift for H3 search.

        Default implementation covers the flow-match-shift schedulers; a family with a
        different scheduler overrides this and keeps the contract: the grid is what SERVES.
        """
        spec = self.stream(stream)
        if spec.scheduler != "flow_match_shift":
            raise ExperimentalSeam(
                f"{self.family}/{stream}: scheduler {spec.scheduler!r} has no grid rule here; "
                "the family adapter must override schedule_grid()"
            )
        use = spec.shift if shift is None else float(shift)
        return ScheduleGrid(
            stream=stream, nfe=nfe, times=flow_match_times(nfe, use), shift=use
        )

    @abstractmethod
    def trainable_set(self, stream: str) -> TrainableSet:
        """Which parameters distilling `stream` may touch."""

    def requires(self) -> Capabilities:
        """The capability gate, checked BEFORE any GPU work (train/recipe.py fails closed).

        Default: no forward-mode AD through attention (C1 — this box deliberately has no
        flash-attn JVP kernel) and no adversarial supervision (C2 — mode collapse in an action
        head is a behavioural failure). A family overriding this to demand either is opting
        into an environment we do not run.
        """
        return Capabilities(jvp_through_attention=False, adversarial=False)

    def default_guidance(self) -> dict[str, tuple[str, float]]:
        """The family's own (mode, scale) per stream -- the base a declaration inherits from.

        Same vocabulary as `descriptors/guidance.py` and the serving AdapterSpec, so the control
        gate canonicalises a schedule's guidance leg identically to the runtime.
        """
        return {s.name: (s.guidance_mode, float(s.guidance_scale)) for s in self.streams()}

    def evidence(self) -> dict[str, Any] | None:
        """The family's registered measured frontier (distill/evidence/), or None if none yet.

        Data, not prose: the rows are what `screen_verdict` decides on and what the control gate
        reads as a swept grid. A family without a campaign returns None and says so.
        """
        from instinctflash.distill import evidence as base

        if self.family not in base.FILES:
            return None
        return base.load_evidence(self.family)

    def guidance_axis_required(self) -> bool:
        """Does screening on this family HAVE to sweep guidance before training is considered?

        Yes whenever any stream serves classifier-free guidance (RFC §11: the control is the
        best untrained configuration over the guidance grid at the same schedule, and Pillar 1
        always includes that axis before Pillar 2). A family with no negative branch anywhere
        (pi05) has no axis to sweep and the gate does not demand one.
        """
        return any(s.guidance_mode == "cfg" for s in self.streams())

    def schedule(self, nfe: Mapping[str, int]) -> dict[str, ScheduleGrid]:
        """Grids for a full operating point, e.g. {'video': 1, 'action': 4}."""
        unknown = sorted(set(nfe) - {s.name for s in self.streams()})
        if unknown:
            raise KeyError(f"{self.family}: nfe names unknown stream(s) {unknown}")
        return {name: self.schedule_grid(name, steps) for name, steps in sorted(nfe.items())}

    # -- training (GPU) -----------------------------------------------------------------------

    @abstractmethod
    def build_student(self, teacher: Any, schedule: Mapping[str, ScheduleGrid]) -> Any:
        """Student init = teacher, trainable per trainable_set(). Family-specific cloning."""

    @abstractmethod
    def consistency_target(
        self, stream: str, batch: Any, teacher: Any, grid: ScheduleGrid
    ) -> Any:
        """The teacher-side target for one step on one stream — H2's plug point."""


# --- the registry -----------------------------------------------------------------------------

_FAMILIES: dict[str, FamilyAdapter] = {}


def register_family(adapter: FamilyAdapter) -> FamilyAdapter:
    if not adapter.family:
        raise ValueError(f"{type(adapter).__name__} declares no family name")
    if adapter.family in _FAMILIES:
        raise ValueError(f"family {adapter.family!r} is already registered")
    _FAMILIES[adapter.family] = adapter
    return adapter


def get_family(name: str) -> FamilyAdapter:
    _ensure_builtin()
    if name not in _FAMILIES:
        raise KeyError(
            f"no distillation adapter for family {name!r}; registered: {sorted(_FAMILIES)}. "
            "A family joins by implementing FamilyAdapter (see docs/rfc/fewstep-distillation.md "
            "for the six sketched families)."
        )
    return _FAMILIES[name]


def registered_families() -> dict[str, FamilyAdapter]:
    _ensure_builtin()
    return dict(_FAMILIES)


def _ensure_builtin() -> None:
    # imported lazily so `import instinctflash.distill.adapter` alone stays dependency-free;
    # by module path, because the accessor above shadows the subpackage as an attribute
    import importlib

    importlib.import_module("instinctflash.distill.families")
