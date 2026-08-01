"""The two descriptors a Backend Adapter provides.

    Backend Adapter -> Descriptors -> Optimization Planner -> Optimization Stack -> Runtime

A Backend Adapter describes its model through exactly two descriptors and nothing else. It never
names an optimization, never sets a flag, and never says "use the fast path". Everything the
planner does is derived from these.

    ExecutionDescriptor   WHAT IS COMPUTED, AND IN WHAT ORDER
                          phases, NFE loops, guidance, commit points, dependencies,
                          the prefill/decode split

    StateDescriptor       WHERE STATE LIVES, AND HOW IT IS ADDRESSED
                          lifetime, ownership, addressing, layout, synchronization

The split is not cosmetic. It is the seam that showed up when six models' control-step graphs were
diffed: five of six draw a line at `commit_context` -- everything above it is prefill, everything
below is decode -- and the two halves want different things. Prefill is where loop-invariant work
can be hoisted; decode is where launches and addressing dominate. A single blob descriptor cannot
express "this is constant across the loop" because it has no notion of the loop.

Why a model must not describe itself in terms of optimizations: the same declaration has to serve
passes that do not exist yet. `guidance = {"action": POSITIVE_ONLY}` is a fact about LingBot-VA
that was written before anything consumed it; `CFGBranchElision` later derived a real optimization
from it without the adapter changing. That only works if adapters state facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from instinctwm.adapter.base import GuidanceRule, PhaseSpec, PurityKey
from instinctwm.state.types import StateManifest


@dataclass(frozen=True)
class ExecutionDescriptor:
    """What is computed per control step, and in what order.

    `phases` is the denoise loop made VISIBLE. Six of six corpus models hide their NFE loop inside
    a single opaque forward(), which is precisely why a runtime cannot vary NFE mid-episode,
    preempt between forwards, interleave phases across episodes, or capture graphs uniformly.
    Declaring the loop is what makes those legal.
    """

    model_id: str
    phases: tuple[PhaseSpec, ...]
    guidance: Mapping[str, GuidanceRule]
    #: conditioning that is a pure function of the named fields. The riskiest declaration in the
    #: system -- hoisting on a wrong purity key produces silently wrong actions -- so passes that
    #: consume it are DECLARED-only, never AUTO.
    purity: tuple[PurityKey, ...] = ()
    #: phase name after which prefill ends and decode begins. Five of six models draw this line
    #: in their own code without naming it.
    prefill_boundary: str | None = None
    #: outputs the caller never reads. Cosmos3-Edge denoises 567 tokens and consumes 17.
    dead_outputs: tuple[str, ...] = ()
    #: False when the solver carries state across steps, which makes truncation illegal:
    #: UniPC's this_order depends on len(timesteps) and set_timesteps rebuilds the sigma table,
    #: so a truncated run is NOT a prefix of the full trajectory. True for 4 of 6, not all.
    nfe_mutable: bool = True

    @property
    def total_forwards(self) -> int:
        return sum(p.nfe for p in self.phases)

    def phase(self, name: str) -> PhaseSpec:
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(name)

    def decode_phases(self) -> tuple[PhaseSpec, ...]:
        if self.prefill_boundary is None:
            return self.phases
        seen = False
        out = []
        for p in self.phases:
            if seen:
                out.append(p)
            if p.name == self.prefill_boundary:
                seen = True
        return tuple(out)


@dataclass(frozen=True)
class StateDescriptor:
    """Where state lives and how it is addressed. Wraps the L3 StateManifest.

    Kept as a thin wrapper rather than merged into ExecutionDescriptor because the two are
    consumed by different layers and change for different reasons: execution changes when the
    model's math changes, state changes when its memory behaviour does.
    """

    manifest: StateManifest

    @property
    def model_id(self) -> str:
        return self.manifest.model_id

    def has_state(self) -> bool:
        return self.manifest.has_state()


@dataclass(frozen=True)
class ModelDescriptor:
    """Everything the planner is allowed to know about a model."""

    execution: ExecutionDescriptor
    state: StateDescriptor
    param_bytes: int = 0
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.execution.model_id != self.state.model_id:
            # A mismatch here means an adapter wired two different models together, which would
            # otherwise surface much later as a baffling correctness failure.
            raise ValueError(
                f"descriptor mismatch: execution={self.execution.model_id!r} "
                f"state={self.state.model_id!r}")

    def summary(self) -> str:
        e, s = self.execution, self.state.manifest
        return (f"{self.execution.model_id}: {len(e.phases)} phases, {e.total_forwards} forwards/step, "
                f"nfe_mutable={e.nfe_mutable} | state: {len(s.arenas)} arena(s), "
                f"E_mat={s.e_materialized()/1e9:.2f} GB, syncs={s.sync_budget}")
