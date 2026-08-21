"""Deployment facts — what `AdapterSpec` deliberately does not know.

`AdapterSpec` answers *what is this model*. It is immutable, checkpoint-scoped, and identical
on every box the checkpoint runs on. This answers *where and how is it being served*, which is
none of those things: the same checkpoint is one GPU here and eight there, and one caller wants
predicted pixels while the next only wants actions.

Keeping them apart is not tidiness. A pass that reads a deployment fact off `AdapterSpec` would
be asking the model author to declare something they cannot know, and the adapter contract says
adapters state facts they can defend. The split is also what makes the same `AdapterSpec` safe
to cache and share across servers.

Fields are added here only when a pass reads one. `FSDPElision` guards on `world_size` and
`ObsDecodeElision` on `want_pixels`; both guards were unreachable before this type existed,
because `Optimizer.compile` had no way to pass them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentSpec:
    """How one particular server is running this model.

    The defaults describe the regime InstinctFlash targets — a single-GPU, actions-only, closed
    control loop — so `DeploymentSpec()` is the right thing for a robot policy server and the
    interesting cases are the ones that deviate.
    """

    #: `torch.distributed` world size. Passes that elide collective machinery are only legal
    #: at 1, where every collective is identity. Take it from the launcher (`WORLD_SIZE`)
    #: rather than from `dist.is_initialized()`, which is True even on one GPU because the
    #: upstream server calls `init_distributed` unconditionally — that is the very bug
    #: `FSDPElision` exists to remove.
    world_size: int = 1

    #: True when the caller consumes predicted observations (video/pixels) and not just
    #: actions. Every WAM surveyed makes the observation-decode tail optional at serving time,
    #: but "optional" is a property of the *call*, not of the checkpoint.
    want_pixels: bool = False

    #: THE MACHINE. `DeviceProfile.probe()` has existed since the pass contract was written and
    #: was called from nowhere, so `HardwareReq` -- the mechanism by which a pass or backend says
    #: which silicon it is legal on -- could not be enforced at plan time. Every
    #: architecture-specific decision was therefore an unguarded extrapolation from the one
    #: machine it was measured on. Hardware is a deployment fact, not a checkpoint fact: the same
    #: weights run on an H100 here and an Orin there, and the checkpoint author cannot know which.
    #:
    #: None means "not probed", which the planner reports rather than assumes. Annotated as a
    #: string so descriptors/ stays free of any import from passes/.
    device: "DeviceProfile | None" = None      # noqa: F821  (passes.contract.DeviceProfile)
