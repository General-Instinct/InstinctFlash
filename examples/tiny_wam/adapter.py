"""A THIRD-PARTY Backend Adapter, written entirely outside `instinctflash/`.

This is the point of the example. Nothing in this file is imported by the runtime, nothing in the
runtime knows it exists, and it is registered at call time with the public `instinctflash.register`.
An author with their own backbone does exactly this: implement `spec()`, register it, publish a
checkpoint whose `execution.backbone` names it.

The adapter answers ONE question for the planner: *what is the shape of a control step for these
weights?* Streams and their lifetimes, phases and their forward counts, guidance per stream. It
never says how the weights were trained, and there is no field here in which it could.
"""

from __future__ import annotations

from typing import Sequence

from instinctflash.adapters.base import (
    AdapterSpec, CommitMode, GuidanceMode, GuidanceRule, KVLifetime, KVStreamSpec, PhaseSpec,
)

BACKBONE = "tiny-wam"


def tiny_wam_spec(model_id: str = "example-org/tiny-wam-2v2a", param_bytes: int = 0) -> AdapterSpec:
    """The declarations, derived from the module's actual structure.

    These are checkable against `examples/tiny_wam/model.py`: two denoise forwards for the video
    stream and two for the action stream, an action chunk of 8 tokens, CFG on video only.
    """
    return AdapterSpec(
        model_id=model_id,
        param_bytes=param_bytes,
        streams=(
            KVStreamSpec(
                name="video", tokens_per_frame=8, lifetime=KVLifetime.EPISODE,
                commit_mode=CommitMode.CONFIRMED,
                # EPISODE: the video KV persists across control steps. CHUNK below: the action KV lives
                # for one control step and is dropped -- a LIFETIME, not a boolean.
            ),
            KVStreamSpec(
                name="action", tokens_per_frame=8, lifetime=KVLifetime.CHUNK,
                commit_mode=CommitMode.CONFIRMED,
            ),
        ),
        phases=(
            PhaseSpec(name="video", nfe=2, writes=frozenset({"video"}), truncatable=True, min_nfe=1),
            PhaseSpec(name="action", nfe=2, reads=frozenset({"video"}),
                      writes=frozenset({"action"}), depends_on=("video",),
                      truncatable=True, min_nfe=1),
        ),
        guidance={
            # the video stream duplicates the batch; the action stream does not. A pass that elides
            # a discarded CFG branch reads THIS, not a model name.
            "video": GuidanceRule(mode=GuidanceMode.CFG, scale=2.0, batchable=True),
            "action": GuidanceRule(mode=GuidanceMode.POSITIVE_ONLY, scale=1.0, batchable=True),
        },
        notes={"example": "examples/tiny_wam -- a real but tiny module, published as a worked "
                          "end-to-end example. Not trained; no accuracy claim."},
    )


class TinyWAMAdapter:
    """Implements the `BackendAdapter` protocol for the tiny model."""

    model_id = BACKBONE

    def __init__(self, checkpoint_dir: str | None = None, param_bytes: int = 0,
                 declared_model_id: str = "example-org/tiny-wam-2v2a"):
        self.checkpoint_dir = checkpoint_dir
        self._param_bytes = param_bytes
        self._model_id = declared_model_id

    def spec(self) -> AdapterSpec:
        return tiny_wam_spec(self._model_id, self._param_bytes)

    #: Passes that act on the SERVING SUBSTRATE rather than on the model: an FSDP wrapper, the
    #: caching allocator's release policy, a debug-dump hook. This example runs in-process and has
    #: none of them, so there is nothing to elide -- the condition each pass creates is already true.
    #: That is different from "installed", and the distinction is the reason install() returns what
    #: it APPLIED rather than what the plan wanted.
    VACUOUS_ON_THIS_DEPLOYMENT = {
        "fsdp_elision": "no torch.distributed wrapper: this example runs in-process, world_size=1",
        "allocator_churn_elision": "no per-step empty_cache() to remove",
        "debug_dump_elision": "no debug-dump hook on the critical path",
    }

    def install(self, server_module: object, plan) -> Sequence[str]:
        """Apply a plan to a concrete serving object. Returns what it ACTUALLY applied.

        A third-party adapter learns something here that is easy to miss: the default pass set
        contains SUBSTRATE passes which apply to every checkpoint, because they describe the serving
        environment rather than the model. An adapter must therefore account for each applied pass --
        install it, or show it is vacuous for this deployment -- and refuse anything it can do
        neither with. Reporting a pass as installed when it was skipped would invalidate every number
        measured against the resulting server.
        """
        applied = [r.name for r in getattr(plan, "results", []) if getattr(r, "applies", False)]
        installed: list[str] = []
        unaccounted = [n for n in applied if n not in self.VACUOUS_ON_THIS_DEPLOYMENT]
        if unaccounted:
            raise RuntimeError(
                f"{self.model_id}: the plan applies {unaccounted}, and this adapter can neither "
                f"install them nor show they are vacuous here. Refusing rather than serving an "
                f"unoptimized model that reports as optimized.")
        return installed

    def vacuous(self, plan) -> dict[str, str]:
        """Applied passes that need no work on this deployment, and why. For the report, not the plan."""
        return {r.name: self.VACUOUS_ON_THIS_DEPLOYMENT[r.name]
                for r in getattr(plan, "results", [])
                if getattr(r, "applies", False) and r.name in self.VACUOUS_ON_THIS_DEPLOYMENT}

    def serve(self, plan, port: int, **kwargs):
        raise NotImplementedError(
            "The example adapter runs inference in-process (see run_end_to_end.py). It deliberately "
            "does not start a server: the workflow being demonstrated is declare -> resolve -> plan "
            "-> run, not deployment.")
