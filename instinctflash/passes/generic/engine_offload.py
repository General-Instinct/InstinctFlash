"""Whether this device should execute through the fused serving engine instead of torch passes.

A PLANNER pass, like `graph_capture`'s planner half: it decides from the declaration and the
device profile alone, with no torch and no engine import. The runtime-side gate — is the engine
package actually importable, does it have a frontend for this backbone — lives in
`runtime/engine_backend.py`, where failing has a cheap, honest fallback (the torch path).

This pass checks the supported device and declared dimensions/denoise schedule.
The Runtime precision policy separately requires an explicit FP8 request.
Performance depends on the checkpoint, action horizon and measured input/output
boundary. The old 56.9/1111 ms pi05 comparison mixed action horizons and must not
be advertised as the gain for a caller's execution. Matched measurements and
current integration limits are in eval/fp8_comparison_2026-09-09/README.md.
"""

from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult, Tier


class EngineOffloadApplicable:
    """Checks declared eligibility for the supported SM110 FP8 executor."""

    name = "engine_offload"

    #: SM110 exactly, for now: the engine ships kernels for thor/sm120/sm89 and this repo has
    #: measured it only on Thor. Widening to other capabilities requires measuring there first.
    hardware = HardwareReq(min_capability=(9, 0), requires=("cuda", "fp8"))

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        dev = getattr(deployment, "device", None)
        cap = getattr(dev, "capability", None) if dev is not None else None
        if cap == (9, 0):
            from instinctflash.runtime.engine_backend import ENGINE_BACKBONES
            backbone = dict(spec.notes or {}).get("backbone")
            if backbone not in ENGINE_BACKBONES:
                return PassResult(self.name, False, Tier.NUMERIC, reason="No H100 FP8 recipe for this backbone")
            return PassResult(self.name, True, Tier.NUMERIC,
                reason="H100 explicit E4M3 projections with native processing and schedule; task quality unverified on this device",
                params={"backend": "engine", "executor": "h100_torch_fp8", "certification": "uncertified"})
        if cap != (11, 0):
            return PassResult(
                self.name, False, Tier.NUMERIC,
                reason=(f"device capability {cap} is not SM110: this engine route is supported "
                        "on Thor only; other devices require their own implementation and measurements"))
        geometry_problem = self._action_geometry_problem(spec)
        if geometry_problem is not None:
            # Declined ON THE MERITS: no params['backend'], so choose_backend('auto') falls back
            # to the torch placement and placement='engine' is refused by EngineBackend's own
            # gate — the two surfaces of one rule (declared_action_dim).
            return PassResult(self.name, False, Tier.NUMERIC, reason=geometry_problem)
        operating_point_problem, served = self._operating_point_problem(spec)
        if operating_point_problem is not None:
            # Same shape, other axis: the engine bakes its (nfe, guidance); a plan priced at any
            # other point declines here, so 'auto' serves the declared point on the torch chain
            # and the placement line names why. Never a silent mismatch between the printed nfe
            # and the executed schedule.
            return PassResult(self.name, False, Tier.NUMERIC, reason=operating_point_problem)
        return PassResult(
            name=self.name,
            applies=True,
            # fp8 weights+activations change numerics by construction; never claim better.
            tier=Tier.NUMERIC,
            reason=("SM110: FP8 engine passes the declared dimension and denoise checks; "
                    "action-horizon and observation/control compatibility still require validation"
                    + (f"; baked denoise operating point: {served}" if served else "")),
            params={
                "backend": "engine",
                "tier_note": "NUMERIC (live execution uncertified)",
                "certification": ("no certificate verified for this live execution; historical "
                                  "paired results for a dedicated server do not certify the "
                                  "generic Runtime's observation and control contract"),
                **({"engine_operating_point": served} if served else {}),
            },
            expected_win=(
                "Measure this checkpoint with matched views, action horizon, denoise steps and "
                "timing scope. The matched Thor study separates native capture, FP16 engine "
                "and FP8 arithmetic; it does not establish robot-facing Runtime parity."),
        )

    @staticmethod
    def _action_geometry_problem(spec: AdapterSpec) -> "str | None":
        """Validate declared output dimensions; Runtime checks the native config too."""
        from instinctflash.runtime.engine_backend import PI05_ENGINE_MAX_ACTION_DIM
        notes = dict(getattr(spec, "notes", None) or {})
        if notes.get("backbone") != "pi05":
            return None
        try:
            declared = int(notes["action_dim"])
        except (KeyError, TypeError, ValueError):
            return ("pi05 action geometry could not be verified: "
                    + notes.get("action_dim_unresolved", "no action_dim fact on the spec"))
        if not 1 <= declared <= PI05_ENGINE_MAX_ACTION_DIM:
            return (f"engine supports action_dim=1..{PI05_ENGINE_MAX_ACTION_DIM}; checkpoint declares "
                    f"{declared}. The torch placement serves its declared geometry without truncation.")
        return None

    @staticmethod
    def _operating_point_problem(spec: AdapterSpec) -> "tuple[str | None, str | None]":
        """Why the engine build cannot serve this plan's operating point — or the point it serves.

        The engine bakes its denoise schedule and serves no CFG (``ENGINE_BAKED_OPERATING_POINTS``
        in ``runtime/engine_backend.py``: pi05_thor pre-scales ``action_out_proj`` by −1/10,
        sizes its time tables for 10 steps and captures ``'steps': 10`` graphs; the VLA-4B, V2
        and GROOT frontends bake theirs the same way). The plan, on the other hand, is priced at
        the operating point the checkpoint declares and the caller overrides
        (``spec.with_nfe`` / ``with_guidance`` in ``facade._compile_declaration``). Before this
        check, ``nfe={"action": 1}`` on Thor printed ``schedule {prefix=1 + action=1}`` and
        the engine ran 11 forwards (iwm_distill/fewstep/cert_pi05_nfe1_thor.md §2).

        The rule itself is ``engine_operating_point_problem`` — shared with ``EngineBackend``
        so plan time and build time cannot disagree. The spec's phases are the requested
        schedule and its guidance rules the requested guidance; the backbone rides on
        ``spec.notes`` as for the geometry check. A backbone no engine build declares keeps
        today's behaviour (choose_backend never builds the engine for it).

        Returns ``(reason, None)`` to decline, ``(None, served_point)`` to proceed.
        """
        from instinctflash.runtime.engine_backend import (  # no torch at module level there
            engine_baked_operating_point, engine_operating_point_problem,
        )
        notes = dict(getattr(spec, "notes", None) or {})
        baked = engine_baked_operating_point(notes.get("backbone"))
        if baked is None:
            return None, None
        requested_steps = {p.name: int(p.nfe) for p in getattr(spec, "phases", ()) or ()}
        requested_guidance = {
            name: (rule.mode.value, float(rule.scale))
            for name, rule in dict(getattr(spec, "guidance", None) or {}).items()
        }
        problem = engine_operating_point_problem(baked, requested_steps, requested_guidance)
        return problem, (None if problem else baked.served_point())
