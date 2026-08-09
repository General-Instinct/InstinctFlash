"""RULED OUT ON LINGBOT-VA BY MEASUREMENT (2026-08-02). Do not enable.

A two-axis liveness test (eval/lingbot_va_robotwin/probe_cfg_liveness.py) found the
action stream's CFG branch 1 live on BOTH axes: corrupting its returned value moved the
final actions by 5.64, and suppressing only its writes to the shared KV pool moved them by
5.39, against a chunk-to-chunk movement of 1.03.

The config genuinely discards that output -- `action_guidance_scale = 1` and the loop takes
`action_noise_pred[:1]` -- so `guidance = {"action": POSITIVE_ONLY}` is a true statement
about OUTPUT USAGE. It is not a statement about dead compute: both CFG branches write the
shared ring KV pool, and the video stream at guidance_scale=5 reads branch 1.

This is why ExecutionDescriptor separates `dead_outputs` from `elidable_computations`.
Kept as the record of a ruled-out optimization, not as a pass to run.

ORIGINAL DOCSTRING FOLLOWS.

CFGBranchElision — stop computing guidance branches whose output is discarded.

The opportunity, on LingBot-VA
------------------------------
`use_cfg` is a single global flag derived from BOTH streams::

    self.use_cfg = (guidance_scale > 1) or (action_guidance_scale > 1)   # server:379

With the RoboTwin config (`guidance_scale=5`, `action_guidance_scale=1`) the video stream turns
CFG on, and `_repeat_input_for_cfg` (server:254-263) then duplicates the batch for *every*
forward — including all 51 action forwards. But the action stream's combine takes the else
branch::

    if action_guidance_scale > 1:  ... combine ...
    else:                          action_noise_pred = action_noise_pred[:1]   # server:552-555

So the negative action branch is computed at full cost and thrown away. Per control step that is
**50 of 51 action forwards running at batch 2 to use batch 1**.

Why the 51st forward is different, and must not be elided
---------------------------------------------------------
The last action forward runs with `update_cache=1` (server:544-546). Its *output* is unused —
there is no `if not last_step` combine for it — but its side effect is to commit action K/V for
BOTH branches into the pool. Later video forwards run CFG, and the negative video branch attends
the negative action K/V. Eliding the negative branch there would silently corrupt the episode,
and only after several chunks, which is the worst possible failure shape.

The same reasoning applies to the video phase's committing forward, so the rule is general:
**a phase's guidance branches may be reduced on every forward except the one that commits KV.**

What makes this a framework pass rather than a hack
---------------------------------------------------
Nothing above is LingBot-VA-specific once the model has declared

    guidance = {"video": GuidanceRule(CFG, 5.0), "action": GuidanceRule(POSITIVE_ONLY)}
    phases   = (..., PhaseSpec("action", nfe=51, commit_step=50, writes={"action"}), ...)

The pass reads those two declarations and derives the elision. It fires on any future model that
declares a POSITIVE_ONLY stream, and correctly declines for DreamZero (whose action stream also
takes the positive branch only, but which runs branches as *separate* forwards rather than a
duplicated batch, so there is nothing to elide — see `batchable`).

Accuracy tier
-------------
NUMERIC, not BITEXACT, and the distinction is real. The arithmetic is identical — we drop a
tensor that was being sliced away regardless. But the surviving branch moves from a batch-2 GEMM
to a batch-1 GEMM, and cuBLAS may select different tiling, which can change the floating-point
reduction order. That is a bounded, structurally-justified difference rather than an
approximation, which is exactly what the NUMERIC tier is for. `probe_bitexact.py` settles it
empirically: if it reports 0, the pass is promoted to BITEXACT for this model+hardware and
recorded as such.

STATUS: NEGATIVE RESULT
Branch 1 is LIVE on both axes -- 5.64 corrupting its returned value, 5.39 suppressing
only its writes to the shared KV pool, against a chunk-to-chunk movement of 1.03. `action_guidance_scale=1`
makes `dead_outputs` a true statement about OUTPUT USAGE; it is not a statement about dead compute.
See HISTORICAL.md.
"""

from __future__ import annotations

from instinctwm.adapters.base import AdapterSpec, GuidanceMode
from instinctwm.descriptors.deployment import DeploymentSpec
from instinctwm.planners.planner import PassResult, Tier


class CFGBranchElision:
    name = "cfg_branch_elision"

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        # Only meaningful if SOME stream forces the batch to be duplicated...
        forces_batch = [
            n for n, g in spec.guidance.items()
            if g.mode is GuidanceMode.CFG and g.batchable
        ]
        # ...while ANOTHER stream pays for that duplication without using it.
        wasteful = [
            n for n, g in spec.guidance.items()
            if g.mode in (GuidanceMode.POSITIVE_ONLY, GuidanceMode.NONE) and g.batchable
        ]

        if not forces_batch:
            return PassResult(self.name, False, Tier.BITEXACT,
                              "no stream requests CFG, so nothing duplicates the batch")
        if not wasteful:
            return PassResult(self.name, False, Tier.BITEXACT,
                              "every stream consumes both guidance branches")

        # Map wasteful streams to the phases that write them, and exempt each phase's
        # committing forward (its K/V is read by the OTHER stream's negative branch).
        targets = {}
        total_elided = 0
        for ph in spec.phases:
            if not (ph.writes & set(wasteful)):
                continue
            commit = {s for s in ph.commit_steps if 0 <= s < ph.nfe}
            n_elidable = ph.nfe - len(commit)
            if n_elidable <= 0:
                # e.g. LingBot-VA's kv_refresh: every forward commits, so nothing is elidable.
                continue
            targets[ph.name] = {
                "elide_except_steps": sorted(commit),
                "n_elided": n_elidable,
                "n_total": ph.nfe,
            }
            total_elided += n_elidable

        if not targets:
            return PassResult(self.name, False, Tier.BITEXACT,
                              f"streams {wasteful} take positive-only guidance but no phase "
                              f"writes them outside a commit step")

        total = spec.total_forwards()
        share = total_elided / (2 * total)  # batch-2 forwards -> batch-1 on the elided ones
        # Report per phase. A bare "N of {total_forwards()}" reads as a claim about the whole
        # control step when it is really a claim about one phase, and its denominator would
        # disagree with the 77-forward denoise count quoted in the write-ups.
        per_phase = ", ".join(
            f"{name}: {t['n_elided']} of {t['n_total']}" for name, t in sorted(targets.items())
        )
        return PassResult(
            name=self.name,
            applies=True,
            tier=Tier.NUMERIC,
            reason=(
                f"streams {sorted(wasteful)} discard their negative branch while {sorted(forces_batch)} "
                f"force batch duplication; forwards that can drop to batch 1 — {per_phase} "
                f"(commit steps exempt: their K/V feeds the other stream's negative branch)"
            ),
            params={"targets": targets, "keep_streams": sorted(forces_batch)},
            expected_win=(
                f"removes {total_elided} of the control step's {2 * total} branch-forwards "
                f"(~{share:.0%}); on a memory-bound batch-1 forward the realized win is bounded "
                f"by the weight traffic that does NOT halve, so treat this as an upper bound "
                f"until measured"
            ),
        )
