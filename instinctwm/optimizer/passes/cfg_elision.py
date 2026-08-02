"""CFGBranchElision — stop computing guidance branches whose output is discarded.

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
"""

from __future__ import annotations

from instinctwm.adapter.base import AdapterSpec, GuidanceMode
from instinctwm.deployment import DeploymentSpec
from instinctwm.optimizer.base import PassResult, Tier


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
        return PassResult(
            name=self.name,
            applies=True,
            tier=Tier.NUMERIC,
            reason=(
                f"streams {sorted(wasteful)} discard their negative branch while {sorted(forces_batch)} "
                f"force batch duplication; {total_elided} of {total} forwards can drop to batch 1 "
                f"(commit steps exempt: their K/V feeds the other stream's negative branch)"
            ),
            params={"targets": targets, "keep_streams": sorted(forces_batch)},
            expected_win=(
                f"removes ~{share:.0%} of denoise batch-work; on a memory-bound batch-1 forward "
                f"the realized win is bounded by the weight traffic that does NOT halve, so treat "
                f"this as an upper bound until measured"
            ),
        )
