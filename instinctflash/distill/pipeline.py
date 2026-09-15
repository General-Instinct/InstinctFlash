"""screen -> train -> verify, with the matched-NFE-control law enforced in code.

THE LAW (campaign law since the PDD post-mortem; docs/rfc/fewstep-distillation.md §4, strengthened
by §11 after the first two campaign verdicts):

    No trained few-step result is reportable unless it is paired, on pinned scenes, against
    the UNTRAINED teacher run at the identical execution configuration — same per-stream step
    counts, same t-grids, same guidance scales and CFG batching, same commit protocol — and
    the report states B−A and C−B separately (A = full teacher, B = untrained matched-NFE
    control, C = trained student): B−A is the cost of step reduction, C−B is what training
    bought.

    THE CONTROL IS THE BEST UNTRAINED CONFIGURATION OVER THE DECLARED GUIDANCE GRID AT THE SAME
    SCHEDULE, not the shipped guidance. An operating point is the tuple (schedule grid,
    per-stream guidance scale, CFG batching); `nfe` alone underspecifies it.

`verify_point` is where the law lives. It REFUSES (`ControlGateViolation`) rather than
degrading: a missing control, a control at a different operating point, a control whose
guidance grid was not swept, a control that is not the best of its grid, or arms that do not
pair are each the exact confound the law exists to remove, and every precedent says the
temptation is to proceed anyway (Flash-WAM's sim tables, the PDD paper, LingBot-VA 2.0, OFP —
none carry the control; our own PDD run's control turned "it works" into "+0.0"; the H1 run's
+14.3 pp vs the shipped-guidance control was +0.0 vs the best untrained knob).

"NO DISTILLATION NEEDED" IS A FRAMEWORK VERDICT, not a failure. Both first instantiations
returned it (LingBot-VA H1: the 1V cliff was a guidance artifact, untrained 1V/4A@w1 = 0.885;
pi05 v044: the untrained frontier is FLAT to nfe1 at 174 -> 51 ms). `screen_verdict` reads a
frontier report and decides, BEFORE any trainer runs, whether the best untrained configuration
at the target schedule already sits within the declared margin of the baseline; if it does,
`train()` raises `NoDistillationNeeded` carrying the artifact the verdict owes — the frontier
table, the screening intervals, and a certification pre-registration stub for the winning
untrained point — instead of spending training compute.

The three stages:

  screen   emit the schedule-sweep spec whose rows become the standing controls (arm B per
           candidate (nfe, guidance) point) — executed by `benchmarks/vla` (`sweep-plan` /
           `run` / `sweep-report`), which is a separate top-level package by design: screening
           is public-eligible, this trainer tier is not, and the seam between them is a JSON
           spec. The sweep-report's `best_untrained_per_schedule` block is the control record.
  train    dispatch to the family adapter (build_student / consistency_target under the
           platform trainer) — only after a screen verdict says training is worth attempting.
           The wan_va and pi05 training halves are ExperimentalSeams owned by their campaigns;
           the pipeline stops AT the seam, it does not route around it.
  verify   the three-arm analysis via `verify.certify` (the same code path every certificate
           in this repo uses), then stamp `provenance.distillation` with the hashes, both
           deltas, the control's guidance and the grid it was chosen from, so the checkpoint
           carries its own honesty (descriptors/distillation.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from instinctflash.descriptors.guidance import canonical_guidance
from instinctflash.distill.adapter import FamilyAdapter, ScheduleGrid, get_family
from instinctflash.verify.certify import Certificate, Outcome, certify
from instinctflash.verify.ranking import rank_candidates, within_one_episode


class ControlGateViolation(RuntimeError):
    """A trained few-step result was offered for reporting without its matched control."""


class NoDistillationNeeded(Exception):
    """The framework verdict "do not train": the best UNTRAINED operating point at the target
    schedule already sits within the declared margin of the baseline. Carries the ScreenVerdict
    (frontier + intervals + certification prereg stub) so the caller emits the artifact the
    verdict owes instead of a trainer run."""

    def __init__(self, verdict: "ScreenVerdict") -> None:
        super().__init__(verdict.headline())
        self.verdict = verdict


_LAW = (
    "the matched-NFE-control law: no trained few-step result is reportable unless paired, on "
    "pinned scenes, against the UNTRAINED teacher at the identical operating point (schedule "
    "grid, per-stream guidance scale, CFG batching), where the control is the BEST untrained "
    "configuration over the guidance grid swept at that schedule, with B−A (cost of step "
    "reduction) and C−B (what training bought) stated separately"
)


def canonical_schedule(
    schedule: Mapping[str, Any], family_guidance: Mapping[str, tuple[str, float]] | None = None
) -> dict[str, Any]:
    """An execution configuration in comparable form: nfe, grids, and canonical guidance.

    Guidance is canonicalised through the declaration schema (descriptors/guidance.py) against
    the family's own defaults, so a schedule that does not mention guidance means "the family's
    shipped point" — and is therefore NOT the same operating point as one declaring the video
    negative branch off. That distinction is the whole H1 lesson.
    """
    out: dict[str, Any] = {}
    if "nfe" in schedule:
        out["nfe"] = {str(k): int(v) for k, v in sorted(dict(schedule["nfe"]).items())}
    if "grids" in schedule and schedule["grids"] is not None:
        out["grids"] = {str(k): [float(t) for t in v] for k, v in sorted(dict(schedule["grids"]).items())}
    out["guidance"] = canonical_guidance(schedule.get("guidance"), family_guidance)
    return json.loads(json.dumps(out, sort_keys=True))


def _schedules_match(
    student: Mapping[str, Any], control: Mapping[str, Any],
    family_guidance: Mapping[str, tuple[str, float]] | None,
) -> bool:
    return canonical_schedule(student, family_guidance) == canonical_schedule(control, family_guidance)


@dataclass(frozen=True)
class ThreeArmReport:
    """A = full teacher, B = untrained matched-NFE control, C = trained student."""

    point: str
    schedule: Mapping[str, Any]
    b_minus_a: Certificate  # cost of step reduction
    c_minus_b: Certificate  # what training bought
    c_minus_a: Certificate  # the composite, reported last and never alone
    #: the control's guidance (canonical) and the grid it was the best of
    control_guidance: Mapping[str, Any] = field(default_factory=dict)
    guidance_grid_swept: tuple[Mapping[str, Any], ...] = ()

    @property
    def training_bought_nothing(self) -> bool:
        """C−B's interval contains zero: training is not distinguishable from the free knob."""
        lo, hi = self.c_minus_b.ci95
        return lo <= 0.0 <= hi

    def summary(self) -> str:
        grid = ", ".join(
            f"{_guidance_label(g.get('guidance'))}={g['success']:.4f}" if g.get("success") is not None
            else _guidance_label(g.get("guidance"))
            for g in self.guidance_grid_swept
        ) or "(none recorded)"
        lines = [
            f"three-arm report for {self.point} (schedule {dict(self.schedule.get('nfe', self.schedule))})",
            f"control operating point: guidance {_guidance_label(self.control_guidance)} — the best "
            f"untrained configuration over the grid swept at this schedule: {grid}",
            "",
            "B − A  (untrained matched-NFE control vs full teacher — the cost of step "
            "reduction):",
            str(self.b_minus_a),
            "",
            "C − B  (trained student vs untrained control — what training bought):",
            str(self.c_minus_b),
            "",
            "C − A  (composite; interpretable only alongside the two above):",
            str(self.c_minus_a),
        ]
        if self.training_bought_nothing:
            lines += [
                "",
                "READING: C−B is not distinguishable from zero — training bought nothing the best "
                "untrained configuration did not already have. What deserves certification is the "
                "UNTRAINED control point (the E6/H1 pattern).",
            ]
        return "\n".join(lines)

    def control_block(
        self,
        *,
        teacher_outcomes_sha256: str,
        control_outcomes_sha256: str,
        student_outcomes_sha256: str,
    ) -> dict[str, Any]:
        """The `matched_nfe_control` object for provenance.distillation."""
        from instinctflash.descriptors.distillation import _arm_summary

        return {
            "teacher_outcomes_sha256": teacher_outcomes_sha256,
            "control_outcomes_sha256": control_outcomes_sha256,
            "student_outcomes_sha256": student_outcomes_sha256,
            "b_minus_a": _arm_summary(self.b_minus_a),
            "c_minus_b": _arm_summary(self.c_minus_b),
            "c_minus_a": _arm_summary(self.c_minus_a),
            "control_guidance": json.loads(json.dumps(dict(self.control_guidance))),
            "guidance_grid_swept": json.loads(json.dumps(list(self.guidance_grid_swept))),
        }


def _guidance_label(guidance) -> str:
    if not guidance:
        return "family default"
    parts = []
    for stream, g in sorted(dict(guidance).items()):
        if isinstance(g, Mapping):
            scale = g.get("scale")
            parts.append(f"{stream}={g.get('mode')}@{scale:g}" if scale is not None else f"{stream}={g.get('mode')}")
        else:
            parts.append(f"{stream}={g}")
    return " ".join(parts)


def _check_control_is_best_of_grid(
    point: str,
    control_schedule: Mapping[str, Any],
    control_grid: Sequence[Mapping[str, Any]] | None,
    family_guidance: Mapping[str, tuple[str, float]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The strong form: the control must be the best of a SWEPT guidance grid at its schedule."""
    if not control_grid:
        raise ControlGateViolation(
            f"{point}: the control's guidance grid was not swept — the control is the BEST "
            f"untrained configuration over the declared guidance grid at this schedule, not the "
            f"shipped guidance; {_LAW}. Screen the guidance axis (schedule_sweep points varying "
            f"guidance at this nfe; the sweep-report's best_untrained_per_schedule block is the "
            f"record) and come back with the grid."
        )
    control_g = canonical_guidance(control_schedule.get("guidance"), family_guidance)
    grid: list[dict[str, Any]] = []
    for entry in control_grid:
        if "success" not in entry:
            raise ControlGateViolation(
                f"{point}: guidance-grid entry {dict(entry)} carries no success — a grid row that "
                f"was not measured cannot be compared; {_LAW}."
            )
        grid.append({
            "point": entry.get("point"),
            "guidance": canonical_guidance(entry.get("guidance"), family_guidance),
            "success": None if entry["success"] is None else float(entry["success"]),
        })
    mine = [g for g in grid if g["guidance"] == control_g]
    if not mine:
        raise ControlGateViolation(
            f"{point}: the control arm's guidance {_guidance_label(control_g)} is not in the swept "
            f"grid {[_guidance_label(g['guidance']) for g in grid]} — a control chosen outside "
            f"the screen is not the screen's best; {_LAW}."
        )
    measured = [g for g in grid if g["success"] is not None]
    best = max((g["success"] for g in measured), default=None)
    own = max((g["success"] for g in mine if g["success"] is not None), default=None)
    n_pairs = min((int(e["n_pairs"]) for e in control_grid if e.get("n_pairs")), default=None)
    if best is not None and not within_one_episode(own, best, n_pairs):
        winners = [_guidance_label(g["guidance"]) for g in measured if g["success"] == best]
        raise ControlGateViolation(
            f"{point}: the control at {_guidance_label(control_g)} (success "
            f"{'unmeasured' if own is None else f'{own:.4f}'}) is not the best untrained "
            f"configuration at this schedule — {', '.join(winners)} scored {best:.4f}. A trained "
            f"point must beat the best free knob, not the shipped one (H1: +14.3 pp vs w=5 was "
            f"+0.0 vs w=3); {_LAW}."
        )
    return control_g, grid


def verify_point(
    *,
    point: str,
    teacher: Sequence[Outcome],
    control: Sequence[Outcome] | None,
    student: Sequence[Outcome],
    student_schedule: Mapping[str, Any],
    control_schedule: Mapping[str, Any] | None,
    margin: float,
    control_grid: Sequence[Mapping[str, Any]] | None = None,
    guidance_axis_required: bool = True,
    family_guidance: Mapping[str, tuple[str, float]] | None = None,
    interval: str = "wald_central95",
    min_pairs: int = 1,
    harness: str = "instinctflash.distill",
    recipe: str = "?",
) -> ThreeArmReport:
    """The three-arm verdict for one trained operating point. Refuses before it degrades.

    `student_schedule`/`control_schedule` are the declared execution configurations (at
    minimum {"nfe": {...}}; grids, and guidance when it departs from the family default). They
    must be EQUAL as operating points — a control at the wrong schedule OR the wrong guidance
    is not a control, it is a different experiment.

    `control_grid` is the guidance grid swept at the control's schedule: rows of
    {guidance, success[, point]} (a sweep-report's `best_untrained_per_schedule[...]
    .candidates`). When `guidance_axis_required` (any stream of the family serves CFG) the grid
    is mandatory and the control must be its best row; a family with no negative branch
    anywhere (pi05) has no axis, and the gate does not demand one.
    """
    if control is None or len(control) == 0:
        raise ControlGateViolation(
            f"{point}: no untrained matched-NFE control arm was provided — {_LAW}. Run the "
            "teacher weights at the student's exact operating point (a schedule-sweep row is the "
            "standing form of this arm) and come back with its outcomes."
        )
    if control_schedule is None:
        raise ControlGateViolation(
            f"{point}: the control arm carries no declared schedule, so it cannot be shown to "
            f"match the student's — {_LAW}."
        )
    if not _schedules_match(student_schedule, control_schedule, family_guidance):
        raise ControlGateViolation(
            f"{point}: control operating point {canonical_schedule(control_schedule, family_guidance)} "
            f"!= student operating point {canonical_schedule(student_schedule, family_guidance)} — a "
            f"control at a different configuration measures a different experiment; {_LAW}."
        )
    control_g: dict[str, Any] = canonical_guidance(control_schedule.get("guidance"), family_guidance)
    grid: list[dict[str, Any]] = []
    if guidance_axis_required:
        control_g, grid = _check_control_is_best_of_grid(point, control_schedule, control_grid, family_guidance)
    elif control_grid:
        grid = [{"point": e.get("point"), "guidance": canonical_guidance(e.get("guidance"), family_guidance),
                 "success": e.get("success")} for e in control_grid]
    # certify() itself enforces pairing (same episodes both arms), duplicate refusal, and the
    # declared margin/interval; NotCertifiable propagates — an unpairable control is no control.
    b_minus_a = certify(
        teacher, control, margin=margin, interval=interval, min_pairs=min_pairs,
        harness=harness, recipe=f"{recipe} [B−A untrained control vs teacher]",
        seeds="paired by episode_id",
    )
    c_minus_b = certify(
        control, student, margin=margin, interval=interval, min_pairs=min_pairs,
        harness=harness, recipe=f"{recipe} [C−B trained vs untrained control]",
        seeds="paired by episode_id",
    )
    c_minus_a = certify(
        teacher, student, margin=margin, interval=interval, min_pairs=min_pairs,
        harness=harness, recipe=f"{recipe} [C−A composite]",
        seeds="paired by episode_id",
    )
    return ThreeArmReport(
        point=point,
        schedule=canonical_schedule(student_schedule, family_guidance),
        b_minus_a=b_minus_a,
        c_minus_b=c_minus_b,
        c_minus_a=c_minus_a,
        control_guidance=control_g,
        guidance_grid_swept=tuple(grid),
    )


# --- the screen verdict -------------------------------------------------------------------------

VERDICT_NO_DISTILLATION_NEEDED = "no_distillation_needed"
VERDICT_TRAIN = "train"
#: the best untrained point's interval straddles the margin at screening n: the cheapest next
#: step is to extend the UNTRAINED screen (or certify it at n >= 500), not to train — the H1
#: 1V/4A@w1 case (−3.5 pp [−0.091, +0.018]) and 1V/2A@w1 (−2.6 pp [−0.068, +0.018]).
VERDICT_EXTEND_SCREEN = "extend_screen"


@dataclass(frozen=True)
class ScreenVerdict:
    """What the screen says about training at one target schedule, decided BEFORE a trainer runs.

    Rule (RFC §11 + frontier prereg §0 framing): over the guidance grid swept at the target
    schedule, take the best untrained configuration. If its deciding lower bound vs the baseline
    clears the declared margin, the operating point is already free — NO DISTILLATION NEEDED,
    and the artifact owed is the frontier + intervals + a certification prereg stub for that
    point (pi05 at every nfe; LingBot-VA 2V/2A@w1). If its interval sits entirely below the
    margin, training has something to buy — TRAIN, with that configuration as the standing
    control (LingBot-VA 1V/1A@w1, −22 pp). Otherwise EXTEND the untrained screen (certify-first,
    n >= 500) before spending training compute (LingBot-VA 1V/4A@w1, 1V/2A@w1). The artifact is
    the same in the first and third cases; only the first is a verdict against training as such.
    """

    family: str
    nfe: Mapping[str, int]
    margin: float
    baseline: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]
    best: Mapping[str, Any] | None
    verdict: str
    reason: str
    frontier_table: str = ""
    source: str = ""

    @property
    def no_distillation_needed(self) -> bool:
        return self.verdict == VERDICT_NO_DISTILLATION_NEEDED

    def control_grid(self) -> list[dict[str, Any]]:
        """The grid rows in the form `verify_point(control_grid=...)` consumes."""
        return [{"point": c.get("point"), "guidance": c.get("guidance"), "success": c.get("success"),
                 "n_pairs": c.get("n_pairs")}
                for c in self.candidates]

    def headline(self) -> str:
        label = _guidance_label(self.best.get("guidance")) if self.best else "?"
        if self.verdict == VERDICT_NO_DISTILLATION_NEEDED:
            return (f"{self.family} @ {dict(self.nfe)}: NO DISTILLATION NEEDED — untrained "
                    f"{self.best['point']} ({label}) already sits within the margin {self.margin:+.3f} "
                    f"of the baseline ({self.reason}); certify the untrained point.")
        if self.verdict == VERDICT_TRAIN:
            return (f"{self.family} @ {dict(self.nfe)}: TRAIN — best untrained {self.best['point']} "
                    f"({label}) is below the margin ({self.reason}); it is the standing control.")
        if self.best is None:
            return f"{self.family} @ {dict(self.nfe)}: EXTEND SCREEN — {self.reason}"
        return (f"{self.family} @ {dict(self.nfe)}: EXTEND SCREEN (certify the untrained point first) — "
                f"best untrained {self.best['point']} ({label}): {self.reason}")

    def artifact(self) -> dict[str, Any]:
        return {
            "kind": "fewstep_screen_verdict",
            "verdict": self.verdict,
            "family": self.family,
            "nfe": dict(self.nfe),
            "margin_declared": self.margin,
            "baseline": dict(self.baseline),
            "guidance_grid_swept": [c.get("guidance") for c in self.candidates],
            "candidates": [dict(c) for c in self.candidates],
            "best_untrained": dict(self.best) if self.best else None,
            "reason": self.reason,
            "source": self.source,
            "frontier_table_markdown": self.frontier_table,
            "certification_prereg_stub_markdown": self.prereg_stub(),
        }

    def prereg_stub(self) -> str:
        """A certification pre-registration STUB for the winning untrained point — DRAFT, not run.

        Mirrors the campaign's `cert_untrained_1v_prereg_DRAFT.md` sections; every number here
        is screening evidence, and the stub says so.
        """
        if self.best is None:
            return "(no measured candidate at this schedule; nothing to pre-register)"
        b = self.best
        g = _guidance_label(b.get("guidance"))
        b1, b2 = b.get("forwards_batch1"), b.get("forwards_batch2")
        fwd = (f"{b1} batch-1 + {b2} batch-2 forwards/cycle" if b1 is not None and b2 is not None
               else "forwards/cycle: n/d")
        lo, hi = (b.get("interval") or [None, None])
        interval = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None and hi is not None else "n/e"
        return "\n".join([
            f"# DRAFT pre-registration (NOT RUN): certification of the UNTRAINED operating point "
            f"{b['point']} ({self.family})",
            "",
            f"Status: DRAFT stub emitted by `instinctflash.distill` screen verdict "
            f"`{self.verdict}`. Nothing here is started until a coordinator says so.",
            "",
            "## 0. What is being certified, and against what",
            f"* Treatment: the stock checkpoint, UNTRAINED, served at the operating point "
            f"(schedule {dict(self.nfe)}, guidance {g}, {fwd}). No adapters, no training artifact.",
            f"* Control (A): the baseline arm {self.baseline.get('id', '?')} RERUN on the same scene set.",
            "* Repeat arm: a second baseline run, last, as the noise floor.",
            "* Matched-NFE untrained controls are not needed — the treatment IS untrained.",
            "",
            "## 1. Screening evidence that motivates this point (NOT certification)",
            f"* success {b.get('success')!s} vs baseline {self.baseline.get('success')!s}; paired "
            f"delta {b.get('delta')!s}, interval {interval} ({b.get('interval_method', '?')}), "
            f"n={b.get('n_pairs', '?')}.",
            f"* guidance grid swept at this schedule: "
            f"{', '.join(_guidance_label(c.get('guidance')) for c in self.candidates)}.",
            "",
            "## 2. Question and success criterion",
            f"Is the point NON-INFERIOR to the baseline with margin {self.margin:+.3f} on paired task "
            "success, judged by the most conservative of the pre-declared intervals? Anything else = "
            "NOT ESTABLISHED, and the point is NOT served as a product operating point.",
            "",
            "## 3. Sample size and scenes",
            "* n >= 500 paired episodes on scene seeds DISJOINT from the screening set by construction; "
            "manifest sha256 recorded before any episode; kept-and-dropped, never retried.",
            "",
            "## 4. Latency block",
            "* Must include the regime where batch-1 vs batch-2 differs (for KV-growing families, the "
            "saturated-pool regime), reported as batch-1 and batch-2 forward counts separately.",
        ])


def _row_lower_bound(row: Mapping[str, Any]) -> float | None:
    if row.get("deciding_lower_bound") is not None:
        return float(row["deciding_lower_bound"])
    interval = row.get("interval")
    if interval and interval[0] is not None:
        return float(interval[0])
    return None


def _row_upper_bound(row: Mapping[str, Any]) -> float | None:
    interval = row.get("interval")
    if interval and len(interval) > 1 and interval[1] is not None:
        return float(interval[1])
    return None


def screen_verdict(
    report: Mapping[str, Any],
    *,
    family: str,
    nfe: Mapping[str, int],
    margin: float | None = None,
    source: str = "",
) -> ScreenVerdict:
    """Decide, from a frontier report, whether training at `nfe` is worth attempting.

    `report` is a sweep-report frontier (`benchmarks.vla.schedule_sweep.build_sweep_report`) or
    an evidence file in the same row shape (`instinctflash/distill/evidence/`): rows carry
    `nfe`, `guidance` (canonical or None = family default), `point_success`, `delta`,
    `interval`, `deciding_lower_bound`, `n_pairs`, `forwards_batch1/2`.
    """
    target = {str(k): int(v) for k, v in dict(nfe).items()}
    if margin is None:
        margin = float(report.get("margin_declared_confers_nothing",
                                  (report.get("sweep") or {}).get("screening", {}).get("margin", -0.05)))
    rows = [r for r in report.get("rows", []) if {str(k): int(v) for k, v in dict(r.get("nfe", {})).items()} == target
            and r.get("status", "SCREENED") == "SCREENED"]
    baseline = dict(report.get("baseline") or {})
    baseline_id = ((report.get("sweep") or {}).get("baseline") or {}).get("id") or baseline.get("id", "baseline")
    baseline.setdefault("id", baseline_id)
    baseline_nfe = ((report.get("sweep") or {}).get("baseline") or {}).get("nfe") or baseline.get("nfe")
    candidates: list[dict[str, Any]] = []
    if baseline_nfe is not None and {str(k): int(v) for k, v in dict(baseline_nfe).items()} == target:
        candidates.append({
            "point": baseline_id, "guidance": baseline.get("guidance"),
            "success": baseline.get("success"), "delta": 0.0, "interval": [0.0, 0.0],
            "deciding_lower_bound": 0.0, "n_pairs": baseline.get("n"),
            "forwards_batch1": baseline.get("forwards_batch1"), "forwards_batch2": baseline.get("forwards_batch2"),
            "cycle_p50_ms": baseline.get("cycle_p50_ms"), "is_baseline": True,
        })
    for r in rows:
        candidates.append({
            "point": r["point"], "guidance": r.get("guidance"), "success": r.get("point_success"),
            "delta": r.get("delta"), "interval": r.get("interval"),
            "interval_method": r.get("interval_method"),
            "deciding_lower_bound": _row_lower_bound(r), "n_pairs": r.get("n_pairs"),
            "forwards_batch1": r.get("forwards_batch1"), "forwards_batch2": r.get("forwards_batch2"),
            "cycle_p50_ms": r.get("cycle_p50_ms"),
        })
    ranked = rank_candidates(candidates)
    if not ranked:
        return ScreenVerdict(family, target, margin, baseline, tuple(candidates), None, VERDICT_EXTEND_SCREEN,
                             f"no screened row at schedule {target} in the report", report.get("table_markdown", ""), source)
    best = ranked[0]
    lo, hi = _row_lower_bound(best), _row_upper_bound(best)
    if lo is not None and lo > margin:
        verdict = VERDICT_NO_DISTILLATION_NEEDED
        reason = (f"deciding lower bound {lo:+.4f} > margin {margin:+.4f} at n={best.get('n_pairs')}; "
                  f"delta {best.get('delta'):+.4f}" if best.get("delta") is not None else
                  f"deciding lower bound {lo:+.4f} > margin {margin:+.4f}")
    elif hi is not None and hi < margin:
        verdict = VERDICT_TRAIN
        reason = f"interval [{lo:+.4f}, {hi:+.4f}] lies entirely below the margin {margin:+.4f}"
    else:
        verdict = VERDICT_EXTEND_SCREEN
        reason = (f"interval [{lo!s}, {hi!s}] straddles the margin {margin:+.4f} at n={best.get('n_pairs')}: "
                  f"extend the untrained screen (or certify it at n >= 500) before training")
    return ScreenVerdict(family, target, margin, baseline, tuple(candidates), best, verdict, reason,
                         report.get("table_markdown", ""), source)


class DistillPipeline:
    """The screen -> train -> verify driver for one family at one target operating point."""

    def __init__(
        self,
        family: str | FamilyAdapter,
        nfe: Mapping[str, int],
        *,
        guidance: Mapping[str, Any] | None = None,
        out_dir: str | Path | None = None,
        dataset: str | None = None,
        recipe_id: str = "fewstep_v0",
    ) -> None:
        self.adapter = family if isinstance(family, FamilyAdapter) else get_family(family)
        self.nfe = dict(nfe)
        #: the student's target guidance leg, in the declaration schema; None = the family's own
        self.guidance = dict(guidance) if guidance is not None else None
        self.grids: dict[str, ScheduleGrid] = self.adapter.schedule(self.nfe)
        self.out_dir = Path(out_dir) if out_dir else None
        self.dataset = dataset
        self.recipe_id = recipe_id

    def student_schedule(self) -> dict[str, Any]:
        """The student's operating point as the provenance block and the gate state it."""
        return canonical_schedule(
            {"nfe": dict(self.nfe),
             "grids": {k: list(g.times) for k, g in sorted(self.grids.items())},
             "guidance": self.guidance},
            self.adapter.default_guidance(),
        )

    # -- stage 1: screen -------------------------------------------------------------------------

    def screening_spec(
        self,
        *,
        name: str,
        driver: Mapping[str, Any],
        baseline_nfe: Mapping[str, int] | None = None,
        extra_points: Sequence[Mapping[str, Any]] = (),
        guidance_grid: Mapping[str, Sequence[Any]] | None = None,
        margin: float = -0.05,
        interval: str = "wald_central95",
        control_steps_per_cycle: int | None = None,
    ) -> dict[str, Any]:
        """The schedule-sweep spec whose rows become this run's standing controls.

        The baseline defaults to the family's CERTIFIED operating point when every stream
        declares one (H2's teacher; for wan_va that is 2V/4A) — screening against the shipped
        configuration, not against the unoptimized full schedule.

        THE GUIDANCE AXIS IS ALWAYS SCREENED on a family that serves CFG (RFC §11, Pillar 1
        before Pillar 2): `guidance_grid` maps a stream to the scales to sweep at the target
        schedule; when omitted on such a family it defaults to {the family's own scale, 1.0}
        — the shipped point and the negative branch off — which is the minimum grid that can
        tell a step-count cliff from a guidance artifact.
        """
        if baseline_nfe is None:
            certified = {s.name: s.certified_nfe for s in self.adapter.streams()}
            missing = sorted(k for k, v in certified.items() if v is None)
            if missing:
                raise ValueError(
                    f"{self.adapter.family}: stream(s) {missing} declare no certified NFE; "
                    "pass baseline_nfe explicitly"
                )
            baseline_nfe = certified

        def _label(nfe: Mapping[str, int]) -> str:
            return "/".join(f"{v}{k[0].upper()}" for k, v in sorted(nfe.items(), reverse=True))

        defaults = self.adapter.default_guidance()
        if guidance_grid is None and self.adapter.guidance_axis_required():
            guidance_grid = {name: sorted({float(scale), 1.0}, reverse=True)
                             for name, (mode, scale) in defaults.items() if mode == "cfg"}
        grid = {k: list(v) for k, v in (guidance_grid or {}).items()}
        unknown = sorted(set(grid) - set(defaults))
        if unknown:
            raise KeyError(f"{self.adapter.family}: guidance_grid names unknown stream(s) {unknown}")

        points: list[dict[str, Any]] = []
        if grid:
            # one point per (nfe, guidance) combination over the (single-stream) grid; multi-stream
            # grids take the cartesian product, which is what "sweep the axis" means
            import itertools

            streams = sorted(grid)
            for combo in itertools.product(*(grid[s] for s in streams)):
                guidance = {s: v for s, v in zip(streams, combo)}
                tag = "".join(f"@{s[0]}w{v:g}" if isinstance(v, (int, float)) else f"@{s[0]}{v}"
                              for s, v in guidance.items())
                points.append({"id": f"{_label(self.nfe)}{tag}", "nfe": dict(self.nfe), "guidance": guidance})
        else:
            points.append({"id": _label(self.nfe), "nfe": dict(self.nfe),
                           **({"guidance": dict(self.guidance)} if self.guidance else {})})
        points += [dict(p) for p in extra_points]
        spec: dict[str, Any] = {
            "schema_version": 1,
            "name": name,
            "baseline": {"id": _label(baseline_nfe), "nfe": dict(baseline_nfe)},
            "points": points,
            "repeat_baseline": True,
            "driver": dict(driver),
            "screening": {"margin": margin, "interval": interval, "min_pairs": 1},
            **(
                {"control_steps_per_cycle": control_steps_per_cycle}
                if control_steps_per_cycle
                else {}
            ),
        }
        if grid:
            spec["guidance_axis"] = grid
        return spec

    def screen_verdict(self, report: Mapping[str, Any], *, margin: float | None = None,
                       source: str = "") -> ScreenVerdict:
        """The verdict at THIS pipeline's target schedule from a frontier report."""
        return screen_verdict(report, family=self.adapter.family, nfe=self.nfe, margin=margin, source=source)

    # -- stage 2: train --------------------------------------------------------------------------

    def train(self, teacher: Any, *, screen: ScreenVerdict | None = None) -> Any:
        """Build the student and hand off to the platform trainer. Seamed families raise HERE.

        Pillar 1 before Pillar 2: a screen verdict is REQUIRED. `NoDistillationNeeded` is raised
        when the screen says the operating point is already free — the caller emits the
        verdict's artifact (frontier + intervals + prereg stub) instead of running a trainer.
        An inconclusive screen is also refused: extend it before spending training compute.

        The capability gate runs next and fails closed (train/recipe.py's discipline): an
        environment that cannot run the family's recipe is a startup error, never a discovery
        hours into a job.
        """
        if screen is None:
            raise ControlGateViolation(
                f"{self.adapter.family}: no screen verdict — training is considered only after the "
                f"guidance x schedule screen at the target operating point (RFC §11, Pillar 1 "
                f"before Pillar 2). Run the schedule sweep and pass screen_verdict(report)."
            )
        if screen.no_distillation_needed:
            raise NoDistillationNeeded(screen)
        if screen.verdict != VERDICT_TRAIN:
            raise ControlGateViolation(
                f"{self.adapter.family}: the screen says {screen.verdict} — {screen.reason}. Training "
                f"compute is not spent on an operating point whose untrained value is unresolved; "
                f"the untrained point is certified (or its screen extended) first."
            )
        from instinctflash.train.recipe import Environment

        ok, why = self.adapter.requires().satisfied_by(Environment())
        if not ok:
            from instinctflash.train.recipe import RecipeRejected

            raise RecipeRejected(f"{self.adapter.family}: {why}")
        return self.adapter.build_student(teacher, self.grids)

    # -- stage 3: verify -------------------------------------------------------------------------

    def verify_and_stamp(
        self,
        *,
        point: str,
        teacher_outcomes: str | Path,
        control_outcomes: str | Path | None,
        student_outcomes: str | Path,
        control_schedule: Mapping[str, Any] | None,
        margin: float,
        control_grid: Sequence[Mapping[str, Any]] | None = None,
        interval: str = "wald_central95",
        min_pairs: int = 1,
        teacher_model_id: str = "?",
        teacher_weights_sha256: str = "?",
        dataset_sha256: str = "?",
    ) -> ThreeArmReport:
        """Three-arm verification from outcome JSONL files, then the provenance stamp.

        The gate fires before any file is read when the control is absent; with all three
        arms present the report and the stamped block carry the same numbers by construction
        (the block is built FROM the report), including the control's guidance and the grid it
        was the best of.
        """
        from instinctflash.verify.certify import load_jsonl

        if control_outcomes is None:
            raise ControlGateViolation(
                f"{point}: no control outcomes file — {_LAW}."
            )
        student_schedule = self.student_schedule()
        report = verify_point(
            point=point,
            teacher=load_jsonl(str(teacher_outcomes)),
            control=load_jsonl(str(control_outcomes)),
            student=load_jsonl(str(student_outcomes)),
            student_schedule=student_schedule,
            control_schedule=control_schedule,
            margin=margin,
            control_grid=control_grid,
            guidance_axis_required=self.adapter.guidance_axis_required(),
            family_guidance=self.adapter.default_guidance(),
            interval=interval,
            min_pairs=min_pairs,
            recipe=self.recipe_id,
        )
        if self.out_dir is not None:
            from instinctflash.descriptors.distillation import build_block, stamp_distillation

            def sha(p: str | Path) -> str:
                return hashlib.sha256(Path(p).read_bytes()).hexdigest()

            block = build_block(
                recipe_id=self.recipe_id,
                family=self.adapter.family,
                teacher_model_id=teacher_model_id,
                teacher_weights_sha256=teacher_weights_sha256,
                dataset_id=self.dataset or "?",
                dataset_sha256=dataset_sha256,
                schedule=student_schedule,
                control=report.control_block(
                    teacher_outcomes_sha256=sha(teacher_outcomes),
                    control_outcomes_sha256=sha(control_outcomes),
                    student_outcomes_sha256=sha(student_outcomes),
                ),
            )
            stamp_distillation(self.out_dir, block)
        return report
