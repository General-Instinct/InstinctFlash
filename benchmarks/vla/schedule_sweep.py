"""Schedule sweeps: N declared schedule points, paired against a baseline arm, as one plan.

This module productizes the untrained low-step frontier campaign (2026-08-31) that was run by
hand for LingBot-VA: seven serving configurations over the same pinned scenes, every point paired
against a re-run baseline arm, a repeat of the baseline as the noise floor, latency measured
per point, and a frontier table -- success and effective Hz per schedule -- as the artifact.
The campaign's prereg and outputs are the reference fixture
(`tests/fixtures/fewstep_frontier/`); what was seven hand-launched server restarts and a
bespoke analysis script becomes one sweep spec compiled into the standard immutable plan.

A POINT IS AN OPERATING POINT, NOT A STEP COUNT. The guidance x NFE campaign (2026-08-31/09-02,
`iwm_distill/fewstep/guidance/gw_autotable.md`, `h1_report.md` §4b) measured the SAME 1V/4A
schedule at 0.752 (w=5), 0.882 (w=3), 0.885 (w=1, batch-1) and 0.276 (w=9) on the same pinned
scenes: `nfe` alone underspecifies quality by 60 points. A sweep point therefore declares the
tuple (schedule grid, per-stream guidance, CFG batching): `nfe` as before, an optional
`guidance` block in the declaration schema (`descriptors/guidance.py`: a mode name, a numeric
scale, or {mode, scale} per stream), and the forward split the latency table reports separately
-- batch-1 vs batch-2 forwards per cycle (declared as `batch2_forwards_per_cycle`, or derived
from the family adapter's phase structure when it is importable). The report groups points by
schedule and names the BEST untrained configuration over the guidance grid swept at each: that,
not the shipped guidance, is the matched-NFE control a trained point must beat (RFC §11).

DESIGN: A SWEEP IS ARMS, NOT A NEW EXECUTION PATH. A schedule point is exactly an arm whose
operating point declares a different step schedule (and guidance) -- so a sweep spec compiles to
the existing arms structure (control = the baseline schedule, one treatment arm per point, plus an optional
repeat of the baseline) and then into a normal `build_plan` plan. Everything the pipeline
already guarantees is inherited unchanged: immutable jobs, deterministic per-pair
counterbalancing of arm order (the campaign's B P1..P5 / B P5..P1 discipline, per pair),
resolved-seed pairing, resumable execution, environment drift refusal. `doctor` and `run`
work on a sweep plan without knowing it is one.

SCREENING, EXPLICITLY NOT CERTIFICATION (frontier prereg §0). The frontier report quotes paired
deltas and intervals per point; the certify() verdict line is recorded in the JSON for the
archive but the table never quotes a SHIP verdict. Certification of any single point remains a
separate pre-registered run at certificate scale. The per-point rows double as the standing
matched-NFE controls that any future distillation run at that schedule must beat
(`docs/rfc/fewstep-distillation.md`).

The gates carried by the generated treatment arms are deliberately vacuous (and say so): the
frontier table is this run's decision surface, and a screening sweep that failed `report`'s
release gates on every slower-or-worse point would be noise, not information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from instinctflash.descriptors.guidance import (
    GuidanceDeclarationError, canonical_guidance, resolve, validate_declared_guidance,
)
from instinctflash.verify.certify import NotCertifiable, Outcome, certify

from .plan import build_plan, validate_plan
from .registry import Registry
from .report import _load_results, _paired
from .util import ConfigurationError, load_json, percentile, require_keys, sha256_json, write_json_atomic

SWEEP_SCHEMA_VERSION = 1

#: The interval methods a sweep may declare; these are `verify.certify`'s methods. The hand
#: campaign decided on the most conservative of three intervals (McNemar-SE / iid bootstrap /
#: task-cluster bootstrap, `eval/lingbot_va_robotwin/certify_operating_point.py`); porting that
#: analysis into `verify.certify` is an open item in the RFC. Until then a sweep declares ONE
#: method up front, which keeps the discipline (declared before the run) if not the exact rule.
INTERVALS = ("wald_central95", "tango_one_sided95")

#: Note stamped into every generated treatment arm's gates.
_GATES_NOTE = (
    "screening sweep: these gates are deliberately vacuous; the frontier table produced by "
    "`sweep-report` is the decision surface, and it issues intervals, never ship verdicts"
)


def load_sweep(path: str | Path) -> dict[str, Any]:
    spec = load_json(Path(path))
    validate_sweep(spec)
    return spec


def _validate_point(point: dict[str, Any], where: str) -> None:
    require_keys(point, ("id", "nfe"), where)
    if not isinstance(point["id"], str) or not point["id"]:
        raise ConfigurationError(f"{where}: id must be a non-empty string")
    nfe = point["nfe"]
    if not isinstance(nfe, dict) or not nfe:
        raise ConfigurationError(
            f"{where}: nfe must map phase name -> steps, e.g. {{'video': 1, 'action': 4}}. "
            "Per-phase, never a bare count: a WAM's two streams are the axis being swept."
        )
    for phase, steps in nfe.items():
        if not isinstance(steps, int) or steps < 1:
            raise ConfigurationError(f"{where}: nfe[{phase!r}] must be a positive integer")
    fwd = point.get("forwards_per_cycle")
    if fwd is not None and (not isinstance(fwd, int) or fwd < 1):
        raise ConfigurationError(f"{where}: forwards_per_cycle must be a positive integer")
    guidance = point.get("guidance")
    if guidance is not None:
        if not isinstance(guidance, dict) or not guidance:
            raise ConfigurationError(
                f"{where}: guidance must map stream name -> a mode name, a numeric scale, or "
                "{'mode': ..., 'scale': ...} (descriptors/guidance.py) -- the second leg of the "
                "operating point, per stream like nfe"
            )
        try:
            validate_declared_guidance(guidance, where=f"{where}.guidance")
        except GuidanceDeclarationError as error:
            raise ConfigurationError(str(error)) from None
    split = point.get("batch2_forwards_per_cycle")
    if split is not None:
        if fwd is None:
            raise ConfigurationError(
                f"{where}: batch2_forwards_per_cycle needs forwards_per_cycle to split"
            )
        if not isinstance(split, int) or split < 0 or split > fwd:
            raise ConfigurationError(
                f"{where}: batch2_forwards_per_cycle must be an integer in [0, forwards_per_cycle]"
            )


def validate_sweep(spec: dict[str, Any]) -> None:
    require_keys(
        spec, ("schema_version", "name", "baseline", "points", "driver", "screening"), "sweep"
    )
    if spec["schema_version"] != SWEEP_SCHEMA_VERSION:
        raise ConfigurationError(f"sweep: unsupported schema_version {spec['schema_version']!r}")
    if not isinstance(spec["name"], str) or not spec["name"]:
        raise ConfigurationError("sweep: name must be a non-empty string")
    _validate_point(spec["baseline"], "sweep.baseline")
    if not isinstance(spec["points"], list) or not spec["points"]:
        raise ConfigurationError("sweep: at least one schedule point is required")
    identifiers = [spec["baseline"]["id"]]
    for index, point in enumerate(spec["points"]):
        _validate_point(point, f"sweep.points[{index}]")
        identifiers.append(point["id"])
    if len(set(identifiers)) != len(identifiers):
        raise ConfigurationError("sweep: point ids (including the baseline) must be unique")
    screening = spec["screening"]
    require_keys(screening, ("margin", "interval"), "sweep.screening")
    margin = float(screening["margin"])
    if not -1.0 < margin < 0.0:
        raise ConfigurationError("sweep.screening.margin must be strictly between -1 and 0")
    if screening["interval"] not in INTERVALS:
        raise ConfigurationError(
            f"sweep.screening.interval must be one of {INTERVALS}; the decision rule is part of "
            "the declaration, never guessed"
        )
    if int(screening.get("min_pairs", 1)) < 1:
        raise ConfigurationError("sweep.screening.min_pairs must be positive")
    steps = spec.get("control_steps_per_cycle")
    if steps is not None and (not isinstance(steps, int) or steps < 1):
        raise ConfigurationError("sweep: control_steps_per_cycle must be a positive integer")
    if not isinstance(spec.get("repeat_baseline", True), bool):
        raise ConfigurationError("sweep: repeat_baseline must be boolean")
    # the driver block is validated for real by validate_arms() once embedded in an arm


def repeat_arm_id(spec: dict[str, Any]) -> str:
    return f"{spec['baseline']['id']}__repeat"


def _arm(spec: dict[str, Any], point: dict[str, Any], *, arm_id: str, role: str) -> dict[str, Any]:
    arm: dict[str, Any] = {
        "id": arm_id,
        "role": role,
        "driver": json.loads(json.dumps(spec["driver"])),
        "operating_point": {
            "name": point["id"],
            # BEHAVIORAL: a schedule point CHANGES the computation; its claim is carried by
            # paired closed-loop evidence, never by an action envelope.
            "tier": "BEHAVIORAL",
            "schedule": {"nfe": dict(point["nfe"])},
        },
    }
    if point.get("forwards_per_cycle") is not None:
        arm["operating_point"]["schedule"]["forwards_per_cycle"] = int(point["forwards_per_cycle"])
    if point.get("guidance") is not None:
        # the guidance leg rides in the arm's schedule, so a driver serves the declared scale
        # (LingBot: `--guidance video=3`) and a repeat/baseline pair stays execution-matched
        arm["operating_point"]["schedule"]["guidance"] = json.loads(json.dumps(point["guidance"]))
    if role == "treatment":
        screening = spec["screening"]
        arm["gates"] = {
            "note": _GATES_NOTE,
            # Vacuous by construction (see the module docstring): a 3V point is legitimately
            # slower than the baseline and a 1V point legitimately changes every action.
            "performance": {"min_speedup": 1e-9},
            "action": {"mode": "numeric", "max_abs": 1e9, "min_cosine": -1.0},
            "success": {
                "margin": float(screening["margin"]),
                "interval": screening["interval"],
                "min_pairs": int(screening.get("min_pairs", 1)),
            },
        }
    return arm


def arms_from_sweep(spec: dict[str, Any]) -> dict[str, Any]:
    """Compile a sweep spec into the standard arms structure, deterministically.

    Arm order is baseline, points in declared order, repeat arm last -- the campaign's run
    order. `build_plan` then counterbalances the whole ordering per pair.
    """
    validate_sweep(spec)
    baseline = spec["baseline"]
    arms = [_arm(spec, baseline, arm_id=baseline["id"], role="control")]
    for point in spec["points"]:
        arms.append(_arm(spec, point, arm_id=point["id"], role="treatment"))
    if spec.get("repeat_baseline", True):
        # The noise floor: the identical configuration, re-run. Kept LAST in declared order so
        # half the pairs (the counterbalanced half) separate it maximally from the control arm,
        # mirroring the campaign's arm-1-first / arm-7-last drift honesty.
        arms.append(_arm(spec, baseline, arm_id=repeat_arm_id(spec), role="treatment"))
    return {"schema_version": 1, "control_arm": baseline["id"], "arms": arms}


def build_sweep_plan(
    registry: Registry,
    spec: dict[str, Any],
    profile_name: str,
    selected_models=None,
) -> dict[str, Any]:
    """A standard immutable plan that also carries its sweep declaration, re-signed.

    The `sweep` block rides in the plan (and is covered by `plan_id`) so a run directory is
    self-describing: `sweep-report` refuses a run whose plan never declared a sweep, and a
    hand-edited sweep block breaks the plan digest exactly as any other tampering does.
    """
    plan = build_plan(registry, arms_from_sweep(spec), profile_name, selected_models)
    plan.pop("plan_id")
    plan["sweep"] = json.loads(json.dumps(spec))
    plan["plan_id"] = sha256_json(plan)
    validate_plan(plan)
    return plan


def forward_split(point: dict[str, Any], backbone: str | None = None) -> tuple[int | None, int | None]:
    """(batch-1 forwards, batch-2 forwards) per cycle for a point, or (None, None) when unknown.

    Precedence: the point's own `batch2_forwards_per_cycle`; else the family adapter's phase
    structure at the point's (nfe, guidance) when the adapter is importable here (wan_va is,
    torch-free); else, when the point's guidance requests no negative branch on any stream,
    every forward is batch-1; else unknown -- reported as n/d, never guessed.
    """
    total = point.get("forwards_per_cycle")
    if point.get("batch2_forwards_per_cycle") is not None and total is not None:
        return total - int(point["batch2_forwards_per_cycle"]), int(point["batch2_forwards_per_cycle"])
    guidance = point.get("guidance")
    if backbone:
        try:
            from instinctflash import load

            spec = load(backbone).spec().with_nfe(dict(point["nfe"])).with_guidance(guidance)
            b = spec.cfg_batching()
            declared = spec.total_forwards()
            b2 = b["batch2_forwards"] + b["separate_branch_forwards"]
            if total is None:
                return declared - b2, b2
            # the declared count excludes protocol forwards (cache-only terminals); scale the
            # phase split onto the declared total the same way the family declares it
            if b2 == 0:
                return total, 0
            if b2 == declared:
                return 0, total
            return total - b2, b2
        except Exception:  # noqa: BLE001 - adapter not importable here; fall through
            pass
    if guidance is not None:
        resolved = resolve(guidance)
        if not any(r.negative_branch for r in resolved.values()):
            return total, 0
    return None, None


def schedule_key(nfe: dict[str, Any]) -> str:
    return json.dumps(dict(sorted(nfe.items())), sort_keys=True, separators=(",", ":"))


def best_untrained_per_schedule(
    spec: dict[str, Any], successes: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Per schedule (nfe), the BEST configuration over the guidance grid swept at it.

    THE CONTROL LAW'S STRONG FORM (RFC §11): a trained point at schedule S is verified against
    the best UNTRAINED configuration over the guidance grid swept at S, not against S at the
    shipped guidance. The H1 screen is the precedent -- vs 1V/4A@w5 the student showed +14.3 pp;
    vs the best untrained knob (1V/4A@w3) it showed +0.0. Ties -- successes within ONE paired
    episode of the top (the campaign's tie rule, h1_report §4b) -- go to the configuration with
    fewer batch-2 forwards (cheaper), then to the lower scale. Points without a success (not
    evaluated) are listed in the grid but cannot win.

    `successes` maps point id -> success, or -> {"success", "n_pairs"} when the pairing size
    is known (it sets the tie tolerance).
    """
    from instinctflash.verify.ranking import rank_candidates

    groups: dict[str, dict[str, Any]] = {}
    candidates = [spec["baseline"], *spec["points"]]
    for point in candidates:
        key = schedule_key(point["nfe"])
        entry = groups.setdefault(key, {"nfe": dict(point["nfe"]), "candidates": []})
        _, b2 = forward_split(point)
        guidance = canonical_guidance(point.get("guidance")) if point.get("guidance") is not None else None
        got = successes.get(point["id"])
        success, n_pairs = (got.get("success"), got.get("n_pairs")) if isinstance(got, dict) else (got, None)
        entry["candidates"].append({
            "point": point["id"],
            "guidance": guidance,
            "success": success,
            "n_pairs": n_pairs,
            "forwards_batch2": b2,
        })
    for entry in groups.values():
        ranked = rank_candidates(entry["candidates"])
        entry["guidance_grid_swept"] = [c["guidance"] for c in entry["candidates"]]
        entry["best"] = ranked[0] if ranked else None
    return groups


# --- the frontier report ---------------------------------------------------------------------


def _closed_loop_outcomes(pairs) -> tuple[list[Outcome], list[Outcome]]:
    control_outcomes, treat_outcomes = [], []
    for control_job, control, _, treatment in pairs:
        request = control_job["request"]
        if request["suite"]["kind"] != "closed_loop":
            continue
        episode = request["pair_id"]
        seed = control["resolved_seed"]
        task = request["task"]
        control_outcomes.append(Outcome(episode, seed, task, control["metrics"]["success"]))
        treat_outcomes.append(Outcome(episode, seed, task, treatment["metrics"]["success"]))
    return control_outcomes, treat_outcomes


def _latency_p50(pairs, side: int) -> float | None:
    """p50 over the latency-suite samples of one side of the pair (1=control, 3=treatment)."""
    samples: list[float] = []
    for pair in pairs:
        if pair[0]["request"]["suite"]["kind"] != "latency":
            continue
        samples.extend(pair[side]["metrics"].get("latency_ms", ()))
    return percentile(samples, 0.5) if samples else None


def _paired_block(
    pairs, screening: dict[str, Any], *, label: str
) -> dict[str, Any]:
    """One point's paired evidence vs the baseline, in screening framing.

    The certify() analysis runs unchanged at the declared margin/interval; its verdict line is
    RECORDED (the archive is the point) and marked screening -- prereg §0: passing a margin here
    confers nothing, and no ship verdict is issued from a sweep.
    """
    control_outcomes, treat_outcomes = _closed_loop_outcomes(pairs)
    if not control_outcomes:
        return {"status": "NOT EVALUATED", "reason": "no paired closed-loop evidence"}
    try:
        certificate = certify(
            control_outcomes,
            treat_outcomes,
            margin=float(screening["margin"]),
            interval=screening["interval"],
            min_pairs=int(screening.get("min_pairs", 1)),
            harness="benchmarks.vla.schedule_sweep",
            recipe=label,
            seeds="explicit in immutable plan",
        )
    except NotCertifiable as error:
        return {"status": "NOT EVALUATED", "reason": str(error)}
    teacher_only, treat_only = certificate.discordant
    deciding = (
        certificate.lower_confidence_bound
        if certificate.lower_confidence_bound is not None
        else certificate.ci95[0]
    )
    interval = (
        list(certificate.tango_central90)
        if certificate.tango_central90 is not None
        else list(certificate.ci95)
    )
    return {
        "status": "SCREENED",
        "screening": True,
        "n_pairs": certificate.n_pairs,
        "baseline_success": certificate.teacher_success,
        "point_success": certificate.student_success,
        "delta": certificate.delta,
        "interval_method": screening["interval"],
        "interval": interval,
        "deciding_lower_bound": deciding,
        "discordant": [teacher_only, treat_only],
        "discordance": (teacher_only + treat_only) / certificate.n_pairs,
        "mcnemar_exact_two_sided_p": certificate.p_value,
        # recorded for the archive, never quoted in the table (prereg §0)
        "certify_verdict_recorded_not_quoted": certificate.verdict,
    }


def _fmt(value, pattern: str, missing: str = "n/e") -> str:
    return missing if value is None else pattern.format(value)


def _row_cells(point: dict[str, Any], block: dict[str, Any], *, cycle_ms, hz,
               split: tuple[int | None, int | None] = (None, None)) -> list[str]:
    b1, b2 = split
    head = [
        point["id"],
        _fmt(b1, "{}", "n/d"),
        _fmt(b2, "{}", "n/d"),
        _fmt(cycle_ms, "{:.0f}"),
        _fmt(hz, "{:.1f}"),
    ]
    if block.get("status") != "SCREENED":
        return head + ["n/e", f"NOT EVALUATED — {block.get('reason', '?')}", "—", "—", "—"]
    lo, hi = block["interval"]
    return head + [
        f"{block['point_success']:.4f}",
        f"{block['delta']:+.4f}",
        f"[{lo:+.4f}, {hi:+.4f}] ({block['interval_method']})",
        str(block["n_pairs"]),
        f"{100.0 * block['discordance']:.1f}%",
    ]


def frontier_table(report: dict[str, Any]) -> str:
    """The frontier table, in the campaign artifact's shape (the reference fixture)."""
    spec = report["sweep"]
    baseline_id = spec["baseline"]["id"]
    lines = [
        f"## {spec['name']} — schedule sweep (SCREENING, paired; baseline {baseline_id}; "
        f"n<={report['max_pairs']}/point)",
        "",
        "Each point is an operating point (schedule grid, per-stream guidance, CFG batching); "
        "forwards per cycle are reported batch-1 and batch-2 separately (n/d = the split is not "
        "declared and could not be derived).",
        "",
        f"| point | batch-1 fwd/cycle | batch-2 fwd/cycle | cycle p50 ms | eff. Hz | success | "
        f"Δ vs {baseline_id} (paired) | 95% interval ({report['interval_method']}) | paired n | "
        f"discordance |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    base = report["baseline"]
    floor = report.get("noise_floor")
    floor_cell = "—"
    if floor and floor.get("status") == "SCREENED":
        floor_cell = (
            f"repeat Δ{floor['delta']:+.4f}, floor {100.0 * floor['discordance']:.1f}%"
        )
    base_b1, base_b2 = base.get("forwards_batch1"), base.get("forwards_batch2")
    lines.append(
        f"| **{baseline_id} (baseline arm)** | {_fmt(base_b1, '{}', 'n/d')} | {_fmt(base_b2, '{}', 'n/d')} "
        f"| {_fmt(base.get('cycle_p50_ms'), '{:.0f}')} | {_fmt(base.get('effective_hz'), '{:.1f}')} "
        f"| {_fmt(base.get('success'), '{:.4f}')} | (control) | {floor_cell} "
        f"| {_fmt(base.get('n'), '{}')} | — |"
    )
    for row in report["rows"]:
        lines.append("| " + " | ".join(row["cells"]) + " |")
    lines.append("")
    if floor and floor.get("status") == "SCREENED":
        lo, hi = floor["interval"]
        lines.append(
            f"Noise floor (baseline repeat, {floor['n_pairs']} pairs): delta "
            f"{floor['delta']:+.4f}, discordance {100.0 * floor['discordance']:.1f}%, "
            f"interval [{lo:+.4f}, {hi:+.4f}] ({floor['interval_method']})."
        )
    best = report.get("best_untrained_per_schedule") or {}
    if any(len(entry["candidates"]) > 1 for entry in best.values()):
        lines += [
            "",
            "Best untrained configuration per schedule (the matched-NFE control for a trained point "
            "at that schedule is the BEST over the guidance grid swept there, not the shipped "
            "guidance):",
            "",
            "| schedule | guidance grid swept | best point | success |",
            "|---|---|---|---|",
        ]
        for entry in best.values():
            grid = ", ".join(_guidance_label(g) for g in entry["guidance_grid_swept"])
            win = entry.get("best")
            lines.append(
                f"| {_nfe_label(entry['nfe'])} | {grid} | "
                + (f"{win['point']} | {win['success']:.4f} |" if win else "n/e | n/e |")
            )
    lines.append("")
    lines.append(
        "Screening sweep, explicitly not certification: intervals are the result; no ship "
        "verdicts are issued from this table (frontier prereg §0). Each row is the standing "
        "matched-NFE control for any future distillation run at that operating point; per "
        "schedule, the control is the best row over the guidance grid swept."
    )
    return "\n".join(lines) + "\n"


def _nfe_label(nfe: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(nfe.items()))


def _guidance_label(guidance) -> str:
    if not guidance:
        return "family default"
    parts = []
    for stream, g in sorted(guidance.items()):
        scale = g.get("scale")
        parts.append(f"{stream}={g.get('mode')}@{scale:g}" if scale is not None else f"{stream}={g.get('mode')}")
    return " ".join(parts)


def build_sweep_report(
    run_dir: str | Path,
    registry: Registry,
    *,
    allow_synthetic: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Pair every schedule point against the baseline arm and emit the frontier table.

    Evidence discipline is `build_report`'s: the plan/manifest/environment bindings must hold,
    every result validates, unpaired or seed-divergent pairs are dropped AND reported. Rows
    carry intervals; verdict lines are archived, not quoted.
    """
    root = Path(run_dir).resolve()
    plan = load_json(root / "plan.json")
    validate_plan(plan)
    spec = plan.get("sweep")
    if not spec:
        raise ConfigurationError(
            f"{root}: this run's plan declares no sweep; `sweep-report` reads only plans built "
            "by `sweep-plan`. For an ordinary paired run use `report`."
        )
    validate_sweep(spec)
    if plan["registry_sha256"] != registry.digest:
        raise ConfigurationError(
            f"run registry {plan['registry_sha256']} differs from current registry {registry.digest}"
        )
    run_manifest = load_json(root / "run_manifest.json")
    expected_manifest = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "pipeline_sha256": plan["pipeline_sha256"],
        "registry_sha256": plan["registry_sha256"],
        "environment_sha256": sha256_json(load_json(root / "environment.json")),
    }
    if run_manifest != expected_manifest:
        raise ConfigurationError("run_manifest.json does not match the plan/environment evidence")

    results, issues = _load_results(root, plan)
    any_synthetic = any(result["provenance"]["synthetic"] for result in results.values())
    screening = spec["screening"]
    steps_per_cycle = spec.get("control_steps_per_cycle")
    backbones = sorted({job["request"]["model"]["backbone"] for job in plan["jobs"]})
    backbone = backbones[0] if len(backbones) == 1 else None

    def effective_hz(p50_ms) -> float | None:
        if p50_ms is None or not steps_per_cycle:
            return None
        return 1000.0 * steps_per_cycle / p50_ms

    # The baseline row reports the baseline ARM's own evidence (all its completed closed-loop
    # episodes and its latency p50), exactly as the campaign table did — not the paired subset
    # of any one comparison, whose n varies with each point's dropped pairs.
    baseline_successes: list[bool] = []
    baseline_latency: list[float] = []
    for job in plan["jobs"]:
        request = job["request"]
        if request["arm"]["id"] != plan["control_arm"]:
            continue
        result = results.get(job["job_id"])
        if result is None:
            continue
        if request["suite"]["kind"] == "closed_loop":
            baseline_successes.append(bool(result["metrics"]["success"]))
        elif request["suite"]["kind"] == "latency":
            baseline_latency.append(percentile(result["metrics"]["latency_ms"], 0.5))
    baseline_p50 = percentile(baseline_latency, 0.5) if baseline_latency else None
    base_b1, base_b2 = forward_split(spec["baseline"], backbone)
    baseline_summary: dict[str, Any] = {
        "success": (
            sum(baseline_successes) / len(baseline_successes) if baseline_successes else None
        ),
        "n": len(baseline_successes) or None,
        "cycle_p50_ms": baseline_p50,
        "effective_hz": effective_hz(baseline_p50),
        "guidance": (canonical_guidance(spec["baseline"].get("guidance"))
                     if spec["baseline"].get("guidance") is not None else None),
        "forwards_per_cycle": spec["baseline"].get("forwards_per_cycle"),
        "forwards_batch1": base_b1,
        "forwards_batch2": base_b2,
    }

    rows = []
    noise_floor = None
    max_pairs = 0
    point_by_id = {point["id"]: point for point in spec["points"]}
    for arm in plan["arms"]:
        if arm["role"] != "treatment":
            continue
        pairs, pair_issues = _paired(plan, results, arm["id"])
        issues.extend(pair_issues)
        block = _paired_block(pairs, screening, label=arm["id"])
        treat_p50 = _latency_p50(pairs, 3)
        if block.get("n_pairs"):
            max_pairs = max(max_pairs, block["n_pairs"])
        if arm["id"] == repeat_arm_id(spec):
            noise_floor = block
            continue
        point = point_by_id[arm["id"]]
        split = forward_split(point, backbone)
        rows.append(
            {
                "point": arm["id"],
                "nfe": dict(point["nfe"]),
                "guidance": (canonical_guidance(point.get("guidance"))
                             if point.get("guidance") is not None else None),
                "forwards_per_cycle": point.get("forwards_per_cycle"),
                "forwards_batch1": split[0],
                "forwards_batch2": split[1],
                "cycle_p50_ms": treat_p50,
                "effective_hz": effective_hz(treat_p50),
                **block,
                "cells": _row_cells(point, block, cycle_ms=treat_p50, hz=effective_hz(treat_p50),
                                    split=split),
            }
        )
    successes: dict[str, Any] = {row["point"]: {"success": row.get("point_success"), "n_pairs": row.get("n_pairs")}
                                 for row in rows}
    successes[spec["baseline"]["id"]] = {"success": baseline_summary["success"], "n_pairs": baseline_summary["n"]}

    complete = len(results) == len(plan["jobs"]) and not issues
    report = {
        "schema_version": 1,
        "kind": "schedule_sweep_frontier",
        "screening": True,
        "plan_id": plan["plan_id"],
        "registry_sha256": registry.digest,
        "environment_sha256": run_manifest["environment_sha256"],
        "sweep": spec,
        "interval_method": screening["interval"],
        "margin_declared_confers_nothing": float(screening["margin"]),
        "complete": complete,
        "synthetic": any_synthetic,
        # Screening evidence is REPORTABLE-AS-SCREENING only when complete and real; it is never
        # reportable as a certification, which is why the field says which claim it carries.
        "reportable_as_screening": complete and (allow_synthetic or not any_synthetic),
        "results": len(results),
        "expected_results": len(plan["jobs"]),
        "issues": issues,
        "max_pairs": max_pairs,
        "baseline": baseline_summary,
        "rows": rows,
        "noise_floor": noise_floor,
        # per schedule, the best configuration over the guidance grid swept: the control a
        # trained point at that schedule must beat (distill.pipeline.verify_point reads this)
        "best_untrained_per_schedule": best_untrained_per_schedule(spec, successes),
    }
    report["table_markdown"] = frontier_table(report)
    report["report_sha256"] = sha256_json(report)
    destination = Path(output).resolve() if output else root / "frontier.json"
    write_json_atomic(destination, report)
    table_path = destination.with_suffix(".md")
    table_path.write_text(report["table_markdown"], encoding="utf-8")
    return report
