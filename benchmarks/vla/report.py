"""Evidence aggregation and release gates for paired benchmark runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from instinctflash.verify.certify import NotCertifiable, Outcome, certify

from .plan import pipeline_digest, validate_plan
from .registry import Registry
from .result import validate_result
from .screening import paired_success_summary
from .util import ConfigurationError, load_json, percentile, sha256_json, write_json_atomic


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left == right else 0.0
    return dot / (left_norm * right_norm)


def _action_mode(model: dict[str, Any], gate: dict[str, Any]) -> tuple[str, float | None, float | None]:
    mode = gate["mode"]
    if mode == "registry":
        if model["determinism"] == "bitexact":
            return "bitexact", None, None
        return "numeric", float(model["repeatability_max_abs"]), float(gate.get("min_cosine", 0.0))
    if mode == "bitexact":
        return mode, None, None
    return mode, float(gate["max_abs"]), float(gate["min_cosine"])


def _load_results(run_dir: Path, plan: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    results: dict[str, dict[str, Any]] = {}
    issues = []
    for job in plan["jobs"]:
        path = run_dir / "results" / f"{job['job_id']}.json"
        if not path.exists():
            issues.append(f"missing result for job {job['job_id']}")
            continue
        try:
            result = load_json(path)
            validate_result(result, job)
        except ConfigurationError as error:
            issues.append(f"invalid result {job['job_id']}: {error}")
            continue
        results[job["job_id"]] = result
    return results, issues


def _paired(
    plan: dict[str, Any], results: dict[str, dict[str, Any]], treatment: str
) -> tuple[list[tuple[dict, dict, dict, dict]], list[str]]:
    by_pair: dict[str, dict[str, tuple[dict, dict]]] = defaultdict(dict)
    for job in plan["jobs"]:
        result = results.get(job["job_id"])
        if result is not None:
            by_pair[job["request"]["pair_id"]][job["request"]["arm"]["id"]] = (job, result)
    pairs = []
    issues = []
    for pair_id, arms in by_pair.items():
        if plan["control_arm"] not in arms or treatment not in arms:
            issues.append(f"pair {pair_id} is missing {plan['control_arm']} or {treatment}")
            continue
        control_job, control = arms[plan["control_arm"]]
        treat_job, treat = arms[treatment]
        if control["resolved_seed"] != treat["resolved_seed"]:
            issues.append(
                f"pair {pair_id} resolved different simulator seeds: "
                f"{control['resolved_seed']} vs {treat['resolved_seed']}"
            )
            continue
        scene_a = control["provenance"].get("scene_sha256")
        scene_b = treat["provenance"].get("scene_sha256")
        if (scene_a is not None or scene_b is not None) and scene_a != scene_b:
            issues.append(f"pair {pair_id} resolved different frozen scenes/instructions")
            continue
        pairs.append((control_job, control, treat_job, treat))
    return pairs, issues


def _action_gate(
    pairs: list[tuple[dict, dict, dict, dict]], model: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    eligible = [
        pair
        for pair in pairs
        if pair[0]["request"]["suite"]["kind"] in {"contract", "latency", "open_loop"}
    ]
    if not eligible:
        return {"verdict": "NOT_APPLICABLE", "pairs": 0}
    mode, max_abs_limit, min_cosine = _action_mode(model, gate)
    digest_matches = 0
    max_abs = 0.0
    minimum_cosine = 1.0
    incomplete = []
    for control_job, control, _, treatment in eligible:
        left_metrics = control["metrics"]
        right_metrics = treatment["metrics"]
        left_digest = left_metrics.get("action_digest")
        # Two absent digests are missing evidence, never agreement: with a custom registry whose
        # suites drop action_digest from required_metrics, None == None must not count bitexact.
        if left_digest is not None and left_digest == right_metrics.get("action_digest"):
            digest_matches += 1
        if mode == "numeric":
            # Drivers owe only their suite's required_metrics (DRIVER_CONTRACT.md), and the
            # latency suite preregisters latency_ms/action_digest/finite without action_values.
            # The numeric value comparison is therefore scoped to suites that require
            # action_values (contract, open_loop); latency pairs keep the digest count above.
            if "action_values" not in control_job["request"]["suite"]["required_metrics"]:
                continue
            left = left_metrics.get("action_values")
            right = right_metrics.get("action_values")
            if not left or not right or len(left) != len(right):
                incomplete.append(control_job["request"]["pair_id"])
                continue
            max_abs = max(max_abs, max(abs(a - b) for a, b in zip(left, right)))
            minimum_cosine = min(minimum_cosine, _cosine(left, right))
    if mode == "bitexact":
        passed = digest_matches == len(eligible)
    else:
        passed = not incomplete and max_abs <= float(max_abs_limit) and minimum_cosine >= float(min_cosine)
    return {
        "verdict": "PASS" if passed else ("INCOMPLETE" if incomplete else "FAIL"),
        "mode": mode,
        "pairs": len(eligible),
        "digest_matches": digest_matches,
        "max_abs": max_abs if mode == "numeric" else None,
        "max_abs_limit": max_abs_limit,
        "minimum_cosine": minimum_cosine if mode == "numeric" else None,
        "min_cosine_limit": min_cosine,
        "incomplete_pairs": incomplete,
    }


# These are versioned driver contracts, not inferred from the presence of a hash.
_CLOSED_LOOP_ACTION_ENCODINGS = {
    "groot-libero-paused-v1": "full applied trace flattened in C order and encoded as big-endian float64",
    "lingbot_vla-joint-robotwin-paused-v1": "per-action 14D controller values encoded as big-endian float64",
    "lingbot_vla_v2-joint-robotwin-paused-v1": "per-action 14D controller values encoded as big-endian float64",
    "pi05-libero-schedule-paused-v1": "full applied trace flattened in C order and encoded as big-endian float64",
    "wan-va-libero-paused-v1": "per-action dtype string followed by contiguous native action bytes (7D)",
    "wan-va-robotwin-paused-v1": "per-action 16D controller values encoded as big-endian float64",
}


def _closed_loop_action_evidence(pairs, jobs, control_id):
    """Observed trajectory equivalence, independent of statistical quality gates."""
    expected = defaultdict(list)
    observed = defaultdict(list)
    for job in jobs:
        req = job["request"]
        if req["arm"]["id"] == control_id and req["suite"]["kind"] == "closed_loop":
            expected[(req["model_id"], req["suite_id"])].append(req)
    for pair in pairs:
        req = pair[0]["request"]
        if req["suite"]["kind"] == "closed_loop":
            observed[(req["model_id"], req["suite_id"])].append(pair)
    evidence = []
    for (model_id, suite_id), requests in sorted(expected.items()):
        entries = observed[(model_id, suite_id)]
        bridges = {r["suite"]["protocol"].get("bridge") for r in requests}
        bridge = next(iter(bridges)) if len(bridges) == 1 else None
        encoding = _CLOSED_LOOP_ACTION_ENCODINGS.get(bridge)
        rows = []
        for cj, control, tj, treatment in entries:
            left, right = control["metrics"], treatment["metrics"]
            counts = (left.get("executed_steps"), right.get("executed_steps"))
            scene = control["provenance"].get("scene_sha256")
            supported = encoding and tj["request"]["suite"]["protocol"].get("bridge") == bridge
            valid = (supported and scene is not None
                     and scene == treatment["provenance"].get("scene_sha256")
                     and all(type(n) is int and n > 0 for n in counts)
                     and left.get("finite") is True and right.get("finite") is True
                     and left.get("action_digest") is not None and right.get("action_digest") is not None)
            equal = valid and counts[0] == counts[1] and left["action_digest"] == right["action_digest"]
            rows.append({
                "pair_id": cj["request"]["pair_id"], "task": cj["request"]["task"],
                "resolved_seed": control["resolved_seed"],
                "verdict": ("PASS" if equal else "FAIL") if valid else "INCOMPLETE",
                "control_steps": counts[0], "treatment_steps": counts[1],
                "control_success": left.get("success"), "treatment_success": right.get("success"),
            })
        missing = sorted({r["pair_id"] for r in requests} - {r["pair_id"] for r in rows})
        incomplete = missing or any(r["verdict"] == "INCOMPLETE" for r in rows)
        different = any(r["verdict"] == "FAIL" for r in rows)
        verdict = ("FAIL" if different else "INCOMPLETE" if incomplete else "PASS") if encoding else "NOT_APPLICABLE"
        evidence.append({
            "model_id": model_id, "suite_id": suite_id, "verdict": verdict,
            "protocol": bridge, "encoding": encoding,
            "scope": "Executed controller action streams on these paired frozen scenes only; not hidden-state or universal bitexact",
            "quality_gate": False, "expected_pairs": len(requests), "pairs": len(rows),
            "matching_pairs": sum(r["verdict"] == "PASS" for r in rows),
            "different_pairs": sum(r["verdict"] == "FAIL" for r in rows),
            "missing_pair_ids": missing,
            "control_successes": sum(r["control_success"] is True for r in rows),
            "treatment_successes": sum(r["treatment_success"] is True for r in rows),
            "details": rows,
        })
    return evidence


def _performance_gate(pairs: list[tuple[dict, dict, dict, dict]], gate: dict[str, Any]) -> dict[str, Any]:
    control_samples: list[float] = []
    treatment_samples: list[float] = []
    for control_job, control, _, treatment in pairs:
        if control_job["request"]["suite"]["kind"] != "latency":
            continue
        control_samples.extend(control["metrics"].get("latency_ms", ()))
        treatment_samples.extend(treatment["metrics"].get("latency_ms", ()))
    if not control_samples or not treatment_samples:
        return {"verdict": "NOT_APPLICABLE", "samples": 0}
    control_p50 = percentile(control_samples, 0.5)
    treatment_p50 = percentile(treatment_samples, 0.5)
    speedup = control_p50 / treatment_p50
    minimum = float(gate["min_speedup"])
    return {
        "verdict": "PASS" if speedup >= minimum else "FAIL",
        "samples": min(len(control_samples), len(treatment_samples)),
        "control_p50_ms": control_p50,
        "treatment_p50_ms": treatment_p50,
        "speedup": speedup,
        "min_speedup": minimum,
    }


def _success_gates(
    pairs: list[tuple[dict, dict, dict, dict]], gate: dict[str, Any], treatment_id: str
) -> list[dict[str, Any]]:
    by_model_suite: dict[tuple[str, str], list[tuple[dict, dict, dict, dict]]] = defaultdict(list)
    for pair in pairs:
        request = pair[0]["request"]
        if request["suite"]["kind"] == "closed_loop":
            by_model_suite[(request["model_id"], request["suite_id"])].append(pair)
    certificates = []
    for (model_id, suite_id), entries in sorted(by_model_suite.items()):
        protocol = entries[0][0]["request"]["suite"]["protocol"]
        margin = float(gate["margin"])
        interval = gate["interval"]
        if "margin" in protocol and float(protocol["margin"]) != margin:
            certificates.append(
                {
                    "model_id": model_id,
                    "suite_id": suite_id,
                    "verdict": "INVALID_PROTOCOL",
                    "reason": (
                        f"arm margin {margin} disagrees with suite preregistration "
                        f"{protocol['margin']}"
                    ),
                }
            )
            continue
        if "interval" in protocol and protocol["interval"] != interval:
            certificates.append(
                {
                    "model_id": model_id,
                    "suite_id": suite_id,
                    "verdict": "INVALID_PROTOCOL",
                    "reason": (
                        f"arm interval {interval} disagrees with suite preregistration "
                        f"{protocol['interval']}"
                    ),
                }
            )
            continue
        teacher = []
        student = []
        for control_job, control, _, treatment in entries:
            request = control_job["request"]
            episode_id = request["pair_id"]
            seed = control["resolved_seed"]
            task = request["task"]
            teacher.append(Outcome(episode_id, seed, task, control["metrics"]["success"]))
            student.append(Outcome(episode_id, seed, task, treatment["metrics"]["success"]))
        try:
            teacher_hash = sha256_json(sorted(entry[0]["request_sha256"] for entry in entries))
            student_hash = sha256_json(sorted(entry[2]["request_sha256"] for entry in entries))
            certificate = certify(
                teacher,
                student,
                margin=margin,
                teacher_hash=teacher_hash,
                student_hash=student_hash,
                harness="benchmarks.vla",
                recipe=treatment_id,
                seeds="explicit in immutable plan",
                min_pairs=int(gate["min_pairs"]),
                interval=interval,
                fail_on_task_collapse=True,
            )
        except NotCertifiable as error:
            certificates.append(
                {"model_id": model_id, "suite_id": suite_id, "verdict": "INCOMPLETE", "reason": str(error)}
            )
            continue
        payload = json.loads(certificate.to_json())
        payload.update(model_id=model_id, suite_id=suite_id)
        certificates.append(payload)
    return certificates


def _repeatability(
    plan: dict[str, Any], results: dict[str, dict[str, Any]], registry: Registry
) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    for job in plan["jobs"]:
        result = results.get(job["job_id"])
        if result is None:
            continue
        request = job["request"]
        key = (
            request["model_id"], request["suite_id"], request["task"],
            request["requested_seed"], request["arm"]["id"],
        )
        grouped[key].append((job, result))
    checks = []
    for key, entries in sorted(grouped.items()):
        if len(entries) < 2:
            continue
        model = registry.models[key[0]]
        first = entries[0][1]["metrics"]
        fingerprints = {entry[1]["provenance"]["environment_fingerprint"] for entry in entries}
        if model["determinism"] == "bitexact":
            reference = first.get("action_digest")
            # Same vacuous-match hardening as the paired gate: a repeat group with no digests
            # is unproven repeatability, not bit-exact repeatability.
            passed = reference is not None and all(
                item[1]["metrics"].get("action_digest") == reference for item in entries[1:]
            )
            measured = 0.0 if passed else None
            limit = 0.0
        else:
            reference = first.get("action_values")
            measured = 0.0
            passed = reference is not None
            if reference is not None:
                for _, result in entries[1:]:
                    values = result["metrics"].get("action_values")
                    if not values or len(values) != len(reference):
                        passed = False
                        break
                    measured = max(measured, max(abs(a - b) for a, b in zip(reference, values)))
            limit = float(model["repeatability_max_abs"])
            passed = passed and measured <= limit
        passed = passed and len(fingerprints) == 1
        checks.append(
            {
                "model_id": key[0], "suite_id": key[1], "task": key[2], "seed": key[3],
                "arm": key[4], "repeats": len(entries), "max_abs": measured, "limit": limit,
                "environment_fingerprints": sorted(fingerprints),
                "verdict": "PASS" if passed else "FAIL",
            }
        )
    return checks


def _environment_consistency(
    plan: dict[str, Any], results: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for job in plan["jobs"]:
        result = results.get(job["job_id"])
        if result is None:
            continue
        request = job["request"]
        grouped[(request["model_id"], request["arm"]["id"])].add(
            result["provenance"]["environment_fingerprint"]
        )
    return [
        {
            "model_id": model_id,
            "arm": arm,
            "environment_fingerprints": sorted(fingerprints),
            "verdict": "PASS" if len(fingerprints) == 1 else "FAIL",
        }
        for (model_id, arm), fingerprints in sorted(grouped.items())
    ]


def build_report(
    run_dir: str | Path,
    registry: Registry,
    *,
    allow_synthetic: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    plan = load_json(root / "plan.json")
    validate_plan(plan)
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
    arms = {arm["id"]: arm for arm in plan["arms"]}
    comparisons = []
    any_synthetic = any(result["provenance"]["synthetic"] for result in results.values())
    for arm_id, arm in arms.items():
        if arm["role"] != "treatment":
            continue
        pairs, pair_issues = _paired(plan, results, arm_id)
        issues.extend(pair_issues)
        per_model: dict[str, list[tuple[dict, dict, dict, dict]]] = defaultdict(list)
        for pair in pairs:
            per_model[pair[0]["request"]["model_id"]].append(pair)
        model_gates = []
        for model_id, model_pairs in sorted(per_model.items()):
            by_suite: dict[str, list[tuple[dict, dict, dict, dict]]] = defaultdict(list)
            for pair in model_pairs:
                by_suite[pair[0]["request"]["suite_id"]].append(pair)
            model_gates.append(
                {
                    "model_id": model_id,
                    "action": _action_gate(model_pairs, registry.models[model_id], arm["gates"]["action"]),
                    "performance": _performance_gate(model_pairs, arm["gates"]["performance"]),
                    "suites": [
                        {
                            "suite_id": suite_id,
                            "action": _action_gate(
                                suite_pairs, registry.models[model_id], arm["gates"]["action"]
                            ),
                            "performance": _performance_gate(
                                suite_pairs, arm["gates"]["performance"]
                            ),
                        }
                        for suite_id, suite_pairs in sorted(by_suite.items())
                    ],
                }
            )
        comparisons.append(
            {
                "control": plan["control_arm"],
                "treatment": arm_id,
                "paired_jobs": len(pairs),
                "models": model_gates,
                "closed_loop": _success_gates(pairs, arm["gates"]["success"], arm_id),
                "closed_loop_actions": _closed_loop_action_evidence(pairs, plan["jobs"], plan["control_arm"]),
                "paired_success": paired_success_summary(pairs, plan["jobs"], plan["control_arm"]),
            }
        )

    repeatability = _repeatability(plan, results, registry)
    environment_consistency = _environment_consistency(plan, results)
    verdicts = []
    for comparison in comparisons:
        for model in comparison["models"]:
            verdicts.extend((model["action"]["verdict"], model["performance"]["verdict"]))
            for suite in model["suites"]:
                verdicts.extend((suite["action"]["verdict"], suite["performance"]["verdict"]))
        verdicts.extend(cert["verdict"] for cert in comparison["closed_loop"])
    verdicts.extend(check["verdict"] for check in repeatability)
    verdicts.extend(check["verdict"] for check in environment_consistency)
    complete = len(results) == len(plan["jobs"]) and not issues
    gates_passed = not any(
        verdict == "INCOMPLETE" or verdict == "INVALID_PROTOCOL" or verdict.startswith("FAIL")
        for verdict in verdicts
    )
    screening = any(job["request"]["suite"].get("protocol", {}).get("screening", False)
                    for job in plan["jobs"])
    reportable = complete and gates_passed and not screening and (allow_synthetic or not any_synthetic)
    report = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "registry_sha256": registry.digest,
        "execution_pipeline_sha256": plan["pipeline_sha256"],
        "analysis_pipeline_sha256": pipeline_digest(),
        "environment_sha256": run_manifest["environment_sha256"],
        "complete": complete,
        "synthetic": any_synthetic,
        "reportable": reportable,
        "gates_passed": gates_passed,
        "screening": screening,
        "results": len(results),
        "expected_results": len(plan["jobs"]),
        "issues": issues,
        "comparisons": comparisons,
        "repeatability": repeatability,
        "environment_consistency": environment_consistency,
    }
    report["report_sha256"] = sha256_json(report)
    destination = Path(output).resolve() if output else root / "report.json"
    write_json_atomic(destination, report)
    return report
