"""Render completed study metrics without changing or promoting measured cells.

Inputs are user_report JSON and an independently audited prompt-update gate.
Only gates[cell_id].status == 'passed' admits an update into the main minimum.
Each passing gate's receipt_sha256 must match that exact report receipt.
The full 31-cell report remains the record of all controls and measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


FAMILIES = {"pi05": "pi05", "vla4": "LingBot-VLA 4B", "vla2": "LingBot-VLA V2 6B",
            "groot": "GR00T N1.7", "edge": "Cosmos3 Edge", "nano": "Cosmos3 Nano",
            "va": "LingBot-VA", "dreamzero": "DreamZero"}
MAIN_ARMS = ("eager_native", "runtime_default", "runtime_selected")
CSV_FIELDS = ("section", "family", "metric", "samples", "eager_cell", "eager_p50_ms",
              "default_cell", "default_route", "default_p50_ms", "default_ratio_vs_eager",
              "fastest_cell", "fastest_route", "fastest_p50_ms", "fastest_ratio_vs_eager",
              "operating_cell", "operating_route", "operating_p50_ms", "operating_ratio_vs_eager",
              "schedule", "report_sha256", "prompt_gate_sha256", "task_quality_validated", "recommended")


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _check(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    _canonical(value)
    return value, hashlib.sha256(data).hexdigest()


def _p50(row):
    value = row["primary"]["p50_ms"]
    _check(type(value) in (int, float) and math.isfinite(value) and value > 0,
           f"{row['id']}: primary p50 must be positive and finite")
    return float(value)


def _route(row):
    if row["arm"] == "eager_native":
        return "eager native"
    precision = row["precision"]
    _check(precision in ("native", "fp8"), "unknown precision")
    tier = row.get("execution_policy", {}).get("tier_ceiling") or row["runtime_kwargs"].get("tier_ceiling")
    tier = tier or ("numeric" if precision == "fp8" else "bitexact")
    _check(tier in ("bitexact", "numeric", "behavioral"), "unknown arithmetic ceiling")
    return f"{'FP8' if precision == 'fp8' else 'native'}; ceiling {tier.upper()}"


def _same_policy(eager, candidate):
    return all(_canonical(eager.get(key)) == _canonical(candidate.get(key))
               for key in ("family", "comparison_group", "model_id", "revision", "effective_schedule", "guidance"))


def _matched_to_eager(row, eager, rows, matched):
    seen = set()
    while row["id"] != eager["id"]:
        _check(row["id"] not in seen and _same_policy(eager, row), "invalid matched-policy comparison chain")
        seen.add(row["id"])
        comparisons = matched.get(row["id"], [])
        _check(len(comparisons) == 1, f"{row['id']}: requires one matched-policy comparison")
        comparison = comparisons[0]
        _check(comparison.get("same_sampling_policy") is True, "matched comparison changed sampling policy")
        if row["arm"] == "runtime_update":
            _check(comparison["baseline"] == row.get("baseline_cell"), "update comparison differs from explicit baseline")
        row = rows[comparison["baseline"]]
        _check(row["arm"] in MAIN_ARMS, "matched comparison cannot pass through an operating point or update")


def _schedule_label(row):
    schedule = row["effective_schedule"]
    nfe = schedule["nfe"]
    if row["family"] == "va":
        return f"{nfe['video']} video / {nfe['action']} action steps"
    _check(row["family"] == "dreamzero" and schedule.get("step_cache") == "dynamic",
           "unexpected operating-point schedule")
    return f"{nfe['video_action']} solver updates; dynamic reuse"


def build_summary(report, prompt_gate):
    """Select descriptive minima only from complete, comparable measured rows."""
    _check(report.get("schema") == "instinctflash.user_report.v1" and report.get("status") == "passed",
           "summary requires a fully passed user report")
    _check(not report.get("errors") and not report.get("pending"), "report still has errors or pending artifacts")
    for flag in ("task_quality_validated", "recommended"):
        _check(report.get(flag) is False, "report must not certify quality or recommend a route")
    _check(report.get("quality_certified", False) is False, "report must not certify quality")
    expected = {"expected_cells": 31, "main_cells": 24, "operating_point_cells": 4, "runtime_update_cells": 3,
                "passed_cells": 31, "failed_cells": 0, "incomplete_cells": 0, "failed_comparisons": 0}
    _check(all(type(report.get("counts", {}).get(key)) is int and report["counts"][key] == value
               for key, value in expected.items()), "summary requires the complete 31-cell 24/4/3 study")
    cells = report["cells"]
    _check(isinstance(cells, list) and len(cells) == 31, "report requires 31 measured cells")
    rows = {row["id"]: row for row in cells}
    _check(len(rows) == len(cells), "duplicate report cell IDs")
    _check(isinstance(prompt_gate.get("gates"), dict), "prompt gate requires a gates mapping")
    for row in cells:
        _check(row.get("status") == "passed" and row.get("family") in FAMILIES, "unpassed or unknown-family row")
        _check(row.get("task_quality_validated") is False and row.get("recommended") is False,
               "cell must not certify quality or recommend a route")
        _check(row["arm"] in (*MAIN_ARMS, "operating_point", "runtime_update"), "unknown report arm")
        _p50(row)
        _check(row["primary"].get("count") == (12 if row["family"] in ("va", "dreamzero") else 20),
               "primary sample count does not match the measured protocol")
    _check(len({row["device"] for row in cells}) == 1 and "thor" in cells[0]["device"].lower(),
           "summary requires the single Thor study")
    matched, changed = {}, {}
    for comparison in report["comparisons"]:
        _check(comparison.get("status") == "passed", "report has an unpassed comparison")
        target = matched if comparison["kind"] == "matched_policy_latency_ratio" else changed
        _check(comparison["kind"] in ("matched_policy_latency_ratio", "changed_policy_latency_ratio"), "unknown comparison kind")
        _check(comparison["candidate"] in rows and comparison["baseline"] in rows, "comparison references an unknown row")
        target.setdefault(comparison["candidate"], []).append(comparison)
    output, excluded = [], []
    for family, label in FAMILIES.items():
        main = [row for row in cells if row["family"] == family and row["arm"] in MAIN_ARMS]
        _check(sorted(row["arm"] for row in main) == sorted(MAIN_ARMS), f"{family}: main arms incomplete")
        eager = next(row for row in main if row["arm"] == "eager_native")
        default = next(row for row in main if row["arm"] == "runtime_default")
        eligible = [row for row in main if row["arm"] != "eager_native"]
        for row in main:
            _matched_to_eager(row, eager, rows, matched)
        for row in cells:
            if row["family"] != family or row["arm"] != "runtime_update":
                continue
            _check(row.get("experimental") is False, "runtime update must be nonexperimental")
            gate = prompt_gate["gates"].get(row["id"], {})
            _check(isinstance(gate, dict), "prompt gate entry must be an object")
            if gate.get("status") != "passed":
                excluded.append(row["id"])
                continue
            _check(_is_sha256(gate.get("receipt_sha256")) and gate["receipt_sha256"] == row["receipt"]["sha256"],
                   "prompt gate receipt SHA-256 mismatch")
            _matched_to_eager(row, eager, rows, matched)
            eligible.append(row)
        fastest = min(eligible, key=lambda row: (_p50(row), row["id"]))
        output.append({"section": "same_sampling_policy", "family": label, "metric": eager["primary_metric"],
                       "samples": eager["primary"]["count"], "eager_cell": eager["id"], "eager_p50_ms": _p50(eager),
                       "default_cell": default["id"], "default_route": _route(default), "default_p50_ms": _p50(default),
                       "default_ratio_vs_eager": _p50(eager) / _p50(default),
                       "fastest_cell": fastest["id"], "fastest_route": _route(fastest), "fastest_p50_ms": _p50(fastest),
                       "fastest_ratio_vs_eager": _p50(eager) / _p50(fastest), "schedule": _canonical(eager["effective_schedule"])})
    operating = [row for row in cells if row["arm"] == "operating_point"]
    _check(len(operating) == 4 and sum(row["arm"] == "runtime_update" for row in cells) == 3,
           "report arm counts disagree with required 24/4/3 layout")
    for row in operating:
        comparisons = changed.get(row["id"], [])
        _check(len(comparisons) == 1, "operating point requires one declared cross-policy comparison")
        comparison = comparisons[0]
        eager = rows[comparison["baseline"]]
        _check(eager["family"] == row["family"] and eager["arm"] == "eager_native",
               "operating-point cross-policy baseline must be its eager reference")
        _check(comparison.get("same_sampling_policy") is False, "operating point must identify its changed policy")
        ratio = comparison["ratio_of_p50"]
        _check(type(ratio) in (int, float) and math.isfinite(ratio)
               and math.isclose(ratio, _p50(eager) / _p50(row), rel_tol=1e-12), "cross-policy ratio disagrees with measured p50")
        output.append({"section": "changed_operating_point", "family": FAMILIES[row["family"]],
                       "metric": row["primary_metric"], "samples": row["primary"]["count"],
                       "eager_cell": eager["id"], "eager_p50_ms": _p50(eager),
                       "operating_cell": row["id"], "operating_route": _route(row), "operating_p50_ms": _p50(row),
                       "operating_ratio_vs_eager": ratio, "schedule": _canonical(row["effective_schedule"]),
                       "schedule_label": _schedule_label(row)})
    return output, excluded


def _csv_line(values):
    stream = io.StringIO()
    csv.writer(stream, quoting=csv.QUOTE_ALL).writerow(values)
    return stream.getvalue().strip()


def render(report_path, prompt_gate_path, csv_path, rst_path):
    report_path, prompt_gate_path, csv_path, rst_path = map(lambda path: Path(path).resolve(),
                                                         (report_path, prompt_gate_path, csv_path, rst_path))
    _check(len({report_path, prompt_gate_path, csv_path, rst_path}) == 4, "summary outputs must not replace either input or each other")
    report, report_hash = _json(report_path)
    gate, gate_hash = _json(prompt_gate_path)
    _check(gate.get("report_sha256") == report_hash, "prompt gate report SHA-256 mismatch")
    rows, excluded = build_summary(report, gate)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "report_sha256": report_hash, "prompt_gate_sha256": gate_hash,
                         "task_quality_validated": False, "recommended": False})
    lines = ["Public API results on Jetson Thor", "================================", "",
             "P50 latency in milliseconds; ratios are eager p50 divided by the measured p50.", "",
             ".. csv-table:: Same declared sampling policy",
             '   :header: "Family", "Eager ms", "Default route", "Default ms", "Eager/default", "Fastest Runtime route (cell)", "Runtime ms", "Eager/Runtime"', ""]
    for row in rows[:8]:
        lines.append("   " + _csv_line((row["family"], f"{row['eager_p50_ms']:.1f}", row["default_route"], f"{row['default_p50_ms']:.1f}",
                                      f"{row['default_ratio_vs_eager']:.2f}x", f"{row['fastest_route']} ({row['fastest_cell']})",
                                      f"{row['fastest_p50_ms']:.1f}", f"{row['fastest_ratio_vs_eager']:.2f}x")))
    lines += ["", ".. csv-table:: Changed operating points (separate)",
              '   :header: "Cell", "Route", "Actual schedule", "P50 ms", "Eager/p50"', ""]
    for row in rows[8:]:
        lines.append("   " + _csv_line((row["operating_cell"], row["operating_route"], row["schedule_label"],
                                      f"{row['operating_p50_ms']:.1f}", f"{row['operating_ratio_vs_eager']:.2f}x")))
    lines += ["", "Ratios above 1 mean lower latency than eager; a default need not be faster.",
              "Fastest Runtime is the minimum measured eligible Runtime p50; eager stays a separate reference.",
              "This is neither task-quality certification nor a recommendation.", "",
              "Ceilings limit permitted Runtime transforms; they do not assert action equality against eager.",
              "Eager disables TorchDynamo; Runtime retains upstream compile behavior. See report.json for action differences.",
              "One Thor, same checkpoint revisions and sampling policy within each main row; precision may differ.",
              "Stateless: five warmups then 20 measured calls with resets and alternating prompts.",
              "History: one warmup episode, then six three-cycle episodes; p50 uses 12 continuation calls.",
              "Recorded cameras and synthetic states are prepared; public preprocessing/predict/commit is timed.",
              "Setup, first calls and pi05's separate 51-call queue remain in the full report.",
              "DreamZero eager setup includes verification of 2,146 loaded tensors.",
              "Changed operating-point ratios do not isolate infrastructure gains or establish task quality.",
              "Prompt updates enter the main minimum only after their corresponding audited gate passes.",
              "The full 31-cell report retains all original controls, including slower FP8 routes.",
              "Updates excluded from the minimum: " + (", ".join(excluded) if excluded else "none") + ".", "",
              f"Report SHA-256: {report_hash}", f"Prompt gate SHA-256: {gate_hash}", ""]
    _check(len(lines) <= 80, "summary exceeds 80 lines")
    csv_path.write_text(stream.getvalue(), encoding="utf-8")
    rst_path.write_text("\n".join(lines), encoding="utf-8")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--report", type=Path, default=root / "report.json")
    parser.add_argument("--prompt-gate", type=Path, default=root / "prompt_update_gate.json")
    parser.add_argument("--csv-output", type=Path, default=root / "results.csv")
    parser.add_argument("--rst-output", type=Path, default=root / "results.rst")
    args = parser.parse_args(argv)
    render(args.report, args.prompt_gate, args.csv_output, args.rst_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
