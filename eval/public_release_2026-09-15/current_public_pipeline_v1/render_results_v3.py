#!/usr/bin/env python3
"""Produce a separate results.rst candidate from the completed V5 summary only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[1]
FORMATTER_SHA = "354ad5c9d6d59c1c1ce796ef3f1339fec5cf773dbf6738d30fb84e37edaf9401"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def formatter():
    path = HERE / "render_readme_v2.py"
    require(digest(path.read_bytes()) == FORMATTER_SHA, "completed-summary formatter source changed")
    spec = importlib.util.spec_from_file_location("bound_readme_formatter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rst_link(label, reference):
    """Format provenance without making a private absolute path a public prerequisite."""
    path = Path(reference["path"])
    if not path.is_absolute():
        path = REPO / path
    require(".." not in path.parts and not any(c in str(path) for c in "\n\r<>`"), "unsafe evidence reference")
    if not path.is_relative_to(REPO):
        return label + " (local-only provenance, SHA256 ``" + reference["sha256"] + "``)"
    target = os.path.relpath(path, STUDY)
    return f"`{label} <{target}>`__"


def csv_table(title, headers, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_ALL)
    writer.writerow(headers)
    writer.writerows(rows)
    return [".. csv-table:: " + title, "   :header-rows: 1", "", *["   " + line for line in buffer.getvalue().splitlines()], ""]


def render(summary, summary_sha256):
    module = formatter()
    rows, cells = module.validate_summary(summary)
    _, derived = module.render(summary)
    lines = ["Current public pipeline results", "===============================", "",
             "The completed public pipeline passed paired inference and WebSocket checks for eight model families, plus six separately qualified foreign-framework captures. The table contains nine operating points, including LingBot-VA at 2V/4A.", "",
             "These are recorded-input inference timings on Jetson Thor. They do not establish task success, transfer the prior Cosmos task SCREEN, or certify complete historical reproduction. Current source qualification and historical numerical comparison are separate.", "",
             "The complete machine-readable evidence is " + rst_link("summary.json", {"path": str(module.EXPECTED_SUMMARY), "sha256": summary_sha256}) + ". Its SHA256 is ``" + summary_sha256 + "``.", "",
             "Prediction p50 (ms)", "-------------------", ""]
    table = []
    for name in module.ROW_ORDER:
        row = rows[name]
        family = "va" if name == "va-2v4a-fp8" else name
        selected = cells[row["published_cell"]]
        native = "Unmeasured" if name == "va-2v4a-fp8" else f"{row['native_p50_ms']:.2f}; {module.schedule_label(family, cells[family + '-eager_native'])}"
        columns = []
        for framework, column in (("lerobot", "LeRobot"), ("vllm-omni", "vLLM_Omni")):
            status = row[column + "_status"]
            columns.append(f"{row[column + '_p50_ms']:.2f}; {module.FOREIGN[framework + '-' + family][2]}" if status == "passed"
                           else {"unsupported": "Unsupported", "not_qualified": "Not qualified", "unmeasured": "Unmeasured"}[status])
        precision = "native, NUMERIC" if family in ("edge", "nano") else "FP8" if selected["precision"] == "fp8" else "native"
        reference = "full 25V/50A native" if name == "va-2v4a-fp8" else "native"
        flash = f"{selected['p50_ms']:.2f}; {precision}, {module.schedule_label(family, selected)}; {derived[name]['speedup_vs_native']:.2f}x versus {reference}"
        table.append([module.LABELS[name], native, *columns, flash])
    lines += csv_table("Declared prediction operating points", ["Model", "Native PyTorch", "LeRobot", "vLLM-Omni", "InstinctFlash"], table)
    dreamzero_roles = {
        "dreamzero-eager_native": "Native reference; vendor encoder compilation",
        "dreamzero-runtime_default": "Runtime default control (BITEXACT)",
        "dreamzero-runtime_selected": "Native runtime-selected control (BITEXACT)",
        "dreamzero-dynamic-fp8": "Published and WebSocket route (BEHAVIORAL)",
    }
    dreamzero_rows = []
    for cell in summary["main"]["dreamzero"]["paired"]["cells"]:
        dreamzero_rows.append([cell["id"], dreamzero_roles[cell["id"]],
                              "FP8" if cell["precision"] == "fp8" else "native",
                              module.schedule_label("dreamzero", cell), f"{cell['p50_ms']:.2f}"])
    lines += csv_table("DreamZero: all four current cells", ["Cell", "Role", "Precision", "Schedule/cache", "p50 (ms)"], dreamzero_rows)
    lines += ["Speedups divide unrounded p50 values. Framework inputs, warmups, prompt/history handling, precision, compilation and schedules differ; foreign columns are not matched-compute comparisons. LeRobot pi05 uses compiled NFE1 while the paired pi05 routes retain NFE10.", "",
              "VA timings select early continuations. Its 2V/4A native cell is unmeasured; that row's speedup explicitly uses the full 25V/50A native reference. DreamZero's native reference keeps vendor encoder compilation and an eager DiT with the checkpoint's fixed 8/16 DiT mask. The published DreamZero route is FP8 with dynamic cache and 16 solver updates; its separate native runtime-selected control is still required by the paired validator.", "",
              "Unsupported means no matching policy in the pinned support registry. Not qualified means an available route has no validated measurement here. WebSocket checks cover six calls across two reset episodes and successful server close; their round-trip times are not used as benchmark p50 values.", "",
              "Historical numerical comparisons", "--------------------------------", "",
              "Each current arm is compared to its own frozen historical arm where precision and request contracts are comparable. Exact bytes are reported without changing thresholds. A valid mismatch remains visible and does not become a task-quality or causal claim. Earlier failed attempts and the negative Nano diagnostic remain preserved even when a later ordinary run matches historical bytes.", ""]
    historical_rows = []
    for family in module.FAMILIES:
        assessment = summary["main"][family]["historical_equivalence"]
        arrays = assessment["arrays"]
        require(arrays, "historical action archive classification missing")
        exact = different = noncomparable = 0
        for row in arrays:
            comparable = row.get("historically_comparable", True)
            equal = row["historical_actions_match_bytes"]
            require(type(comparable) is bool, "invalid historical comparability")
            if not comparable:
                require(family == "dreamzero" and row["cell"] == "dreamzero-runtime_selected" and equal is None,
                        "unexpected non-comparable historical route")
                noncomparable += 1
            else:
                require(type(equal) is bool, "invalid historical exactness")
                exact += equal
                different += not equal
        historical_rows.append([module.LABELS[family], f"{exact}/{exact + different}", str(different), str(noncomparable), assessment["status"]])
    lines += csv_table("Own-arm action archives", ["Family", "Exact / comparable", "Different", "Non-comparable", "Assessment"], historical_rows)
    lines += ["The new DreamZero native runtime-selected control is not comparable to the older same-ID FP8 control; its exactness and action difference are null. The other three declared DreamZero routes are compared separately. This table counts action archives, not successful tasks or episodes.", "",
              "Raw evidence and reproduction", "-----------------------------", ""]
    for family in module.FAMILIES:
        main = summary["main"][family]
        links = [rst_link("plan", main["paired"]["plan"]), rst_link("run", main["paired"]["run"]),
                 rst_link("paired CPU replay", main["paired"]["replay_report"])]
        links += [rst_link("WS " + result["cell"], result["receipt"]) for result in main["serving"]["results"]]
        lines.append("* " + module.LABELS[family] + ": " + "; ".join(links) + ".")
    lines.append("")
    for name in module.FOREIGN:
        row = summary["foreign"][name]
        lines.append("* " + name + ": " + rst_link("capture", row["capture"]) + "; " + rst_link("report", row["published_report"]) + ".")
    lines += ["", "Use the portable `reproduction guide <../../REPRODUCE.rst>`_ and `installation guide <../../INSTALL.rst>`_ for fresh environments, pinned checkpoint preparation, paired capture, reporting and serving checks. The installed entrypoints are ``python -m benchmarks.regression.reproduce``, ``python -m benchmarks.regression.serve_smoke`` and ``python -m benchmarks.regression.framework_compare``.", "",
              "Hash-bound operational receipts retain their original paths. References outside this repository are local-only provenance; they are not requirements for the portable reproduction commands. Source-stage manifest SHA256: ``" + summary["source_stage_manifest"]["sha256"] + "``.", "",
              "Experimental SDE1/CFG1 recipes are separate from this full UniPC4/CFG3 Cosmos table. See the `SDE1 guide <../../examples/cosmos3_sde1/README.rst>`_; any separately published SDE1 latency receipts carry their own inputs, compiler/cache scope and unqualified task-quality label. No SDE1 value is inferred from this summary.", ""]
    return "\n".join(lines), derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "snapshot_v5/summary.json")
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local candidate directory; never the published results path")
    args = parser.parse_args()
    path = args.summary.absolute()
    require(path == HERE / "snapshot_v5/summary.json" and path.resolve() == path, "only regular snapshot_v5 summary is accepted")
    raw = path.read_bytes()
    require(digest(raw) == args.summary_sha256, "summary SHA256 differs")
    content, rows = render(json.loads(raw), args.summary_sha256)
    output = args.output.absolute()
    require(output.resolve() == output and output.is_relative_to(HERE) and not output.exists(), "output must be a new local candidate directory")
    binding = {"schema": "instinctflash.results_candidate.v3", "status": "candidate_from_completed_qualification",
               "summary": {"path": str(path), "sha256": digest(raw)},
               "renderer": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).read_bytes())},
               "summary_validator": {"path": str(HERE / "render_readme_v2.py"), "sha256": FORMATTER_SHA},
               "candidate": {"path": str(output / "results.rst"), "sha256": digest(content.encode())},
               "intended_reader_entry": "eval/public_release_2026-09-15/results.rst", "rows": rows,
               "README_modified": False, "published_results_modified": False, "SDE1_values_included": False}
    output.mkdir()
    for name, data in (("results.rst", content.encode()), ("binding.json", (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())):
        with (output / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"candidate": binding["candidate"], "binding_sha256": digest((output / "binding.json").read_bytes())}))


if __name__ == "__main__":
    main()
