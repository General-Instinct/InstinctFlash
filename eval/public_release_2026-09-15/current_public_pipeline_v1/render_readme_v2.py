#!/usr/bin/env python3
"""Format a separate README candidate from the completed, SHA-bound V5 summary only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_SUMMARY = HERE / "snapshot_v5/summary.json"
ASSEMBLER_SHA = "e21227ad17fbd4b46857aa0f41a0a255e4ab27afd7f9b804381ca303be2f7cbc"
CONTRACT_SHA = "1086a7e236219f6b8a073bee9e695d870981877cd52dda8646d75dace0146f0d"
STAGE_SHA = "a316bd918ae4426460f0779f90f34985b31a8639f3b6be7f9dd20e86c9cbdbc4"
FAMILIES = ("pi05", "groot", "vla4", "vla2", "va", "edge", "nano", "dreamzero")
PUBLISHED = {family: family + "-runtime_selected" for family in FAMILIES}
PUBLISHED["dreamzero"] = "dreamzero-dynamic-fp8"
ROW_ORDER = ("va", "va-2v4a-fp8", "vla4", "vla2", "edge", "nano", "pi05", "groot", "dreamzero")
LABELS = {
    "va": "LingBot-VA", "va-2v4a-fp8": "↳ LingBot-VA @2V/4A",
    "vla4": "LingBot-VLA-4B", "vla2": "LingBot-VLA-V2-6B",
    "edge": "Cosmos3 Edge DROID", "nano": "Cosmos3 Nano DROID",
    "pi05": "pi05", "groot": "GR00T N1.7", "dreamzero": "DreamZero DROID",
}
FOREIGN = {
    "lerobot-pi05": ("lerobot", "pi05", "compiled, NFE1"),
    "lerobot-groot": ("lerobot", "groot", "native, NFE4"),
    "lerobot-va": ("lerobot", "va", "native, 2V/4A"),
    "vllm-omni-edge": ("vllm-omni", "edge", "compiled, UniPC4 CFG3"),
    "vllm-omni-nano": ("vllm-omni", "nano", "compiled, UniPC4 CFG3"),
    "vllm-omni-dreamzero": ("vllm-omni", "dreamzero", "compiled, upstream step cache"),
}
FALSE_CLAIMS = (
    "complete_historical_reproduction", "task_quality_certified",
    "prior_cosmos_SCREEN_transferred", "diagnostic_latency_used",
    "foreign_operating_points_uniformly_matched",
)
EVIDENCE_LINK = "eval/public_release_2026-09-15/results.rst"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def positive(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            "p50 must be a finite positive number")
    return value


def schedule_label(family, cell):
    schedule = cell["schedule"]
    if family in ("edge", "nano"):
        require(schedule["sampler"] == "unipc" and schedule["steps"] == 4
                and schedule["guidance"] == 3 and schedule["nfe"]["action"] == 4,
                "Cosmos must retain full UniPC4/CFG3")
        return "UniPC4 CFG3"
    if family == "va":
        pair = schedule["video_steps"], schedule["action_steps"]
        require(pair in ((25, 50), (2, 4)), "unknown VA schedule")
        require(schedule["nfe"] == {"video": pair[0], "action": pair[1]}, "VA NFE mismatch")
        return f"{pair[0]}V/{pair[1]}A"
    if family == "dreamzero":
        require(schedule["solver_updates"] == schedule["nfe"]["video_action"] == 16
                and schedule["checkpoint_num_dit_steps"] == 8,
                "DreamZero solver/checkpoint schedule changed")
        require(schedule["step_cache"] in ("checkpoint", "dynamic"), "unknown DreamZero cache")
        return "16 solver updates, dynamic cache" if schedule["step_cache"] == "dynamic" else "fixed 8/16 DiT"
    nfe = 4 if family == "groot" else 10
    require(schedule["nfe"]["action"] == nfe, "original action schedule changed")
    return f"NFE{nfe}"


def validate_summary(summary):
    require(summary["schema"] == "instinctflash.current_public_pipeline_qualification.v1"
            and summary["status"] == "current_pipeline_qualified", "summary is not complete current qualification")
    require(summary["assembler"]["sha256"] == ASSEMBLER_SHA
            and summary["contract"]["sha256"] == CONTRACT_SHA
            and summary["source_stage_manifest"]["sha256"] == STAGE_SHA, "unexpected summary source binding")
    require(summary["failures"] == [] and all(summary[key] is False for key in FALSE_CLAIMS),
            "failure or unsupported qualification claim")
    counts = summary["counts"]
    require(all(counts[key] == value for key, value in {
        "main_expected": 8, "main_passed": 8, "foreign_expected": 6,
        "foreign_passed": 6, "historical_families_assessed": 8,
    }.items()), "incomplete family coverage")
    require(summary["source_correspondence"]["status"] == summary["omni_dependency_closure"]["status"] == "passed",
            "source/dependency qualification incomplete")
    require(summary["published_selected_cell_ids"] == PUBLISHED, "published selection changed")
    require(set(summary["main"]) == set(FAMILIES) and set(summary["foreign"]) == set(FOREIGN), "family IDs differ")
    cells = {}
    for family, main in summary["main"].items():
        require(main["status"] == main["paired"]["status"] == main["serving"]["status"] == "passed",
                "paired or WS evidence incomplete")
        require(main["historical_equivalence"]["assessment_complete"] is True, "historical assessment incomplete")
        expected = [family + suffix for suffix in ("-eager_native", "-runtime_default", "-runtime_selected")]
        expected += ["va-2v4a-fp8"] if family == "va" else ["dreamzero-dynamic-fp8"] if family == "dreamzero" else []
        require([cell["id"] for cell in main["paired"]["cells"]] == expected, "paired cell coverage changed")
        serving = main["serving"]["results"]
        expected_ws = {PUBLISHED[family]} | ({"va-2v4a-fp8"} if family == "va" else set())
        require(len(serving) == len(expected_ws) and {row["cell"] for row in serving} == expected_ws
                and all(row["status"] == "passed" for row in serving), "wrong published WS selection")
        for cell in main["paired"]["cells"]:
            positive(cell["p50_ms"])
            schedule_label(family, cell)
            selected_fp8 = (cell["id"] == PUBLISHED[family] and family in ("pi05", "vla4", "vla2", "va", "dreamzero")) or cell["id"] == "va-2v4a-fp8"
            require(cell["precision"] == ("fp8" if selected_fp8 else "native"), "precision label differs")
            cells[cell["id"]] = cell
        if family == "dreamzero":
            require(cells[PUBLISHED[family]]["schedule"]["step_cache"] == "dynamic", "DreamZero published cache is not dynamic")
    for name, row in summary["foreign"].items():
        framework, family, _ = FOREIGN[name]
        require(row["status"] == "passed" and row["cell"] == name
                and row["framework"] == framework and row["family"] == family, "foreign selection differs")
        positive(row["latency"]["p50_ms"])
    require(len(summary["table"]) == 9, "table must contain exactly nine rows")
    rows = {row["model"]: row for row in summary["table"]}
    require(set(rows) == set(ROW_ORDER), "table rows missing or duplicated")
    for name, row in rows.items():
        family = "va" if name == "va-2v4a-fp8" else name
        selected = name if name == "va-2v4a-fp8" else PUBLISHED[family]
        require(row["published_cell"] == selected and row["current_pipeline_status"] == "passed",
                "table published route differs")
        require(row["InstinctFlash_p50_ms"] == cells[selected]["p50_ms"], "table selected timing differs")
        for column, suffix in (("native", "-eager_native"), ("runtime_default", "-runtime_default")):
            if name == "va-2v4a-fp8":
                require(row[column + "_p50_ms"] is None, "2V4A has no direct native/default measurement")
                ref_column = "native" if column == "native" else "default"
                require(row["comparison_reference_" + ref_column + "_p50_ms"] == cells[family + suffix]["p50_ms"],
                        "VA full-schedule comparison reference differs")
            else:
                require(row[column + "_p50_ms"] == cells[family + suffix]["p50_ms"], "table control timing differs")
        for framework, column in (("lerobot", "LeRobot"), ("vllm-omni", "vLLM_Omni")):
            foreign_id = framework + "-" + family
            if name == "va" and framework == "lerobot":
                expected_status, expected_p50 = "unmeasured", None
            elif foreign_id in FOREIGN:
                expected_status, expected_p50 = "passed", summary["foreign"][foreign_id]["latency"]["p50_ms"]
            else:
                expected_status = "not_qualified" if family == "groot" and framework == "vllm-omni" else "unsupported"
                expected_p50 = None
            require(row[column + "_status"] == expected_status and row[column + "_p50_ms"] == expected_p50,
                    "foreign table mapping differs")
    return rows, cells


def render(summary):
    rows, cells = validate_summary(summary)
    derived = {}
    table = ["| Model | Native PyTorch | LeRobot | vLLM-Omni | InstinctFlash |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ROW_ORDER:
        row = rows[name]
        family = "va" if name == "va-2v4a-fp8" else name
        cell = cells[row["published_cell"]]
        reference = row["comparison_reference_native_p50_ms"] if name == "va-2v4a-fp8" else row["native_p50_ms"]
        speedup = reference / row["InstinctFlash_p50_ms"]
        derived[name] = {"published_cell": cell["id"], "p50_ms": cell["p50_ms"],
                         "native_reference_p50_ms": reference, "speedup_vs_native": speedup,
                         "speedup_scope": "full VA 25V/50A native reference" if name == "va-2v4a-fp8" else "same-row native reference"}
        native = "—" if name == "va-2v4a-fp8" else f"{row['native_p50_ms']:.2f} · {schedule_label(family, cells[family + '-eager_native'])}"
        foreign_columns = []
        for framework, column in (("lerobot", "LeRobot"), ("vllm-omni", "vLLM_Omni")):
            status = row[column + "_status"]
            if status == "passed":
                foreign_columns.append(f"{row[column + '_p50_ms']:.2f} · {FOREIGN[framework + '-' + family][2]}")
            else:
                foreign_columns.append({"unsupported": "Unsupported", "not_qualified": "Not qualified", "unmeasured": "—"}[status])
        precision = "native, NUMERIC" if family in ("edge", "nano") else "FP8" if cell["precision"] == "fp8" else "native"
        scope = " vs full 25V/50A native" if name == "va-2v4a-fp8" else " vs native"
        flash = f"**{cell['p50_ms']:.2f}** · {precision}, {schedule_label(family, cell)} · {speedup:.2f}×{scope}"
        table.append("| " + " | ".join((LABELS[name], native, *foreign_columns, flash)) + " |")
    whats_new = [
        "## What's new 🔥", "",
        "- **Full source and eight model families.** Public install, paired inference and WebSocket serving paths are qualified for all eight models below.",
        f"- **Cosmos3 at full UniPC4/CFG3.** Edge: **{derived['edge']['p50_ms']:.2f} ms**; Nano: **{derived['nano']['p50_ms']:.2f} ms**, both native precision with NUMERIC optimizations.",
        f"- **LingBot-VA @2V/4A.** **{derived['va-2v4a-fp8']['p50_ms']:.2f} ms / {derived['va-2v4a-fp8']['speedup_vs_native']:.2f}×** versus the full 25V/50A native reference in early continuations; full 25V/50A FP8: **{derived['va']['p50_ms']:.2f} ms**.",
        f"- **pi05 FP8.** **{derived['pi05']['p50_ms']:.2f} ms / {derived['pi05']['speedup_vs_native']:.2f}×** versus native, retaining NFE10.",
    ]
    notes = [
        "Prediction p50 on **Jetson Thor** (ms), using the current public recorded-input pipeline. Speedups use the displayed native reference and unrounded p50 values.",
        "",
        *table,
        "",
        "Framework protocols, warmups, prompt/history handling, precision and schedules differ; foreign columns are not matched-compute comparisons. VA measures early continuations; its 2V/4A speedup uses the full 25V/50A native row. DreamZero's native reference retains vendor encoder compilation and an eager DiT; its FP8 route explicitly enables dynamic cache with 16 solver updates.",
        "",
        "— means unmeasured; Unsupported means no matching policy in the pinned registry; Not qualified means no validated measurement for an available route. Historical numerical comparisons and failed attempts are retained separately. These timing and serving checks do not certify task quality, and no prior Cosmos task SCREEN is transferred.",
        "",
        f"[Current protocol, raw results and historical comparison scope]({EVIDENCE_LINK}) · [Reproduction commands](REPRODUCE.rst)",
    ]
    return "\n".join(whats_new + ["", "## Results", ""] + notes) + "\n", derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=EXPECTED_SUMMARY)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local directory for candidate and binding")
    args = parser.parse_args()
    path = args.summary.absolute()
    require(path == EXPECTED_SUMMARY and path.resolve() == path, "only the regular completed snapshot_v5 summary is accepted")
    raw = path.read_bytes()
    require(digest(raw) == args.summary_sha256, "summary SHA256 differs")
    summary = json.loads(raw)
    fragment, derived = render(summary)
    output = args.output.absolute()
    require(output.resolve() == output and output.is_relative_to(HERE), "output must be a new regular local study directory")
    require(not output.exists(), "output already exists")
    source = Path(__file__).read_bytes()
    candidate = fragment.encode()
    binding = {"schema": "instinctflash.readme_candidate.v2", "status": "candidate_from_completed_qualification",
               "summary": {"path": str(path), "sha256": digest(raw)},
               "renderer": {"path": str(Path(__file__).resolve()), "sha256": digest(source)},
               "assembler": summary["assembler"],
               "candidate": {"path": str(output / "candidate.md"), "sha256": digest(candidate)},
               "published_cell_ids": PUBLISHED, "rows": derived,
               "README_modified": False, "publication_authorized_by_formatter": False,
               "scope": "Formatting only; all input comes from the completed expected-SHA V5 summary. No scientific replay or new qualification is performed."}
    output.mkdir()
    for name, data in (("candidate.md", candidate), ("binding.json", (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())):
        with (output / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"candidate": binding["candidate"], "binding_sha256": digest((output / "binding.json").read_bytes())}))


if __name__ == "__main__":
    main()
