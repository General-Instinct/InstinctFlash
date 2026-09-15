"""Small synthetic presentation checks; no actual current results are rendered."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("readme_candidate", Path(__file__).with_name("render_readme_v2.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def representative_summary():
    summary = {
        "schema": "instinctflash.current_public_pipeline_qualification.v1", "status": "current_pipeline_qualified",
        "assembler": {"sha256": m.ASSEMBLER_SHA}, "contract": {"sha256": m.CONTRACT_SHA},
        "source_stage_manifest": {"sha256": m.STAGE_SHA}, "failures": [],
        "counts": {"main_expected": 8, "main_passed": 8, "foreign_expected": 6,
                   "foreign_passed": 6, "historical_families_assessed": 8},
        "source_correspondence": {"status": "passed"}, "omni_dependency_closure": {"status": "passed"},
        "published_selected_cell_ids": dict(m.PUBLISHED), "main": {}, "foreign": {}, "table": [],
        **dict.fromkeys(m.FALSE_CLAIMS, False),
    }
    for family in m.FAMILIES:
        ids = [family + suffix for suffix in ("-eager_native", "-runtime_default", "-runtime_selected")]
        ids += ["va-2v4a-fp8"] if family == "va" else ["dreamzero-dynamic-fp8"] if family == "dreamzero" else []
        cells = []
        for index, name in enumerate(ids):
            schedule = {"nfe": {"action": 4 if family == "groot" else 10}}
            if family == "va":
                v, a = (2, 4) if name == "va-2v4a-fp8" else (25, 50)
                schedule = {"video_steps": v, "action_steps": a, "nfe": {"video": v, "action": a}}
            elif family in ("edge", "nano"):
                schedule = {"sampler": "unipc", "steps": 4, "guidance": 3, "nfe": {"action": 4}}
            elif family == "dreamzero":
                schedule = {"solver_updates": 16, "nfe": {"video_action": 16}, "checkpoint_num_dit_steps": 8,
                            "step_cache": "dynamic" if name == "dreamzero-dynamic-fp8" else "checkpoint"}
            fp8 = (name == m.PUBLISHED[family] and family not in ("groot", "edge", "nano")) or name == "va-2v4a-fp8"
            cells.append({"id": name, "p50_ms": [8000, 4000, 2000, 1200][index],
                          "precision": "fp8" if fp8 else "native", "schedule": schedule})
        ws = [m.PUBLISHED[family]] + (["va-2v4a-fp8"] if family == "va" else [])
        summary["main"][family] = {"status": "passed", "paired": {"status": "passed", "cells": cells},
            "serving": {"status": "passed", "results": [{"status": "passed", "cell": name} for name in ws]},
            "historical_equivalence": {"assessment_complete": True, "status": "failed_historical_equivalence"}}
    for name, (framework, family, _) in m.FOREIGN.items():
        summary["foreign"][name] = {"status": "passed", "cell": name, "framework": framework,
                                   "family": family, "latency": {"p50_ms": 1171.23456}}
    for name in m.ROW_ORDER:
        sub = name == "va-2v4a-fp8"
        family = "va" if sub else name
        selected = name if sub else m.PUBLISHED[family]
        cells = {row["id"]: row for row in summary["main"][family]["paired"]["cells"]}
        row = {"model": name, "published_cell": selected, "current_pipeline_status": "passed",
               "native_p50_ms": None if sub else 8000, "runtime_default_p50_ms": None if sub else 4000,
               "comparison_reference_native_p50_ms": 8000 if sub else None,
               "comparison_reference_default_p50_ms": 4000 if sub else None,
               "InstinctFlash_p50_ms": cells[selected]["p50_ms"]}
        for framework, column in (("lerobot", "LeRobot"), ("vllm-omni", "vLLM_Omni")):
            status = "passed" if framework + "-" + family in m.FOREIGN else "unsupported"
            if family == "groot" and framework == "vllm-omni":
                status = "not_qualified"
            if name == "va" and framework == "lerobot":
                status = "unmeasured"
            row[column + "_status"] = status
            row[column + "_p50_ms"] = 1171.23456 if status == "passed" else None
        summary["table"].append(row)
    return summary


def test_only_intended_nine_rows_and_precision_schedule_labels():
    text, derived = m.render(representative_summary())
    lines = text.splitlines()
    assert len([line for line in lines if line.startswith("| ")]) == 11
    assert "| Model | Native PyTorch | LeRobot | vLLM-Omni | InstinctFlash |" in lines
    assert "| LingBot-VA | 8000.00 · 25V/50A | — |" in text
    assert "| ↳ LingBot-VA @2V/4A | — | 1171.23 · native, 2V/4A |" in text
    assert derived["dreamzero"]["published_cell"] == "dreamzero-dynamic-fp8"
    assert derived["dreamzero"]["p50_ms"] == 1200
    assert "FP8, 16 solver updates, dynamic cache" in text
    assert "6.67× vs full 25V/50A native" in text
    assert derived["va-2v4a-fp8"]["speedup_vs_native"] == 8000 / 1200
    assert "native, NUMERIC, UniPC4 CFG3" in text
    assert "Native PyTorch (eager)" not in text and "SDE1" not in text
    assert "failed attempts" in text and "do not certify task quality" in text


@pytest.mark.parametrize("bad", ["partial", "dreamzero_control", "va_foreign", "va_native", "step_reduction", "bad_ws", "quality", "nonfinite"])
def test_pending_or_wrong_mapping_cannot_format(bad):
    s = representative_summary()
    if bad == "partial":
        s["counts"]["main_passed"] = 7
    elif bad == "dreamzero_control":
        s["published_selected_cell_ids"]["dreamzero"] = "dreamzero-runtime_selected"
    elif bad == "va_foreign":
        s["table"][0]["LeRobot_p50_ms"] = 1171.23456
    elif bad == "va_native":
        s["table"][1]["native_p50_ms"] = 8000
    elif bad == "step_reduction":
        s["main"]["nano"]["paired"]["cells"][-1]["schedule"]["steps"] = 1
    elif bad == "bad_ws":
        s["main"]["dreamzero"]["serving"]["results"][0]["cell"] = "dreamzero-runtime_selected"
    elif bad == "quality":
        s["task_quality_certified"] = True
    else:
        s["main"]["pi05"]["paired"]["cells"][-1]["p50_ms"] = float("nan")
    with pytest.raises(ValueError):
        m.render(s)
