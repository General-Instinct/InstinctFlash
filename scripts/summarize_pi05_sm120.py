#!/usr/bin/env python3
"""Export a complete, source-verified 500-pair campaign as portable CPU evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.vla.pi05_sm120_libero import paired_statistics
from benchmarks.vla.util import sha256_file, write_json_atomic


def summarize(directory, source_root):
    raw = json.loads((directory / "results.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    if sha256_file(directory / "calibration.json") != plan["calibration_sha256"]:
        raise ValueError("frozen calibration manifest changed before export")
    if raw["status"] not in {"PASS", "FAIL"} or raw["summary"]["pairs"] != 500:
        raise ValueError("only a complete 500-pair qualification can be exported")
    for name, expected in plan["source_sha256"].items():
        if sha256_file(source_root / name) != expected:
            raise ValueError(f"campaign source changed: {name}")
    pairs, arms, all_noise_calls = [], {}, 0
    for task in range(10):
        receipts = {}
        for arm in ("native", "fp8"):
            name = f"{arm}-{task}.json"
            if sha256_file(directory / name) != raw["receipt_sha256"][name]:
                raise ValueError(f"receipt changed: {name}")
            receipt = json.loads((directory / name).read_text())
            if receipt["plan_sha256"] != sha256_file(directory / "plan.json"):
                raise ValueError("receipt plan changed")
            records = receipt["episodes"]
            if [r["seed"] for r in records] != list(range(40100, 40150)):
                raise ValueError("missing, duplicate or unplanned seeds")
            for row in records:
                if type(row["success"]) is not bool or not 1 <= row["steps"] <= 280:
                    raise ValueError("invalid success/step record")
                if len(row["noise_sha256"]) != (row["steps"] + 9) // 10:
                    raise ValueError("incomplete noise trace")
                if row["scene"]["init_state_index"] != row["seed"] % 50:
                    raise ValueError("wrong initial state index")
                if not np.isfinite(row["latency_ms"]).all() or min(row["latency_ms"]) <= 0:
                    raise ValueError("invalid latency trace")
            receipts[arm] = receipt
            timings = [v for row in records for v in row["latency_ms"]]
            arms[f"{arm}-{task}"] = {"successes": sum(r["success"] for r in records),
                "episodes": 50, "settings": receipt["settings"],
                "peak_allocated_bytes": receipt["peak_allocated_bytes"],
                "resident_allocated_bytes": receipt["resident_allocated_bytes"],
                "infer_p50_ms": float(np.median(timings))}
        for native, fp8 in zip(receipts["native"]["episodes"], receipts["fp8"]["episodes"]):
            common = min(len(native["noise_sha256"]), len(fp8["noise_sha256"]))
            if native["scene"] != fp8["scene"] or native["noise_sha256"][:common] != fp8["noise_sha256"][:common]:
                raise ValueError("initial observation/state or noise differs between paired arms")
            all_noise_calls += common
            pairs.append({"task_id": task, "seed": native["seed"],
                "native": native["success"], "fp8": fp8["success"],
                "native_steps": native["steps"], "fp8_steps": fp8["steps"],
                "initial_observation_sha256": native["scene"]["initial_observation_sha256"],
                "native_action_digest": native["action_digest"], "fp8_action_digest": fp8["action_digest"]})
    summary = paired_statistics(pairs)
    if summary != raw["summary"]:
        raise ValueError("summary does not match the executed episode receipts")
    protocol = {k: v for k, v in plan.items() if k not in ("config", "source_sha256", "environment")}
    tasks = [{"task_id": task, **paired_statistics([p for p in pairs if p["task_id"] == task])}
             for task in range(10)]
    limitations = ["Confidence statements are conditional on this fixed ten-task suite.",
                   "Calibration outlier selection used task-7 demonstration frames, not its episode outcomes.",
                   "The former 30-episode screen has a different protocol and is not a paired calibration ablation."]
    if not plan["performance_target_met"]:
        limitations.append("Calibration speed measurements did not meet the 1.5x target; see study_scores.")
    return {"schema_version": 1, "status": raw["status"], "hardware": "RTX 5090 / SM120",
        "scope": "FlashRT BF16 versus FP8 with official checkpoint processors, computed chunk 50, executed horizon 10",
        "protocol": protocol, "source_sha256": plan["source_sha256"], "environment": plan["environment"],
        "calibration": raw["calibration"], "summary": summary, "tasks": tasks, "arms": arms,
        "pairing": {"matched_initial_states_and_observations": 500, "matched_noise_calls": all_noise_calls},
        "pairs": pairs, "execution_plan_sha256": sha256_file(directory / "plan.json"),
        "exporter_source_sha256": sha256_file(Path(__file__)),
        "receipt_sha256": raw["receipt_sha256"],
        "limitations": limitations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.campaign.resolve(), args.source_root.resolve())
    write_json_atomic(args.output, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
