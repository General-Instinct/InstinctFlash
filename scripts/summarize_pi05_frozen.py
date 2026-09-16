#!/usr/bin/env python3
"""Export source-verified frozen replay, repeated rollout and 500-pair evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.vla import pi05_sm120_libero as matched
from benchmarks.vla.pi05_frozen_replay import compare_rollouts, summarize as summarize_replay
from benchmarks.vla.pi05_frozen_libero import check as check_qualification
from benchmarks.vla.pi05_sm120_diagnostic import validate_pair
from benchmarks.vla.util import sha256_file, write_json_atomic
from flash_rt.core import frozen_state as state
from flash_rt.frontends.torch.pi05_frozen import observation_hashes


def read(path):
    return json.loads(Path(path).read_text())


def export(replay_dir, qualification_dir):
    replay = summarize_replay(replay_dir, 3)
    catalog = state.load(replay_dir / "state.json")
    rollout_receipts = [read(replay_dir / f"rollout-both-{r}.json") for r in range(2)]
    for receipt in rollout_receipts:
        if receipt["plan_sha256"] != sha256_file(replay_dir / "plan.json") or receipt["state_sha256"] != sha256_file(replay_dir / "state.json"):
            raise ValueError("repeated rollout belongs to a different plan/state")
    rollout = compare_rollouts(*[r["episodes"] for r in rollout_receipts])
    previous_rollout = read(replay_dir / "rollout-results.json")
    if any(rollout[k] != previous_rollout[k] for k in rollout):
        raise ValueError("repeated rollout summary differs from receipts")
    preparation, previous = check_qualification(qualification_dir)
    for name, expected in preparation["prerequisite_sha256"].items():
        if sha256_file(replay_dir / name) != expected:
            raise ValueError("qualification prerequisite changed")
    plan = read(qualification_dir / "plan.json")
    if plan["frozen_catalog_sha256"] != sha256_file(replay_dir / "state.json"):
        raise ValueError("qualification uses a different algorithm catalog")
    if plan["source_extension_sha256"] != preparation["extensions"]:
        raise ValueError("qualification extension sources changed")
    raw = matched.report(qualification_dir, list(range(10)), 50)
    raw["protocol"] = plan["protocol"]
    write_json_atomic(qualification_dir / "results.json", raw)
    pairs, task_states, noise_calls = [], {}, 0
    for task in range(10):
        task_path = qualification_dir / f"state-{task}.json"
        if sha256_file(task_path) != plan["frozen_state_sha256"][str(task)]:
            raise ValueError("task frozen state changed")
        task_state = state.load(task_path, catalog["identity"])
        if task_state["gemm"] != catalog["gemm"] or task_state["token_lengths"] != catalog["token_lengths"]:
            raise ValueError("task state uses a different algorithm catalog/profile set")
        prompt = read(qualification_dir / f"scenes-{task}.json")["40100"]["prompt"]
        calibration, _ = matched.calibration_frames(qualification_dir, prompt)
        if task_state["prompt"] != prompt or task_state["calibration"] != {
                "observations_sha256": observation_hashes(calibration), "percentile": plan["selected"]["percentile"]}:
            raise ValueError("task state calibration provenance changed")
        task_states[str(task)] = {"state_sha256": sha256_file(task_path), "prompt": prompt,
                                 "scale_sha256": state.digest(task_state["scale_bits"]),
                                 "calibration": task_state["calibration"]}
        native = read(qualification_dir / f"native-{task}.json")
        fp8 = read(qualification_dir / f"fp8-{task}.json")
        if fp8["frozen_state_sha256"] != task_states[str(task)]["state_sha256"]:
            raise ValueError("FP8 receipt does not use the task's frozen state")
        if state.scale_bits(fp8["scales"]) != task_state["scale_bits"]:
            raise ValueError("FP8 receipt scales differ from frozen state")
        for left, right in zip(native["episodes"], fp8["episodes"]):
            noise_calls += validate_pair(left, right)
            pairs.append({"task_id": task, "seed": left["seed"], "native": left["success"], "fp8": right["success"],
                          "native_steps": left["steps"], "fp8_steps": right["steps"],
                          "native_action_digest": left["action_digest"], "fp8_action_digest": right["action_digest"],
                          "initial_observation_sha256": left["scene"]["initial_observation_sha256"]})
    if matched.paired_statistics(pairs) != raw["summary"]:
        raise ValueError("qualification summary does not match the receipts")
    third_task5 = compare_rollouts(rollout_receipts[0]["episodes"],
                                  read(qualification_dir / "fp8-5.json")["episodes"])
    qualified = {"status": raw["status"], "summary": raw["summary"], "tasks": raw["tasks"], "pairs": pairs,
        "pairing": {"matched_initial_states_and_observations": 500, "matched_noise_calls": noise_calls},
        "execution_plan_sha256": sha256_file(qualification_dir / "plan.json"),
        "receipt_sha256": raw["receipt_sha256"], "task_states": task_states,
        "protocol": {k: v for k, v in plan.items() if k not in ("config", "environment", "source_sha256", "source_extension_sha256")}}
    repeated_pairs = [{"seed": a["seed"], "success": a["success"], "steps": a["steps"],
                       "repeat_success": b["success"], "repeat_steps": b["steps"],
                       "action_digest": a["action_digest"], "repeat_action_digest": b["action_digest"],
                       "noise_sha256": a["noise_sha256"], "repeat_noise_sha256": b["noise_sha256"]}
                      for a, b in zip(*[r["episodes"] for r in rollout_receipts])]
    statuses = [replay["status"], rollout["status"], qualified["status"], third_task5["status"]]
    sources = {**plan["source_sha256"], **plan["source_extension_sha256"],
               "benchmarks/vla/pi05_sm120_diagnostic.py": sha256_file(matched.ROOT / "benchmarks/vla/pi05_sm120_diagnostic.py")}
    return {"schema_version": 1, "status": "PASS" if all(s == "PASS" for s in statuses) else "FAIL",
        "scope": "opt-in full-FP8 frozen frontend; pinned checkpoint, software, GPU UUID and registered token lengths",
        "source_sha256": sources, "exporter_source_sha256": sha256_file(Path(__file__)),
        "environment": plan["environment"], "execution_identity": catalog["identity"],
        "state_format": {"schema": catalog["schema"], "catalog_sha256": sha256_file(replay_dir / "state.json"),
                         "scale_count": len(catalog["scale_bits"]), "algorithm_count": len(catalog["gemm"]["entries"]),
                         "token_lengths": catalog["token_lengths"]},
        "numeric_replay": {k: replay[k] for k in ("status", "scope", "summary", "native", "receipts", "state_sha256")},
        "repeated_task5": {**rollout, "pairs": repeated_pairs, "full_campaign_repeat": third_task5,
                           "receipt_sha256": {str(r): sha256_file(replay_dir / f"rollout-both-{r}.json") for r in range(2)}},
        "qualification": qualified,
        "limitations": ["Bit-exact replay was tested on this GPU and pinned software; no cross-device/version guarantee.",
                        "Global PyTorch numeric settings and policy RNG remain caller-owned; the drivers pin eval flags and episode seeds.",
                        "The fresh ablation records each shape once, unlike the legacy repeatedly retuned path.",
                        "Memory metrics use the PyTorch allocator, not total process VRAM.",
                        "Task success claims are conditional on the fixed ten-task Spatial suite.",
                        "The default load_model path is unchanged; use FrozenPi05Frontend explicitly."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = export(args.replay.resolve(), args.qualification.resolve())
    write_json_atomic(args.output, result)
    print(json.dumps({"status": result["status"], "replay": result["numeric_replay"]["summary"]["both"],
                      "rollout": result["repeated_task5"]["successes"],
                      "qualification": result["qualification"]["summary"]}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
