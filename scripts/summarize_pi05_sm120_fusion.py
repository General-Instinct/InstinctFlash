"""Export auditable SM120 fusion evidence without relabeling the old 500-pair gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from benchmarks.vla.pi05_sm120_fusion import ROOT, read, source_hashes
from benchmarks.vla.util import sha256_file, write_json_atomic


def checked_abba(folder, mode, frozen, *, require_improvement=True):
    result = read(folder / "results.json")
    rows = result["rows"]
    if result["kind"] != "ABBA" or result["status"] != "PASS" or len(rows) != 4:
        raise ValueError("incomplete ABBA")
    if [r["arm"] for r in rows] != ["baseline", mode, mode, "baseline"]:
        raise ValueError("wrong ABBA order")
    for row in rows:
        if (row["source_sha256"] != source_hashes() or row["profiled"] or row["cases"] != 8
                or not row["all_action_bytes_equal"] or not row["all_noise_equal"]):
            raise ValueError("invalid real-model replay/provenance")
        if row["base_identity"] != frozen["execution_identity"]:
            raise ValueError("base execution identity changed")
        reference = frozen["numeric_replay"]["receipts"]["both"][0]
        if row["actions_sha256"] != reference["actions_sha256"] or row["reference_arrays_sha256"] != reference["arrays_sha256"]:
            raise ValueError("replay differs from committed action bytes")
        if row["base_state_sha256"] != frozen["state_format"]["catalog_sha256"]:
            raise ValueError("different frozen calibration/GEMM state")
        if len(row["timings_ms"]) != row["iterations"] or row["iterations"] < 64:
            raise ValueError("incomplete timings")
        if statistics.median(row["timings_ms"]) != row["p50_ms"]:
            raise ValueError("median differs from raw timings")
        if row["fusion"] and not row["uninstall_exact"]:
            raise ValueError("uninstall did not restore original actions")
    base = statistics.mean(rows[i]["p50_ms"] for i in (0, 3))
    candidate = statistics.mean(rows[i]["p50_ms"] for i in (1, 2))
    if result["baseline_ms"] != base or result["candidate_ms"] != candidate or result["speedup"] != base / candidate:
        raise ValueError("ABBA arithmetic differs")
    spread = {"baseline": abs(rows[0]["p50_ms"] - rows[3]["p50_ms"]) / base * 100,
              "candidate": abs(rows[1]["p50_ms"] - rows[2]["p50_ms"]) / candidate * 100}
    improved = base > candidate * 1.005 and max(spread.values()) <= 1.0
    if result["arm_spread_percent"] != spread or result["performance_improved"] != improved:
        raise ValueError("stability decision differs from raw measurements")
    if require_improvement and not improved:
        raise ValueError("independent performance check did not pass")
    return result


def export(args):
    frozen = read(ROOT / "examples/pi05_vla/sm120_frozen_results.json")
    arms = {mode: checked_abba(folder, mode, frozen)
            for mode, folder in (("all8", args.fusion), ("hoist", args.hoist), ("all8-hoist", args.combined))}
    binaries = {row["fusion"]["extension_sha256"] for result in arms.values()
                for row in result["rows"] if row["fusion"]}
    if len(binaries) != 1:
        raise ValueError("candidate binaries differ")
    binary = next(iter(binaries))
    additional = ([checked_abba(args.additional_fusion, "all8", frozen, require_improvement=False)]
                  if args.additional_fusion else [])
    if any(row["fusion"] and row["fusion"]["extension_sha256"] != binary for run in additional for row in run["rows"]):
        raise ValueError("additional timing used a different binary")
    micro = read(args.micro)
    if (micro["source_sha256"] != sha256_file(ROOT / "benchmarks/vla/pi05_sm120_fusion_micro.py")
            or micro["binary_sha256"]["fusion"] != binary):
        raise ValueError("microbenchmark source/binary differs")
    rollout = read(args.rollout / "results.json")
    if rollout["status"] != "PASS" or rollout["source_sha256"] != source_hashes() or rollout["native_arm_rerun"]:
        raise ValueError("invalid historical trajectory replay")
    pairs = []
    published = {(p["task_id"], p["seed"]): p for p in frozen["qualification"]["pairs"]}
    for task in rollout["rows"]:
        signature = task["fusion"]
        if signature["extension_sha256"] != binary or signature["mode"] != "all8" or not signature["hoist_scales"]:
            raise ValueError("rollout used a different fusion")
        if signature["base_state_sha256"] != frozen["qualification"]["task_states"][str(task["task"])]["state_sha256"]:
            raise ValueError("rollout task calibration state differs")
        for row in task["episodes"]:
            reference = published[task["task"], row["seed"]]
            if (row["mismatch_fields"] or row["action_digest"] != reference["fp8_action_digest"]
                    or row["success"] != reference["fp8"] or row["steps"] != reference["fp8_steps"]
                    or row["scene"]["initial_observation_sha256"] != reference["initial_observation_sha256"]):
                raise ValueError("trajectory differs from historical frozen FP8")
            pairs.append({"task": task["task"], **{k: row[k] for k in
                ("seed", "success", "steps", "action_digest", "reference_action_digest", "noise_sha256", "mismatch_fields")}})
    expected = {(t, s) for t in range(10) for s in range(40100, 40100 + rollout["episodes_per_task"])}
    if len(pairs) != len(expected) or {(p["task"], p["seed"]) for p in pairs} != expected:
        raise ValueError("incomplete task/seed coverage")
    checks = {}
    for name, path in (("memcheck", args.memcheck), ("synccheck", args.synccheck), ("racecheck", args.racecheck)):
        log = path.read_text()
        marker = "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" if name == "racecheck" else "ERROR SUMMARY: 0 errors"
        if marker not in log or "Target application returned an error" in log:
            raise ValueError(f"{name} has no clean completion")
        checks[name] = {"log_sha256": sha256_file(path), "error_summary": "0 errors"}
    suites = ET.parse(args.junit).getroot().findall("testsuite")
    if len(suites) != 1 or any(int(suites[0].get(key, "-1")) != 0 for key in ("failures", "errors", "skipped")):
        raise ValueError("GPU operator tests did not all pass")
    if int(suites[0].get("tests", "0")) != 26:
        raise ValueError("incomplete GPU operator matrix")
    return {"schema_version": 1, "status": "PASS", "source_sha256": source_hashes(),
        "exporter_source_sha256": sha256_file(Path(__file__)), "extension_sha256": binary,
        "scope": "explicit RTX 5090 SM120 pi0.5 two-view QKV bias/split and FFN bias/GELU/FP8 fusion; original normalization unchanged",
        "base_evidence_sha256": sha256_file(ROOT / "examples/pi05_vla/sm120_frozen_results.json"),
        "gpu_operator_tests": {"passed": 26, "junit_sha256": sha256_file(args.junit),
            "source_sha256": sha256_file(ROOT / "tests/test_pi05_sm120_fusion_cuda.py")},
        "ablation": arms, "microbenchmarks": micro, "sanitizer": checks,
        "additional_timing_runs": additional,
        "trajectory_replay": {"status": "PASS", "kind": rollout["kind"], "native_arm_rerun": False,
            "episodes_per_task": rollout["episodes_per_task"], "pairs": pairs,
            "raw_receipt_sha256": {str(t): sha256_file(args.rollout / f"task-{t}.json") for t in range(10)}},
        "limitations": ["Default load_model and original frozen evidence are unchanged.",
            "LayerNorm fusion was rejected after shared-memory race and instrumented numeric checks; it is not shipped or counted in these results.",
            "Scale-upload hoisting is a separate host-side optimization, not a CUDA-kernel speedup.",
            "Microbenchmarks include an identical input reset in both arms; use ABBA for end-to-end latency.",
            "The new fusion has only the explicitly listed historical trajectory replays, not a fresh Native/FP8 500-pair gate.",
            "No cross-GPU or software-version bit-exact guarantee; source changes require requalification."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fusion", "hoist", "combined", "micro", "rollout", "memcheck", "synccheck", "racecheck", "junit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--additional-fusion", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = export(args)
    write_json_atomic(args.output, result)
    print(json.dumps({"status": result["status"], "speedups": {k: v["speedup"] for k, v in result["ablation"].items()},
                      "exact_episodes": len(result["trajectory_replay"]["pairs"])}, indent=2))


if __name__ == "__main__":
    main()
