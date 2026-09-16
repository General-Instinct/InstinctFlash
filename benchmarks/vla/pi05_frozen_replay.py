"""Real SM120 ablations and cross-process replay of frozen pi0.5 state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from benchmarks.vla import pi05_sm120_libero as matched
from benchmarks.vla.pi05_sm120_diagnostic import validate_pair
from benchmarks.vla.util import sha256_file, write_json_atomic

MODES = ("fresh", "scales", "algorithms", "both")


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    names = ["benchmarks/vla/pi05_frozen_replay.py", "serving/csrc/gemm/gemm_runner.h",
             "serving/csrc/gemm/gemm_runner.cu", "serving/csrc/bindings.cpp",
             "benchmarks/vla/pi05_sm120_diagnostic.py"]
    return {**matched.source_hashes(), **{n: sha256_file(matched.ROOT / n) for n in names}}


def verify_baseline(path):
    plan = read(path / "plan.json")
    result = read(path / "results.json")
    if result["status"] != "PASS" or result["summary"]["pairs"] != 500:
        raise ValueError("requires the completed qualified baseline")
    # Additive opt-in source files are allowed; the original rollout is unchanged.
    for name, expected in plan["source_sha256"].items():
        if sha256_file(matched.ROOT / name) != expected:
            raise ValueError(f"original qualified Python source changed: {name}")
    if result["plan"] != plan or sha256_file(path / "calibration.json") != plan["calibration_sha256"]:
        raise ValueError("baseline plan/calibration changed")
    if sha256_file(path / "calibration.npz") != read(path / "calibration.json")["archive_sha256"]:
        raise ValueError("calibration archive changed")
    if sha256_file(Path(plan["config"]["checkpoint"]) / "model.safetensors") != matched.MODEL_SHA256:
        raise ValueError("checkpoint changed")
    return plan


def prepare(output, baseline, task, profile_min=128, profile_max=200):
    previous = verify_baseline(baseline)
    environment = matched.environment_identity()
    old = {k: v for k, v in previous["environment"].items() if k != "binary_sha256"}
    new = {k: v for k, v in environment.items() if k != "binary_sha256"}
    if old != new:
        raise ValueError("qualified dependency/simulator environment changed")
    plan = {"protocol": "pi05-frozen-replay-v1", "baseline": str(baseline),
            "baseline_plan_sha256": sha256_file(baseline / "plan.json"),
            "checkpoint": previous["config"]["checkpoint"], "task": task,
            "source_sha256": sources(), "environment": environment,
            "profile_lengths": list(range(profile_min, profile_max + 1)),
            "settings": previous["selected"], "calibration_sha256": previous["calibration_sha256"]}
    if output == baseline or output.is_relative_to(baseline):
        raise ValueError("do not write into the qualified baseline")
    if (output / "plan.json").exists():
        if read(output / "plan.json") != plan:
            raise ValueError("resume requires the identical plan")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be empty")
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / "plan.json", plan)
    return plan


def checked_plan(output):
    plan = read(output / "plan.json")
    if plan["source_sha256"] != sources() or plan["environment"] != matched.environment_identity():
        raise ValueError("frozen experiment sources/environment changed")
    verify_baseline(Path(plan["baseline"]))
    if sha256_file(Path(plan["baseline"]) / "plan.json") != plan["baseline_plan_sha256"]:
        raise ValueError("baseline plan changed")
    return plan


def worker(output, mode, repetition, episodes=0):
    import torch
    from flash_rt.core.frozen_state import digest, scale_bits
    from flash_rt.frontends.torch.pi05_frozen import FrozenPi05Frontend
    plan = checked_plan(output)
    baseline = Path(plan["baseline"])
    scenes = read(baseline / f"scenes-{plan['task']}.json")
    prompt = scenes["40100"]["prompt"]
    calibration, holdout = matched.calibration_frames(baseline, prompt)
    matched._apply_eval_numeric_environment()
    matched.seed_everything(5090120)
    torch.cuda.reset_peak_memory_stats()
    if mode == "native":
        runtime = matched.make_runtime({"checkpoint": plan["checkpoint"]}, "native")
    else:
        runtime = FrozenPi05Frontend(plan["checkpoint"])
    runtime.set_prompt(prompt)
    artifact = output / "state.json"
    started = time.perf_counter()
    if mode in ("record", "fresh", "native"):
        runtime.calibrate(calibration, percentile=plan["settings"]["percentile"])
        if mode == "record":
            runtime.register_profiles(calibration[0], plan["profile_lengths"])
            runtime.save_state(artifact)
    else:
        runtime.load_state(artifact, parts=mode)
        if mode == "algorithms":
            runtime.calibrate(calibration, percentile=plan["settings"]["percentile"])
    setup_s = time.perf_counter() - started
    if mode == "record":
        runtime.close()
        print(f"Recorded {artifact}", flush=True)
        return
    result = {"mode": mode, "repetition": repetition, "plan_sha256": sha256_file(output / "plan.json"),
        "state_sha256": sha256_file(artifact) if artifact.exists() else None,
        "setup_after_weight_load_s": setup_s,
        "scale_sha256": digest(scale_bits(runtime._scales)),
        "scales_equal_to_recording": scale_bits(runtime._scales) == read(artifact)["payload"]["scale_bits"] if mode != "native" else None}
    if episodes:
        vec = matched.make_vec(plan["task"])
        result["episodes"] = []
        name = f"rollout-{mode}-{repetition}.json"
        try:
            for seed in range(40100, 40100 + episodes):
                row = matched.run_episode(vec, runtime, seed, scenes[str(seed)])
                result["episodes"].append(row)
                write_json_atomic(output / name, result)
                print(f"{name} seed={seed} success={row['success']} steps={row['steps']}", flush=True)
        finally:
            vec.close()
    else:
        values, noises = [], []
        for i, obs in enumerate(holdout):
            matched.seed_everything(123456 + i)
            values.append(runtime.infer(obs)["actions"])
            noises.append(runtime._noise_buf.float().cpu().numpy())
        result["actions_sha256"] = hashlib.sha256(np.asarray(values).tobytes()).hexdigest()
        result["noise_sha256"] = hashlib.sha256(np.asarray(noises).tobytes()).hexdigest()
        result["cases"] = len(values)
        times = []
        for i in range(24):
            t0 = time.perf_counter()
            runtime.infer(holdout[i % len(holdout)])
            if i >= 8:
                times.append((time.perf_counter() - t0) * 1000)
        result["infer_p50_ms"] = float(np.median(times))
        result["resident_allocated_bytes"] = torch.cuda.memory_allocated()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        data_path = output / f"{mode}-{repetition}.npz"
        np.savez(data_path, actions=np.asarray(values), noises=np.asarray(noises))
        result["arrays_sha256"] = sha256_file(data_path)
        if mode != "native":
            entries = [[*r[:4], r[4].hex()] for r in runtime.frontend.gemm.export_algo_cache()]
            result["algorithms_sha256"] = digest(entries)
            result["algorithms_equal_to_recording"] = entries == read(artifact)["payload"]["gemm"]["entries"]
        write_json_atomic(output / f"{mode}-{repetition}.json", result)
        print(json.dumps(result, indent=2), flush=True)
    runtime.close()


def child(output, mode, repetition=0, episodes=0):
    name = f"{'rollout-' if episodes else ''}{mode}-{repetition}"
    command = [sys.executable, "-m", "benchmarks.vla.pi05_frozen_replay", "--output", str(output),
               "--worker", mode, "--repetition", str(repetition), "--episodes", str(episodes)]
    with (output / f"{name}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)


def summarize(output, repeats):
    plan = checked_plan(output)
    baseline = read(output / "native-0.json")
    if baseline["plan_sha256"] != sha256_file(output / "plan.json") or baseline["arrays_sha256"] != sha256_file(output / "native-0.npz"):
        raise ValueError("native numeric reference changed")
    with np.load(output / "native-0.npz") as arrays:
        native = arrays["actions"].astype(float)
        noise = arrays["noises"].copy()
    rows, summary = {}, {}
    for mode in MODES:
        receipts = [read(output / f"{mode}-{r}.json") for r in range(repeats)]
        cosines = []
        for r, receipt in enumerate(receipts):
            if receipt["plan_sha256"] != sha256_file(output / "plan.json") or receipt["state_sha256"] != sha256_file(output / "state.json"):
                raise ValueError("replay receipt plan or state differs")
            if receipt["arrays_sha256"] != sha256_file(output / f"{mode}-{r}.npz"):
                raise ValueError("replay arrays changed")
            with np.load(output / f"{mode}-{r}.npz") as arrays:
                if (hashlib.sha256(arrays["actions"].tobytes()).hexdigest() != receipt["actions_sha256"]
                        or hashlib.sha256(arrays["noises"].tobytes()).hexdigest() != receipt["noise_sha256"]):
                    raise ValueError("action/noise receipt digest differs from arrays")
                if not np.array_equal(noise, arrays["noises"]):
                    raise ValueError("unmatched ablation noise")
                a, b = native.reshape(len(native), -1), arrays["actions"].astype(float).reshape(len(native), -1)
                cosines.extend(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
        rows[mode] = receipts
        summary[mode] = {"repeats": repeats,
            "bit_exact_actions": len({r["actions_sha256"] for r in receipts}) == 1,
            "identical_scales": len({r["scale_sha256"] for r in receipts}) == 1,
            "scales_match_recording": all(r["scales_equal_to_recording"] for r in receipts),
            "algorithms_match_recording": all(r["algorithms_equal_to_recording"] for r in receipts),
            "minimum_native_cosine": float(min(cosines)),
            "speedup": baseline["infer_p50_ms"] / float(np.median([r["infer_p50_ms"] for r in receipts])),
            "peak_allocated_ratio": max(r["peak_allocated_bytes"] for r in receipts) / baseline["peak_allocated_bytes"]}
    both = summary["both"]
    passed = repeats >= 3 and both["bit_exact_actions"] and both["scales_match_recording"] and both["algorithms_match_recording"]
    passed = passed and both["minimum_native_cosine"] >= .98 and both["speedup"] >= 1.05 and both["peak_allocated_ratio"] <= .75
    result = {"status": "PASS" if passed else "FAIL", "scope": "eight real held-out inputs, independent process restarts",
              "summary": summary, "native": baseline, "receipts": rows, "plan": plan,
              "state_sha256": sha256_file(output / "state.json")}
    write_json_atomic(output / "replay-results.json", result)
    return result


def compare_rollouts(left, right):
    seeds = list(range(40100, 40150))
    if [r["seed"] for r in left] != seeds or [r["seed"] for r in right] != seeds:
        raise ValueError("rollout comparison requires all 50 unique planned seeds")
    mismatches = []
    for a, b in zip(left, right):
        validate_pair(a, b)
        if any(a[k] != b[k] for k in ("success", "steps", "noise_sha256", "action_digest")):
            mismatches.append(a["seed"])
    return {"status": "FAIL" if mismatches else "PASS", "episodes_per_repeat": 50,
            "successes": [sum(r["success"] for r in rows) for rows in (left, right)],
            "bit_exact_trajectories": not mismatches, "mismatch_seeds": mismatches}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile-min", type=int, default=128)
    p.add_argument("--profile-max", type=int, default=200)
    p.add_argument("--worker", choices=("record", "native", *MODES))
    p.add_argument("--repetition", type=int, default=0)
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--rollout", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    if args.worker:
        worker(output, args.worker, args.repetition, args.episodes)
        return
    if args.baseline is None or not 0 <= args.task < 10 or args.repeats < 1 or not 1 <= args.profile_min <= args.profile_max <= 200:
        p.error("invalid baseline, task, repeat count or profile range")
    prepare(output, args.baseline.resolve(), args.task, args.profile_min, args.profile_max)
    if not (output / "state.json").exists():
        child(output, "record")
    if args.rollout:
        for r in range(2):
            child(output, "both", r, episodes=50)
        checked_plan(output)
        receipts = [read(output / f"rollout-both-{r}.json") for r in range(2)]
        for receipt in receipts:
            if receipt["plan_sha256"] != sha256_file(output / "plan.json") or receipt["state_sha256"] != sha256_file(output / "state.json"):
                raise ValueError("rollout plan or state changed")
        result = {**compare_rollouts(*[r["episodes"] for r in receipts]),
                  "state_sha256": sha256_file(output / "state.json")}
        write_json_atomic(output / "rollout-results.json", result)
    else:
        child(output, "native")
        for mode in MODES:
            for r in range(args.repeats):
                child(output, mode, r)
                print(f"Completed {mode}, repetition {r}", flush=True)
        result = summarize(output, args.repeats)
    print(json.dumps(result.get("summary", result), indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
