"""Fresh 500-pair LIBERO qualification with one immutable FP8 state per task.

The recorded algorithm catalog is reused across prompts, but each task gets its
own demonstration-derived scales and prompt-bound state. No task outcome is used
to select algorithms, scales, horizons or seeds. Native runs anew on this binary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from benchmarks.vla import pi05_sm120_libero as matched
from benchmarks.vla.pi05_frozen_replay import verify_baseline
from benchmarks.vla.util import sha256_file, write_json_atomic


def read(path):
    return json.loads(Path(path).read_text())


def extensions():
    names = ("benchmarks/vla/pi05_frozen_libero.py", "benchmarks/vla/pi05_frozen_replay.py",
             "serving/csrc/gemm/gemm_runner.h", "serving/csrc/gemm/gemm_runner.cu", "serving/csrc/bindings.cpp")
    return {n: sha256_file(matched.ROOT / n) for n in names}


def check(output):
    preparation = read(output / "preparation.json")
    if preparation["sources"] != matched.source_hashes() or preparation["extensions"] != extensions():
        raise ValueError("frozen qualification source changed")
    if preparation["environment"] != matched.environment_identity():
        raise ValueError("frozen qualification environment changed")
    if sha256_file(Path(preparation["catalog"])) != preparation["catalog_sha256"]:
        raise ValueError("algorithm catalog changed")
    previous = verify_baseline(Path(preparation["baseline"]))
    if sha256_file(output / "calibration.json") != previous["calibration_sha256"]:
        raise ValueError("calibration manifest changed")
    return preparation, previous


def build_state(output, task):
    from flash_rt.core.frozen_state import load
    from flash_rt.frontends.torch.pi05_frozen import FrozenPi05Frontend
    preparation, previous = check(output)
    prompt = read(output / f"scenes-{task}.json")["40100"]["prompt"]
    calibration, _ = matched.calibration_frames(output, prompt)
    matched._apply_eval_numeric_environment()
    matched.seed_everything(5090120)
    runtime = FrozenPi05Frontend(previous["config"]["checkpoint"])
    catalog = load(preparation["catalog"], runtime.identity())
    # Algorithm descriptors depend on the model/environment and matrix shapes,
    # not prompt text. Do not copy the catalog's task-specific calibration.
    rows = [(*row[:4], bytes.fromhex(row[4])) for row in catalog["gemm"]["entries"]]
    runtime.frontend.gemm.import_algo_cache(catalog["gemm"]["identity"], rows)
    runtime.set_prompt(prompt)
    runtime.calibrate(calibration, percentile=previous["selected"]["percentile"])
    runtime.register_profiles(calibration[0], catalog["token_lengths"])
    runtime.save_state(output / f"state-{task}.json")
    runtime.close()


def worker(output, task, arm):
    import torch
    from flash_rt.frontends.torch.pi05_frozen import FrozenPi05Frontend
    check(output)
    plan = read(output / "plan.json")
    if arm == "native":
        matched.worker(plan["config"], output, task, arm)
        return
    state_path = output / f"state-{task}.json"
    if sha256_file(state_path) != plan["frozen_state_sha256"][str(task)]:
        raise ValueError("task execution state changed")
    matched._apply_eval_numeric_environment()
    matched.seed_everything(5090120)
    torch.cuda.reset_peak_memory_stats()
    runtime = FrozenPi05Frontend(plan["config"]["checkpoint"])
    runtime.load_state(state_path)
    scenes = read(output / f"scenes-{task}.json")
    result = {"task_id": task, "arm": arm, "settings": plan["selected"],
              "frozen_state_sha256": sha256_file(state_path), "plan_sha256": sha256_file(output / "plan.json"),
              "scales": {k: float(v) for k, v in runtime._scales.items()}, "episodes": []}
    vec = matched.make_vec(task)
    try:
        for seed in range(40100, 40150):
            episode = matched.run_episode(vec, runtime, seed, scenes[str(seed)])
            result["episodes"].append(episode)
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["resident_allocated_bytes"] = torch.cuda.memory_allocated()
            write_json_atomic(output / f"fp8-{task}.json", result)
            print(f"fp8 task={task} seed={seed} success={episode['success']} steps={episode['steps']}", flush=True)
    finally:
        vec.close()
        runtime.close()


def child(output, task, arm):
    with (output / f"{arm}-{task}.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "benchmarks.vla.pi05_frozen_libero", "--output", str(output),
                        "--worker", arm, "--task", str(task)], stdout=log, stderr=subprocess.STDOUT,
                        check=True, timeout=7200)


def campaign(baseline, catalog, output):
    previous = verify_baseline(baseline)
    replay = read(catalog.parent / "replay-results.json")
    repeated = read(catalog.parent / "rollout-results.json")
    catalog_sha = sha256_file(catalog)
    if any(r["status"] != "PASS" or r["state_sha256"] != catalog_sha for r in (replay, repeated)):
        raise ValueError("catalog must first pass numeric and 50-seed repeated-rollout gates")
    if output == baseline or output.is_relative_to(baseline):
        raise ValueError("do not overwrite the baseline")
    preparation = {"baseline": str(baseline), "catalog": str(catalog), "catalog_sha256": catalog_sha,
        "prerequisite_sha256": {n: sha256_file(catalog.parent / n) for n in ("replay-results.json", "rollout-results.json")},
        "sources": matched.source_hashes(), "extensions": extensions(), "environment": matched.environment_identity()}
    if (output / "preparation.json").exists():
        if read(output / "preparation.json") != preparation:
            raise ValueError("resume requires an identical preparation plan")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be empty")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("calibration.json", "calibration.npz", *[f"scenes-{t}.json" for t in range(10)]):
            shutil.copyfile(baseline / name, output / name)
        write_json_atomic(output / "preparation.json", preparation)
    # Freeze every task state before running any new closed-loop episodes.
    for task in range(10):
        if not (output / f"state-{task}.json").exists():
            child(output, task, "state")
        print(f"Prepared immutable state for task {task}", flush=True)
    config = {**previous["config"], "output": str(output)}
    plan = {**previous, "config": config, "source_sha256": matched.source_hashes(),
        "environment": preparation["environment"], "protocol": "pi05-sm120-frozen-matched-v1",
        "source_extension_sha256": extensions(), "frozen_catalog_sha256": preparation["catalog_sha256"],
        "frozen_state_sha256": {str(t): sha256_file(output / f"state-{t}.json") for t in range(10)},
        "baseline_execution_plan_sha256": sha256_file(baseline / "plan.json"),
        "execution_state": "per-task scales; shared immutable shape-specific GEMM algorithms",
        "selection": "reuse the prior demonstration-selected percentile; no new outcome-based selection"}
    for key in ("study_scores", "performance_target_met", "candidates", "selected_name"):
        plan.pop(key, None)
    plan["historical_calibration_selection"] = {"selected_name": previous["selected_name"],
        "baseline_execution_plan_sha256": plan["baseline_execution_plan_sha256"]}
    if (output / "plan.json").exists():
        if read(output / "plan.json") != plan:
            raise ValueError("qualification plan changed")
    else:
        write_json_atomic(output / "plan.json", plan)
    for task in range(10):
        for arm in (("native", "fp8") if task % 2 == 0 else ("fp8", "native")):
            path = output / f"{arm}-{task}.json"
            if path.exists() and len(read(path).get("episodes", [])) == 50:
                continue
            child(output, task, arm)
            print(f"Completed task {task}, {arm}, 50 episodes", flush=True)
    check(output)
    result = matched.report(output, list(range(10)), 50)
    result["protocol"] = plan["protocol"]
    write_json_atomic(output / "results.json", result)
    print(json.dumps(result["summary"], indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--catalog", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", choices=("state", "native", "fp8"))
    p.add_argument("--task", type=int, default=5)
    args = p.parse_args()
    output = args.output.resolve()
    if args.worker:
        if not 0 <= args.task < 10:
            p.error("invalid task")
        if args.worker == "state": build_state(output, args.task)
        else: worker(output, args.task, args.worker)
        return
    if args.baseline is None or args.catalog is None:
        p.error("--baseline and --catalog are required")
    result = campaign(args.baseline.resolve(), args.catalog.resolve(), output)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
