"""Exact trajectory replay against the already qualified frozen FP8 LIBERO arm.

This is NOT a new Native-versus-FP8 non-inferiority campaign. Run all 50 seeds
per task to cover the full historical FP8 arm; the default three are a screen.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

from benchmarks.vla import pi05_frozen_libero as frozen
from benchmarks.vla import pi05_sm120_libero as matched
from benchmarks.vla.pi05_sm120_fusion import read, source_hashes
from benchmarks.vla.util import sha256_file, write_json_atomic


def checked_baseline(folder):
    frozen.check(folder)
    evidence = read(matched.ROOT / "examples/pi05_vla/sm120_frozen_results.json")
    plan = read(folder / "plan.json")
    if sha256_file(folder / "plan.json") != evidence["qualification"]["execution_plan_sha256"]:
        raise ValueError("baseline plan differs from committed frozen qualification")
    for name, expected in evidence["qualification"]["receipt_sha256"].items():
        if sha256_file(folder / name) != expected:
            raise ValueError(f"historical receipt changed: {name}")
    return evidence, plan


def worker(args):
    from flash_rt.frontends.torch.pi05_frozen import FrozenPi05Frontend
    from instinctflash.runtime.pi05_sm120_fusion import install_pi05_sm120_fusion
    sources = source_hashes()
    script_sha = sha256_file(Path(__file__))
    evidence, plan = checked_baseline(args.baseline)
    task = args.task
    state = args.baseline / f"state-{task}.json"
    if sha256_file(state) != plan["frozen_state_sha256"][str(task)]:
        raise ValueError("task state changed")
    references = {r["seed"]: r for r in read(args.baseline / f"fp8-{task}.json")["episodes"]}
    published = {p["seed"]: p for p in evidence["qualification"]["pairs"] if p["task_id"] == task}
    scenes = read(args.baseline / f"scenes-{task}.json")
    matched._apply_eval_numeric_environment()
    matched.seed_everything(5090120)
    runtime = FrozenPi05Frontend(plan["config"]["checkpoint"])
    runtime.load_state(state)
    handle = install_pi05_sm120_fusion(runtime, state, mode=args.mode, hoist_scales=args.hoist_scales)
    result = {"task": task, "fusion": handle.signature, "source_sha256": sources,
              "driver_sha256": script_sha, "episodes": [], "status": "RUNNING"}
    vec = matched.make_vec(task)
    try:
        for seed in range(40100, 40100 + args.episodes):
            reference = references[seed]
            if reference["action_digest"] != published[seed]["fp8_action_digest"]:
                raise ValueError("historical action trace differs from committed evidence")
            if reference["success"] != published[seed]["fp8"]:
                raise ValueError("historical outcome differs from committed evidence")
            row = matched.run_episode(vec, runtime, seed, scenes[str(seed)])
            fields = ("seed", "success", "scene", "steps", "action_digest", "noise_sha256")
            row["mismatch_fields"] = [key for key in fields if row[key] != reference[key]]
            row["reference_action_digest"] = reference["action_digest"]
            result["episodes"].append(row)
            write_json_atomic(args.output / f"task-{task}.json", result)
            if row["mismatch_fields"]:
                raise ValueError(f"trajectory mismatch task={task} seed={seed}: {row['mismatch_fields']}")
            print(f"task={task} seed={seed} success={row['success']} steps={row['steps']} exact=True", flush=True)
    finally:
        vec.close()
        handle.close()
        runtime.close()
    if sources != source_hashes() or script_sha != sha256_file(Path(__file__)):
        raise ValueError("sources changed during rollout")
    result["status"] = "PASS"
    write_json_atomic(args.output / f"task-{task}.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("all4", "all8"), default="all8")
    parser.add_argument("--hoist-scales", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--task", type=int, choices=range(10))
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        parser.error("episodes must be in [1, 50]")
    if args.task is not None:
        worker(args)
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    for task in range(10):
        command = [sys.executable, "-m", "benchmarks.vla.pi05_sm120_fusion_rollout",
            "--baseline", str(args.baseline), "--output", str(args.output),
            "--mode", args.mode, "--episodes", str(args.episodes), "--task", str(task)]
        if args.hoist_scales:
            command.append("--hoist-scales")
        with (args.output / f"task-{task}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
        print(f"Completed task {task}: {args.episodes} exact historical trajectory replays", flush=True)
    rows = [read(args.output / f"task-{task}.json") for task in range(10)]
    if any(row["status"] != "PASS" or row["source_sha256"] != sources for row in rows) or source_hashes() != sources:
        raise ValueError("incomplete replay or source drift")
    result = {"status": "PASS", "kind": "SCREEN" if args.episodes < 50 else "FULL_FP8_REPLAY",
              "native_arm_rerun": False, "episodes_per_task": args.episodes,
              "source_sha256": sources, "rows": rows}
    write_json_atomic(args.output / "results.json", result)


if __name__ == "__main__":
    main()
