"""Real-model SM120 fusion screen, independent ABBA timing and exact replay checks."""
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
from benchmarks.vla.util import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    names = ("benchmarks/vla/pi05_sm120_fusion.py", "instinctflash/runtime/pi05_sm120_fusion.py",
             "benchmarks/vla/pi05_sm120_fusion_micro.py", "benchmarks/vla/pi05_sm120_fusion_rollout.py",
             "serving/csrc/kernels/pi05_sm120_fusion.cu", "serving/csrc/pi05_sm120_fusion_bindings.cpp",
             "serving/csrc/kernels/common.cuh", "serving/CMakeLists.txt")
    return {name: sha256_file(ROOT / name) for name in names}


def read(path): return json.loads(Path(path).read_text())


def worker(args):
    import torch
    from flash_rt.frontends.torch.pi05_frozen import FrozenPi05Frontend
    from instinctflash.runtime.pi05_sm120_fusion import install_pi05_sm120_fusion
    sources_at_start = source_hashes()
    payload = read(args.state)["payload"]
    _, observations = matched.calibration_frames(args.baseline, payload["prompt"])
    reference = args.state.parent / "both-0.npz"
    receipt = read(args.state.parent / "both-0.json")
    if receipt["arrays_sha256"] != sha256_file(reference) or receipt["state_sha256"] != sha256_file(args.state):
        raise ValueError("baseline reference/state changed")
    matched._apply_eval_numeric_environment()
    matched.seed_everything(5090120)
    runtime = FrozenPi05Frontend(args.checkpoint)
    runtime.load_state(args.state)
    handle = None
    if args.worker != "baseline":
        mode = "none" if args.worker == "hoist" else args.worker.removesuffix("-hoist")
        handle = install_pi05_sm120_fusion(runtime, args.state, mode=mode,
                                          hoist_scales=args.worker.endswith("hoist"))
    actions, noises = [], []
    with np.load(reference, allow_pickle=False) as expected:
        for i, obs in enumerate(observations):
            matched.seed_everything(123456 + i)
            actual = runtime.infer(obs)["actions"]
            noise = runtime._noise_buf.float().cpu().numpy()
            if not np.array_equal(actual.view(np.uint8), expected["actions"][i].view(np.uint8)):
                raise ValueError(f"action bytes changed for real holdout {i}, max delta {np.abs(actual - expected['actions'][i]).max()}")
            if not np.array_equal(noise.view(np.uint8), expected["noises"][i].view(np.uint8)):
                raise ValueError("noise bytes changed")
            actions.append(actual.copy()); noises.append(noise.copy())
    for i in range(args.warmup):
        matched.seed_everything(123456 + i % len(observations))
        runtime.infer(observations[i % len(observations)])
    torch.cuda.synchronize()
    if args.profile: torch.cuda.profiler.start()
    timings = []
    for i in range(args.iterations):
        matched.seed_everything(123456 + i % len(observations))
        start = time.perf_counter()
        runtime.infer(observations[i % len(observations)])
        timings.append((time.perf_counter() - start) * 1000)
    torch.cuda.synchronize()
    if args.profile: torch.cuda.profiler.stop()
    result = {"arm": args.worker, "tag": args.tag, "profiled": args.profile,
        "cases": len(actions), "all_action_bytes_equal": True, "all_noise_equal": True,
        "actions_sha256": hashlib.sha256(np.asarray(actions).tobytes()).hexdigest(),
        "p50_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)), "timings_ms": timings,
        "warmup": args.warmup, "iterations": args.iterations, "source_sha256": source_hashes(),
        "reference_arrays_sha256": sha256_file(reference), "reference_receipt_sha256": sha256_file(args.state.parent / "both-0.json"),
        "observation_sha256": matched.sha256_json([{k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                                                    for k, v in obs.items()} for obs in observations]),
        "base_state_sha256": sha256_file(args.state), "base_identity": payload["identity"],
        "fusion": handle.signature if handle else None,
        "profile_installs": handle.profile_installs if handle else None,
        "skipped_scale_installs": handle.skipped_scale_installs if handle else 0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "resident_allocated_bytes": torch.cuda.memory_allocated()}
    if handle:
        handle.close()
        matched.seed_everything(123456)
        restored = runtime.infer(observations[0])["actions"]
        if not np.array_equal(restored.view(np.uint8), np.asarray(actions)[0].view(np.uint8)):
            raise ValueError("uninstall failed to restore baseline action bytes")
        result["uninstall_exact"] = True
    runtime.close()
    if source_hashes() != sources_at_start:
        raise ValueError("sources changed during inference")
    write_json_atomic(args.output / f"{args.tag}.json", result)
    print(json.dumps({k: result[k] for k in ("arm", "p50_ms", "p95_ms", "all_action_bytes_equal", "skipped_scale_installs")}), flush=True)


def run_child(args, arm, tag):
    command = [sys.executable, "-m", "benchmarks.vla.pi05_sm120_fusion", "--worker", arm,
        "--tag", tag, "--output", str(args.output), "--checkpoint", str(args.checkpoint),
        "--state", str(args.state), "--baseline", str(args.baseline),
        "--iterations", str(args.iterations), "--warmup", str(args.warmup)]
    with (args.output / f"{tag}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    return read(args.output / f"{tag}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("baseline", "qkv", "ffn4", "ffn8", "all4", "all8", "hoist", "all4-hoist", "all8-hoist"))
    parser.add_argument("--tag", default="worker")
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--abba", choices=("qkv", "ffn4", "ffn8", "all4", "all8", "hoist", "all4-hoist", "all8-hoist"))
    args = parser.parse_args()
    if args.iterations < 8 or args.warmup < 8:
        parser.error("at least eight warmup and timed calls are required")
    if Path(args.tag).name != args.tag:
        parser.error("tag must be a plain filename")
    if args.worker and (args.output / f"{args.tag}.json").exists():
        parser.error("worker result already exists")
    if not args.worker and args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    if args.abba:
        arms = [("baseline", "a1"), (args.abba, "b1"), (args.abba, "b2"), ("baseline", "a2")]
    else:
        arms = [(a, a) for a in ("baseline", "qkv", "ffn4", "ffn8", "all4", "all8", "hoist", "all8-hoist")]
    rows = [run_child(args, arm, tag) for arm, tag in arms]
    if any(row["source_sha256"] != source_hashes() for row in rows):
        raise ValueError("source changed during timing")
    for key in ("base_state_sha256", "base_identity", "reference_arrays_sha256", "reference_receipt_sha256",
                "observation_sha256", "actions_sha256"):
        if any(row[key] != rows[0][key] for row in rows):
            raise ValueError(f"arms differ in {key}")
    signatures = [r["fusion"] for r in rows if r["fusion"] is not None]
    if len({r["extension_sha256"] for r in signatures}) > 1:
        raise ValueError("extension changed between arms")
    result = {"status": "PASS", "kind": "ABBA" if args.abba else "SCREEN", "rows": rows,
              "source_sha256": source_hashes()}
    if args.abba:
        base = np.mean([rows[0]["p50_ms"], rows[3]["p50_ms"]])
        candidate = np.mean([rows[1]["p50_ms"], rows[2]["p50_ms"]])
        result["baseline_ms"] = float(base); result["candidate_ms"] = float(candidate)
        result["speedup"] = float(base / candidate); result["saved_ms"] = float(base - candidate)
        result["arm_spread_percent"] = {
            "baseline": float(abs(rows[0]["p50_ms"] - rows[3]["p50_ms"]) / base * 100),
            "candidate": float(abs(rows[1]["p50_ms"] - rows[2]["p50_ms"]) / candidate * 100)}
        # PASS above means exact replay/provenance, not a promise of speedup.
        result["performance_improved"] = bool(base > candidate * 1.005
            and max(result["arm_spread_percent"].values()) <= 1.0)
    write_json_atomic(args.output / "results.json", result)
    print(json.dumps({"summary": [(r["arm"], r["p50_ms"]) for r in rows],
                      "speedup": result.get("speedup")}, indent=2))


if __name__ == "__main__": main()
