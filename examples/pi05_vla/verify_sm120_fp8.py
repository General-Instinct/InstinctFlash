#!/usr/bin/env python3
"""Qualify the Pi0.5 FlashRT FP8 path on RTX 5090 / SM120.

The public invocation runs native BF16 and FP8 in separate processes, using the same pinned
checkpoint, real LIBERO observations, prompt, and diffusion-noise seeds.  The child mode is an
implementation detail that prevents CUDA allocator and graph state from leaking between arms.

Example::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=serving:. python \
      examples/pi05_vla/verify_sm120_fp8.py \
      --checkpoint /path/to/pi05_libero \
      --dataset /path/to/libero_spatial/data/chunk-000/file-000.parquet \
      --output examples/pi05_vla/sm120_fp8_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from flash_rt import load_model


MODEL_REPO = "lerobot/pi05_libero_finetuned_v044"
MODEL_REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
MODEL_SHA256 = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
DATASET_REPO = "lerobot/libero_spatial_image"
DATASET_REVISION = "d86c0b94922572b3b657e1d1a3d01f0952ddeb46"
DATASET_SHA256 = "cc4681188f4c5eeec4253d51ddbaa5ad90ad01bcd7ed4c23fc767a1e0c14686b"
PROMPT = "pick up the black bowl next to the cookie box and place it on the plate"
CALIBRATION_INDICES = (0, 30, 60, 90)
EVALUATION_INDICES = (10, 50, 100)
SEEDS = (424242, 5090, 120120)
TIMING_ITERATIONS = 9
MIN_ACTION_COSINE = 0.98
MIN_SPEEDUP = 1.05
MAX_FP8_RESIDENT_RATIO = 0.75
MAX_FP8_PEAK_RATIO = 0.75
MIN_RELEASED_BF16_BYTES = 5_000_000_000
EXPECTED_RELEASED_BF16_KEYS = 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_observations(path: Path, indices: tuple[int, ...]) -> list[dict]:
    columns = [
        "observation.images.image",
        "observation.images.wrist_image",
        "observation.state",
        "task_index",
    ]
    rows = pq.read_table(path, columns=columns).take(list(indices)).to_pylist()
    observations = []
    for index, row in zip(indices, rows):
        if int(row["task_index"]) != 0:
            raise AssertionError(f"row {index} is task {row['task_index']}, expected task 0")

        def image(key: str) -> np.ndarray:
            raw = np.asarray(
                Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB"),
                dtype=np.uint8,
            )
            # The checkpoint processor owns image normalization and resizing.
            return raw.copy()

        observations.append({
            "image": image("observation.images.image"),
            "wrist_image": image("observation.images.wrist_image"),
            "state": np.asarray(row["observation.state"], dtype=np.float32),
        })
    return observations


def run_arm(checkpoint: Path, dataset: Path, mode: str, output: Path) -> None:
    from flash_rt import flash_rt_fa2, flash_rt_kernels

    calibration = load_observations(dataset, CALIBRATION_INDICES)
    evaluation = load_observations(dataset, EVALUATION_INDICES)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("this gate requires one visible SM120 CUDA device")

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = load_model(
        checkpoint,
        num_views=2,
        use_fp8=mode == "fp8",
        hardware="rtx_sm120",
        action_horizon=10,
    )
    runtime = model._pipe
    frontend = runtime.frontend
    load_ms = (time.perf_counter() - started) * 1000.0
    if frontend.fp8_layout != "nk":
        raise AssertionError(f"SM120 selected {frontend.fp8_layout!r}, expected 'nk'")

    runtime.set_prompt(PROMPT)
    initial_tokens = runtime.prepare(calibration[0])[0]["token_ids"]
    changed_state = {**calibration[0], "state": calibration[0]["state"].copy()}
    changed_state["state"][0] += 0.5
    if np.array_equal(initial_tokens, runtime.prepare(changed_state)[0]["token_ids"]):
        raise AssertionError("state changes do not reach the checkpoint token inputs")
    try:
        runtime.prepare({k: v for k, v in calibration[0].items() if k != "state"})
    except ValueError:
        pass
    else:
        raise AssertionError("missing checkpoint state was accepted")
    torch.manual_seed(5090120)
    started = time.perf_counter()
    model.calibrate(calibration, prompt=PROMPT, percentile=99.9)
    calibrate_ms = (time.perf_counter() - started) * 1000.0

    # The first replay after capture is deliberately not part of the timing or numeric pair.
    torch.manual_seed(1)
    def predict(obs):
        return model.predict(images=[obs["image"], obs["wrist_image"]], state=obs["state"])

    predict(evaluation[0])
    torch.manual_seed(2)
    predict(evaluation[1])
    frontend.latency_records.clear()

    outputs, noises, latencies = [], [], []
    for observation, seed in zip(evaluation, SEEDS):
        torch.manual_seed(seed)
        started = time.perf_counter()
        actions = predict(observation).astype(np.float32)
        latencies.append((time.perf_counter() - started) * 1000.0)
        if not np.isfinite(actions).all():
            raise AssertionError(f"{mode} produced non-finite actions for seed {seed}")
        outputs.append(actions)
        noises.append(runtime._noise_buf.float().cpu().numpy().copy())

    paired_latency_ms = latencies
    latencies = []
    for index in range(TIMING_ITERATIONS):
        torch.manual_seed(9000 + index)
        started = time.perf_counter()
        predict(evaluation[index % len(evaluation)])
        latencies.append((time.perf_counter() - started) * 1000.0)

    actions = np.stack(outputs)
    np.savez(output.with_suffix(".npz"), actions=actions, noises=np.stack(noises))
    payload = {
        "mode": mode,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "fp8_layout": frontend.fp8_layout,
        "fp8_weight_count": len(frontend._fp8_weights),
        "computed_action_chunk": runtime.config.chunk_size,
        "executed_action_horizon": runtime.action_horizon,
        "state_tokenized": True,
        "state_changes_tokens": True,
        "missing_state_refused": True,
        "normalization": "checkpoint MEAN_STD",
        "binary_sha256": {
            "flash_rt_kernels": sha256(Path(flash_rt_kernels.__file__)),
            "flash_rt_fa2": sha256(Path(flash_rt_fa2.__file__)),
        },
        "load_ms": load_ms,
        "calibrate_ms": calibrate_ms,
        "paired_case_latency_ms": paired_latency_ms,
        "latency_ms": latencies,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resident_allocated_bytes": torch.cuda.memory_allocated(),
        "resident_reserved_bytes": torch.cuda.memory_reserved(),
        "released_bf16_bytes": getattr(frontend, "_released_bf16_bytes", 0),
        "released_bf16_key_count": len(getattr(frontend, "_released_bf16_keys", ())),
        "action_shape": list(actions.shape),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
    }
    # Shape-only stress, excluded from the action/timing/memory comparison above.
    # Exercise more token lengths than the eight-entry cache and verify that the
    # evicted and replaced CUDA graph handles are actually destroyed.
    prepared, batch = runtime.prepare(evaluation[0])
    original_prepare = runtime.prepare
    graphs = []
    try:
        for offset in range(12):
            staged = {**prepared, "token_ids": prepared["token_ids"][:len(prepared["token_ids"]) - offset]}
            runtime.prepare = lambda obs, staged=staged: (staged, batch)
            if not np.isfinite(predict(evaluation[0])).all():
                raise AssertionError("non-finite output during graph profile stress")
            graphs.append(runtime.pipeline._graph)
            if len(runtime._profiles) > 8:
                raise AssertionError("graph profile cache exceeded its bound")
        evicted = sum(not graph.captured for graph in graphs)
        if evicted < 4:
            raise AssertionError("evicted CUDA graphs were not destroyed")
    finally:
        runtime.prepare = original_prepare
    model.close()
    if any(graph.captured for graph in graphs):
        raise AssertionError("frontend close left live CUDA graphs")
    payload["graph_lifecycle"] = {"shape_only_stress": True, "profiles": 12,
                                  "evicted_and_destroyed": evicted, "all_closed": True}
    output.write_text(json.dumps(payload, indent=2) + "\n")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64).ravel()
    b = right.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_child(mode: str, args, output: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint", str(args.checkpoint),
        "--dataset", str(args.dataset),
        "--output", str(output),
        "--arm", mode,
    ]
    completed = subprocess.run(
        command,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=args.timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{mode} arm failed ({completed.returncode})\n"
            f"stdout tail:\n{completed.stdout[-4000:]}\n"
            f"stderr tail:\n{completed.stderr[-4000:]}"
        )
    print(f"{mode} arm complete", flush=True)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    relatives = (
        "examples/pi05_vla/verify_sm120_fp8.py",
        "serving/flash_rt/api.py",
        "serving/flash_rt/frontends/torch/pi05_rtx.py",
        "serving/flash_rt/frontends/torch/pi05_checkpoint.py",
        "serving/flash_rt/models/pi05/pipeline_rtx.py",
        "serving/flash_rt/hardware/rtx/attn_backend.py",
        "serving/flash_rt/core/utils/actions.py",
        "serving/flash_rt/core/cuda_graph.py",
    )
    return {relative: sha256(root / relative) for relative in relatives}


def qualify(args) -> int:
    args.checkpoint = args.checkpoint.resolve()
    args.dataset = args.dataset.resolve()
    if not (args.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint / "model.safetensors")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    model_sha256 = sha256(args.checkpoint / "model.safetensors")
    dataset_sha256 = sha256(args.dataset)
    if model_sha256 != MODEL_SHA256:
        raise RuntimeError(
            f"checkpoint hash {model_sha256} does not match pinned {MODEL_SHA256}"
        )
    if dataset_sha256 != DATASET_SHA256:
        raise RuntimeError(
            f"dataset hash {dataset_sha256} does not match pinned {DATASET_SHA256}"
        )

    with tempfile.TemporaryDirectory(prefix="pi05-sm120-") as directory:
        temporary = Path(directory)
        paths = {mode: temporary / f"{mode}.json" for mode in ("native", "fp8")}
        for mode in ("native", "fp8"):
            run_child(mode, args, paths[mode])

        arms = {mode: json.loads(paths[mode].read_text()) for mode in paths}
        tensors = {mode: np.load(paths[mode].with_suffix(".npz")) for mode in paths}
        failures = []
        if not np.array_equal(tensors["native"]["noises"], tensors["fp8"]["noises"]):
            failures.append("native and FP8 diffusion-noise tensors differ")

        comparisons = []
        for position, (row, seed) in enumerate(zip(EVALUATION_INDICES, SEEDS)):
            native = tensors["native"]["actions"][position]
            fp8 = tensors["fp8"]["actions"][position]
            value = cosine(native, fp8)
            comparison = {
                "dataset_row": row,
                "seed": seed,
                "noise_bitwise_equal": bool(np.array_equal(
                    tensors["native"]["noises"][position],
                    tensors["fp8"]["noises"][position],
                )),
                "action_cosine": value,
                "max_abs_action_delta": float(np.max(np.abs(native - fp8))),
                "mean_abs_action_delta": float(np.mean(np.abs(native - fp8))),
            }
            comparisons.append(comparison)
            if value < MIN_ACTION_COSINE:
                failures.append(
                    f"row {row} action cosine {value:.6f} < {MIN_ACTION_COSINE:.6f}"
                )

        native_p50 = float(np.median(arms["native"]["latency_ms"]))
        fp8_p50 = float(np.median(arms["fp8"]["latency_ms"]))
        speedup = native_p50 / fp8_p50
        resident_ratio = (
            arms["fp8"]["resident_allocated_bytes"]
            / arms["native"]["resident_allocated_bytes"]
        )
        peak_ratio = arms["fp8"]["peak_allocated_bytes"] / arms["native"]["peak_allocated_bytes"]
        if peak_ratio > MAX_FP8_PEAK_RATIO:
            failures.append(f"FP8 startup peak ratio {peak_ratio:.4f} > {MAX_FP8_PEAK_RATIO:.4f}")
        if resident_ratio > MAX_FP8_RESIDENT_RATIO:
            failures.append(
                f"FP8 resident ratio {resident_ratio:.4f} > {MAX_FP8_RESIDENT_RATIO:.4f}"
            )
        if arms["fp8"]["released_bf16_bytes"] < MIN_RELEASED_BF16_BYTES:
            failures.append("FP8 arm did not release the certified BF16 weight payload")
        if arms["fp8"]["released_bf16_key_count"] != EXPECTED_RELEASED_BF16_KEYS:
            failures.append("FP8 arm released an unexpected BF16 key set")
        if speedup < MIN_SPEEDUP:
            failures.append(f"FP8 speedup {speedup:.4f} < {MIN_SPEEDUP:.4f}")
        if arms["fp8"]["fp8_layout"] != "nk":
            failures.append("FP8 arm did not select the SM120 nk layout")
        if arms["fp8"]["fp8_weight_count"] <= 0:
            failures.append("FP8 arm registered no FP8 weights")

        result = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if not failures else "FAIL",
            "hardware": "RTX 5090 / SM120",
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "model_sha256": model_sha256,
            },
            "dataset": {
                "repo": DATASET_REPO,
                "revision": DATASET_REVISION,
                "file": "data/chunk-000/file-000.parquet",
                "file_sha256": dataset_sha256,
                "prompt": PROMPT,
                "calibration_indices": list(CALIBRATION_INDICES),
                "evaluation_indices": list(EVALUATION_INDICES),
                "preprocessing": "LeRobot/OpenPI centered bilinear resize-with-pad to 224x224",
            },
            "protocol": {
                "process_isolation": True,
                "warmup_replays": 2,
                "timing_replays": TIMING_ITERATIONS,
                "seeds": list(SEEDS),
                "calibration_seed": 5090120,
                "action_operating_point": "checkpoint chunk 50, executed horizon 10, action dim 7",
                "thresholds": {
                    "min_action_cosine": MIN_ACTION_COSINE,
                    "min_speedup": MIN_SPEEDUP,
                    "max_fp8_resident_ratio": MAX_FP8_RESIDENT_RATIO,
                    "max_fp8_peak_ratio": MAX_FP8_PEAK_RATIO,
                    "min_released_bf16_bytes": MIN_RELEASED_BF16_BYTES,
                    "expected_released_bf16_keys": EXPECTED_RELEASED_BF16_KEYS,
                    "noise_bitwise_equal": True,
                },
            },
            "source_sha256": source_hashes(),
            "arms": arms,
            "comparisons": comparisons,
            "summary": {
                "min_action_cosine": min(row["action_cosine"] for row in comparisons),
                "max_abs_action_delta": max(
                    row["max_abs_action_delta"] for row in comparisons
                ),
                "native_p50_ms": native_p50,
                "fp8_p50_ms": fp8_p50,
                "speedup": speedup,
                "fp8_to_native_resident_ratio": resident_ratio,
                "fp8_to_native_peak_ratio": peak_ratio,
                "resident_memory_reduction_bytes": (
                    arms["native"]["resident_allocated_bytes"]
                    - arms["fp8"]["resident_allocated_bytes"]
                ),
            },
            "failures": failures,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--arm", choices=("native", "fp8"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.arm:
        run_arm(args.checkpoint.resolve(), args.dataset.resolve(), args.arm, args.output)
        return 0
    return qualify(args)


if __name__ == "__main__":
    raise SystemExit(main())
