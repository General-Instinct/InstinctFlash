#!/usr/bin/env python3
"""One deterministic real-model arm for P009-A3 A/B measurement."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

CAMS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
PROMPT = "Use the left arm to lift the plastic drink bottle head-up"


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def frame(rng):
    return {
        key: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        for key in CAMS
    }


def stage3_buffer_pointers(transformer):
    return {
        str(index): {
            str(key): [tensor.data_ptr() for tensor in tensors]
            for key, tensors in getattr(block, "_ifl_stage3_buffers", {}).items()
        }
        for index, block in enumerate(transformer.blocks)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stage3", action="store_true")
    parser.add_argument("--reset-replay", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    result = {
        "variant": "stage3" if args.stage3 else "stage2",
        "git_branch": git(repo, "branch", "--show-current"),
        "git_commit": git(repo, "rev-parse", "HEAD"),
        "git_status": git(repo, "status", "--short"),
        "checkpoint": str(args.checkpoint),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cycles": [],
        "status": "running",
    }
    runtime = None
    actions = []
    try:
        from instinctflash import Runtime

        runtime = Runtime.from_pretrained(
            args.checkpoint,
            placement="in_process",
            seed=args.seed,
            tier_ceiling="numeric",
            exclude_passes=("cfg_branch_elision", "conv_layout_ndhwc"),
        )
        runtime.reset(prompt=PROMPT)
        transformer = runtime._backend._impl._server.transformer
        a1 = getattr(transformer, "_ifl_sm120_gated_residual_kernel", None)
        a2 = getattr(transformer, "_ifl_wan_stage2_kernels", None)
        if a1 is None or a2 is None:
            raise RuntimeError("real-model A3 gate requires installed A1 and A2")
        a3 = getattr(transformer, "_ifl_wan_stage3_kernels", None)
        if args.stage3 and a3 is None:
            # Direct-install fallback keeps this runner useful before planner integration.
            from instinctflash.backends.sm120_wan_stage3 import (
                SM120WanStage3Kernels,
                install_wan_stage3,
            )
            a3 = install_wan_stage3(
                transformer, SM120WanStage3Kernels(args.library))
        if not args.stage3 and a3 is not None:
            raise RuntimeError("baseline arm unexpectedly auto-installed A3")
        result["a1_library"] = str(a1.path)
        result["a2_library"] = str(a2.path)
        result["a3_library"] = None if a3 is None else str(a3.path)
        result["a3_installed"] = a3 is not None
        begin_a1, begin_a2 = a1.calls, a2.calls
        begin_a3 = 0 if a3 is None else a3.calls
        torch.cuda.reset_peak_memory_stats()

        def run_sequence():
            sequence = []
            records = []
            rng = np.random.default_rng(0)
            for cycle in range(args.cycles):
                nframes = 1 if cycle == 0 else (4 if cycle == 1 else 8)
                observation = [frame(rng) for _ in range(nframes)]
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = runtime.predict(
                    {"obs": observation, "prompt": PROMPT, "save_visualization": False})
                torch.cuda.synchronize()
                action = np.asarray(output["action"] if isinstance(output, dict) else output)
                sequence.append(action.copy())
                records.append({
                    "index": cycle,
                    "frames": nframes,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "finite": bool(np.isfinite(action).all()),
                    "shape": list(action.shape),
                })
            return sequence, records

        actions, result["cycles"] = run_sequence()
        if args.reset_replay:
            pointers_before = stage3_buffer_pointers(transformer)
            runtime.reset(prompt=PROMPT)
            replay_actions, replay_cycles = run_sequence()
            pointers_after = stage3_buffer_pointers(transformer)
            equal = [
                bool(np.array_equal(first, second))
                for first, second in zip(actions, replay_actions, strict=True)
            ]
            result["reset_replay"] = {
                "equal_cycles": sum(equal),
                "all_equal": all(equal),
                "max_abs": max(
                    float(np.max(np.abs(first - second)))
                    for first, second in zip(actions, replay_actions, strict=True)),
                "buffer_pointers_stable": pointers_before == pointers_after,
                "cycles": replay_cycles,
            }
            if not result["reset_replay"]["all_equal"]:
                raise RuntimeError("A3 reset replay changed actions")
            if not result["reset_replay"]["buffer_pointers_stable"]:
                raise RuntimeError("A3 reset replay changed persistent buffer pointers")
        result["a1_calls"] = a1.calls - begin_a1
        result["a2_calls"] = a2.calls - begin_a2
        result["a3_calls"] = 0 if a3 is None else a3.calls - begin_a3
        result["max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 4)
        result["status"] = "pass"
    except Exception as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if actions:
            np.savez_compressed(
                args.actions,
                **{f"cycle_{index}": action for index, action in enumerate(actions)})
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
