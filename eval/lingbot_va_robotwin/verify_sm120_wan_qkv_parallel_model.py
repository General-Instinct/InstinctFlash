#!/usr/bin/env python3
"""Real-model 42-cycle A-B-B-A and reset gate for P009-A7."""

from __future__ import annotations

import argparse
import json
import statistics
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


def frame(rng):
    return {k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in CAMS}


def region(arms, lo, hi):
    b = (
        statistics.mean(x["latency_ms"] for x in arms[0][lo:hi])
        + statistics.mean(x["latency_ms"] for x in arms[3][lo:hi])
    ) / 2
    c = (
        statistics.mean(x["latency_ms"] for x in arms[1][lo:hi])
        + statistics.mean(x["latency_ms"] for x in arms[2][lo:hi])
    ) / 2
    return {
        "baseline_mean_ms": b,
        "candidate_mean_ms": c,
        "delta_ms": b - c,
        "speedup": b / c,
        "latency_reduction_percent": (b - c) / b * 100,
        "samples_per_arm": hi - lo,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cycles", type=int, default=42)
    args = ap.parse_args()
    from instinctflash import Runtime

    rt = Runtime.from_pretrained(
        args.checkpoint,
        placement="in_process",
        seed=123,
        tier_ceiling="numeric",
        exclude_passes=(
            "cfg_branch_elision",
            "conv_layout_ndhwc",
            "sm120_wan_qkv_parallel",
        ),
    )
    rt.reset(prompt=PROMPT)
    transformer = rt._backend._impl._server.transformer
    for attr in (
        "_ifl_wan_stage3_kernels",
        "_ifl_wan_qk_rope_kernels",
        "_ifl_wan_gemm_kernels",
        "_ifl_wan_ring_concat_kernels",
    ):
        if getattr(transformer, attr, None) is None:
            raise RuntimeError(f"missing A6 prerequisite {attr}")
    from instinctflash.backends.sm120_wan_qkv_parallel import (
        SM120WanParallelQKVKernels,
        install_wan_qkv_parallel,
    )

    kernels = install_wan_qkv_parallel(
        transformer, SM120WanParallelQKVKernels(args.library)
    )
    sites = [block.attn1 for block in transformer.blocks]
    if len(sites) != 30:
        raise RuntimeError(f"expected 30 A7 sites, got {len(sites)}")

    def enable(on):
        for site in sites:
            linears = (site.to_q, site.to_k, site.to_v)
            forwards = (
                site._ifl_wan_qkv_parallel_candidates
                if on
                else site._ifl_wan_qkv_parallel_originals
            )
            for linear, forward in zip(linears, forwards):
                linear.forward = forward

    def pointers():
        return tuple(
            output.data_ptr()
            for site in sites
            for outputs in site._ifl_wan_qkv_parallel_buffers.values()
            for output in outputs
        )

    # Populate both A5 and A7 static outputs before the measured ABBA order. Without this neutral
    # warmup, the first baseline arm alone pays unrelated first-forward allocations and makes the
    # full-cycle delta look much larger than the steady-state effect under test.
    for on in (True, False):
        enable(on)
        rt.reset(prompt=PROMPT)
        rng = np.random.default_rng(991)
        for cycle in range(3):
            torch.manual_seed(991 + cycle)
            nframes = 1 if cycle == 0 else (4 if cycle == 1 else 8)
            rt.predict(
                {
                    "obs": [frame(rng) for _ in range(nframes)],
                    "prompt": PROMPT,
                    "save_visualization": False,
                }
            )
        torch.cuda.synchronize()

    names = ("baseline_a", "candidate_b1", "candidate_b2", "baseline_a2")
    arms = []
    actions = []
    peaks = []
    calls = []
    for name, on in zip(names, (False, True, True, False)):
        enable(on)
        rt.reset(prompt=PROMPT)
        rng = np.random.default_rng(0)
        seq = []
        records = []
        begin = sum(kernels.calls.values())
        torch.cuda.reset_peak_memory_stats()
        for cycle in range(args.cycles):
            torch.manual_seed(1234 + cycle)
            nframes = 1 if cycle == 0 else (4 if cycle == 1 else 8)
            obs = [frame(rng) for _ in range(nframes)]
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = rt.predict(
                {"obs": obs, "prompt": PROMPT, "save_visualization": False}
            )
            torch.cuda.synchronize()
            action = np.asarray(
                output["action"] if isinstance(output, dict) else output
            ).copy()
            seq.append(action)
            records.append(
                {
                    "index": cycle,
                    "frames": nframes,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
            print(name, cycle, f"{records[-1]['latency_ms']:.3f}", flush=True)
        arms.append(records)
        actions.append(seq)
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
        calls.append(sum(kernels.calls.values()) - begin)
    reference = actions[0]
    equal = {}
    for name, seq in zip(names, actions):
        same = [np.array_equal(a, b) for a, b in zip(reference, seq)]
        equal[name] = {
            "all_equal": all(same),
            "equal_cycles": sum(same),
            "max_abs": max(
                float(np.max(np.abs(a - b))) for a, b in zip(reference, seq)
            ),
        }
    before = pointers()
    enable(True)
    rt.reset(prompt=PROMPT)
    rng = np.random.default_rng(0)
    reset_actions = []
    for cycle in range(3):
        torch.manual_seed(1234 + cycle)
        nframes = 1 if cycle == 0 else (4 if cycle == 1 else 8)
        o = rt.predict(
            {
                "obs": [frame(rng) for _ in range(nframes)],
                "prompt": PROMPT,
                "save_visualization": False,
            }
        )
        reset_actions.append(
            np.asarray(o["action"] if isinstance(o, dict) else o).copy()
        )
    after = pointers()
    reset_equal = [np.array_equal(actions[1][i], reset_actions[i]) for i in range(3)]
    result = {
        "schema_version": 1,
        "model": "robbyant/lingbot-va-posttrain-robotwin",
        "model_revision": "8c9dea8abbc5c91cc9e18bc3264b8915083bbe70",
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tier": "BITEXACT",
        "registered_modules": len(sites),
        "action_equality": equal,
        "arm_means_ms": dict(
            zip(names, [statistics.mean(x["latency_ms"] for x in arm) for arm in arms])
        ),
        "arm_spread_percent": {
            "baseline": abs(
                statistics.mean(x["latency_ms"] for x in arms[0])
                - statistics.mean(x["latency_ms"] for x in arms[3])
            )
            / statistics.mean(
                [
                    statistics.mean(x["latency_ms"] for x in arms[0]),
                    statistics.mean(x["latency_ms"] for x in arms[3]),
                ]
            )
            * 100,
            "candidate": abs(
                statistics.mean(x["latency_ms"] for x in arms[1])
                - statistics.mean(x["latency_ms"] for x in arms[2])
            )
            / statistics.mean(
                [
                    statistics.mean(x["latency_ms"] for x in arms[1]),
                    statistics.mean(x["latency_ms"] for x in arms[2]),
                ]
            )
            * 100,
        },
        "full_cycles_0_41": region(arms, 0, args.cycles),
        "cold_cycles_0_3": region(arms, 0, 4),
        "late_cycles_32_41": region(arms, 32, args.cycles),
        "kernel_calls": dict(zip(names, calls)),
        "count_histogram": dict(kernels.calls),
        "peak_memory_gib": dict(zip(names, peaks)),
        "reset_replay": {
            "cycles": 3,
            "all_actions_equal": all(reset_equal),
            "equal_cycles": sum(reset_equal),
            "max_abs": max(
                float(np.max(np.abs(actions[1][i] - reset_actions[i])))
                for i in range(3)
            ),
            "buffer_pointers_stable": before == after,
        },
    }
    result["status"] = (
        "pass"
        if all(v["all_equal"] for v in equal.values())
        and result["reset_replay"]["all_actions_equal"]
        and result["reset_replay"]["buffer_pointers_stable"]
        else "fail"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    enable(False)
    rt.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
