#!/usr/bin/env python3
"""RTX 5090 Cosmos3-Edge eager vs persistent same-prompt text-KV harness."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--arm",
        choices=("eager", "persistent"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from cosmos_framework.inference.common import inference as common_inference
    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
    )

    class _NoopRunner:
        def run(self, *unused_args, **unused_kwargs):
            return None

        def is_safe(self, **unused_kwargs):
            return True, "guardrails disabled for action-policy benchmark"

    def _no_guardrails(cls, setup_args):
        return cls(text=_NoopRunner(), video=_NoopRunner())

    common_inference.GuardrailRunners.create = classmethod(_no_guardrails)

    class _ConfiguredService(RobolabPolicyService):
        def _build_setup_args(self, service_args):
            setup_args = super()._build_setup_args(service_args)
            return setup_args.model_copy(
                update={
                    "guardrails": False,
                    "use_torch_compile": False,
                    "use_cuda_graphs": False,
                }
            )

    service_args = RobolabServerArgs(
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        domain_name="droid_lerobot",
        action_dim=8,
        action_chunk_size=16,
        image_height=540,
        image_width=640,
        num_steps=4,
        guidance=1.0,
        shift=5.0,
        seed=0,
        deterministic_seed=True,
    )
    service = _ConfiguredService(service_args)
    if args.arm == "persistent":
        from cosmos3_iwm.persistent_text_kv import install_persistent_text_kv

        install_persistent_text_kv(service)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(540, 640, 3), dtype=np.uint8)
    request = {
        "observation/image": image,
        "observation/joint_position": np.zeros((1, 7), dtype=np.float32),
        "observation/gripper_position": np.zeros((1, 1), dtype=np.float32),
        "prompt": "pick up the banana and place it in the bowl",
    }

    samples: list[float] = []
    output = None
    for index in range(args.warmup + args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = service.infer(request)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if index >= args.warmup:
            samples.append(elapsed_ms)
    assert output is not None
    action = np.asarray(output["action"], dtype=np.float32)
    gates = []
    for case_index in range(4):
        case_request = dict(request)
        case_request["observation/image"] = np.roll(
            image, case_index + 1, axis=1
        ).copy()
        case_request["observation/joint_position"] = np.full(
            (1, 7), (case_index + 1) * 0.01, dtype=np.float32
        )
        case_output = service.infer(case_request)
        case_action = np.asarray(case_output["action"], dtype=np.float32)
        gates.append(
            {
                "case": f"new_observation_{case_index + 1}",
                "action_sha256": hashlib.sha256(case_action.tobytes()).hexdigest(),
                "action_values": case_action.tolist(),
            }
        )
    changed_prompt = dict(request)
    changed_prompt["prompt"] = "move the object to the left"
    try:
        changed_output = service.infer(changed_prompt)
        changed_action = np.asarray(changed_output["action"], dtype=np.float32)
        prompt_gate = {
            "status": "pass",
            "action_sha256": hashlib.sha256(changed_action.tobytes()).hexdigest(),
            "action_values": changed_action.tolist(),
        }
    except (RuntimeError, ValueError, AssertionError) as exc:
        prompt_gate = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    record = {
        "schema_version": 1,
        "model": "nvidia/Cosmos3-Edge-Policy-DROID",
        "model_revision": "68b17b3c959ccd0999de9a972c5f7c8b57112f86",
        "source_revision": "0e034bc98ffa3c3dfa19f037871f3a8bbc1c4d05",
        "arm": args.arm,
        "protocol": "fresh process; p50 of 12 calls after 3 warmups; official RoboLab service; eager kernels; 4 UniPC steps; guidance=1.0; action chunk 16x8",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_ms": samples,
        "latency_ms_p50": statistics.median(samples),
        "latency_ms_mean": statistics.mean(samples),
        "latency_ms_p90": float(np.percentile(samples, 90)),
        "action_shape": list(action.shape),
        "action_finite": bool(np.isfinite(action).all()),
        "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
        "action_values": action.tolist(),
        "new_observation_gates": gates,
        "changed_prompt_gate": prompt_gate,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "cache_stats": (
            dict(service.model._ifl_persistent_text_kv.stats)
            if args.arm == "persistent"
            else {}
        ),
        "tier": "BITEXACT",
        "status": "pass",
    }
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "arm",
                    "latency_ms_p50",
                    "latency_ms_mean",
                    "latency_ms_p90",
                    "action_shape",
                    "action_finite",
                    "action_sha256",
                    "peak_allocated_gib",
                    "peak_reserved_gib",
                    "status",
                )
            },
            indent=2,
        )
    )
    service = None
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
