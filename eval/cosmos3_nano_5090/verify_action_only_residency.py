#!/usr/bin/env python3
"""Real-weight Nano gate: complete CPU lm_head reference vs action-only sentinel."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch


class CPUOnlyLinear(torch.nn.Linear):
    """Complete checkpoint target that parent .to/.to_empty cannot migrate to CUDA."""

    def _apply(self, fn, recurse=True):
        return self


def arm_cpu_reference():
    from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLTextForCausalLM

    original = Qwen3VLTextForCausalLM.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        old = self.lm_head
        self.lm_head = CPUOnlyLinear(
            old.in_features,
            old.out_features,
            bias=False,
            device="cpu",
            dtype=torch.bfloat16,
        )

    Qwen3VLTextForCausalLM.__init__ = init


def make_service(checkpoint, arm):
    from cosmos_framework.inference.common import inference as common_inference
    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
    )

    class NoopRunner:
        def run(self, *args, **kwargs):
            return None

        def is_safe(self, **kwargs):
            return True, "disabled"

    def no_guardrails(cls, setup_args):
        return cls(text=NoopRunner(), video=NoopRunner())

    common_inference.GuardrailRunners.create = classmethod(no_guardrails)
    if arm == "cpu_head":
        arm_cpu_reference()
    else:
        from cosmos3_iwm.nano_action_only import arm_nano_action_only_lm_head

        arm_nano_action_only_lm_head()

    class EagerService(RobolabPolicyService):
        def _build_setup_args(self, service_args):
            setup = super()._build_setup_args(service_args)
            return setup.model_copy(
                update={
                    "guardrails": False,
                    "use_torch_compile": False,
                    "use_cuda_graphs": False,
                }
            )

    args = RobolabServerArgs(
        checkpoint_path=str(checkpoint.resolve()),
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
    service = EagerService(args)
    network = service.model.net
    if network.predict_text_tokens:
        raise RuntimeError("Nano checkpoint unexpectedly requests text logits")
    if arm == "elided":
        from cosmos3_iwm.nano_action_only import verify_nano_action_only_model

        verify_nano_action_only_model(service.model)
    else:
        head = network.language_model.lm_head
        if not isinstance(head, CPUOnlyLinear) or head.weight.device.type != "cpu":
            raise RuntimeError("Nano reference did not retain its complete CPU lm_head")
    from cosmos3_iwm.persistent_text_kv import install_persistent_text_kv

    install_persistent_text_kv(service)
    return service


def memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "free_gib": free / 2**30,
        "total_gib": total / 2**30,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("cpu_head", "elided"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()
    torch.cuda.reset_peak_memory_stats()
    service = make_service(args.checkpoint, args.arm)
    after_load = memory()
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(540, 640, 3), dtype=np.uint8)
    request = {
        "observation/image": image,
        "observation/joint_position": np.zeros((1, 7), dtype=np.float32),
        "observation/gripper_position": np.zeros((1, 1), dtype=np.float32),
        "prompt": "pick up the banana and place it in the bowl",
    }
    samples = []
    output = None
    for index in range(args.warmup + args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = service.infer(request)
        torch.cuda.synchronize()
        if index >= args.warmup:
            samples.append((time.perf_counter() - started) * 1000)
    action = np.asarray(output["action"], dtype=np.float32)
    gates = [
        {
            "case": "base",
            "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
            "action_values": action.tolist(),
        }
    ]
    for case_index in range(4):
        case = dict(request)
        case["observation/image"] = np.roll(image, case_index + 1, axis=1).copy()
        case["observation/joint_position"] = np.full(
            (1, 7), (case_index + 1) * 0.01, dtype=np.float32
        )
        action = np.asarray(service.infer(case)["action"], dtype=np.float32)
        gates.append(
            {
                "case": f"new_observation_{case_index + 1}",
                "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
                "action_values": action.tolist(),
            }
        )
    changed = dict(request)
    changed["prompt"] = "move the object to the left"
    action = np.asarray(service.infer(changed)["action"], dtype=np.float32)
    gates.append(
        {
            "case": "changed_prompt",
            "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
            "action_values": action.tolist(),
        }
    )
    result = {
        "schema_version": 1,
        "model": "nvidia/Cosmos3-Nano-Policy-DROID",
        "model_revision": "2b9f9517efcfbf26e222945b386ae9b65c0930ac",
        "source_revision": "0e034bc98ffa3c3dfa19f037871f3a8bbc1c4d05",
        "arm": args.arm,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tier": "BITEXACT",
        "protocol": "fresh process; original BF16 eager kernels; persistent text-KV in both arms; p50 of 6 after 2 warmups; six action cases",
        "latency_ms": samples,
        "latency_p50_ms": statistics.median(samples),
        "latency_mean_ms": statistics.mean(samples),
        "memory_after_load": after_load,
        "memory_after_gate": memory(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "cache_stats": dict(service.model._ifl_persistent_text_kv.stats),
        "gates": gates,
        "status": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "gates"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
