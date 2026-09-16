#!/usr/bin/env python3
"""Compare the SM120 checkpoint frontend to fully loaded official LeRobot weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.vla.pi05_weights import verify_loaded_weights
from flash_rt.frontends.torch.pi05_checkpoint import Pi05CheckpointFrontend
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from verify_sm120_fp8 import (CALIBRATION_INDICES, EVALUATION_INDICES, SEEDS, PROMPT,
                             MODEL_SHA256, DATASET_SHA256, load_observations, cosine, sha256, source_hashes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if sha256(args.checkpoint / "model.safetensors") != MODEL_SHA256 or sha256(args.dataset) != DATASET_SHA256:
        raise ValueError("reference requires the pinned checkpoint and real dataset")
    runtime = Pi05CheckpointFrontend(args.checkpoint, use_fp8=True)
    runtime.set_prompt(PROMPT)
    runtime.calibrate(load_observations(args.dataset, CALIBRATION_INDICES))
    policy = PI05Policy.from_pretrained(args.checkpoint, config=runtime.config).eval()
    identity = verify_loaded_weights(policy, args.checkpoint)
    comparisons = []
    with torch.no_grad():
        for obs, seed in zip(load_observations(args.dataset, EVALUATION_INDICES), SEEDS):
            torch.manual_seed(seed)
            actions = runtime.infer(obs)["actions"]
            _, batch = runtime.prepare(obs)
            # Preserve the same BF16-rounded values; stock action_in_proj is FP32.
            noise = runtime._noise_buf.float().clone().unsqueeze(0)
            expected = policy.predict_action_chunk(batch, noise=noise)
            expected = runtime._post(expected)[0, :10].float().cpu().numpy()
            comparisons.append({"seed": seed, "action_cosine": cosine(expected, actions),
                "max_abs_delta": float(np.abs(expected - actions).max()),
                "mean_abs_delta": float(np.abs(expected - actions).mean()),
                "token_count": int(batch["observation.language.attention_mask"].sum())})
    result = {"status": "PASS" if all(c["action_cosine"] >= .98 for c in comparisons) else "FAIL",
              "loaded_weights": identity, "comparisons": comparisons,
              "source_sha256": {**source_hashes(),
                  "examples/pi05_vla/verify_checkpoint_reference.py": sha256(Path(__file__))},
              "packages": {name: importlib.metadata.version(name) for name in ("lerobot", "transformers", "torch")},
              "scope": "official LeRobot 50-step computation and processors; identical explicit noise values"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
