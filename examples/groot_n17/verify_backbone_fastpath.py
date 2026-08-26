"""Gate cached Qwen3-VL metadata and the logits-free backbone path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_runtime import _synthetic_observation
from groot_n17_iwm.backbone_fastpath import install_backbone_fastpath
from instinctflash import Runtime


def _skip_transformers_mistral_hub_probe():
    # transformers 4.57.3 _patch_mistral_regex calls the Hub API even in offline mode;
    # the checkpoint's Qwen3 tokenizer is not a mistral model, skip it. Same workaround
    # as verify_fastpaths.py; run these gates with HF_HUB_OFFLINE=1.
    import transformers.tokenization_utils_base as tub

    def _no_mistral_patch(cls, tokenizer, *args, **kwargs):
        return tokenizer

    tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_no_mistral_patch)


def main() -> int:
    _skip_transformers_mistral_hub_probe()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=None,
        help="N1.7 weights dir; defaults to $GR00T_N17_CHECKPOINT or the package's "
             "base_weights pointer (HF cache)",
    )
    parser.add_argument(
        "--source-root", default=None,
        help="Isaac-GR00T checkout; defaults to $GR00T_ROOT or the documented conventions",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    package = Path(__file__).resolve().parent
    if args.source_root:
        os.environ["GR00T_ROOT"] = str(Path(args.source_root).expanduser().resolve())
    if args.checkpoint:
        os.environ["GR00T_N17_CHECKPOINT"] = str(Path(args.checkpoint).expanduser().resolve())
    os.environ["IFL_GROOT_STATIC_CAPTURE"] = "1"
    os.environ["IFL_GROOT_FAST_DECODE"] = "1"
    os.environ["IFL_GROOT_BACKBONE_FASTPATH"] = "0"
    runtime = Runtime.from_pretrained(
        package,
        device=args.device,
        placement="in_process",
        nfe={"action": 4},
    )

    zero = _synthetic_observation()
    rng = np.random.default_rng(1701)
    textured = {
        "images": [
            rng.integers(0, 256, size=image.shape, dtype=np.uint8)
            for image in zero["images"]
        ],
        "state": zero["state"].copy(),
    }
    cases = (
        ("zero", "pick up the object", zero, 1701),
        ("textured", "pick up the object", textured, 1702),
        (
            "long_prompt",
            "carefully pick up the leftmost small red block and place it in the drawer",
            textured,
            1703,
        ),
    )

    def infer(prompt, observation, seed):
        runtime.reset(prompt=prompt)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return runtime.predict(observation)["action"].copy()

    eager = {
        name: infer(prompt, observation, seed)
        for name, prompt, observation, seed in cases
    }
    model = runtime._backend._impl._policy.model
    handle = install_backbone_fastpath(model)
    optimized = {
        name: infer(prompt, observation, seed)
        for name, prompt, observation, seed in cases
    }
    exact = {name: bool(np.array_equal(eager[name], optimized[name])) for name in eager}
    max_abs = {
        name: float(np.max(np.abs(eager[name] - optimized[name]))) for name in eager
    }

    prompt, observation = cases[0][1:3]
    runtime.reset(prompt=prompt)
    for _ in range(args.warmup):
        runtime.predict(observation)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        runtime.predict(observation)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1_000.0)
    values = np.asarray(samples, dtype=np.float64)
    report = {
        "array_equal": exact,
        "max_abs": max_abs,
        "cache_hits": handle.hits,
        "cache_misses": handle.misses,
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "min_ms": float(values.min()),
    }
    print(json.dumps(report, indent=2))
    handle.close()
    runtime.close()
    return 0 if all(exact.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
