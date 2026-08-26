"""Compare object-free OXE decoding with the upstream object-by-object path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_runtime import _synthetic_observation
from groot_n17_iwm.fast_decode import FastOXEDecoder
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
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    package = Path(__file__).resolve().parent
    if args.source_root:
        os.environ["GR00T_ROOT"] = str(Path(args.source_root).expanduser().resolve())
    if args.checkpoint:
        os.environ["GR00T_N17_CHECKPOINT"] = str(Path(args.checkpoint).expanduser().resolve())
    os.environ["IFL_GROOT_STATIC_CAPTURE"] = "1"
    runtime = Runtime.from_pretrained(package, device=args.device, placement="in_process")
    runtime.reset(prompt="pick up the object")
    runtime.predict(_synthetic_observation())
    torch.cuda.synchronize()

    policy = runtime._backend._impl._policy
    processor = policy.processor
    decoder = getattr(processor, "_instinctflash_fast_decoder", None)
    if decoder is None:
        decoder = FastOXEDecoder(processor)
        upstream_decode = processor.decode_action
    else:
        upstream_decode = decoder._fallback
    rng = np.random.default_rng(1701)
    action = rng.standard_normal((1, 40, 132), dtype=np.float32)
    state = {
        "eef_9d": rng.standard_normal((1, 1, 9), dtype=np.float32),
        "gripper_position": rng.standard_normal((1, 1, 1), dtype=np.float32),
        "joint_position": rng.standard_normal((1, 1, 7), dtype=np.float32),
    }

    upstream = upstream_decode(action, policy.embodiment_tag, state)
    optimized = decoder(action, policy.embodiment_tag, state)

    def measure(callable_):
        started = time.perf_counter()
        for _ in range(args.iterations):
            callable_(action, policy.embodiment_tag, state)
        return (time.perf_counter() - started) * 1_000.0 / args.iterations

    upstream_ms = measure(upstream_decode)
    optimized_ms = measure(decoder)

    observation = _synthetic_observation()
    processor.decode_action = upstream_decode
    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    reference_action = runtime.predict(observation)["action"].copy()
    torch.cuda.synchronize()
    processor.decode_action = decoder
    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    candidate_action = runtime.predict(observation)["action"].copy()
    torch.cuda.synchronize()
    float64_exact = {
        key: bool(np.array_equal(upstream[key], optimized[key])) for key in upstream
    }
    float32_exact = {
        key: bool(
            np.array_equal(upstream[key].astype(np.float32), optimized[key].astype(np.float32))
        )
        for key in upstream
    }
    max_abs = {
        key: float(np.max(np.abs(upstream[key] - optimized[key]))) for key in upstream
    }
    print(json.dumps({
        "supported": decoder.supported,
        "upstream_ms": upstream_ms,
        "optimized_ms": optimized_ms,
        "speedup": upstream_ms / optimized_ms,
        "float64_array_equal": float64_exact,
        "float32_array_equal": float32_exact,
        "max_abs_float64": max_abs,
        "end_to_end_array_equal": bool(np.array_equal(reference_action, candidate_action)),
        "end_to_end_max_abs": float(np.max(np.abs(reference_action - candidate_action))),
    }, indent=2))
    runtime.close()
    return (
        0
        if all(float64_exact.values()) and np.array_equal(reference_action, candidate_action)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
