"""Profile the major GR00T N1.7 serving stages on one CUDA device.

This deliberately uses synchronization at stage boundaries.  It is a diagnosis
tool, not a throughput benchmark: ``benchmark_runtime.py`` remains the source
for uninstrumented end-to-end latency.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from benchmark_runtime import _synthetic_observation
from instinctflash import Runtime


class _StageTimer:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._restore: list[tuple[Any, str, Any]] = []

    def wrap(self, owner: Any, attribute: str, label: str) -> None:
        original = getattr(owner, attribute)

        def measured(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                self.samples[label].append((time.perf_counter() - started) * 1_000.0)

        setattr(owner, attribute, measured)
        self._restore.append((owner, attribute, original))

    def close(self) -> None:
        for owner, attribute, original in reversed(self._restore):
            setattr(owner, attribute, original)
        self._restore.clear()


class _ProcessorProxy:
    def __init__(self, processor: Any, timer: _StageTimer) -> None:
        self._processor = processor
        self._timer = timer

    def __call__(self, *args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return self._processor(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            self._timer.samples["processor"].append(
                (time.perf_counter() - started) * 1_000.0
            )

    def __getattr__(self, name: str):
        return getattr(self._processor, name)


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "min_ms": float(array.min()),
    }


def main() -> int:
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
    parser.add_argument("--backend", choices=("eager", "cuda_graph"), default="cuda_graph")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Limit PyTorch and OpenCV preprocessing threads (0 keeps their defaults).",
    )
    args = parser.parse_args()

    if args.cpu_threads > 0:
        import cv2

        torch.set_num_threads(args.cpu_threads)
        cv2.setNumThreads(args.cpu_threads)

    package = Path(__file__).resolve().parent
    if args.source_root:
        os.environ["GR00T_ROOT"] = str(Path(args.source_root).expanduser().resolve())
    if args.checkpoint:
        os.environ["GR00T_N17_CHECKPOINT"] = str(Path(args.checkpoint).expanduser().resolve())
    os.environ["IFL_GROOT_STATIC_CAPTURE"] = "1" if args.backend == "cuda_graph" else "0"

    runtime = Runtime.from_pretrained(package, device=args.device, placement="in_process")
    runtime.reset(prompt="pick up the object")
    observation = _synthetic_observation()
    for _ in range(args.warmup):
        runtime.predict(observation)
    torch.cuda.synchronize()

    loop = runtime._backend._impl
    policy = loop._policy
    model = policy.model
    head = model.action_head
    timer = _StageTimer()

    timer.wrap(policy, "collate_fn", "collate")
    timer.wrap(policy.processor, "decode_action", "decode_action")
    policy.processor = _ProcessorProxy(policy.processor, timer)
    timer.wrap(model, "prepare_input", "model_prepare_input")
    timer.wrap(model.backbone, "forward", "backbone")
    qwen = getattr(model.backbone, "model", None)
    if qwen is not None:
        base = getattr(qwen, "model", None)
        visual = getattr(qwen, "visual", None)
        language = getattr(qwen, "language_model", None)
        lm_head = getattr(qwen, "lm_head", None)
        if base is not None:
            timer.wrap(base, "get_rope_index", "backbone_rope_index")
        if visual is not None:
            timer.wrap(visual, "fast_pos_embed_interpolate", "vision_pos_embed")
            timer.wrap(visual, "rot_pos_emb", "vision_rope")
            timer.wrap(visual, "forward", "vision_total")
        if language is not None:
            timer.wrap(language, "forward", "language_total")
        if lm_head is not None:
            timer.wrap(lm_head, "forward", "backbone_lm_head")
    timer.wrap(head, "_encode_features", "action_encode_features")
    timer.wrap(head, "get_action_with_features", "action_flow_total")
    timer.wrap(head.action_encoder, "forward", "action_encoder")
    timer.wrap(head.model, "forward", "dit")
    timer.wrap(head.action_decoder, "forward", "action_decoder")
    timer.wrap(model, "get_action", "model_total")

    totals = []
    try:
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            runtime.predict(observation)
            torch.cuda.synchronize()
            totals.append((time.perf_counter() - started) * 1_000.0)
    finally:
        policy.processor = policy.processor._processor
        timer.close()

    stages = {name: _percentiles(values) for name, values in sorted(timer.samples.items())}
    for stage in stages.values():
        stage["calls_per_request"] = stage["count"] / args.iterations
    print(json.dumps({
        "backend": args.backend,
        "gpu": torch.cuda.get_device_name(),
        "cpu_threads": torch.get_num_threads(),
        "iterations": args.iterations,
        "end_to_end": _percentiles(totals),
        "stages": stages,
    }, indent=2))
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
