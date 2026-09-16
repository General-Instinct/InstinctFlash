#!/usr/bin/env python3
"""Bounded SM120 primitive qualification using the unchanged shared checks.

Use each actual pinned model environment. This verifies synthetic arithmetic and
CPU residency only; it does not construct a model or qualify task quality.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import qualify_sm89_fp8 as checks
from qualify_sm89_fp8 import _check_packing, _check_projection, _check_residency, _sha


def qualify(device, report):
    import torch

    report["runtime"] = {"python": sys.version, "python_executable": sys.executable,
                         "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                         "torch_git_version": torch.version.git_version}
    device = torch.device(device)
    if (device.type != "cuda" or not torch.cuda.is_available()
            or torch.cuda.get_device_capability(device) != (12, 0)):
        raise RuntimeError("Qualification requires a visible SM120 CUDA device")
    torch.cuda.set_device(device)
    import triton

    from instinctflash.runtime import (
        desktop_fp8,
        fp8_pack,
        module_residency,
        sm89_fp8,
        sm120_fp8,
        torch_fp8_linear,
    )

    report["runtime"]["triton"] = triton.__version__
    props = torch.cuda.get_device_properties(device)
    report["device"] = {"name": props.name, "capability": [props.major, props.minor],
                        "uuid": str(props.uuid),
                        "total_memory_bytes": props.total_memory, "selected_device": str(device)}
    report["source"] = {
        Path(module.__file__).name: {"path": str(Path(module.__file__).resolve()),
                                    "sha256": _sha(module.__file__)}
        for module in (desktop_fp8, fp8_pack, sm89_fp8, sm120_fp8, torch_fp8_linear, module_residency, checks)}
    report["executor"] = sm120_fp8.EXECUTOR
    report["recipe"] = torch_fp8_linear.SM120FP8Linear.recipe
    torch.manual_seed(731)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _check_packing(torch, fp8_pack.dynamic_pack_bf16_e4m3, device, report)
        report["projections"] = []
        for shape, bias in (((1, 16, 16), False), ((7, 768, 768), True),
                            ((40, 2048, 2048), False), ((16, 3072, 3072), True),
                            ((3, 4096, 1024), False), ((7, 5120, 13824), True)):
            report["projections"].append(_check_projection(
                torch, torch_fp8_linear.SM120FP8Linear, device, shape, bias))
            # CUDAGraph's private pool becomes reclaimable with the previous
            # case's local references; keep the largest case as the peak bound.
            torch.cuda.empty_cache()
        _check_residency(torch, torch_fp8_linear.SM120FP8Linear, device, report)
    report["memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True, help="new JSON receipt path")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new path to preserve earlier evidence")
    report = {"schema": "instinctflash.sm120_fp8_primitive_qualification.v1",
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "script_sha256": _sha(__file__), "passed": False,
              "scope": "synthetic primitives only", "task_quality_status": "unverified",
              "model_constructed": False, "model_weights_downloaded": False}
    try:
        qualify(args.device, report)
        report["passed"] = True
    except Exception as error:  # noqa: BLE001 - persist any failed primitive receipt
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output),
                      "scope": report["scope"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
