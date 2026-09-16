#!/usr/bin/env python3
"""Bounded SM89 FP8 primitive checks; no model weights or task-quality claim.

Run this file with each deployed environment's Python and its installed package,
or set PYTHONPATH to the source checkout being qualified. At most one synthetic
projection case is live at a time. The output records the imported source hashes
so a successful report cannot be mistaken for qualification of different code.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import traceback


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference_pack(torch, x):
    scale = x.abs().amax().float().clamp_min(1e-12).reshape(1) / 448.0
    packed = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return packed, scale


def _equal_bytes(torch, actual, expected, label):
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        count = (actual.view(torch.uint8) != expected.view(torch.uint8)).sum().item()
        raise AssertionError(f"{label}: {count} differing bytes")


def _check_packing(torch, pack, device, report):
    cases = []
    for shape, magnitude in (((40, 2048), 1), ((256, 5120), 1), ((1, 16), 0),
                             ((1, 16), 1e-20), ((1, 16), 1e30)):
        x = torch.randn(shape, device=device, dtype=torch.bfloat16) * magnitude
        actual, scale = pack(x)
        expected, expected_scale = _reference_pack(torch, x)
        _equal_bytes(torch, scale, expected_scale, "activation scale")
        _equal_bytes(torch, actual, expected, "activation pack")
        cases.append({"shape": list(shape), "magnitude": magnitude, "passed": True})
    patterns = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16)
    patterns = patterns.view(torch.bfloat16)
    finite = patterns[torch.isfinite(patterns)].reshape(1, -1).contiguous()
    actual, scale = pack(finite)
    expected, expected_scale = _reference_pack(torch, finite)
    _equal_bytes(torch, scale, expected_scale, "all finite BF16 scale")
    _equal_bytes(torch, actual, expected, "all finite BF16 pack")
    cases.append({"case": "all_finite_bf16_bit_patterns", "elements": finite.numel(),
                  "passed": True})
    x = torch.ones((4, 16), device=device, dtype=torch.bfloat16)
    for nonfinite in (float("nan"), float("inf"), -float("inf")):
        x[0, 0] = nonfinite
        actual, scale = pack(x)
        expected, expected_scale = _reference_pack(torch, x)
        # Different FP8 NaN signs/payloads are immaterial here; propagating the
        # reference nonfinite values instead of sanitizing them is the contract.
        torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0,
                                   equal_nan=True)
    cases.append({"case": "nonfinite_propagation", "passed": True})
    report["packing"] = cases


def _check_projection(torch, linear_type, device, shape, bias):
    rows, width, outputs = shape
    source = torch.nn.Linear(width, outputs, bias=bias, dtype=torch.bfloat16)
    cpu_packed = linear_type(source, device=device)
    cpu_stored = linear_type(source, device=device, storage_device="cpu")
    if any(value.device.type != "cpu" for value in cpu_stored.buffers()):
        raise AssertionError("CPU packed storage allocated a CUDA buffer")
    cpu_stored.to(device=device)
    _equal_bytes(torch, cpu_stored.weight_fp8, cpu_packed.weight_fp8, "CPU stored weights")
    _equal_bytes(torch, cpu_stored.weight_scale, cpu_packed.weight_scale, "CPU stored scale")
    # This comparison specifically qualifies the preplacement memory bridge:
    # source CPU packing must produce the same FP8 recipe as source CUDA packing.
    gpu_source = source.to(device)
    gpu_packed = linear_type(gpu_source, device=device)
    _equal_bytes(torch, cpu_packed.weight_fp8, gpu_packed.weight_fp8, "CPU/GPU packed weights")
    _equal_bytes(torch, cpu_packed.weight_scale, gpu_packed.weight_scale, "CPU/GPU weight scale")
    if bias:
        _equal_bytes(torch, cpu_packed.bias, gpu_packed.bias, "CPU/GPU bias")

    x = torch.randn((rows, width), device=device, dtype=torch.bfloat16)
    packed, scale = _reference_pack(torch, x)
    expected = torch._scaled_mm(
        packed, cpu_packed.weight_fp8.t(), scale_a=scale, scale_b=cpu_packed.weight_scale,
        bias=cpu_packed.bias, out_dtype=torch.bfloat16, use_fast_accum=False)
    actual = cpu_packed(x)
    _equal_bytes(torch, actual, expected, "projection recipe")
    if actual.dtype != torch.bfloat16 or not torch.isfinite(actual).all():
        raise AssertionError("projection did not return finite BF16 outputs")
    # Descriptive only: primitive quantization error is not task-quality evidence.
    native = gpu_source(x)
    diff = actual.float() - native.float()
    metrics = {"max_abs": diff.abs().max().item(),
               "relative_l2": (diff.norm() / native.float().norm().clamp_min(1e-12)).item()}

    empty = cpu_packed(x[:0])
    if empty.shape != (0, outputs) or empty.dtype != torch.bfloat16:
        raise AssertionError("empty token partition contract changed")
    _equal_bytes(torch, cpu_packed(x.reshape(1, rows, width)).reshape(rows, outputs),
                 actual, "rank-three input")
    _equal_bytes(torch, cpu_packed(x.t().contiguous().t()), actual, "strided input")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _equal_bytes(torch, cpu_packed(x.float()), actual, "native BF16 autocast")

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            cpu_packed(x)
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = cpu_packed(x)
    for magnitude in (0, 1e-4, 20):
        x.copy_(torch.randn_like(x) * magnitude)
        graph.replay()
        eager = cpu_packed(x)
        _equal_bytes(torch, captured, eager, "changed-input graph replay")
    torch.cuda.synchronize(device)
    return {"rows": rows, "in_features": width, "out_features": outputs, "bias": bias,
            "cpu_gpu_packing_equal": True, "reference_recipe_equal": True,
            "changed_input_graph_replay_equal": True,
            "bf16_difference_descriptive_only": metrics, "passed": True}


def _check_residency(torch, linear_type, device, report):
    from instinctflash.runtime.module_residency import ModuleResidency

    report["module_residency"] = []
    for precision in ("native", "fp8"):
        blocks = []
        for _ in range(2):
            linear = torch.nn.Linear(128, 128, dtype=torch.bfloat16)
            if precision == "fp8":
                linear = linear_type(linear, device=device, storage_device="cpu")
            blocks.append(torch.nn.Sequential(linear, torch.nn.GELU()))
        root = torch.nn.Sequential(*blocks).eval()
        native = copy.deepcopy(root).to(device=device)
        originals = list(root.parameters()) + list(root.buffers())
        budget = sum(t.numel() * t.element_size()
                     for t in list(blocks[0].parameters()) + list(blocks[0].buffers()))
        owner = ModuleResidency(root, blocks, device=device, budget_bytes=budget,
                                reserve_bytes=0)
        try:
            for magnitude in (0, 1e-4, 20):
                x = torch.randn((7, 128), device=device, dtype=torch.bfloat16) * magnitude
                _equal_bytes(torch, root(x), native(x), "streamed/resident module outputs")
                current = list(root.parameters()) + list(root.buffers())
                if any(a is not b or a.device.type != "cpu" for a, b in zip(current, originals)):
                    raise AssertionError("Residency did not restore its original CPU master tensors")
            report["module_residency"].append({"precision": precision, "passed": True,
                                                "residency": owner.report()})
        finally:
            owner.close()


def qualify(device, report):
    import torch

    report["runtime"] = {"python": sys.version, "python_executable": sys.executable,
                         "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                         "torch_git_version": torch.version.git_version}
    device = torch.device(device)
    if (device.type != "cuda" or not torch.cuda.is_available()
            or torch.cuda.get_device_capability(device) != (8, 9)):
        raise RuntimeError("Qualification requires a visible SM89 CUDA device")
    torch.cuda.set_device(device)
    import triton
    from instinctflash.runtime import fp8_pack, sm89_fp8, torch_fp8_linear, module_residency

    report["runtime"]["triton"] = triton.__version__
    props = torch.cuda.get_device_properties(device)
    report["device"] = {"name": props.name, "capability": [props.major, props.minor],
                        "total_memory_bytes": props.total_memory, "selected_device": str(device)}
    report["source"] = {
        Path(module.__file__).name: {"path": str(Path(module.__file__).resolve()),
                                    "sha256": _sha(module.__file__)}
        for module in (fp8_pack, sm89_fp8, torch_fp8_linear, module_residency)}
    report["recipe"] = torch_fp8_linear.SM89FP8Linear.recipe
    torch.manual_seed(731)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _check_packing(torch, fp8_pack.dynamic_pack_bf16_e4m3, device, report)
        report["projections"] = []
        for shape, bias in (((1, 16, 16), False), ((7, 768, 768), True),
                            ((40, 2048, 2048), False), ((16, 3072, 3072), True),
                            ((3, 4096, 1024), False), ((7, 5120, 13824), True)):
            report["projections"].append(_check_projection(
                torch, torch_fp8_linear.SM89FP8Linear, device, shape, bias))
            # CUDAGraph's private pool becomes reclaimable with the previous
            # case's local references; keep the largest case as the peak bound.
            torch.cuda.empty_cache()
        _check_residency(torch, torch_fp8_linear.SM89FP8Linear, device, report)
    report["memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True, help="new JSON receipt path")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new path to preserve earlier evidence")
    report = {"schema": "instinctflash.sm89_fp8_primitive_qualification.v1",
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "script_sha256": _sha(__file__), "passed": False,
              "scope": "synthetic primitives only", "task_quality_status": "unverified",
              "model_constructed": False, "model_weights_downloaded": False}
    try:
        qualify(args.device, report)
        report["passed"] = True
    except Exception as error:
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
