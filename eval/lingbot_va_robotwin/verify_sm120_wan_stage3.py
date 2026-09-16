#!/usr/bin/env python3
"""RTX 5090 operator gate for P009-A3 exact norm1 + Ada modulation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from instinctflash.backends.sm120_wan_stage3 import SM120WanStage3Kernels

DIM = 3072
EPS = 1e-6


def eager(hidden, scale, shift):
    normed, mean, rstd = torch.native_layer_norm(
        hidden.float(), (DIM,), None, None, EPS)
    output = (normed * (1.0 + scale) + shift).to(hidden.dtype)
    return output, mean.reshape(-1), rstd.reshape(-1)


def timed(fn, warm=20, iters=200):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def one_case(kernels, rows, pattern, exponent):
    gen = torch.Generator(device="cuda").manual_seed(1000 + rows * 7 + exponent)
    if pattern == "random":
        hidden = torch.randn(1, rows, DIM, device="cuda", dtype=torch.bfloat16,
                             generator=gen)
    elif pattern == "constant":
        hidden = torch.full((1, rows, DIM), 2.0 ** exponent,
                            device="cuda", dtype=torch.bfloat16)
    elif pattern == "alternating":
        values = torch.empty(rows * DIM, device="cuda", dtype=torch.float32)
        values[0::2] = 2.0 ** exponent
        values[1::2] = -(2.0 ** exponent)
        hidden = values.reshape(1, rows, DIM).to(torch.bfloat16)
    else:
        raise ValueError(pattern)

    styles = torch.randn(1, rows, 6, DIM, device="cuda", dtype=torch.float32,
                         generator=gen) * 0.05
    shift = styles[:, :, 0, :]
    scale = styles[:, :, 1, :]
    output = torch.empty_like(hidden)
    means = torch.empty(rows, device="cuda", dtype=torch.float32)
    rstds = torch.empty_like(means)

    ref, ref_mean, ref_rstd = eager(hidden, scale, shift)
    kernels.norm1_ada_into(hidden, scale, shift, output, means, rstds)
    torch.cuda.synchronize()

    record = {
        "rows": rows,
        "pattern": pattern,
        "exponent": exponent,
        "style_row_stride": scale.stride(-2),
        "output": {
            "exact": torch.equal(ref, output),
            "differing_words": int(torch.count_nonzero(ref.view(torch.int16) != output.view(torch.int16))),
            "max_abs": float((ref.float() - output.float()).abs().max()),
        },
        "mean": {
            "exact": torch.equal(ref_mean, means),
            "differing_words": int(torch.count_nonzero(ref_mean.view(torch.int32) != means.view(torch.int32))),
            "max_abs": float((ref_mean - means).abs().max()),
        },
        "rstd": {
            "exact": torch.equal(ref_rstd, rstds),
            "differing_words": int(torch.count_nonzero(ref_rstd.view(torch.int32) != rstds.view(torch.int32))),
            "max_abs": float((ref_rstd - rstds).abs().max()),
        },
    }
    if pattern == "random" and exponent == 0:
        eager_ms = timed(lambda: eager(hidden, scale, shift)[0])
        fused_ms = timed(
            lambda: kernels.norm1_ada_certified(
                hidden, scale, shift, output, means, rstds))
        record["timing"] = {
            "eager_ms": eager_ms,
            "fused_ms": fused_ms,
            "speedup": eager_ms / fused_ms,
        }
    return record


def invalid_gates(kernels):
    hidden = torch.zeros(1, 64, DIM, device="cuda", dtype=torch.bfloat16)
    styles = torch.zeros(1, 64, 6, DIM, device="cuda", dtype=torch.float32)
    shift, scale = styles[:, :, 0, :], styles[:, :, 1, :]
    output = torch.empty_like(hidden)
    means = torch.empty(64, device="cuda", dtype=torch.float32)
    rstds = torch.empty_like(means)
    cases = [
        ("dtype", (hidden, scale.to(torch.bfloat16), shift, output, means, rstds)),
        ("shape", (hidden[:, :63], scale[:, :63], shift[:, :63], output[:, :63],
                   means[:63], rstds[:63])),
        ("alias", (hidden, scale, shift, hidden, means, rstds)),
    ]
    out = []
    for label, args in cases:
        try:
            kernels.norm1_ada_into(*args)
        except (TypeError, ValueError, RuntimeError) as error:
            out.append({"label": label, "rejected": True, "error": str(error)})
        else:
            out.append({"label": label, "rejected": False})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kernels = SM120WanStage3Kernels(args.library)
    records = []
    for rows in (64, 480):
        for pattern in ("random", "constant", "alternating"):
            for exponent in (-4, 0, 4):
                records.append(one_case(kernels, rows, pattern, exponent))
    guards = invalid_gates(kernels)
    all_exact = all(
        row[name]["exact"] for row in records for name in ("output", "mean", "rstd"))
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "library": str(args.library.resolve()),
        "records": records,
        "guards": guards,
        "kernel_calls": kernels.calls,
        "all_bitexact": all_exact,
        "all_invalid_inputs_rejected": all(row["rejected"] for row in guards),
    }
    result["status"] = (
        "pass" if result["all_bitexact"] and result["all_invalid_inputs_rejected"]
        else "fail")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
