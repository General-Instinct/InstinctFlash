#!/usr/bin/env python3
"""Formal operator and performance gate for P009-A7 parallel Q/K/V."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from instinctflash.backends.sm120_wan_qkv_parallel import (
    CERTIFIED_CONFIGS,
    SM120WanParallelQKVKernels,
)


def fill(linears, pattern, exponent, m, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scale = 2.0**exponent
    if pattern == "random":
        x = torch.randn(
            m, 3072, device="cuda", dtype=torch.bfloat16, generator=generator
        ).mul_(scale)
    elif pattern == "constant":
        x = torch.full((m, 3072), 0.25 * scale, device="cuda", dtype=torch.bfloat16)
    else:
        signs = (torch.arange(3072, device="cuda") % 2 * 2 - 1).to(torch.bfloat16)
        x = signs.expand(m, 3072).clone().mul_(scale)
    with torch.no_grad():
        for linear in linears:
            linear.weight.copy_(
                torch.randn(
                    3072,
                    3072,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                ).mul_(0.125)
            )
            linear.bias.copy_(
                torch.randn(
                    3072,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
            )
    return x


def bench(reference, candidate):
    for _ in range(15):
        reference()
        candidate()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)

    def arm(function):
        values = []
        for _ in range(31):
            begin.record()
            for _ in range(30):
                function()
            end.record()
            end.synchronize()
            values.append(begin.elapsed_time(end) * 1000 / 30)
        return {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "spread_percent": (max(values) - min(values))
            / statistics.mean(values)
            * 100,
        }

    baseline_a = arm(reference)
    candidate_b1 = arm(candidate)
    candidate_b2 = arm(candidate)
    baseline_a2 = arm(reference)
    baseline = (baseline_a["median_us"] + baseline_a2["median_us"]) / 2
    treatment = (candidate_b1["median_us"] + candidate_b2["median_us"]) / 2
    return {
        "baseline_arms": [baseline_a, baseline_a2],
        "candidate_arms": [candidate_b1, candidate_b2],
        "baseline_mean_us": baseline,
        "candidate_mean_us": treatment,
        "delta_us": baseline - treatment,
        "speedup": baseline / treatment,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kernels = SM120WanParallelQKVKernels(args.library)
    records, timings = [], {}
    compared = differences = 0
    for m in (64, 480):
        linears = tuple(
            torch.nn.Linear(3072, 3072, device="cuda", dtype=torch.bfloat16)
            .eval()
            .requires_grad_(False)
            for _ in range(3)
        )
        attention = SimpleNamespace(to_q=linears[0], to_k=linears[1], to_v=linears[2])
        for pattern_index, pattern in enumerate(("random", "constant", "alternating")):
            for exponent in (-4, 0, 4):
                x = fill(
                    linears,
                    pattern,
                    exponent,
                    m,
                    7000 + m + pattern_index * 10 + exponent,
                )
                plan = kernels.register_attention(attention, m)
                outputs = tuple(
                    torch.empty(m, 3072, device="cuda", dtype=torch.bfloat16)
                    for _ in range(3)
                )
                references = tuple(
                    F.linear(x, linear.weight, linear.bias) for linear in linears
                )
                kernels.project_into(plan, x, linears, outputs)
                torch.cuda.synchronize()
                per_output = [
                    int((a.view(torch.int16) != b.view(torch.int16)).sum())
                    for a, b in zip(references, outputs)
                ]
                first = tuple(output.clone() for output in outputs)
                deterministic = True
                for _ in range(4):
                    kernels.project_into(plan, x, linears, outputs)
                    torch.cuda.synchronize()
                    deterministic &= all(
                        torch.equal(a.view(torch.int16), b.view(torch.int16))
                        for a, b in zip(first, outputs)
                    )
                words = sum(reference.numel() for reference in references)
                row = {
                    "M": m,
                    "pattern": pattern,
                    "exponent": exponent,
                    "words": words,
                    "differences": per_output,
                    "deterministic": deterministic,
                }
                records.append(row)
                compared += words
                differences += sum(per_output)
                print(row, flush=True)
                if pattern == "random" and exponent == 0:
                    timings[str(m)] = bench(
                        lambda x=x, linears=linears: tuple(
                            F.linear(x, linear.weight, linear.bias)
                            for linear in linears
                        ),
                        lambda plan=plan, x=x, linears=linears, outputs=outputs: (
                            kernels.project_into(plan, x, linears, outputs)
                        ),
                    )
                    print("TIMING", m, timings[str(m)], flush=True)
    guard_rows = []
    guard_cases = (
        (
            "wrong_dtype",
            lambda: kernels.project_into(plan, x.float(), linears, outputs),
        ),
        ("wrong_rows", lambda: kernels.project_into(plan, x[:-1], linears, outputs)),
        (
            "aliased_outputs",
            lambda: kernels.project_into(
                plan, x, linears, (outputs[0], outputs[0], outputs[2])
            ),
        ),
    )
    for label, function in guard_cases:
        try:
            function()
        except (TypeError, ValueError, RuntimeError) as error:
            guard_rows.append({"label": label, "rejected": True, "error": str(error)})
        else:
            guard_rows.append({"label": label, "rejected": False})
    with torch.no_grad():
        linears[0].weight.add_(0)
    try:
        kernels.project_into(plan, x, linears, outputs)
    except (TypeError, ValueError, RuntimeError) as error:
        guard_rows.append(
            {"label": "mutated_weight", "rejected": True, "error": str(error)}
        )
    else:
        guard_rows.append({"label": "mutated_weight", "rejected": False})

    source = (
        Path(__file__).resolve().parents[2]
        / "instinctflash/native/wan_qkv_parallel_sm120.cu"
    )
    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tier": "BITEXACT",
        "configs": {str(key): list(value) for key, value in CERTIFIED_CONFIGS.items()},
        "compared_words": compared,
        "differing_words": differences,
        "all_bitexact": differences == 0,
        "all_deterministic": all(row["deterministic"] for row in records),
        "records": records,
        "guards": guard_rows,
        "all_invalid_inputs_rejected": all(row["rejected"] for row in guard_rows),
        "timings": timings,
        "kernel_calls": dict(kernels.calls),
        "artifacts": {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        },
    }
    result["status"] = (
        "pass"
        if result["all_bitexact"]
        and result["all_deterministic"]
        and result["all_invalid_inputs_rejected"]
        else "fail"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
