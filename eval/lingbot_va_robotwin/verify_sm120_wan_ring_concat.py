#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from instinctflash.backends.sm120_wan_ring_concat import (
    CERTIFIED_BATCH as B,
)
from instinctflash.backends.sm120_wan_ring_concat import (
    CERTIFIED_HEAD_DIM as D,
)
from instinctflash.backends.sm120_wan_ring_concat import (
    CERTIFIED_HEADS as H,
)
from instinctflash.backends.sm120_wan_ring_concat import (
    CERTIFIED_TOTAL as T,
)
from instinctflash.backends.sm120_wan_ring_concat import (
    SM120WanRingConcatKernels,
)

parser = argparse.ArgumentParser()
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
kernels = SM120WanRingConcatKernels(args.library)
generator = torch.Generator(device="cuda").manual_seed(5090)
key = torch.randn(
    (B, T, H, D), device="cuda", dtype=torch.bfloat16, generator=generator
)
value = torch.randn(
    (B, T, H, D), device="cuda", dtype=torch.bfloat16, generator=generator
)
gate_records = []
timing_records = {}


def candidate(start, count):
    return kernels.concat(key, value, start=start, count=count, total=T)


def reference(start, count):
    end = start + count - T
    return (
        torch.cat((key[:, :end], key[:, start:]), dim=1),
        torch.cat((value[:, :end], value[:, start:]), dim=1),
    )


for start, count in ((9000, 1000), (7000, 4000), (5000, 7000), (2000, 9000)):
    rk, rv = reference(start, count)
    ck, cv = candidate(start, count)
    torch.cuda.synchronize()
    dk = int((rk.view(torch.int16) != ck.view(torch.int16)).sum())
    dv = int((rv.view(torch.int16) != cv.view(torch.int16)).sum())
    record = {
        "start": start,
        "count": count,
        "key_differences": dk,
        "value_differences": dv,
    }
    gate_records.append(record)
    print("gate", record, flush=True)
    assert dk == 0 and dv == 0


def bench(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    vals = []
    for _ in range(31):
        begin.record()
        for _ in range(20):
            fn()
        end.record()
        end.synchronize()
        vals.append(begin.elapsed_time(end) / 20)
    return {
        "p50_ms": statistics.median(vals),
        "min_ms": min(vals),
        "spread_pct": (max(vals) - min(vals)) / statistics.mean(vals) * 100,
    }


for start, count in ((9000, 1000), (7000, 4000), (5000, 7000), (2000, 9000)):
    b1 = bench(lambda start=start, count=count: reference(start, count))
    c1 = bench(lambda start=start, count=count: candidate(start, count))
    c2 = bench(lambda start=start, count=count: candidate(start, count))
    b2 = bench(lambda start=start, count=count: reference(start, count))
    bm = (b1["p50_ms"] + b2["p50_ms"]) / 2
    cm = (c1["p50_ms"] + c2["p50_ms"]) / 2
    record = {
        "baseline_ms": bm,
        "candidate_ms": cm,
        "speedup": bm / cm,
        "baseline_arms": [b1, b2],
        "candidate_arms": [c1, c2],
    }
    timing_records[str(count)] = record
    print("timing", start, count, record, flush=True)

source = (
    Path(__file__).resolve().parents[2]
    / "instinctflash/native/wan_ring_concat_sm120.cu"
)
result = {
    "schema_version": 1,
    "device": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "tier": "BITEXACT",
    "gates": gate_records,
    "all_bitexact": all(
        row["key_differences"] == 0 and row["value_differences"] == 0
        for row in gate_records
    ),
    "timings": timing_records,
    "kernel_calls": dict(kernels.calls),
    "artifacts": {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
    },
}
result["status"] = "pass" if result["all_bitexact"] else "fail"
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
raise SystemExit(0 if result["status"] == "pass" else 1)
