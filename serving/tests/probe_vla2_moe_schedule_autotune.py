#!/usr/bin/env python3
"""GPU smoke for the vla2_moe_schedule autotune site. Runs ON THOR, under the GPU lock.

    flock /tmp/thor_gpu.lock -c "PYTHONPATH=<flash_rt>:<instinctflash> \
        python serving/tests/probe_vla2_moe_schedule_autotune.py"

Four claims, measured:
  1. BIT-IDENTITY on this device: loop and batched launches over the same fp8 bytes and the
     same shared scales produce torch.equal outputs (the M2 unit-test fact, re-established
     live -- the site's verify() hook refuses the swap otherwise, so this doubles as the gate).
  2. FIRST LOAD MEASURES: schedule="auto" benches both launch shapes and reports medians.
  3. SECOND LOAD IS A CACHE HIT: same key, same winner, no kernels timed.
  4. THE OVERRIDE IS REAL: IFL_AUTOTUNE_VLA2_MOE_SCHEDULE=loop serves the loop schedule.

Random fp32 weights at the real shapes (E=32, T=51, moe_h=512, h=768); LAYERS below keeps the
burst short. Exit 0 only if all four hold.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

LAYERS = 6


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    import flash_rt.flash_rt_kernels as fvk
    from flash_rt.models.vla2.moe_engine import (
        E, HIDDEN, MOE_H, Vla2MoeEngine, quantize_fp8_pt,
    )

    cache = Path(tempfile.mkdtemp()) / "autotune.json"
    os.environ["IFL_AUTOTUNE_CACHE"] = str(cache)
    os.environ.pop("IFL_AUTOTUNE", None)
    os.environ.pop("IFL_AUTOTUNE_VLA2_MOE_SCHEDULE", None)

    g = torch.Generator().manual_seed(11)
    layers = [{
        "gate.weight": torch.randn(E, HIDDEN, generator=g),
        "e_score_correction_bias": torch.randn(E, generator=g),
        "experts.gate_proj": torch.randn(E, MOE_H, HIDDEN, generator=g) * 0.02,
        "experts.up_proj": torch.randn(E, MOE_H, HIDDEN, generator=g) * 0.02,
        "experts.down_proj": torch.randn(E, HIDDEN, MOE_H, generator=g) * 0.02,
    } for _ in range(LAYERS)]

    failures = []
    dev = torch.cuda.get_device_properties(0)
    print(f"device: {dev.name} sm{dev.major}{dev.minor}, layers={LAYERS}, T=51")

    # descriptor caches build on first call; one eager pass per shape before anything is timed
    # (the vla4b/_capture_graphs discipline).
    eng = Vla2MoeEngine.from_fp32_layers(fvk, layers, mode="batched")
    x16 = torch.randn(51, HIDDEN, device="cuda", dtype=torch.float16)
    xq, _ = quantize_fp8_pt(x16.float())
    x = xq.cuda().contiguous()
    out = torch.zeros(51, HIDDEN, dtype=torch.float16, device="cuda")

    def run_all(schedule):
        eng.schedule = schedule
        for l in range(LAYERS):
            eng.routed_moe_fn(l, 0, x.data_ptr(), out.data_ptr(), 0)
        torch.cuda.synchronize()
        return out.clone()

    # 1. bit-identity, the headline
    a = run_all("batched")
    b = run_all("loop")
    ident = torch.equal(a, b)
    print(f"  bit-identity (batched vs loop, shared grid): "
          f"{'torch.equal = True' if ident else 'FAILED, max|d| = %g' % (a - b).abs().max()}")
    if not ident:
        failures.append("loop vs batched not bit-identical at fixed scales")

    # 2. first load measures (schedule='auto' resolves through the autotune runner)
    eng2 = Vla2MoeEngine.from_fp32_layers(fvk, layers, mode="batched", schedule="auto")
    d1 = eng2.autotune_decision
    print(f"  first : {d1.reason}")
    print(f"          timings_ms (full {LAYERS}-layer pass) = "
          f"{ {k: round(v, 3) for k, v in d1.timings_ms.items()} }")
    if d1.source != "measured":
        failures.append(f"first load should measure, got {d1.source}")
    if eng2.schedule != d1.chosen:
        failures.append("engine schedule does not match the decision")

    # 3. second load hits the cache
    eng3 = Vla2MoeEngine.from_fp32_layers(fvk, layers, mode="batched", schedule="auto")
    d2 = eng3.autotune_decision
    print(f"  second: {d2.reason}")
    if d2.source != "cache" or d2.chosen != d1.chosen:
        failures.append(f"second load should be a cache hit on the same winner, got "
                        f"{d2.source}/{d2.chosen}")

    # 4. override
    os.environ["IFL_AUTOTUNE_VLA2_MOE_SCHEDULE"] = "loop"
    eng4 = Vla2MoeEngine.from_fp32_layers(fvk, layers, mode="batched", schedule="auto")
    print(f"  forced: {eng4.autotune_decision.reason}")
    if eng4.schedule != "loop" or eng4.autotune_decision.source != "forced":
        failures.append(f"override failed: {eng4.schedule}/{eng4.autotune_decision.source}")
    del os.environ["IFL_AUTOTUNE_VLA2_MOE_SCHEDULE"]

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS: bit-identical, measured, cached, overridable on this device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
