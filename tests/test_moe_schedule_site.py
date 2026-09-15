#!/usr/bin/env python3
"""The vla2 MoE launch-schedule site: the parts that hold without the fvk kernels.

The load-bearing distinction being pinned: `mode` is the SCALE GRID (a numerics decision made
at R3/M2d, maxabs 9.7-20.3 between grids) and `schedule` is the LAUNCH SHAPE (bit-identical at
a fixed grid, M2 unit tests maxabs_batched_vs_batch1_loop = 0.0). Only the second is
autotunable, and pairing the batched schedule with the per-expert grid -- one B-scale per
strided-batched launch descaling 32 experts -- must be refused, not served.

Needs a CUDA device for buffer allocation but NO flash_rt kernels (fvk_mod is only dereferenced
by routed_moe_fn). The kernel-level half -- real timings and the torch.equal verification -- is
serving/tests/probe_vla2_moe_schedule_autotune.py, run on Thor.

    CUDA_VISIBLE_DEVICES=7 python tests/test_moe_schedule_site.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "serving")]

import pytest  # noqa: E402

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device for the engine's buffer allocation (no kernels needed)",
                allow_module_level=True)

from flash_rt.models.vla2.moe_engine import (  # noqa: E402
    E, HIDDEN, MOE_H, Vla2MoeEngine, moe_schedule_site, resolve_moe_schedule,
)
from instinctflash.passes.contract import Tier  # noqa: E402


def _layers(n=1):
    g = torch.Generator().manual_seed(7)
    return [{
        "gate.weight": torch.randn(E, HIDDEN, generator=g),
        "e_score_correction_bias": torch.randn(E, generator=g),
        "experts.gate_proj": torch.randn(E, MOE_H, HIDDEN, generator=g) * 0.02,
        "experts.up_proj": torch.randn(E, MOE_H, HIDDEN, generator=g) * 0.02,
        "experts.down_proj": torch.randn(E, HIDDEN, MOE_H, generator=g) * 0.02,
    } for _ in range(n)]


def test_schedule_follows_mode_by_default():
    b = Vla2MoeEngine.from_fp32_layers(None, _layers(), mode="batched")
    assert b.schedule == "batched" and b._gu_scale.dim() == 1
    lo = Vla2MoeEngine.from_fp32_layers(None, _layers(), mode="loop")
    assert lo.schedule == "loop" and lo._gu_scale.dim() == 2, \
        "default must keep yesterday's coupling bit-for-bit"


def test_batched_schedule_on_per_expert_grid_is_refused():
    try:
        Vla2MoeEngine.from_fp32_layers(None, _layers(), mode="loop", schedule="batched")
    except ValueError as e:
        assert "B-scale" in str(e), e
    else:
        raise AssertionError("batched launches over per-expert scales must be refused: "
                             "one descale per launch cannot serve 32 grids")


def test_loop_schedule_on_shared_grid_reads_the_shared_slot():
    eng = Vla2MoeEngine.from_fp32_layers(None, _layers(), mode="batched", schedule="loop")
    assert eng.schedule == "loop"
    base = eng._gu_scale.data_ptr()
    # every expert of layer 0 reads the SAME slot -- that is what makes the swap bit-identical
    assert {eng._gu_scale_ptr(0, e) for e in range(E)} == {base}
    per = Vla2MoeEngine.from_fp32_layers(None, _layers(), mode="loop")
    pbase = per._gu_scale.data_ptr()
    assert per._gu_scale_ptr(0, 3) == pbase + 3 * 4
    assert per._dn_scale_ptr(0, 5) == per._dn_scale.data_ptr() + 5 * 4


def test_auto_on_per_expert_grid_is_loop_without_a_bench():
    eng = Vla2MoeEngine.from_fp32_layers(None, _layers(), mode="loop", schedule="auto")
    assert eng.schedule == "loop", "there is nothing to tune on the per-expert grid"
    assert not hasattr(eng, "autotune_decision"), "no bench may have run"


def test_site_declaration_names_its_evidence():
    eng = Vla2MoeEngine.from_fp32_layers(None, _layers(), mode="batched")
    site = moe_schedule_site(eng)
    assert site.baseline == "batched"
    loop = site.candidate("loop")
    assert loop.tier is Tier.BITEXACT and site.candidate("batched").tier is Tier.BITEXACT
    for needle in ("maxabs_batched_vs_batch1_loop = 0.0", "scale-mode delta", "torch.equal"):
        assert needle in loop.evidence, f"evidence must name the M2 measurement ({needle!r})"
    assert "grid=shared" in site.shape_signature
    # resolve on the shared grid without kernels: the bench raises per candidate, and the
    # runner's contract is that a candidate that cannot run loses loudly -- the baseline stands.
    import os
    import tempfile
    os.environ["IFL_AUTOTUNE_CACHE"] = tempfile.mktemp(suffix=".json")
    try:
        chosen = resolve_moe_schedule(eng)
        assert chosen == "batched", chosen
        assert eng.autotune_decision.source == "verify-failed", eng.autotune_decision
    finally:
        del os.environ["IFL_AUTOTUNE_CACHE"]


if __name__ == "__main__":
    from run_tests import run_module_tests
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(run_module_tests(globals()))
