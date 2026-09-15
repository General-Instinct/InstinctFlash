#!/usr/bin/env python3
"""The VA conv-layout autotune site, CPU-checkable parts.

The GPU half -- that the bench measures real kernels and the cache round-trips on real silicon
-- is eval/lingbot_va_robotwin/probe_conv_autotune.py, run on one idle GPU. What is pinned here
is the wiring that must hold everywhere: the site's declaration (NUMERIC candidate carries the
555-episode certificate as evidence, the baseline is the stock layout), the Decision -> ConvPlan
mapping, the no-CUDA fallback, and the plan surfacing with its tier consequence.

    python tests/test_conv_autotune_site.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402  (module under test imports it; a torch-less env skips this file)

from instinctflash.autotune import Decision, SITES, record_decision  # noqa: E402
import instinctflash.backends.conv.apply as conv_apply  # noqa: E402
from instinctflash.backends.conv.apply import (  # noqa: E402
    CONV_LAYOUT_SITE, autotune_conv_layout, conv_plan_from_decision,
)
from instinctflash.backends.conv.semantics import MemoryLayout  # noqa: E402
from instinctflash.passes.contract import Tier  # noqa: E402
from instinctflash.planners.planner import Plan  # noqa: E402
from instinctflash.runtime.lingbot_install import install_conv_layout_autotune  # noqa: E402


def test_site_declaration():
    assert SITES["va_conv_layout"] is CONV_LAYOUT_SITE
    assert CONV_LAYOUT_SITE.baseline == "stock"
    ndhwc = CONV_LAYOUT_SITE.candidate("ndhwc")
    assert ndhwc.tier is Tier.NUMERIC
    # The tier claim must carry its certificate, not a vibe: the 555-episode numbers and the
    # sm_90 provenance are what a reader needs to judge whether the claim transfers.
    for needle in ("555", "McNemar", "sm_90", "does not transfer"):
        assert needle in ndhwc.evidence, f"evidence must name the certificate ({needle!r} missing)"
    assert CONV_LAYOUT_SITE.candidate("stock").tier is Tier.BITEXACT


def test_decision_to_conv_plan_mapping():
    d = Decision("va_conv_layout", "ndhwc", "stock", 10.3, Tier.NUMERIC, "measured",
                 reason="autotuned: ...")
    p = conv_plan_from_decision(d)
    assert p.backend_name == "cudnn_conv3d" and p.use_layout is MemoryLayout.NDHWC
    assert p.convert_subgraph and p.tier is Tier.NUMERIC

    d2 = Decision("va_conv_layout", "stock", "", 1.0, Tier.BITEXACT, "disabled", reason="...")
    p2 = conv_plan_from_decision(d2)
    assert not p2.convert_subgraph and p2.tier is Tier.BITEXACT
    assert p2.backend_name == "torch_fallback" and p2.use_layout is MemoryLayout.NCDHW


def test_no_cuda_keeps_the_incumbent():
    if torch.cuda.is_available():
        print("skip: this asserts the no-CUDA path and a GPU is visible")
        return
    d = autotune_conv_layout()
    assert d.chosen == "stock" and d.source == "no-device", d
    assert "incumbent" in d.reason


def test_plan_surfacing_prices_the_swap():
    plan = Plan("lingbot-va", [])
    d = Decision("va_conv_layout", "ndhwc", "stock", 1.3, Tier.NUMERIC, "cache",
                 reason="autotuned: va_conv_layout chose ndhwc over stock (1.30x), "
                        "equivalence NUMERIC [cache]")
    record_decision(plan, d)
    assert plan.tier() is Tier.NUMERIC, "a NUMERIC layout swap must make the plan NUMERIC"
    text = plan.explain()
    assert "autotune:va_conv_layout" in text and "equivalence NUMERIC" in text
    # and the explain NOTE machinery fires, telling the reader what a claim now costs
    assert "paired" in text and "non-inferiority" in text.lower()


def test_installer_activation_is_plan_scoped():
    calls = []
    old = conv_apply.install_conv_layout
    try:
        conv_apply.install_conv_layout = (
            lambda server, **kwargs: calls.append(server) or ["synthetic decision"]
        )

        class VA:
            pass

        assert install_conv_layout_autotune(object(), VA) == ["conv_layout_ndhwc"]
        enabled = VA()
        assert calls == [enabled]

        # This later plan excluded P007 and therefore never calls its installer.
        excluded = VA()
        assert calls == [enabled]

        assert install_conv_layout_autotune(object(), VA) == ["conv_layout_ndhwc"]
        reenabled = VA()
        assert calls == [enabled, reenabled]
    finally:
        conv_apply.install_conv_layout = old


def test_missing_vae_is_a_loud_install_failure():
    old = conv_apply.autotune_conv_layout
    conv_apply.autotune_conv_layout = lambda **kwargs: Decision(
        "va_conv_layout", "stock", "", 1.0, Tier.BITEXACT, "disabled", reason="test"
    )
    try:
        try:
            conv_apply.install_conv_layout(SimpleNamespace())
        except RuntimeError as exc:
            assert "neither streaming_vae nor streaming_vae_half" in str(exc)
        else:
            raise AssertionError("an applied P007 plan must not silently miss every VAE")
    finally:
        conv_apply.autotune_conv_layout = old


def test_foreign_device_profile_is_refused_and_cache_identity_is_versioned():
    if not torch.cuda.is_available():
        print("skip: device/cache identity guard needs a visible CUDA device")
        return
    from instinctflash.passes.contract import DeviceProfile

    actual = DeviceProfile.probe()
    foreign = DeviceProfile(
        name=actual.name + "-other", capability=actual.capability,
        total_memory=actual.total_memory, features=actual.features,
    )
    try:
        autotune_conv_layout(model_id="foreign-profile-test", device=foreign)
    except RuntimeError as exc:
        assert "measuring one GPU under another GPU's cache key is refused" in str(exc)
    else:
        raise AssertionError("P007 measured the current GPU under a foreign cache key")

    ident = conv_apply._p007_runtime_identity(torch.cuda.current_device())
    for needle in ("p007abi=", "gpu=", "driver=", "torch=", "cuda=", "cudnn="):
        assert needle in ident


if __name__ == "__main__":
    from run_tests import run_module_tests
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(run_module_tests(globals()))
