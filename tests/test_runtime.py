"""Tests for plan installation.

The property under test is narrow and important: a server must never come up claiming a pass
that was not applied. `plan.explain()` is what every measurement is labelled with, so an
installer that silently skips a pass turns a real number into a mislabelled one.

Skipped when torch is absent — the runtime layer needs it, the rest of the package does not.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instinctflash import Optimizer, Tier, load  # noqa: E402  (needs the insert above)

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_TORCH = False


class _FakeServerModule:
    """Enough of `wan_va_server` for the installers to bind to."""

    def __init__(self):
        self.save_async = lambda obj, path: ("wrote", path)
        self._configure_model = lambda *a, **k: None
        self.VA_Server = type("VA_Server", (), {
            "_infer": lambda self, obs, frame_st_id=0: None,
            "_reset": lambda self, prompt=None: None,
            "_compute_kv_cache": lambda self, obs: None,
        })


def test_install_plan_applies_the_bitexact_substrate_passes():
    if not HAVE_TORCH:
        return
    from instinctflash.runtime.lingbot_install import install_plan

    model = load("lingbot-va-posttrain-robotwin")
    plan = Optimizer(tier_ceiling=Tier.BITEXACT).compile(model.spec())
    server = _FakeServerModule()

    # The three passes that patch the upstream model classes import `modules.model`, which this fake
    # module does not provide; they are exercised against the real tree elsewhere
    # (test_action_terminal_elision.py installs P010 on the real classes).
    applied = install_plan(
        server, server.VA_Server,
        plan.without("conditioning_prefill", "ring_kv_addressing", "action_terminal_forward_elision"),
    )

    assert "fsdp_elision" in applied
    assert "debug_dump_elision" in applied
    assert "allocator_churn_elision" in applied
    # obs_decode_elision installs a constructor wrapper; this fake is deliberately not built.
    assert any(a.startswith("obs_decode_elision") for a in applied)
    assert server.save_async(None, "x") is None, "debug dump should be neutered"


def test_install_plan_refuses_a_pass_it_cannot_install():
    if not HAVE_TORCH:
        return
    from instinctflash.runtime.lingbot_install import install_plan

    model = load("lingbot-va-posttrain-robotwin")
    # cfg_branch_elision is analysed but has no installer yet.
    plan = Optimizer(tier_ceiling=Tier.NUMERIC).compile(model.spec())
    assert "cfg_branch_elision" in [r.name for r in plan.applied]

    server = _FakeServerModule()
    try:
        install_plan(server, server.VA_Server, plan)
    except NotImplementedError as exc:
        assert "cfg_branch_elision" in str(exc)
        assert "plan.without(" in str(exc)
        return
    raise AssertionError("install_plan must refuse a plan it cannot fully apply")


def test_installers_refuse_a_server_that_changed_shape():
    if not HAVE_TORCH:
        return
    from instinctflash.runtime.lingbot_install import install_debug_dump_elision

    class _Renamed:
        pass

    try:
        install_debug_dump_elision(_Renamed())
    except RuntimeError as exc:
        assert "save_async" in str(exc)
        return
    raise AssertionError("an installer must raise rather than no-op on an unknown server")


def test_obs_decode_elision_is_plan_scoped_and_one_shot():
    """The stripping must reach exactly the next server built by the arming thread's plan.

    A permanent __init__ rebind meant one actions-only plan stripped the decoders of EVERY
    later VA_Server in the process -- including a want_pixels Runtime whose plan declined the
    pass, which then raised at the exact boundary it paid to keep.
    """
    if not HAVE_TORCH:
        return
    import torch

    from instinctflash.runtime.lingbot_install import install_obs_decode_elision

    class _VAE:
        def __init__(self):
            self.decoder = torch.nn.Linear(2, 2)

    class _Wrapper:
        def __init__(self):
            self.vae = _VAE()

    class _Server:
        def __init__(self):
            self.streaming_vae = _Wrapper()
            self.streaming_vae_half = _Wrapper()

    module = _FakeServerModule()
    stripped = lambda s: type(s.streaming_vae.vae.decoder).__name__ == "_ElidedObservationDecoder"

    install_obs_decode_elision(module, _Server)
    assert stripped(_Server()), "the armed plan's server must be stripped"
    assert not stripped(_Server()), "a later, unarmed server must keep its decoders"
    install_obs_decode_elision(module, _Server)
    assert stripped(_Server()), "re-arming must strip exactly the next server again"
    assert not stripped(_Server())


def test_prompt_encoder_staging_is_a_plan_line_with_enforced_pairing():
    """Constrained-memory T5 staging must be declared, not a silent device sniff."""
    if not HAVE_TORCH:
        return
    from instinctflash.descriptors.deployment import DeploymentSpec
    from instinctflash.passes.contract import DeviceProfile
    from instinctflash.runtime.lingbot_install import (
        _PROMPT_ENCODER_STAGING,
        install_plan,
    )

    spec = load("lingbot-va-posttrain-robotwin").spec()

    def compiled(capability, memory):
        device = DeviceProfile(name="synthetic", capability=capability,
                               total_memory=memory, features=frozenset({"cuda"}))
        return Optimizer(tier_ceiling=Tier.BITEXACT).compile(
            spec, DeploymentSpec(device=device))

    by_name = {r.name: r for r in compiled((12, 0), 32 << 30).results}
    assert by_name["prompt_encoder_staging"].applies, "low-memory SM120 must declare staging"
    assert "empty_cache" in by_name["prompt_encoder_staging"].params["action"]
    by_name = {r.name: r for r in compiled((8, 9), 24 << 30).results}
    assert by_name["prompt_encoder_staging"].applies
    assert not by_name["allocator_churn_elision"].applies
    assert by_name["obs_decode_elision"].applies
    by_name = {r.name: r for r in compiled((9, 0), 80 << 30).results}
    assert not by_name["prompt_encoder_staging"].applies, "sm90 must keep the encoder resident"
    deviceless = {r.name: r for r in
                  Optimizer(tier_ceiling=Tier.BITEXACT).compile(spec, DeploymentSpec()).results}
    assert not deviceless["prompt_encoder_staging"].applies, "unprobed target must decline"

    # the mechanism lives in conditioning_prefill's reset wrapper; the pairing is enforced
    plan = compiled((12, 0), 32 << 30).without("conditioning_prefill")
    try:
        install_plan(_FakeServerModule(), type("VA", (), {}), plan)
    except RuntimeError as exc:
        assert "requires conditioning_prefill" in str(exc)
    else:
        raise AssertionError("install_plan must refuse staging without conditioning_prefill")

    # a plan that dropped staging disarms the one-shot token for the next server
    _PROMPT_ENCODER_STAGING.pending = True
    plan = Optimizer(tier_ceiling=Tier.BITEXACT).compile(spec, DeploymentSpec())
    try:
        install_plan(_FakeServerModule(), type("VA", (), {}), plan.without("ring_kv_addressing"))
    except Exception:
        pass  # later installers may need the upstream tree; the disarm precedes them
    assert getattr(_PROMPT_ENCODER_STAGING, "pending", None) is False


def test_resolve_lingbot_root_reports_what_is_missing():
    if not HAVE_TORCH:
        return
    from instinctflash.runtime.lingbot_install import resolve_lingbot_root

    try:
        resolve_lingbot_root("/definitely/not/here")
    except FileNotFoundError as exc:
        assert "LINGBOT_ROOT" in str(exc)
        return
    raise AssertionError("resolve_lingbot_root should raise on a missing checkout")


if __name__ == "__main__":
    # Script-style entry, matching the rest of this directory. pytest still collects the
    # test_* functions above directly.
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
