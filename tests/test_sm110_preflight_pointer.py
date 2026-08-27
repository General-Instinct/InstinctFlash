#!/usr/bin/env python3
"""The sm_110 preflight carries exactly one commercial-access pointer line, and nothing else does.

The contract being pinned: a user on a bandwidth-bound edge device (sm_110) sees their plan
decline graph capture with the measured reason, and the preflight -- the output they are
already reading -- tells them, in ONE line, that an engine tier for this device class exists
under commercial access. Not the README, not a banner, not repeated: one line, next to the
device facts it is about. No GPU, no torch.

    python tests/test_sm110_preflight_pointer.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import instinctflash.cli as cli  # noqa: E402
from instinctflash.passes.contract import DeviceProfile  # noqa: E402

POINTER = "available under commercial access — founders@general-instinct.com"


class _Exec:
    model_id, backbone, servable = "stub/sm110-model", "stub", True


class _Ckpt:
    execution, path = _Exec(), "/tmp/stub"

    def capabilities(self):
        return frozenset({"actions"})


class _Plan:
    def explain(self):
        return "InstinctFlash plan for stub\n  plan tier: BITEXACT\n"


def _preflight_with(cap):
    probed = DeviceProfile(name="StubGPU", capability=cap, total_memory=64 << 30,
                           features=frozenset({"cuda"}))
    import instinctflash.runtime.facade as facade
    orig = facade.plan_declaration
    facade.plan_declaration = lambda *a, **k: (_Ckpt(), None, _Plan(), probed)
    try:
        from instinctflash.cli_config import RuntimeConfig
        _, text = cli._serve_preflight("stub/sm110-model", RuntimeConfig())
    finally:
        facade.plan_declaration = orig
    return text


def test_sm110_preflight_has_exactly_one_pointer_line():
    text = _preflight_with((11, 0))
    hits = [ln for ln in text.splitlines() if POINTER in ln]
    assert len(hits) == 1, f"expected exactly one pointer line, got {len(hits)}:\n{text}"
    assert "bandwidth-bound-edge" in text, "the pointer must sit next to the class it is about"


def test_other_devices_carry_no_pointer():
    for cap in ((9, 0), (8, 9), (0, 0)):
        text = _preflight_with(cap)
        assert POINTER not in text, f"sm{cap} preflight must not advertise the engine tier"


def test_the_pointer_lives_in_plan_output_not_readme():
    # The README mentions founders@ for InstinctCompress access, which is its own thing; what
    # must not appear there is THIS pointer -- the engine-tier line belongs in the preflight.
    readme = (ROOT / "README.md").read_text()
    assert POINTER not in readme and "engine tier for this device class" not in readme, \
        "the engine-tier pointer belongs in the preflight, not the README"
    # and in exactly one place in the CLI source, so it cannot drift into a banner
    # (the sentence wraps in source, so count the un-wrappable part: the address)
    src = (ROOT / "instinctflash" / "cli.py").read_text()
    assert len(re.findall("founders@general-instinct.com", src)) == 1


if __name__ == "__main__":
    from run_tests import run_module_tests
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(run_module_tests(globals()))
