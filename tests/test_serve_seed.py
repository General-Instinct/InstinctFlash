#!/usr/bin/env python3
"""--serve.seed: two serves with the same seed and the same inputs are comparable value-for-value.

WHY A SEED FLAG IS A FIRST-CLASS CITIZEN HERE. The models this runtime serves draw noise on every
control step, upstream serving is unseeded, so two STOCK servers already disagree — which means no
user can do a value-for-value regression against the framework's own BITEXACT claim (fresh-user
walkthrough, item 5). The seed is per-episode at the runtime floor; adapters that accept `seed=`
thread it per request (wan_va re-seeds every `_infer` draw as seed+frame_st_id).

Stub-runtime loopback, CPU only: a registered stub backbone whose predict draws torch AND numpy
noise, loaded through the real `Runtime.from_pretrained(..., seed=)` path.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import instinctflash  # noqa: E402
from instinctflash.adapters.base import AdapterSpec, PhaseSpec  # noqa: E402

BACKBONE = "seed-stub"


class _NoiseImpl:
    """Draws from both RNG families the runtime promises to seed."""

    def predict(self, observation):
        return {"action": (torch.randn(8).numpy() + np.random.randn(8)).astype("float64")}

    def reset(self, **conditioning):
        pass


class _SeedStubAdapter:
    model_id = BACKBONE

    def spec(self) -> AdapterSpec:
        return AdapterSpec(model_id=BACKBONE, param_bytes=0, streams=(),
                           phases=(PhaseSpec(name="action", nfe=1),), guidance={})

    def install(self, server_module, plan):
        return []

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        return _NoiseImpl()


instinctflash.register(BACKBONE, _SeedStubAdapter)


def _pkg(td: Path) -> Path:
    d = td / "pkg"
    d.mkdir()
    (d / "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": "example-org/seed-stub", "backbone": BACKBONE,
                      "servable": True},
    }))
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    return d


def _episode_actions(pkg: Path, seed, n: int = 2) -> list:
    from instinctflash import Runtime
    with Runtime.from_pretrained(pkg, seed=seed) as rt, rt.episode(prompt="t") as ep:
        return [ep.predict({"x": 0})["action"] for _ in range(n)]


def test_same_seed_identical_actions():
    with tempfile.TemporaryDirectory() as td:
        pkg = _pkg(Path(td))
        a = _episode_actions(pkg, seed=1234)
        b = _episode_actions(pkg, seed=1234)
    assert all((x == y).all() for x, y in zip(a, b)), "same seed, same inputs -> same actions"
    assert not (a[0] == a[1]).all(), "within an episode the draws still advance (not frozen noise)"


def test_different_seeds_differ():
    with tempfile.TemporaryDirectory() as td:
        pkg = _pkg(Path(td))
        a = _episode_actions(pkg, seed=1234)
        b = _episode_actions(pkg, seed=4321)
    assert not (a[0] == b[0]).all(), "a different seed must actually change the draw"


def test_seed_is_per_episode():
    from instinctflash import Runtime
    with tempfile.TemporaryDirectory() as td:
        pkg = _pkg(Path(td))
        with Runtime.from_pretrained(pkg, seed=7) as rt:
            with rt.episode(prompt="t") as ep:
                first = ep.predict({"x": 0})["action"]
            with rt.episode(prompt="t") as ep:
                again = ep.predict({"x": 0})["action"]
    assert (first == again).all(), "a new episode replays the same seeded stream"


def test_unseeded_stays_stock():
    with tempfile.TemporaryDirectory() as td:
        pkg = _pkg(Path(td))
        a = _episode_actions(pkg, seed=None, n=1)
        b = _episode_actions(pkg, seed=None, n=1)
    assert not (a[0] == b[0]).all(), "no seed -> the model's own unseeded behaviour is untouched"


def test_cli_flag_parses_and_reaches_from_pretrained():
    import inspect

    from instinctflash import Runtime
    from instinctflash.cli import ServeConfig
    from instinctflash.cli_config import parse_config
    cfg = parse_config(ServeConfig, ["--serve.model=m", "--serve.seed=7"])
    assert cfg.serve.seed == 7
    assert parse_config(ServeConfig, ["--serve.model=m"]).serve.seed is None
    assert "seed" in inspect.signature(Runtime.from_pretrained).parameters
    # the wan_va adapter threads it deeper: per-request in-process, --deterministic-seed on the
    # worker command line
    from instinctflash.adapters.lingbot_va import LingBotVA
    assert "seed" in inspect.signature(LingBotVA.build_in_process).parameters
    assert "seed" in inspect.signature(LingBotVA.worker_command).parameters


def test_readme_documents_the_flag():
    text = (ROOT / "README.md").read_text()
    assert "--serve.seed" in text, "the serve section documents the A/B seed flag"


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
