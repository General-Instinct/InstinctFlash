#!/usr/bin/env python3
"""`validate` on a training-output tree says what to do, not just what is missing.

THE FAILURE THIS PINS. Trainers write `<run>/transformer/*.safetensors`; the package convention
is the transformer contents FLAT at the package root (adapters/lingbot_va.py, materialize()).
A fresh user pointing `validate` at the training output got "MISSING config.json" plus "no local
weight files; referenced by execution.base_weights" — both true, both useless, and the second one
actively misleading while ten gigabytes of weights sat one directory down (2026-08-26 usability
journal, item 2). The existing errors stay; one actionable line is appended, and only on FAILING
reports — a composed upstream package legitimately carries transformer/ and is not told to
flatten itself.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instinctflash.descriptors.package import validate_package  # noqa: E402

HINT_MARK = "training-output layout"


def _training_tree(td: Path, with_declaration: bool) -> Path:
    d = td / "run_out"
    (d / "transformer").mkdir(parents=True)
    (d / "transformer" / "config.json").write_text("{}")
    (d / "transformer" / "model.safetensors").write_bytes(b"\x00")
    if with_declaration:
        (d / "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "example-org/fans-8000", "backbone": "wan_va",
                          "servable": True,
                          "base_weights": "robbyant/lingbot-va-posttrain-robotwin"},
        }))
    return d


def test_journal_repro_declared_training_tree_gets_the_hint():
    with tempfile.TemporaryDirectory() as td:
        d = _training_tree(Path(td), with_declaration=True)
        rep = validate_package(d)
        text = rep.explain()
    assert not rep.ok, "a training-output tree is not a valid package"
    assert "MISSING  config.json" in text, "the original error is kept"
    assert "no local weight files" in text, "the original base_weights note is kept"
    assert HINT_MARK in text and "mv " in text and "transformer/*" in text, text
    assert "instinctflash.json" in text.split(HINT_MARK, 1)[1], \
        "the hint states where the flat contents belong"


def test_undeclared_training_tree_gets_the_hint_too():
    with tempfile.TemporaryDirectory() as td:
        d = _training_tree(Path(td), with_declaration=False)
        rep = validate_package(d)
        text = rep.explain()
    assert not rep.ok
    assert "no declaration" in text, "the original problem is kept"
    assert HINT_MARK in text and "mv " in text, text


def test_valid_package_with_a_transformer_component_is_not_nagged():
    # The upstream-composed layout (a declared view of an original release) carries transformer/
    # as a component and is VALID; the hint fires only when validation fails.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pkg"
        (d / "transformer").mkdir(parents=True)
        (d / "transformer" / "model.safetensors").write_bytes(b"\x00")
        (d / "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "example-org/x", "backbone": "wan_va", "servable": True},
        }))
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\x00")
        rep = validate_package(d)
    assert rep.ok, rep.explain()
    assert HINT_MARK not in rep.explain()


def test_failing_dir_without_transformer_gets_no_layout_hint():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "empty"
        d.mkdir()
        (d / "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "example-org/x", "backbone": "wan_va", "servable": True},
        }))
        rep = validate_package(d)
    assert not rep.ok
    assert HINT_MARK not in rep.explain(), "the hint is about layout, not a generic banner"


def test_the_cli_verb_shows_the_hint():
    from instinctflash.cli import main
    with tempfile.TemporaryDirectory() as td:
        d = _training_tree(Path(td), with_declaration=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["validate", str(d)])
    out = buf.getvalue()
    assert rc == 1, out
    assert HINT_MARK in out and "mv " in out, out


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
