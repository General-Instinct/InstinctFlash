#!/usr/bin/env python3
"""The plan preflight header names the checkpoint, not the adapter's default example.

THE FAILURE THIS PINS. A local package whose declaration says
`model_id: general-instinct/lingbot-va-fans-8000` was preflighted as "InstinctFlash plan for
lingbot-va-posttrain-robotwin" — the adapter spec's own sample id — because the Plan is built
from `adapter.spec()` and nothing substituted the declared identity (fresh-user walkthrough,
item 4). The wrong name lands on exactly the line a user quotes in a bug report.

No GPU, no weights: everything runs through the declaration-only preflight.
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

DECLARED = "general-instinct/lingbot-va-fans-8000"


def _pkg(td: Path) -> Path:
    d = td / "pkg"
    d.mkdir()
    (d / "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": DECLARED, "backbone": "wan_va", "servable": True,
                      "guidance": {"video": "cfg", "action": "positive_only"},
                      "nfe": {"video": 2, "action": 4},
                      # weights by reference: serve now runs the loader's package gate before
                      # preflight, so the fixture must be a LOADABLE package, not a bare
                      # declaration — the header invariant under test is unchanged.
                      "base_weights": "robbyant/lingbot-va-posttrain-robotwin"},
    }))
    (d / "config.json").write_text("{}")
    return d


def test_plan_header_prefers_the_declared_model_id():
    from instinctflash.runtime.facade import plan_declaration
    with tempfile.TemporaryDirectory() as td:
        _, _, plan, _ = plan_declaration(_pkg(Path(td)), probe_device=False)
    text = plan.explain()
    assert f"InstinctFlash plan for {DECLARED}" in text, text.splitlines()[0]
    assert "lingbot-va-posttrain-robotwin" not in text.splitlines()[0]


def test_serve_dry_run_shows_it_end_to_end():
    from instinctflash.cli import main
    with tempfile.TemporaryDirectory() as td:
        pkg = _pkg(Path(td))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["serve", str(pkg), "--serve.dry_run=true"])
    out = buf.getvalue()
    assert rc == 0, out
    assert f"InstinctFlash plan for {DECLARED}" in out, out
    assert f"serve preflight for '{DECLARED}'" in out


def test_a_declaration_without_model_id_keeps_the_adapter_name():
    # An empty declared id must not blank the header; the adapter's spec id is the honest
    # fallback for a declaration that names nothing.
    from instinctflash.descriptors.package import Checkpoint
    from instinctflash.descriptors.checkpoint import ExecutionDeclaration
    from instinctflash.runtime.facade import _compile_declaration
    decl = ExecutionDeclaration(model_id="", backbone="wan_va", servable=True)
    _, plan, _ = _compile_declaration(Checkpoint("nowhere", decl), probe_device=False)
    assert "InstinctFlash plan for lingbot-va-posttrain-robotwin" in plan.explain()


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
