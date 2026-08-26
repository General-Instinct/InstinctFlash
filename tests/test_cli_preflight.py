"""Gates for the declaration-only preflight and the startup_timeout_s plumb-through.

The failure these pin: `instinctflash plan <hub-id>` used to construct a full Runtime, which
resolved the whole weight snapshot — a coffee-break download to answer a question one metadata
file answers. `plan_declaration` must never call `snapshot_download`, and the worker placement
must receive the caller's startup timeout (cold 10 GB loads exceed the old hardcoded 900 s only
on the worker path, which is exactly where the knob was being dropped).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _patched_hub(declaration: Path):
    import huggingface_hub

    calls = []

    def one_file(repo, name, revision=None):
        calls.append((repo, name, revision))
        return str(declaration)

    def no_snapshot(*args, **kwargs):
        raise AssertionError("preflight tried to download a weight snapshot")

    return huggingface_hub, one_file, no_snapshot, calls


def _write_declaration(td: str) -> Path:
    declaration = Path(td) / "instinctflash.json"
    declaration.write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": "org/model", "backbone": "wan_va", "servable": True,
                      "guidance": {"video": "cfg", "action": "positive_only"},
                      "nfe": {"video": 2, "action": 4}},
    }))
    return declaration


def test_remote_preflight_never_calls_snapshot_download():
    from instinctflash.runtime.facade import plan_declaration

    with tempfile.TemporaryDirectory() as td:
        declaration = _write_declaration(td)
        hub, one_file, no_snapshot, calls = _patched_hub(declaration)
        old_file, old_snapshot = hub.hf_hub_download, hub.snapshot_download
        hub.hf_hub_download, hub.snapshot_download = one_file, no_snapshot
        try:
            _, _, plan, device = plan_declaration("org/model", probe_device=False)
        finally:
            hub.hf_hub_download, hub.snapshot_download = old_file, old_snapshot
    assert calls == [("org/model", "instinctflash.json", None)]
    assert plan.results
    assert device is None


def test_plan_verb_is_declaration_only():
    import contextlib
    import io

    from instinctflash.cli import main

    with tempfile.TemporaryDirectory() as td:
        declaration = _write_declaration(td)
        hub, one_file, no_snapshot, calls = _patched_hub(declaration)
        old_file, old_snapshot = hub.hf_hub_download, hub.snapshot_download
        hub.hf_hub_download, hub.snapshot_download = one_file, no_snapshot
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = main(["plan", "org/model"])
        finally:
            hub.hf_hub_download, hub.snapshot_download = old_file, old_snapshot
    out = buf.getvalue()
    assert rc == 0, out
    assert "declaration-only" in out
    assert "APPLY" in out or "skip" in out, "the verb still prints the plan"
    assert calls == [("org/model", "instinctflash.json", None)]


def test_plan_verb_exclusion_shows_in_the_plan():
    import contextlib
    import io

    from instinctflash.cli import main

    with tempfile.TemporaryDirectory() as td:
        declaration = _write_declaration(td)
        hub, one_file, no_snapshot, _calls = _patched_hub(declaration)
        old_file, old_snapshot = hub.hf_hub_download, hub.snapshot_download
        hub.hf_hub_download, hub.snapshot_download = one_file, no_snapshot
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = main(["plan", "org/model", "--exclude-pass", "graph_capture"])
        finally:
            hub.hf_hub_download, hub.snapshot_download = old_file, old_snapshot
    out = buf.getvalue()
    assert rc == 0, out
    assert "dropped by caller" in out


def test_startup_timeout_reaches_the_worker_and_only_the_worker():
    from types import SimpleNamespace

    from instinctflash.runtime.execution import (
        InProcessBackend, WorkerBackend, choose_backend,
    )

    ckpt = SimpleNamespace(execution=SimpleNamespace(backbone="tiny-wam"))
    plan = SimpleNamespace(results=[])

    class _Adapter:
        def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
            raise AssertionError("never built in this test")

    worker, _ = choose_backend("worker", _Adapter(), ckpt, plan, startup_timeout_s=3600.0)
    assert isinstance(worker, WorkerBackend)
    assert worker._timeout == 3600.0

    # The in-process backend takes no lifecycle timeout; the knob must be dropped, not crash.
    inproc, _ = choose_backend("in_process", _Adapter(), ckpt, plan, startup_timeout_s=3600.0)
    assert isinstance(inproc, InProcessBackend)


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
