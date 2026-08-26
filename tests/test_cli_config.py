"""Gates for the typed CLI plumbing: config precedence, hard errors, the certify verb, and the
weights-index validation gate.

The properties pinned here are the ones a machine consumer depends on: YAML-then-dotted-CLI
precedence, unknown fields as hard errors (a misspelled knob must never silently no-op), one
stable JSON error schema, atomic --output.path writes, and validate refusing a package whose
declared shards are missing or escape the package.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instinctflash.cli_config import ConfigError, ModelConfig, RuntimeConfig, parse_config  # noqa: E402


@dataclass
class _RunLike:
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def test_yaml_then_dotted_override_precedence():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "run.yaml"
        p.write_text("model:\n  path: org/model\nruntime:\n  nfe:\n    action: 2\n")
        cfg = parse_config(_RunLike, [f"--config_path={p}", "--runtime.nfe.action=4"])
    assert cfg.model.path == "org/model"        # from the file
    assert cfg.runtime.nfe == {"action": 4}     # the CLI override wins
    assert cfg.runtime.tier_ceiling == "bitexact"


def test_unknown_field_is_a_hard_error_not_a_silent_noop():
    try:
        parse_config(_RunLike, ["--runtime.tir_ceiling=numeric"])
    except ConfigError as e:
        assert "tir_ceiling" in str(e)
    else:
        raise AssertionError("a misspelled field must be an error, not an ignored knob")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "run.yaml"
        p.write_text("runtme:\n  device: cuda\n")
        try:
            parse_config(_RunLike, [f"--config_path={p}"])
        except ConfigError as e:
            assert "runtme" in str(e)
        else:
            raise AssertionError("an unknown config-file section must be an error")


def _outcomes_jsonl(path: Path, successes, tag="ep"):
    with open(path, "w") as f:
        for i, s in enumerate(successes):
            f.write(json.dumps({"episode_id": f"{tag}{i}", "seed": 10000 * (1 + i),
                                "task": "adjust_bottle", "success": bool(s)}) + "\n")


def test_certify_verb_certifies_and_writes_atomically():
    from instinctflash.cli import main

    with tempfile.TemporaryDirectory() as td:
        teacher = Path(td) / "teacher.jsonl"
        student = Path(td) / "student.jsonl"
        _outcomes_jsonl(teacher, [1] * 92 + [0] * 8)
        _outcomes_jsonl(student, [1] * 92 + [0] * 8)
        out_path = Path(td) / "cert" / "certificate.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main([
                "certify",
                f"--certify.teacher_outcomes={teacher}",
                f"--certify.student_outcomes={student}",
                "--certify.margin=-0.05",
                f"--output.path={out_path}",
            ])
        assert rc == 0
        payload = json.loads(out_path.read_text())
        assert payload["instinctflash_cli_schema"] == 1
        assert payload["command"] == "certify"
        assert payload["ok"] is True
        assert payload["result"]["passed"] is True
        assert payload["result"]["n_pairs"] == 100
        assert payload["result"]["margin_declared"] == -0.05
        leftovers = [n for n in os.listdir(out_path.parent) if n.startswith(".")]
        assert not leftovers, f"atomic write left temp files: {leftovers}"


def test_certify_verb_fails_a_real_regression():
    from instinctflash.cli import main

    with tempfile.TemporaryDirectory() as td:
        teacher = Path(td) / "teacher.jsonl"
        student = Path(td) / "student.jsonl"
        _outcomes_jsonl(teacher, [1] * 92 + [0] * 8)
        _outcomes_jsonl(student, [1] * 85 + [0] * 15)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main([
                "certify",
                f"--certify.teacher_outcomes={teacher}",
                f"--certify.student_outcomes={student}",
                "--certify.margin=-0.05",
            ])
    assert rc == 1, "a 7-point drop must fail a 5-point margin through the CLI too"


def test_certify_json_error_is_one_schema_object():
    from instinctflash.cli import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["certify", "--certify.margin=-0.05", "--output.format=json"])
    payload = json.loads(buf.getvalue())
    assert rc == 2
    assert payload["instinctflash_cli_schema"] == 1
    assert payload["command"] == "certify"
    assert payload["ok"] is False
    assert payload["error"]["code"] == "CONFIG_ERROR"


def _package(td: Path) -> Path:
    pkg = td / "pkg"
    pkg.mkdir()
    (pkg / "config.json").write_text("{}")
    (pkg / "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": "example-org/x", "backbone": "tiny-wam", "servable": True,
                      "nfe": {"action": 2}, "base_weights": "upstream/none"},
    }))
    return pkg


def test_validate_gates_on_weights_index_integrity():
    from instinctflash.cli import main

    with tempfile.TemporaryDirectory() as td:
        pkg = _package(Path(td))
        index = pkg / "model.safetensors.index.json"

        # A declared shard that does not exist: the package is not valid, and now the exit says so.
        index.write_text(json.dumps({"weight_map": {"w": "model-00001-of-00002.safetensors"}}))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["validate", str(pkg)])
        assert rc == 1
        assert "referenced shard is missing" in buf.getvalue()

        # A shard path escaping the package: refused (a 'validated' package must not read outside).
        (Path(td) / "outside.safetensors").write_bytes(b"\x00")
        index.write_text(json.dumps({"weight_map": {"w": "../outside.safetensors"}}))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["validate", str(pkg)])
        assert rc == 1
        assert "escapes package" in buf.getvalue()

        # A complete index validates as before.
        (pkg / "model-00001-of-00001.safetensors").write_bytes(b"\x00")
        index.write_text(json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["validate", str(pkg)])
        assert rc == 0, buf.getvalue()


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
