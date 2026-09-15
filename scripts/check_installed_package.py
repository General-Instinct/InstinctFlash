#!/usr/bin/env python3
"""Check an installed wheel from any directory, without model loads, networking or GPU probes.

Run with the target environment's Python and -I; no repository imports are added.
The generated fixtures contain metadata only and are never used for inference.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile


REQUIRED_FILES = (
    "instinctflash/runtime/step_cache.py",
    "instinctflash/runtime/step_cache_policy.py",
    "instinctflash/runtime/telemetry.py",
    "instinctflash/runtime/tensor_cache.py",
    "instinctflash/runtime/compiled_region.py",
    "instinctflash/native/CMakeLists.txt",
    "instinctflash/native/wan_residual_sm120.cu",
    "instinctflash/native/wan_stage2_sm120.cu",
    "instinctflash/native/bf16/CMakeLists.txt",
    "instinctflash/native/bf16/build.sh",
    "instinctflash/native/bf16/linear_relu2_sm110.cu",
    "benchmarks/vla/config/adapters.json",
    "benchmarks/regression/runtime_bundle.py",
    "benchmarks/regression/systemd/instinctflash-thor-regression.service",
    "benchmarks/regression/systemd/instinctflash-thor-regression.timer",
)


def check(*, require_all_adapters: bool, require_torch_free: bool) -> dict:
    import huggingface_hub
    import instinctflash
    import yaml
    from instinctflash.cli import main as cli_main
    from instinctflash.descriptors.known import KNOWN_DECLARATIONS
    from instinctflash.passes.contract import DeviceProfile
    from instinctflash.runtime.facade import plan_declaration
    from instinctflash.runtime.loader import available_models, discover_plugins

    distribution = importlib.metadata.distribution("instinctflash")
    origin = Path(instinctflash.__file__).resolve()
    installed_root = Path(distribution.locate_file("instinctflash")).resolve()
    assert origin.parent == installed_root, f"import does not come from installed distribution: {origin}"
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    assert not direct_url.get("dir_info", {}).get("editable"), "this check requires a wheel installation"
    registered = available_models()
    problems = discover_plugins()
    assert not problems, problems
    if require_all_adapters:
        required = {doc["execution"]["backbone"] for doc in KNOWN_DECLARATIONS.values()}
        assert required <= set(registered), f"missing adapters: {sorted(required - set(registered))}"
    torch_present = importlib.util.find_spec("torch") is not None
    if require_torch_free:
        assert not torch_present, "the clean core audit environment contains Torch"

    assets = {}
    for relative in REQUIRED_FILES:
        path = Path(distribution.locate_file(relative))
        assert path.is_file(), f"wheel omits {relative}"
        assets[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    def forbidden(*args, **kwargs):
        raise AssertionError("installed-wheel check attempted network access or a weight download")

    def no_device(*args, **kwargs):
        raise RuntimeError("CPU packaging audit: device probing disabled")

    huggingface_hub.hf_hub_download = forbidden
    huggingface_hub.snapshot_download = forbidden
    DeviceProfile.probe = staticmethod(no_device)
    commands = []

    def cli(args, *, json_result=False, expected_code=0):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli_main(args)
            except SystemExit as error:
                code = error.code
        assert code == expected_code, (args, code, out.getvalue(), err.getvalue())
        payload = json.loads(out.getvalue()) if json_result else None
        commands.append({"argv": args, "exit_code": code,
                         "stdout_sha256": hashlib.sha256(out.getvalue().encode()).hexdigest()})
        return payload

    for args in (["--help"], ["serve", "--help"], ["validate", "--help"],
                 ["eval", "--help"], ["models"], ["models", "--json"], ["eval", "adapters"]):
        cli(list(args), json_result=args == ["models", "--json"])
    planned = []
    with tempfile.TemporaryDirectory(prefix="instinctflash-wheel-fixtures-") as temporary:
        fixture_root = Path(temporary)
        for index, (model_id, document) in enumerate(sorted(KNOWN_DECLARATIONS.items())):
            if document["execution"]["backbone"] not in registered:
                continue
            package = fixture_root / str(index)
            package.mkdir()
            (package / "instinctflash.json").write_text(json.dumps(document))
            config = {}
            if document["execution"]["backbone"] == "dreamzero":
                config = {"action_head_cfg": {"config": {"diffusion_model_cfg": {
                    "frame_seqlen": 880, "dim": 5120, "num_layers": 40}}}}
            (package / "config.json").write_text(json.dumps(config))
            checkpoint, _, plan, device = plan_declaration(package, probe_device=False, step_cache="checkpoint")
            assert checkpoint.path == str(package) and device is None
            planned.append({"model_id": model_id, "backbone": checkpoint.execution.backbone,
                            "scheduler_nfe": dict(checkpoint.execution.nfe), "plan_entries": len(plan.results)})
            cli(["describe", str(package), "--json"], json_result=True)
            for suffix in ("json", "yaml"):
                document = {"serve": {"model": str(package), "dry_run": True},
                            "runtime": {"step_cache": "checkpoint"}, "output": {"format": "json"}}
                config_file = package / f"serve.{suffix}"
                config_file.write_text(json.dumps(document) if suffix == "json" else yaml.safe_dump(document))
                result = cli(["serve", f"--config_path={config_file}"], json_result=True)
                assert result["ok"], result
            result = cli(["validate", str(package), "--output.format=json"], json_result=True)
            assert result["ok"], result
        error = cli(["serve", "--runtime.precison=fp8", "--output.format=json"],
                    json_result=True, expected_code=2)
        assert error["ok"] is False
    assert "torch" not in sys.modules, "metadata-only operations imported Torch"
    return {"schema": 1, "ok": True, "scope": "installed wheel metadata only; no model inference",
            "python": sys.version, "executable": sys.executable, "cwd": os.getcwd(),
            "isolated_mode": bool(sys.flags.isolated), "instinctflash_origin": str(origin),
            "instinctflash_version": distribution.version, "torch_installed": torch_present,
            "torch_imported": False, "registered_adapters": registered,
            "required_asset_sha256": assets, "metadata_fixture_plans": planned, "cli_checks": commands,
            "installed_distributions": {dist.metadata["Name"]: dist.version
                                        for dist in importlib.metadata.distributions()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all-adapters", action="store_true")
    parser.add_argument("--require-torch-free", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(require_all_adapters=args.require_all_adapters, require_torch_free=args.require_torch_free)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(json.dumps({"ok": True, "report": str(args.output),
                          "metadata_plans": len(result["metadata_fixture_plans"]),
                          "cli_checks": len(result["cli_checks"]),
                          "torch_imported": result["torch_imported"]}))
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
