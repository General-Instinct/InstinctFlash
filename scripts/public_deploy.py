#!/usr/bin/env python3
"""Plan a deployment or check its CPU dependencies, without downloading weights.

Run with the intended vendor environment's Python, from any working directory::

    python /path/to/InstinctFlash/scripts/public_deploy.py plan all
    python /path/to/InstinctFlash/scripts/public_deploy.py doctor pi05
    python /path/to/InstinctFlash/scripts/public_deploy.py plan edge --execution numeric

JSON goes to stdout. Exit 0 means the requested plan/CPU checks succeeded;
exit 1 means a dependency check failed; exit 2 means invalid input.
Neither command installs packages, loads a checkpoint, tests CUDA or certifies quality.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "release" / "deployment_profiles.json"
SCHEMA = "instinctflash.deployment_profiles.v1"
FAMILIES = {"va", "vla4", "vla2", "pi05", "groot", "edge", "nano", "dreamzero"}
IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z")


def load_profiles(path: Path = PROFILE_PATH) -> dict:
    """Read the portable catalog; historical eval files are not runtime dependencies."""
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA or document.get("target") != "jetson_thor":
        raise ValueError("unsupported deployment profile schema or target")
    rows = document.get("models", [])
    if len(rows) != 8 or {row["id"] for row in rows} != FAMILIES:
        raise ValueError("catalog must contain exactly the eight distinct model variants")
    for row in rows:
        if not re.fullmatch(r"[0-9a-f]{40}", row["checkpoint"]["revision"]):
            raise ValueError("checkpoint revision must be an immutable 40-character commit")
        if "native" not in row["execution_modes"]:
            raise ValueError("each model requires an explicit native default")
        for dependency in row["dependency_imports"]:
            if not IDENTIFIER.fullmatch(dependency["module"]) or any(
                not IDENTIFIER.fullmatch(a) for a in dependency["attributes"]
            ):
                raise ValueError("invalid dependency module or API name")
        for deferred in row.get("deferred_native_imports", []):
            if (deferred["module"] not in row["source_only_modules"]
                    or any(d["module"] == deferred["module"] for d in row["dependency_imports"])
                    or not deferred["source_sha256"]):
                raise ValueError("deferred native imports must be source-only and hash-bound")
            for name, digest in deferred["source_sha256"].items():
                if (Path(name).is_absolute() or ".." in Path(name).parts
                        or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                    raise ValueError("invalid deferred native source binding")
        for name, mode in row["execution_modes"].items():
            kwargs = mode["runtime_kwargs"]
            precision, tier = kwargs["precision"], kwargs["tier_ceiling"]
            if precision not in ("native", "fp8") or tier not in (
                "bitexact", "numeric", "behavioral"
            ):
                raise ValueError("execution precision and transformation ceiling must be explicit")
            if precision == "fp8" and tier == "bitexact":
                raise ValueError("FP8 cannot use a bitexact ceiling")
            if kwargs["step_cache"] not in ("checkpoint", "dynamic"):
                raise ValueError("step cache selection must be explicit")
            changed = "nfe" in kwargs or kwargs["step_cache"] == "dynamic"
            if changed != mode["schedule_changed"] or (changed and tier != "behavioral"):
                raise ValueError("changed schedules require the behavioral ceiling and label")
            if name == "native" and (precision, tier, kwargs["step_cache"], changed) != (
                "native", "bitexact", "checkpoint", False
            ):
                raise ValueError("native default must preserve checkpoint precision and schedule")
            if any(value.startswith("/") for value in mode["environment"].values()):
                raise ValueError("execution profiles cannot embed host-specific absolute paths")
    return document


def make_plan(profile: dict, execution: str = "native", *, python: str = sys.executable) -> dict:
    modes = profile["execution_modes"]
    if execution not in modes:
        raise ValueError(f"{profile['id']} supports execution modes: {', '.join(modes)}")
    checkpoint, mode = profile["checkpoint"], copy.deepcopy(modes[execution])
    install = [python, "-m", "pip", "install", str(ROOT) + "[serve]"]
    if profile["adapter"]["package_directory"]:
        install.append(str(ROOT / profile["adapter"]["package_directory"]))
    prepare = (
        "from instinctflash.descriptors.package import from_pretrained; "
        f"print(from_pretrained({checkpoint['model_id']!r}, "
        f"revision={checkpoint['revision']!r}).path)"
    )
    runtime = copy.deepcopy(mode["runtime_kwargs"])
    serve_config = {
        "serve": {"model": "<resolved_checkpoint_path>", "host": "127.0.0.1", "port": 8000},
        "runtime": runtime,
        "output": {"format": "json"},
    }
    return {
        "id": profile["id"], "label": profile["label"], "checkpoint": checkpoint,
        "adapter": profile["adapter"], "upstream": profile["upstream"],
        "python": profile["python"], "bootstrap": profile["bootstrap"],
        "execution": execution, "available_execution_modes": list(modes),
        "selected_execution": mode,
        "commands": {
            "install_core_and_adapter_after_vendor_setup": install,
            "doctor_before_weights": [python, str(Path(__file__).resolve()), "doctor",
                                      profile["id"], "--execution", execution],
            "prepare_pinned_checkpoint_after_doctor": [python, "-c", prepare],
            "serve_after_saving_config": ["instinctflash", "serve", "--config_path=serve.json"],
        },
        "serve_config_template": serve_config,
        "python_runtime_call": {"model_id_or_path": checkpoint["model_id"],
                                "revision": checkpoint["revision"], **runtime},
        "usage": [
            "Commands are a plan; this utility never executes installation or weight preparation.",
            "Run doctor inside the prepared vendor environment. Missing vendor pins are disclosed below.",
            "After explicit weight preparation, replace <resolved_checkpoint_path> with the returned declared path.",
            "Apply selected_execution.environment before Python/serve; supply any required library paths explicitly.",
            "CLI serve has no revision option; use the resolved pinned local checkpoint, not the unpinned Hub ID.",
            "Use runtime.observation.describe() for this checkpoint's cameras/state/history; reset at episode boundaries.",
            "Use one Runtime per observation stream. A synthetic action smoke is not robot/task validation.",
        ],
        "release_gaps": profile["release_gaps"],
        "not_verified": ["fresh_vendor_install", "checkpoint_download_and_assets", "GPU_execution",
                         "latency", "WebSocket_roundtrip", "robot_actions", "task_quality"],
    }


def _check(name: str, passed: bool, **detail) -> dict:
    return {"name": name, "status": "passed" if passed else "failed", **detail}


def _find_without_import(name: str):
    """Find nested source/extension paths without running package initializers."""
    from importlib.machinery import PathFinder

    path = None
    parts = name.split(".")
    spec = None
    for index in range(len(parts)):
        spec = PathFinder.find_spec(".".join(parts[:index + 1]), path)
        if spec is None:
            return None
        path = spec.submodule_search_locations
        if index < len(parts) - 1 and path is None:
            return None
    return spec


def _native_file_check(requirement: dict) -> dict:
    """Check file/host ABI presence only; never dlopen a CUDA library."""
    from importlib.machinery import EXTENSION_SUFFIXES
    import platform
    import struct

    kind, name = requirement["kind"], requirement["name"]
    path = None
    if kind == "file_env":
        if os.environ.get(name):
            path = Path(os.environ[name]).expanduser()
    elif kind == "extension":
        spec = _find_without_import(name)
        if spec and spec.origin and any(spec.origin.endswith(s) for s in EXTENSION_SUFFIXES):
            path = Path(spec.origin)
    elif kind == "package_file":
        package, relative = name.split("/", 1)
        spec = _find_without_import(package)
        if spec and spec.submodule_search_locations:
            path = Path(next(iter(spec.submodule_search_locations))) / relative
    else:
        raise ValueError("unknown native requirement kind")
    result = _check(f"native:{name}", False, remedy=requirement["remedy"],
                    scope="ELF file and host architecture only; CUDA/exports/linking untested")
    if path is None or not path.is_file():
        result["reason"] = "matching_native_file_missing"
        return result
    with path.open("rb") as stream:
        header = stream.read(20)
        digest = hashlib.sha256(header)
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    expected = {"aarch64": 183, "arm64": 183, "x86_64": 62, "AMD64": 62}.get(platform.machine())
    valid = len(header) == 20 and header[:6] == b"\x7fELF\x02\x01"
    actual = struct.unpack("<H", header[18:20])[0] if valid else None
    result.update(path=str(path.resolve()), sha256=digest.hexdigest(),
                  status="passed" if valid and expected is not None and actual == expected else "failed",
                  reason="host_ELF_checked" if valid and actual == expected else "incompatible_host_ELF")
    return result


def _probe_worker(payload: dict) -> dict:
    """CPU import checks run in a disposable interpreter, never the caller's process.

    These Python operation guards are diagnostic controls, not a sandbox for untrusted
    vendor code. External/native code is still the user's installed dependency stack.
    """
    import contextlib
    import functools
    import importlib
    import importlib.metadata
    import io
    import shutil
    import socket
    import traceback

    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1",
                      NO_ALBUMENTATIONS_UPDATE="1")
    denied = []
    denied_origins = []
    allowed_cpu_plumbing = []

    def import_plumbing(event, args):
        if event not in ("socket.bind", "subprocess.Popen"):
            return None
        # urllib3 checks IPv6 availability with a local ephemeral bind and
        # immediately closes it. No connection or data transmission is allowed.
        caller = sys._getframe(2)
        if (event == "socket.bind" and len(args) >= 2 and
                getattr(args[0], "family", None) == socket.AF_INET6 and args[1] == ("::1", 0) and
                caller.f_globals.get("__name__") == "urllib3.util.connection" and
                caller.f_code.co_name == "_has_ipv6"):
            return "urllib3_ipv6_loopback_capability_probe"
        # CPython platform.processor() invokes only `uname -p` on Linux.
        # Require the real system executable and the stdlib call site; neither
        # arbitrary commands nor shell/subprocess wrappers are admitted.
        argv = ([args[1]] if isinstance(args[1], str) else list(args[1])) if event == "subprocess.Popen" else []
        if event == "subprocess.Popen" and argv in (["uname", "-p"], ["lscpu"]):
            binary = shutil.which(str(args[0]))
            expected = "/usr/bin/uname" if argv == ["uname", "-p"] else "/usr/bin/lscpu"
            if not binary or Path(binary).resolve() != Path(expected).resolve():
                return None
            frame = caller
            for _ in range(10):
                if frame is None:
                    break
                if (argv == ["uname", "-p"] and frame.f_globals.get("__name__") == "platform" and
                        frame.f_code.co_name == "from_subprocess" and
                        Path(frame.f_code.co_filename).resolve() ==
                        Path(sys.modules["platform"].__file__).resolve()):
                    return "stdlib_platform_uname_processor_probe"
                if (argv == ["lscpu"] and frame.f_globals.get("__name__") == "numpy.testing._private.utils" and
                        frame.f_code.co_name == "check_support_sve"):
                    return "numpy_testing_cpu_sve_capability_probe"
                frame = frame.f_back
        return None

    def guard(event, args):
        plumbing = import_plumbing(event, args)
        if plumbing:
            allowed_cpu_plumbing.append(plumbing)
            return
        network = event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo",
                            "socket.sendto", "socket.sendmsg", "socket.bind")
        process = event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.exec")
        weights = event == "open" and isinstance(args[0], (str, bytes)) and str(args[0]).endswith(
            (".safetensors", ".pt", ".pth", ".ckpt")
        )
        if network or process or weights:
            denied.append(event)
            denied_origins.append({"event": event, "stack": [
                {"file": f.filename, "line": f.lineno, "function": f.name}
                for f in traceback.extract_stack(limit=14)[:-1]
            ]})
            raise RuntimeError("operation_disabled_in_cpu_doctor")

    sys.addaudithook(guard)
    checks = []
    for entry in payload["source_roots"]:
        sys.path.insert(0, entry)
    # Drop arbitrary import output rather than echoing environment variables or tokens
    # which a third-party module might log. Exceptions are reported by type only.
    class Discard(io.TextIOBase):
        def write(self, value):
            return len(value)

    with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(Discard()):
        for dependency in payload["dependencies"]:
            name = dependency["module"]
            try:
                module = importlib.import_module(name)
                if name == "torch":
                    def refuse_cuda(original):
                        # Distinct wrappers preserve callable identity: Dynamo's
                        # import-time dispatch maps reject one function assigned
                        # to several semantically distinct CUDA API names.
                        @functools.wraps(original)
                        def blocked(*args, **kwargs):
                            denied.append("torch.cuda.initialization")
                            raise RuntimeError("GPU_initialization_disabled_in_cpu_doctor")
                        return blocked
                    module.cuda.is_available = lambda: False
                    module.cuda.device_count = lambda: 0
                    for method in ("_lazy_init", "init", "set_device", "current_device",
                                   "get_device_capability", "get_device_properties"):
                        original = getattr(module.cuda, method, lambda: None)
                        setattr(module.cuda, method, refuse_cuda(original))
                for attribute in dependency["attributes"]:
                    value = module
                    for component in attribute.split("."):
                        value = getattr(value, component)
                checks.append(_check(f"import:{name}", True,
                                     origin=getattr(module, "__file__", None),
                                     required_attributes=dependency["attributes"]))
            except Exception as error:
                checks.append(_check(f"import:{name}", False, error_type=type(error).__name__,
                                     error_frames=[{"file": f.filename, "line": f.lineno, "function": f.name}
                                                   for f in traceback.extract_tb(error.__traceback__)[-12:]],
                                     remedy="Prepare the family's compatible vendor stack/API in this interpreter."))
        for name in payload["source_only_modules"]:
            try:
                spec = _find_without_import(name)
                origin = Path(spec.origin) if spec and spec.origin else None
                checks.append(_check(f"source:{name}", bool(origin and origin.is_file()),
                                     origin=str(origin) if origin else None,
                                     scope="located without executing this module's initializer"))
            except Exception as error:
                checks.append(_check(f"source:{name}", False, error_type=type(error).__name__))
        try:
            from instinctflash.runtime.loader import load

            adapter = load(payload["backbone"])
            actual = f"{type(adapter).__module__}:{type(adapter).__name__}"
            checks.append(_check("adapter_registration", actual == payload["adapter_entry_point"],
                                 actual=actual, expected=payload["adapter_entry_point"]))
            ok, _reason = adapter.can_host_in_process()
            checks.append(_check("adapter_host_imports", bool(ok),
                                 scope="existing adapter CPU host probe; model not constructed",
                                 remedy="Check the declared source root, installed vendor APIs and dependency checks."))
        except Exception as error:
            checks.append(_check("adapter_host_imports", False, error_type=type(error).__name__,
                                 remedy="Install core and this adapter into the vendor interpreter."))
        for requirement in payload["native_requirements"]:
            try:
                checks.append(_native_file_check(requirement))
            except Exception as error:
                checks.append(_check(f"native:{requirement['name']}", False,
                                     error_type=type(error).__name__, remedy=requirement["remedy"]))
        versions = {}
        for name in payload["distributions"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
    # An attempted network/device operation cannot be hidden by a vendor catching it.
    checks.append(_check("offline_cpu_operation_guard", not denied,
                         blocked_operations=sorted(set(denied)), blocked_origins=denied_origins,
                         allowed_cpu_import_plumbing=sorted(set(allowed_cpu_plumbing))))
    return {"checks": checks, "distributions": versions, "GPU_verified": False,
            "guard_scope": "CUDA hidden; PyTorch initialization and Python network/process/weight-file operations blocked"}


def run_probe(payload: dict, timeout: float) -> dict:
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1",
                       NO_ALBUMENTATIONS_UPDATE="1")
    environment.update(payload.get("root_environment", {}))
    # Resolve user-relative paths before changing the child working directory.
    for requirement in payload.get("native_requirements", []):
        if requirement["kind"] == "file_env" and environment.get(requirement["name"]):
            environment[requirement["name"]] = str(
                Path(environment[requirement["name"]]).expanduser().resolve()
            )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--_probe"],
            input=json.dumps(payload), text=True, capture_output=True, env=environment,
            cwd=str(Path(__file__).resolve().parent), timeout=timeout, check=False,
        )
        if result.returncode != 0:
            return {"checks": [_check("probe_process", False, reason="nonzero_exit",
                                      returncode=result.returncode)]}
        parsed = json.loads(result.stdout)
        if not isinstance(parsed.get("checks"), list) or not parsed["checks"]:
            raise ValueError("missing probe checks")
        return parsed
    except subprocess.TimeoutExpired:
        return {"checks": [_check("probe_process", False, reason="timeout", timeout_seconds=timeout)]}
    except (OSError, ValueError):
        return {"checks": [_check("probe_process", False, reason="invalid_or_unavailable_probe")]}


def doctor(profile: dict, execution: str = "native", *, timeout: float = 45,
           probe_runner=run_probe) -> dict:
    plan = make_plan(profile, execution)
    checks = [_check("core_python", sys.version_info >= (3, 10))]
    source_roots = []
    root_environment = {}
    deferred_checks = []
    upstream = profile["upstream"]
    variable = upstream["root_env"]
    if variable:
        value = os.environ.get(variable)
        root = Path(value).expanduser().resolve() if value else None
        valid = bool(root and root.is_dir())
        checks.append(_check(f"source_root:{variable}", valid,
                             remedy=f"Set {variable} to the matching {upstream['repository']} checkout."))
        if valid:
            source_roots.append(str(root))
            root_environment[variable] = str(root)
            for marker in upstream["source_markers"]:
                source = root / marker
                present = source.is_file()
                checks.append(_check(f"source_file:{marker}", present,
                                     sha256=hashlib.sha256(source.read_bytes()).hexdigest() if present else None))
        for deferred in profile.get("deferred_native_imports", []):
            bindings = []
            for name, expected in deferred["source_sha256"].items():
                source = root / name if valid else None
                actual = (hashlib.sha256(source.read_bytes()).hexdigest()
                          if source and source.is_file() and source.resolve().is_relative_to(root) else None)
                binding = _check(f"deferred_import_source:{name}", actual == expected,
                                 sha256=actual, expected_sha256=expected)
                checks.append(binding)
                bindings.append(binding)
            deferred_checks.append({"name": f"full_native_import:{deferred['module']}",
                                    "status": "deferred", "reason": deferred["reason"],
                                    "source_binding_verified": all(b["status"] == "passed" for b in bindings),
                                    "GPU_verified": False, "full_import_passed": False})
    mode = plan["selected_execution"]
    if profile["id"] == "dreamzero" and mode["runtime_kwargs"]["precision"] == "fp8":
        checks.append(_check("DreamZero_FP8_TensorRT_disabled",
                             "LOAD_TRT_ENGINE" not in os.environ and
                             os.environ.get("ENABLE_TENSORRT", "0").lower() in ("", "0", "false"),
                             remedy="Unset LOAD_TRT_ENGINE and disable ENABLE_TENSORRT for DreamZero FP8."))
    probe = probe_runner({
        "source_roots": source_roots, "root_environment": root_environment,
        "dependencies": profile["dependency_imports"],
        "source_only_modules": profile["source_only_modules"], "backbone": profile["backbone"],
        "adapter_entry_point": profile["adapter"]["entry_point"],
        "native_requirements": mode["native_requirements"],
        "distributions": list(dict.fromkeys(["instinctflash", profile["adapter"]["distribution"],
                                            "torch", "transformers", "lerobot", "flash-rt"])),
    }, timeout)
    checks.extend(probe["checks"])
    passed = all(check["status"] == "passed" for check in checks)
    return {"id": profile["id"], "execution": execution, "ok": passed,
            "status": ("cpu_preflight_passed_with_deferred_gpu_import" if deferred_checks
                       else "cpu_preflight_passed") if passed else "blocked",
            "deferred_checks": deferred_checks,
            "scope": "CPU dependency/source checks before download; not a deployment qualification",
            "checks": checks, "distributions": probe.get("distributions", {}),
            "python": {"executable": sys.executable, "version": sys.version.split()[0],
                       "historical_thor": profile["python"]["historical_thor"]},
            "bootstrap_status": profile["bootstrap"]["status"],
            "upstream_pin_status": upstream["pin_status"],
            "GPU_verified": False, "weights_downloaded": False, "model_constructed": False,
            "task_quality_certified": False, "not_verified": plan["not_verified"],
            "next_step": "Review plan and unresolved bootstrap/library requirements before explicit checkpoint preparation."}


def main(argv=None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--_probe"]:
        result = _probe_worker(json.load(sys.stdin))
        print(json.dumps(result, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("plan", "doctor"))
    parser.add_argument("model", choices=("all", *sorted(FAMILIES)))
    parser.add_argument("--execution", default="native", help="Explicit named recipe; see plan output for supported modes.")
    parser.add_argument("--timeout", type=float, default=45, help="Maximum seconds for each CPU dependency subprocess.")
    args = parser.parse_args(arguments)
    if not 0 < args.timeout <= 300:
        parser.error("--timeout must be greater than 0 and at most 300 seconds")
    try:
        catalog = load_profiles()
        selected = [row for row in catalog["models"] if args.model in ("all", row["id"])]
        # Validate all selections before any imports so a mixed unsupported recipe fails early.
        plans = [make_plan(row, args.execution) for row in selected]
        results = plans if args.command == "plan" else [
            doctor(row, args.execution, timeout=args.timeout) for row in selected
        ]
        ok = all(row.get("ok", True) for row in results)
        report = {"schema": "instinctflash.public_deploy.v1", "command": args.command, "ok": ok,
                  "target": catalog["target"], "profiles_sha256": hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
                  "results": results, "evidence_scope": catalog["evidence_scope"]}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if ok else 1
    except (OSError, KeyError, TypeError, ValueError) as error:
        # No raw vendor output or environment dump is ever included.
        print(json.dumps({"schema": "instinctflash.public_deploy.v1", "ok": False,
                          "error_type": type(error).__name__, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
