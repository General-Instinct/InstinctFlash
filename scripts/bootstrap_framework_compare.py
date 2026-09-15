#!/usr/bin/env python3
"""Create an isolated public LeRobot or vLLM-Omni comparison environment.

This installs packages and performs CPU import checks only. It does not load
checkpoints, run inference, or replace an existing environment/source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import urllib.request


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "benchmarks/regression/fixtures/frameworks"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def load():
    catalog = json.loads((DATA / "catalog.json").read_text())
    require(sha(DATA / "sources.json") == catalog["source_inventory_sha256"], "source profile changed")
    require(sha(DATA / catalog["startup_patch_file"]) == catalog["startup_patch_sha256"], "patch profile changed")
    require(sha(DATA / "omni_thor_startup_patch.json") == catalog["historical_startup_patch_sha256"], "historical patch changed")
    return catalog, json.loads((DATA / "sources.json").read_text())


def apply_startup_patch(root, *, acknowledged):
    require(acknowledged, "Omni historical startup patch requires --allow-startup-patch")
    catalog, _ = load()
    patch = json.loads((DATA / catalog["startup_patch_file"]).read_text())
    changes = []
    for item in patch["changes"]:
        path = Path(root) / item["path"]
        require(path.is_file() and not path.is_symlink() and sha(path) == item["before_sha256"], "unexpected original Omni source")
        text = path.read_text()
        require(text.count(item["old"]) == 1, "startup patch must match exactly once")
        updated = text.replace(item["old"], item["new"])
        require(hashlib.sha256(updated.encode()).hexdigest() == item["after_sha256"], "startup patch output differs")
        changes.append((path, updated))
    # Check every input before applying either edit. Only the fresh owned clone is passed here.
    for path, updated in changes:
        path.write_text(updated)
    return patch


def cosmos_metadata(root):
    """Declare the Omni utility-import stack; no Cosmos inference implementation changes."""
    path = Path(root) / "pyproject.toml"
    before = path.read_bytes()
    old = b'"transformers>=4.57.1,<5.0.0",'
    new = b'"transformers>=5.13.0,<5.15",\n    "iopath==0.1.10",'
    require(before.count(old) == 1, "unexpected Cosmos dependency declaration")
    after = before.replace(old, new)
    path.write_bytes(after)
    return {"path": "pyproject.toml", "before_sha256": hashlib.sha256(before).hexdigest(),
            "after_sha256": hashlib.sha256(after).hexdigest(), "old": old.decode(), "new": new.decode(),
            "scope": "Metadata-only compatibility for Omni's pinned Cosmos pose/transforms/UniPC helpers; not a qualified standalone Cosmos environment."}


def make_plan(framework, root, python, uv, *, allow_startup_patch=False, env_dir=None):
    catalog, sources = load()
    require(framework in {"lerobot", "vllm-omni"}, "unknown framework")
    require(framework != "vllm-omni" or allow_startup_patch, "Omni requires explicit startup-patch acknowledgement")
    root = Path(root).absolute()
    names = [framework] + (["cosmos-framework"] if framework == "vllm-omni" else [])
    versions = catalog["packages"][framework]
    packages = [f"{name}=={version}" for name, version in versions.items()]
    pins = catalog["public_wheels"]
    public_packages = [f"{name} @ {pins[name + '==' + version]['url']}#sha256={pins[name + '==' + version]['sha256']}"
                       if name + "==" + version in pins else f"{name}=={version}" for name, version in versions.items()]
    environment = Path(env_dir).absolute() if env_dir else root / "env"
    require(environment != root and not root.is_relative_to(environment), "environment must not contain the evidence root")
    return {"schema": "instinctflash.framework_bootstrap.v1", "framework": framework,
            "catalog_sha256": sha(DATA / "catalog.json"), "sources_sha256": sha(DATA / "sources.json"),
            "root": str(root), "environment": str(environment), "python": python, "uv": uv,
            "python_minor": "3.12", "machine": "aarch64", "sources": {name: sources[name] for name in names},
            "constraints": packages, "public_requirements": public_packages,
            "source_extras": ["pi", "groot", "lingbot_va"] if framework == "lerobot" else [],
            "build_requirements": ["pip", "setuptools>=77,<81", "wheel", "hatchling", "setuptools-scm==8.3.1"],
            "wheel_metadata_repair": catalog["native_wheel_repairs"][framework]["id"],
            "allow_startup_patch": allow_startup_patch, "gpu_qualified": False,
            "cosmos_dependency_metadata_overlay": framework == "vllm-omni",
            "scope": "Independent inference benchmark environment; fresh public source/wheels, no inherited site-packages or editable install."}


def clean_env(cache_dir=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "UV_", "IFL_", "CONDA_"))
           and k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    env.update(CUDA_VISIBLE_DEVICES="", GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1",
               UV_NO_CONFIG="1", UV_CONCURRENT_BUILDS="1", UV_CONCURRENT_DOWNLOADS="2",
               PYTHONDONTWRITEBYTECODE="1", PIP_CONFIG_FILE=os.devnull,
               VLLM_OMNI_TARGET_DEVICE="cuda", VLLM_OMNI_VERSION_OVERRIDE="0.29.0.dev0+gf7d9deb45")
    if cache_dir:
        env["UV_CACHE_DIR"] = str(Path(cache_dir).absolute())
    return env


def repaired_cusparselt(root, framework, *, wheel=None, receipt_path=None):
    """Reuse or reproduce the pinned NVIDIA metadata-only AArch64 correction."""
    path = REPO / "scripts/repair_vendor_wheel.py"
    spec = importlib.util.spec_from_file_location("framework_wheel_repair", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    catalog, _ = load()
    selected = catalog["native_wheel_repairs"][framework]
    rule = selected["rule"]
    folder = root / "wheel_repair"
    folder.mkdir()
    if wheel or receipt_path:
        require(wheel and receipt_path, "supplied repaired wheel and receipt must be paired")
        wheel = Path(wheel).resolve(strict=True)
        receipt = json.loads(Path(receipt_path).read_text())
        require(receipt["status"] == "passed" and receipt["all_other_payload_bytes_preserved"] is True
                and receipt["repair_source_sha256"] == sha(path) and receipt["public_source"] == rule
                and receipt["repaired"]["sha256"] == selected["repaired_sha256"]
                and sha(wheel) == receipt["repaired"]["sha256"], "supplied wheel repair is not the audited public artifact")
    else:
        original = folder / rule["filename"]
        with urllib.request.urlopen(rule["url"], timeout=60) as source, original.open("xb") as target:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                target.write(block)
        wheel = folder / "repaired" / rule["filename"]
        receipt = helper.repair(original, wheel, rule)
    require(sha(wheel) == selected["repaired_sha256"], "reproduced wheel repair differs from the pinned artifact")
    write(folder / "receipt.json", receipt)
    return wheel


def install(args):
    plan = make_plan(args.framework, args.root, args.python, args.uv, allow_startup_patch=args.allow_startup_patch, env_dir=args.env_dir)
    root = Path(plan["root"])
    require(not root.exists() and not root.is_symlink(), "bootstrap root already exists")
    require(not Path(plan["environment"]).exists() and not Path(plan["environment"]).is_symlink(), "environment destination already exists")
    root.mkdir(parents=True, exist_ok=False)
    write(root / "plan.json", plan)
    env = clean_env(args.cache_dir)
    commands = []
    receipt = {"schema": plan["schema"], "status": "failed", "plan_sha256": sha(root / "plan.json"),
               "bootstrap_sha256": sha(__file__), "commands": commands, "gpu_qualified": False}

    def command(argv, *, timeout=3600):
        row = {"argv": list(map(str, argv)), "log": f"command-{len(commands):02d}.log"}
        commands.append(row)
        with (root / row["log"]).open("x") as log:
            completed = subprocess.run(row["argv"], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        row["exit_code"] = completed.returncode
        require(completed.returncode == 0, f"command failed; preserved log: {root / row['log']}")

    try:
        command([args.python, "-I", "-c", "import platform,sys; assert sys.version_info[:2]==(3,12); assert platform.machine()=='aarch64'"])
        clones = {}
        for name, source in plan["sources"].items():
            clone = root / "sources" / name
            clone.parent.mkdir(exist_ok=True)
            command(["git", "clone", "--no-checkout", "--filter=blob:none", source["repository"], clone])
            command(["git", "-C", clone, "checkout", "--detach", source["revision"]])
            actual = subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], env=env, text=True).strip()
            require(actual == source["revision"], "source checkout revision differs")
            for name_notice, expected in source["notices"].items():
                require(sha(clone / name_notice) == expected, "upstream license/notice differs")
            clones[name] = clone
        if args.framework == "vllm-omni":
            receipt["startup_patch"] = apply_startup_patch(clones["vllm-omni"], acknowledged=args.allow_startup_patch)
            receipt["cosmos_metadata_overlay"] = cosmos_metadata(clones["cosmos-framework"])
        for name, clone in clones.items():
            spec = plan["sources"][name]
            for rel, expected in spec["installed_inventory"].items():
                require(sha(clone / spec["source_package_path"] / rel) == expected, f"prepared source differs: {name}/{rel}")
        constraint = root / "constraints.txt"
        constraint.write_text("\n".join(plan["constraints"]) + "\n")
        command([args.uv, "venv", "--python", args.python, plan["environment"]])
        python = Path(plan["environment"]) / "bin/python"
        common = [args.uv, "pip", "install", "--python", python, "--constraint", constraint, "--index-url", "https://pypi.org/simple"]
        native = [str(repaired_cusparselt(root, args.framework, wheel=args.repaired_wheel, receipt_path=args.repair_receipt))]
        command([*common, *plan["build_requirements"], *plan["public_requirements"], *native])
        requirements = ([str(clones["lerobot"]) + "[pi,groot,lingbot_va]"] if args.framework == "lerobot"
                        else [str(clones["cosmos-framework"]), str(clones["vllm-omni"])])
        command([*common, "--no-build-isolation", *requirements])
        command([args.uv, "pip", "check", "--python", python])
        command([python, "-I", "-m", "pip", "check"])
        # The check script uses stdlib metadata/path reads and imports classes only;
        # CUDA visibility is empty and no policy is constructed.
        command([python, "-I", __file__, "_verify", "--framework", args.framework,
                 "--receipt", str(root / "installed.json")])
        receipt.update(status="installed_cpu_imports_passed", installed_sha256=sha(root / "installed.json"),
                       environment=str(Path(plan["environment"]).resolve()))
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(root / "completion.json", receipt)
    return receipt


def verify(framework, output):
    source = REPO / "benchmarks/regression/framework_compare.py"
    spec = importlib.util.spec_from_file_location("framework_compare_bootstrap_check", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.catalog()
    sources = module.installed_sources(framework)
    packages = module.installed_packages(value["packages"][framework])
    if framework == "lerobot":
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        classes = [GrootPolicy, LingBotVAPolicy, PI05Policy]
    else:
        # A first-request utility import is not covered by importing Omni's
        # scheduler/model classes. Check it in its own disposable CPU process.
        guarded = str(output) + ".cosmos.json"
        subprocess.run([sys.executable, "-I", __file__, "_verify-cosmos", "--framework", framework,
                        "--receipt", guarded], env=clean_env(), timeout=120, check=True)
        cpu = json.loads(Path(guarded).read_text())
        require(cpu["status"] == "passed" and not cpu["blocked_operations"], "Cosmos CPU import guard failed")
        from vllm_omni import Omni
        from vllm_omni.diffusion.models.cosmos3.utils import build_robolab_unipc_scheduler
        from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
        classes = [Omni, build_robolab_unipc_scheduler, DreamZeroPipeline]
    write(output, {"status": "passed", "packages": packages, "sources": sources,
                   "imports": [c.__module__ + "." + c.__name__ for c in classes], "gpu_inference": False,
                   "cosmos_transform_cpu_guard": ({"path": guarded, "sha256": sha(guarded)}
                                                   if framework == "vllm-omni" else None)})


def verify_cosmos_transform(output):
    """Import the actual first-request symbol without model/device/weight work.

    Python operation controls diagnose trusted installed dependencies; they are
    not a native-code security sandbox. This runs only in a disposable child.
    """
    import functools
    import importlib
    import shutil
    import socket

    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", NO_ALBUMENTATIONS_UPDATE="1")
    blocked, plumbing = [], []

    def guard(event, args):
        caller = sys._getframe(1) if event in {"socket.bind", "subprocess.Popen"} else None
        if (event == "socket.bind" and getattr(args[0], "family", None) == socket.AF_INET6
                and args[1] == ("::1", 0) and caller.f_globals.get("__name__") == "urllib3.util.connection"
                and caller.f_code.co_name == "_has_ipv6"):
            plumbing.append("urllib3_ipv6_loopback_capability_probe")
            return
        if event == "subprocess.Popen":
            argv = list(args[1]) if not isinstance(args[1], str) else [args[1]]
            expected = {("uname", "-p"): ("platform", "from_subprocess", "/usr/bin/uname"),
                        ("lscpu",): ("numpy.testing._private.utils", "check_support_sve", "/usr/bin/lscpu"),
                        ("/sbin/ldconfig", "-p"): ("ctypes.util", "_findSoname_ldconfig", "/sbin/ldconfig")}.get(tuple(argv))
            executable = shutil.which(str(args[0]))
            if expected and executable and Path(executable).resolve() == Path(expected[2]).resolve():
                frame = caller
                for _ in range(10):
                    if frame is None:
                        break
                    if frame.f_globals.get("__name__") == expected[0] and frame.f_code.co_name == expected[1]:
                        plumbing.append("cpu_capability:" + " ".join(argv))
                        return
                    frame = frame.f_back
        weights = (event == "open" and isinstance(args[0], (str, bytes))
                   and os.fsdecode(args[0]).endswith((".safetensors", ".pt", ".pth", ".ckpt")))
        if weights or event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto",
                               "socket.sendmsg", "socket.bind", "subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"}:
            blocked.append(event)
            raise RuntimeError("operation_disabled_in_cpu_import")

    sys.addaudithook(guard)
    receipt = {"status": "failed", "gpu_inference": False, "model_constructed": False,
               "scope": "actual Cosmos ActionTransformPipeline symbol import; not a model or GPU qualification"}
    try:
        import torch

        def refuse(original):
            @functools.wraps(original)
            def denied(*args, **kwargs):
                blocked.append("torch.cuda.initialization")
                raise RuntimeError("GPU_initialization_disabled_in_cpu_import")
            return denied

        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        for name in ("_lazy_init", "init", "set_device", "current_device", "get_device_capability", "get_device_properties"):
            setattr(torch.cuda, name, refuse(getattr(torch.cuda, name)))
        module = importlib.import_module("cosmos_framework.data.generator.action.utils.transforms")
        actual = module.ActionTransformPipeline
        require(callable(actual), "native ActionTransformPipeline is unavailable")
        require(not blocked and not torch.cuda.is_initialized(), "CPU import attempted a forbidden operation")
        receipt.update(status="passed", module=module.__name__, symbol=actual.__name__,
                       source=str(Path(module.__file__).resolve()), source_sha256=sha(module.__file__))
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(blocked_operations=blocked, allowed_cpu_import_plumbing=plumbing)
        write(output, receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "install", "_verify", "_verify-cosmos"))
    parser.add_argument("--framework", choices=("lerobot", "vllm-omni"), required=True)
    parser.add_argument("--root")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--cache-dir")
    parser.add_argument("--env-dir", help="Optional NEW environment path, e.g. tmpfs; persistent evidence remains under --root")
    parser.add_argument("--repaired-wheel", help="Optional audited framework-specific cuSPARSELt wheel, consumed read-only")
    parser.add_argument("--repair-receipt", help="Required exact payload-preservation receipt for --repaired-wheel")
    parser.add_argument("--allow-startup-patch", action="store_true")
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)
    if args.operation in {"_verify", "_verify-cosmos"}:
        require(args.receipt, "verification receipt required")
        if args.operation == "_verify-cosmos":
            require(args.framework == "vllm-omni", "Cosmos utility check requires Omni")
            return verify_cosmos_transform(args.receipt)
        return verify(args.framework, args.receipt)
    require(args.root, "new bootstrap root required")
    value = install(args) if args.operation == "install" else make_plan(args.framework, args.root, args.python, args.uv,
                                                                      allow_startup_patch=args.allow_startup_patch, env_dir=args.env_dir)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
