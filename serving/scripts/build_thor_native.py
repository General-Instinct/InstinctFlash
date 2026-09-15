"""Build the public Torch Thor native stack in new directories only.

First build FA2 with build_thor_fa2.sh. Pass its completed library and SHA here.
This then builds the main engine, FMHA and BF16 library from exact source snapshots,
and packages a CPython 3.12 / Linux aarch64 flash-rt wheel. No model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from build_thor_fa2 import encoded, resolved_tool, run_capture, sha, write_json


CUTLASS_COMMIT = "da5e086dab31d63815acafdac9a9c5893b1c69e2"
CUTLASS_GROUPS = ("include", "tools/util/include", "examples/77_blackwell_fmha", "LICENSE.txt")
SKIP_PARTS = {"__pycache__", ".git", "build", "dist"}
SKIP_SUFFIXES = {".so", ".o", ".a", ".pyc", ".pyo"}


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--cutlass-root", type=Path, required=True)
    result.add_argument("--fa2-artifact", type=Path, required=True)
    result.add_argument("--fa2-sha256", required=True)
    result.add_argument("--build-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--python", default="python3.12")
    result.add_argument("--cmake", default="cmake")
    result.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    result.add_argument("--cxx", default="g++")
    result.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    result.add_argument("--timeout-seconds", type=int, default=3600)
    result.add_argument("--plan-only", action="store_true")
    return result


def regular_files(root, groups):
    result = []
    for name in groups:
        item = root / name
        if not item.exists():
            raise ValueError(f"missing build input: {item}")
        for path in sorted(item.rglob("*")) if item.is_dir() else [item]:
            relative = path.relative_to(root)
            if set(relative.parts) & SKIP_PARTS or path.suffix in SKIP_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(f"materialize source links explicitly: {path}")
            if path.is_file():
                result.append(path)
    return sorted(set(result))


def repository_groups(repo):
    groups = ["serving/CMakeLists.txt", "serving/csrc", "serving/flash_rt",
              "serving/pyproject.toml", "serving/setup.py", "serving/LICENSE",
              "instinctflash/native/bf16"]
    metadata = (repo / "serving/pyproject.toml").read_text()
    declared = re.search(r"(?m)^readme\s*=\s*(['\"])([^'\"]+)\1\s*$", metadata)
    if declared:
        readme = Path(declared.group(2))
        if readme.is_absolute() or ".." in readme.parts:
            raise ValueError("package readme must remain inside serving/")
        groups.append(str(Path("serving") / readme))
    elif re.search(r"(?m)^readme\s*=", metadata):
        raise ValueError("builder needs an explicit readme filename or no readme metadata")
    return tuple(groups)


def make_plan(args):
    repo = args.repo_root.resolve(strict=True)
    cutlass = args.cutlass_root.resolve(strict=True)
    fa2 = args.fa2_artifact.resolve(strict=True)
    build, output = args.build_root.resolve(), args.output_dir.resolve()
    if build.exists() or output.exists():
        raise ValueError("build and output directories must be new")
    for left, right in ((build, output), (output, build), (repo, build), (repo, output),
                        (cutlass, build), (cutlass, output)):
        if left == right or left in right.parents:
            raise ValueError("build/output/input roots must be disjoint")
    if not re.fullmatch(r"[0-9a-f]{64}", args.fa2_sha256) or sha(fa2) != args.fa2_sha256:
        raise ValueError("FA2 artifact hash does not match")
    if "cpython-312-aarch64-linux-gnu" not in fa2.name:
        raise ValueError("FA2 input must be CPython 3.12 aarch64")
    if not 1 <= args.timeout_seconds <= 7200:
        raise ValueError("timeout must be in 1..7200 seconds")
    commit = subprocess.check_output(["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True).strip()
    if commit != CUTLASS_COMMIT:
        raise ValueError(f"this qualified build recipe pins CUTLASS {CUTLASS_COMMIT}; got {commit}")
    source_files = regular_files(repo, repository_groups(repo))
    cutlass_files = regular_files(cutlass, CUTLASS_GROUPS)
    return {"kind": "instinctflash_thor_native_build_v1", "repo_root": str(repo),
            "cutlass_root": str(cutlass), "cutlass_commit": commit,
            "fa2_artifact": str(fa2), "fa2_sha256": args.fa2_sha256,
            "build_root": str(build), "output_dir": str(output), "jobs": args.jobs,
            "tools": {key: resolved_tool(getattr(args, key)) for key in ("python", "cmake", "nvcc", "cxx")},
            "source_files": {str(p.relative_to(repo)): sha(p) for p in source_files},
            "cutlass_files": {str(p.relative_to(cutlass)): sha(p) for p in cutlass_files},
            "builder_sha256": sha(__file__), "helper_sha256": sha(Path(__file__).with_name("build_thor_fa2.py")),
            "gpu_inference_requested": False}


def copy_snapshot(source, destination, inventory):
    for relative, expected in inventory.items():
        src, dst = source / relative, destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(src) != expected or sha(dst) != expected:
            raise ValueError(f"build input changed: {src}")


def validate_wheel(path):
    if not path.name.endswith("-cp312-cp312-linux_aarch64.whl"):
        raise ValueError(f"incorrect native wheel tag: {path.name}")
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        needed = ["flash_rt/flash_rt_kernels.cpython-312-aarch64-linux-gnu.so",
                  "flash_rt/flash_rt_fa2.cpython-312-aarch64-linux-gnu.so",
                  "flash_rt/libfmha_fp16_strided.so"]
        if not all(name in names for name in needed):
            raise ValueError("native wheel lacks one of kernels, FA2 or FMHA")
        if any(name.endswith((".pyc", ".pyo", ".o", ".a")) for name in names):
            raise ValueError("native wheel contains stale build/bytecode artifacts")
        if any(name.endswith(".so") and name not in needed for name in names):
            raise ValueError("native wheel contains an unqualified extra library")
        if "flash_rt/frontends/torch/vla4b_thor.py" not in names:
            raise ValueError("public native wheel must retain executable frontend source")
        notices = [f"flash_rt/notices/{name}" for name in
                   ("FlashAttention-LICENSE.txt", "CUTLASS-FA2-LICENSE.txt", "CUTLASS-engine-LICENSE.txt")]
        if not all(name in names and wheel.read(name).strip() for name in notices):
            raise ValueError("native wheel lacks one of the three third-party license notices")
        metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1 or "Root-Is-Purelib: false" not in wheel.read(metadata[0]).decode():
            raise ValueError("native wheel is mislabeled as pure Python")
        return {name: hashlib.sha256(wheel.read(name)).hexdigest() for name in names}


def build(args, spec):
    started = time.time()
    root, output = Path(spec["build_root"]), Path(spec["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", spec)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    tools, commands = spec["tools"], []

    def logged(command, name):
        commands.append(command)
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {name}", flush=True)
        with (output / name).open("x") as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout_seconds)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)

    try:
        if platform.machine() != "aarch64":
            raise ValueError("native Thor build requires aarch64")
        probe = "import json,sys,sysconfig;print(json.dumps({'version':list(sys.version_info[:2]),'soabi':sysconfig.get_config_var('SOABI')}))"
        python = json.loads(run_capture([tools["python"], "-I", "-B", "-c", probe], env=env))
        if python["version"] != [3, 12] or python["soabi"] != "cpython-312-aarch64-linux-gnu":
            raise ValueError("selected Python must be CPython 3.12 aarch64")
        write_json(output / "toolchain.json", {"python": python,
                   "versions": {k: run_capture([v, "--version"], env=env) for k, v in tools.items()},
                   "sha256": {k: sha(v) for k, v in tools.items()},
                   "build_environment_is_clean_runtime": False})
        source = root / "source"
        copy_snapshot(Path(spec["repo_root"]), source, spec["source_files"])
        cutlass = source / "serving/third_party/cutlass"
        copy_snapshot(Path(spec["cutlass_root"]), cutlass, spec["cutlass_files"])
        cmake, native_build = tools["cmake"], root / "objects"
        common = ["-G", "Unix Makefiles", "-DCMAKE_BUILD_TYPE=Release",
                  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_CUDA_ARCHITECTURES=OFF",
                  f"-DCMAKE_CUDA_COMPILER={tools['nvcc']}", f"-DCMAKE_CXX_COMPILER={tools['cxx']}"]
        logged([cmake, "-S", str(source / "serving"), "-B", str(native_build), *common,
                f"-DPython3_EXECUTABLE={tools['python']}", "-DGPU_ARCH=110",
                "-DFLASH_RT_BUILD_JAX_FFI=OFF", "-DCCACHE_PROGRAM=OFF"],
               "configure.log")
        logged([cmake, "--build", str(native_build), "--target", "flash_rt_kernels", "fmha_fp16_strided",
                "--parallel", str(args.jobs), "--verbose"], "compile.log")
        bf16_build = root / "bf16_objects"
        logged([cmake, "-S", str(source / "instinctflash/native/bf16"), "-B", str(bf16_build),
                *common, f"-DCUTLASS_ROOT={cutlass}"], "bf16_configure.log")
        logged([cmake, "--build", str(bf16_build), "--parallel", str(args.jobs), "--verbose"],
               "bf16_compile.log")
        package = source / "serving/flash_rt"
        (package / "notices").mkdir(exist_ok=True)
        shutil.copyfile(cutlass / "LICENSE.txt", package / "notices/CUTLASS-engine-LICENSE.txt")
        fa2 = Path(spec["fa2_artifact"])
        if sha(fa2) != spec["fa2_sha256"]:
            raise ValueError("FA2 changed before packaging")
        shutil.copyfile(fa2, package / fa2.name)
        libraries = list(package.glob("*.so")) + [bf16_build / "libinstinctflash_bf16.so"]
        if len(libraries) != 4 or any(not p.is_file() for p in libraries):
            raise ValueError("expected exactly four newly built/bound native libraries")
        native_dir = output / "native"
        native_dir.mkdir()
        elf = {}
        for lib in libraries:
            shutil.copyfile(lib, native_dir / lib.name)
            cuobjdump = Path(tools["nvcc"]).resolve().parent / "cuobjdump"
            cubins = run_capture([str(cuobjdump), "--list-elf", str(lib)], env=env)
            architectures = set(re.findall(r"sm_\d+a?", cubins))
            if not architectures or not architectures <= {"sm_110", "sm_110a"}:
                raise ValueError(f"unexpected device architecture in {lib}: {architectures}")
            elf[lib.name] = {"sha256": sha(lib), "bytes": lib.stat().st_size,
                             "device_architectures": sorted(architectures), "cubins": cubins,
                             "dynamic": run_capture(["readelf", "-d", str(lib)], env=env)}
        write_json(output / "native_libraries.json", elf)
        # Load extension modules without package import and resolve C entry points only.
        probe = """import ctypes,importlib.util,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);result={}
for name,exports in [('flash_rt_kernels',('GemmRunner','FvkContext')),('flash_rt_fa2',('fwd_fp16','fwd_bf16','fwd_bf16_causal'))]:
 f=next(p.glob(name+'.*.so'));s=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 assert all(callable(getattr(m,key,None)) for key in exports);result[name]=list(exports)
lib=ctypes.CDLL(str(p/'libfmha_fp16_strided.so'));assert getattr(lib,'fmha_fp16_strided',None)
result['fmha_fp16_strided']='resolved_only'
lib=ctypes.CDLL(str(p/'libinstinctflash_bf16.so'));assert getattr(lib,'instinctflash_bf16_linear_relu2',None)
assert lib.instinctflash_bf16_abi_version()==1;result['instinctflash_bf16_abi_version']=1
result['inference_calls']=0
print(json.dumps(result))"""
        imported = json.loads(run_capture([tools["python"], "-I", "-B", "-c", probe,
                                          str(native_dir)], env=env))
        write_json(output / "cpu_import.json", imported)
        # No dependency resolution or installation; setuptools builds this snapshot only.
        logged([tools["python"], "-I", "-B", "-m", "pip", "wheel", "--no-deps", "--no-index",
                "--no-build-isolation", "--wheel-dir", str(output / "wheels"), str(source / "serving")],
               "wheel.log")
        wheels = list((output / "wheels").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("expected exactly one flash-rt wheel")
        inventory = validate_wheel(wheels[0])
        for name, metadata in elf.items():
            if name != "libinstinctflash_bf16.so" and inventory.get(f"flash_rt/{name}") != metadata["sha256"]:
                raise ValueError(f"packaged native library differs from the qualified build: {name}")
        for name in ("FlashAttention-LICENSE.txt", "CUTLASS-FA2-LICENSE.txt", "CUTLASS-engine-LICENSE.txt"):
            if inventory[f"flash_rt/notices/{name}"] != sha(package / "notices" / name):
                raise ValueError(f"packaged license notice differs from the build input: {name}")
        write_json(output / "wheel_inventory.json", inventory)
        dependencies = {}
        for tree in (native_build, bf16_build):
            for file in tree.rglob("*.o.d"):
                for word in shlex.split(file.read_text().replace("\\\n", " "))[1:]:
                    path = Path(word)
                    if not path.is_absolute():
                        path = tree / path
                    if path.is_file():
                        dependencies[str(path.resolve())] = sha(path)
        write_json(output / "compiler_dependencies.json", dependencies)
        for label, tree in (("engine", native_build), ("bf16", bf16_build)):
            for name in ("CMakeCache.txt", "compile_commands.json"):
                shutil.copyfile(tree / name, output / f"{label}_{name}")
        result = {"kind": "instinctflash_thor_native_build_result_v1", "status": "passed",
                  "plan_sha256": sha(output / "plan.json"), "commands": commands,
                  "wheel": str(wheels[0]), "wheel_sha256": sha(wheels[0]),
                  "native_libraries": {k: v["sha256"] for k, v in elf.items()},
                  "elapsed_seconds": time.time() - started, "gpu_inference_calls": 0,
                  "clean_runtime_install_qualified": False, "numerical_or_performance_qualified": False}
        write_json(output / "completion.json", result)
        print(encoded(result).decode(), flush=True)
        return result
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": repr(error),
                   "commands": commands, "elapsed_seconds": time.time() - started})
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    spec = make_plan(args)
    if args.plan_only:
        print(encoded(spec).decode())
        return 0
    build(args, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
