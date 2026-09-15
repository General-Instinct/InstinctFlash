"""Fresh, FA2-only Thor build with source receipts and no environment installation.

This raw-pointer extension uses CPython/pybind11 and CUDA, not the Torch C++ ABI.
The isolated CMake project avoids the main engine's historical SM110 FA2 exclusion.
No GPU discovery, CUDA context, or inference is used by the builder or import check.
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
import tarfile
import time


SOURCE_GROUPS = (
    "csrc/fa2_bindings.cpp",
    "csrc/attention/fa2_wrapper.cu",
    "csrc/attention/fa2_wrapper_causal.cu",
    "csrc/attention/fa2_causal_inst",
    "csrc/attention/flash_attn_2_src",
)
DTYPES = ("fp16", "bf16")
HEAD_DIMS = (96, 128, 256)
EXPORTS = ("fwd_fp16", "fwd_bf16", "fwd_bf16_causal")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())


def source_files(root):
    result = []
    for name in SOURCE_GROUPS:
        path = root / name
        if not path.exists():
            raise ValueError(f"missing vendored source: {path}")
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for item in files:
            if item.is_symlink():
                raise ValueError(f"source symlink requires explicit materialization: {item}")
            if item.is_file():
                result.append(item)
    return sorted(set(result))


def kernel_sources():
    base = "csrc/attention/flash_attn_2_src/flash_attn"
    files = [
        f"{base}/flash_fwd{split}_hdim{head}_{dtype}_sm80.cu"
        for head in HEAD_DIMS
        for dtype in DTYPES
        for split in ("", "_split")
    ]
    files += [
        "csrc/attention/fa2_wrapper.cu",
        "csrc/attention/fa2_wrapper_causal.cu",
        "csrc/attention/fa2_causal_inst/flash_fwd_hdim128_bf16_sm80_causal.cu",
        "csrc/attention/fa2_causal_inst/flash_fwd_split_hdim128_bf16_sm80_causal.cu",
    ]
    return files


def cmake_source():
    sources = "\n  ".join(kernel_sources())
    return f"""cmake_minimum_required(VERSION 3.24)
project(instinctflash_thor_fa2 LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES OFF)
find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${{Python3_EXECUTABLE}}" -I -B -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_library(fa2_vendor_obj OBJECT
  {sources})
set_target_properties(fa2_vendor_obj PROPERTIES CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
target_include_directories(fa2_vendor_obj PRIVATE csrc
  csrc/attention/flash_attn_2_src csrc/attention/flash_attn_2_src/flash_attn
  csrc/attention/flash_attn_2_src/cutlass/include)
target_compile_definitions(fa2_vendor_obj PRIVATE FLASH_NAMESPACE=fa2_vendor
  FA2_HAS_HDIM_96=1 FA2_HAS_HDIM_128=1 FA2_HAS_HDIM_256=1
  FA2_HAS_FP16=1 FA2_HAS_BF16=1)
target_compile_options(fa2_vendor_obj PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda
    -O3 --use_fast_math -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    "SHELL:-Xcompiler -Wno-deprecated-declarations"
    "SHELL:-gencode arch=compute_110,code=sm_110" "SHELL:--threads 1">)
pybind11_add_module(flash_rt_fa2 NO_EXTRAS csrc/fa2_bindings.cpp)
set_target_properties(flash_rt_fa2 PROPERTIES CXX_STANDARD 17
  LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/library/flash_rt")
target_sources(flash_rt_fa2 PRIVATE $<TARGET_OBJECTS:fa2_vendor_obj>)
target_link_libraries(flash_rt_fa2 PRIVATE CUDA::cudart)
"""


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
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


def resolved_tool(value):
    path = shutil.which(str(value))
    if path is None:
        raise ValueError(f"required build tool not found: {value}")
    return str(Path(path).absolute())


def plan(args):
    source = args.source_root.resolve(strict=True)
    build = args.build_root.resolve()
    output = args.output_dir.resolve()
    if build.exists() or build.is_symlink() or output.exists() or output.is_symlink():
        raise ValueError("build-root and output-dir must both be new; existing evidence is never reused")
    for first, second in ((build, output), (output, build), (source, build), (source, output)):
        if first == second or first in second.parents:
            raise ValueError("source, build-root and output-dir must be disjoint")
    if args.timeout_seconds < 1 or args.timeout_seconds > 7200:
        raise ValueError("build timeout must be in 1..7200 seconds")
    files = source_files(source)
    for relative in kernel_sources():
        if source / relative not in files:
            raise ValueError(f"missing required kernel instantiation: {relative}")
    return {
        "kind": "instinctflash_thor_fa2_build_v1",
        "source_root": str(source), "build_root": str(build), "output_dir": str(output),
        "architecture": "sm_110", "python_abi": "cp312", "jobs": args.jobs,
        "nvcc_threads_per_job": 1, "head_dims": list(HEAD_DIMS), "dtypes": list(DTYPES),
        "causal": {"head_dim": 128, "dtype": "bf16"},
        "torch_cpp_abi_dependency": False, "gpu_inference_requested": False,
        "tools": {key: resolved_tool(getattr(args, key)) for key in ("python", "cmake", "nvcc", "cxx")},
        "source_files": {str(p.relative_to(source)): sha(p) for p in files},
        "builder_sha256": sha(__file__), "cmake_source_sha256": hashlib.sha256(cmake_source().encode()).hexdigest(),
    }


def run_capture(command, *, env, cwd=None, timeout=60):
    return subprocess.check_output(command, cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT,
                                   timeout=timeout).strip()


def no_ptx_in_listing(output):
    """CUDA 13.2 prints an informational line for an empty PTX inventory."""
    return not output.strip() or bool(re.fullmatch(
        r"cuobjdump info\s*: No PTX file found to extract from '[^\n]+'. "
        r"You may try with -all option\.", output.strip()))


def build_artifact(args, spec):
    started = time.time()
    output = Path(spec["output_dir"])
    root = Path(spec["build_root"])
    output.mkdir(parents=True, exist_ok=False)
    root.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", spec)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    tools = spec["tools"]
    commands = []

    def logged(command, filename):
        commands.append(command)
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {filename}", flush=True)
        with (output / filename).open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                returncode = process.wait(timeout=args.timeout_seconds)
            except BaseException:
                # Bound this builder's compiler descendants, never another process group.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)

    try:
        if platform.machine() != "aarch64":
            raise ValueError("this Thor-native builder requires an aarch64 host")
        probe = """import importlib.metadata as m,json,platform,sys,sysconfig
def version(name):
 try: return m.version(name)
 except m.PackageNotFoundError: return None
print(json.dumps({'python':sys.version,'executable':sys.executable,'version':list(sys.version_info[:2]),
 'machine':platform.machine(),'soabi':sysconfig.get_config_var('SOABI'),
 'python_include':sysconfig.get_path('include'),'pybind11':version('pybind11'),
 'torch_metadata_only':version('torch')}))"""
        python = json.loads(run_capture([tools["python"], "-I", "-B", "-c", probe], env=env))
        if python["version"] != [3, 12] or python["machine"] != "aarch64" or not python["pybind11"]:
            raise ValueError("selected build Python must be CPython 3.12 aarch64 with pybind11 installed")
        versions = {key: run_capture([value, "--version"], env=env) for key, value in tools.items()}
        toolchain = {"python": python, "versions": versions,
                     "executables_sha256": {key: sha(value) for key, value in tools.items()},
                     "build_environment_is_clean_runtime": False}
        write_json(output / "toolchain.json", toolchain)
        source = root / "source"
        for relative, expected in spec["source_files"].items():
            src = Path(spec["source_root"]) / relative
            dst = source / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if sha(src) != expected or sha(dst) != expected:
                raise ValueError(f"source changed while taking build snapshot: {src}")
        (source / "CMakeLists.txt").write_text(cmake_source())
        shutil.copyfile(__file__, output / "build_thor_fa2.py")
        shutil.copyfile(Path(__file__).with_suffix(".sh"), output / "build_thor_fa2.sh")
        build = root / "objects"
        logged([tools["cmake"], "-S", str(source), "-B", str(build), "-G", "Unix Makefiles",
                "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                "-DCMAKE_CUDA_ARCHITECTURES=OFF", f"-DCMAKE_CUDA_COMPILER={tools['nvcc']}",
                f"-DCMAKE_CXX_COMPILER={tools['cxx']}", f"-DPython3_EXECUTABLE={tools['python']}"],
               "configure.log")
        logged([tools["cmake"], "--build", str(build), "--target", "flash_rt_fa2", "--parallel",
                str(args.jobs), "--verbose"], "compile.log")
        libraries = list((build / "library" / "flash_rt").glob("flash_rt_fa2*.so"))
        if len(libraries) != 1 or "cpython-312-aarch64-linux-gnu" not in libraries[0].name:
            raise ValueError("expected exactly one CPython 3.12 aarch64 FA2 library")
        library = libraries[0]
        cuobjdump = Path(tools["nvcc"]).resolve().parent / "cuobjdump"
        cubins = run_capture([str(cuobjdump), "--list-elf", str(library)], env=env)
        ptx = run_capture([str(cuobjdump), "--list-ptx", str(library)], env=env)
        architectures = set(re.findall(r"sm_\d+a?", cubins))
        if architectures != {"sm_110"} or not no_ptx_in_listing(ptx):
            raise ValueError(f"unexpected compiled architectures/PTX: {architectures}, {ptx!r}")
        (output / "cubins.txt").write_text(cubins + "\n")
        (output / "ptx.txt").write_text(ptx + "\n")
        import_probe = """import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('flash_rt_fa2',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
names=('fwd_fp16','fwd_bf16','fwd_bf16_causal')
assert all(callable(getattr(module,name,None)) for name in names)
print(json.dumps({'module':module.__file__,'exports':list(names),'inference_calls':0}))"""
        imported = json.loads(run_capture([tools["python"], "-I", "-B", "-c", import_probe,
                                          str(library)], env=env))
        write_json(output / "cpu_import.json", imported)
        dependencies = {}
        for depfile in build.rglob("*.o.d"):
            words = shlex.split(depfile.read_text().replace("\\\n", " "))
            for word in words[1:]:
                dependency = Path(word)
                if not dependency.is_absolute():
                    dependency = build / dependency
                if dependency.is_file():
                    dependencies[str(dependency.resolve())] = sha(dependency)
        if not dependencies or not any("pybind11" in path for path in dependencies):
            raise ValueError("compiler dependency inventory lacks pybind11 headers")
        write_json(output / "compiler_dependencies.json", dependencies)
        for filename in ("CMakeCache.txt", "compile_commands.json"):
            shutil.copyfile(build / filename, output / filename)
        shutil.copyfile(source / "CMakeLists.txt", output / "CMakeLists.txt")
        installed = output / "flash_rt" / library.name
        installed.parent.mkdir()
        shutil.copyfile(library, installed)
        if sha(installed) != sha(library):
            raise ValueError("library copy hash mismatch")
        # Preserve the exact vendored source/licenses with the portable binary.
        source_archive = output / "source_snapshot.tar.gz"
        with tarfile.open(source_archive, "w:gz") as archive:
            archive.add(source, arcname="source")
        bundle = output / "instinctflash-thor-fa2-cp312-linux_aarch64.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for name in ("flash_rt", "plan.json", "toolchain.json", "CMakeLists.txt",
                         "build_thor_fa2.py", "build_thor_fa2.sh", "compiler_dependencies.json",
                         "cpu_import.json", "cubins.txt", "source_snapshot.tar.gz"):
                archive.add(output / name, arcname=name)
        result = {"status": "passed", "kind": "instinctflash_thor_fa2_build_result_v1",
                  "plan_sha256": sha(output / "plan.json"), "library": str(installed),
                  "library_sha256": sha(installed), "library_bytes": installed.stat().st_size,
                  "bundle": str(bundle), "bundle_sha256": sha(bundle),
                  "commands": commands, "elapsed_seconds": time.time() - started,
                  "gpu_inference_calls": 0, "runtime_clean_install_qualified": False,
                  "numerical_or_performance_qualified": False}
        write_json(output / "completion.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": repr(error),
                   "commands": commands, "elapsed_seconds": time.time() - started})
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    spec = plan(args)
    if args.plan_only:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return 0
    build_artifact(args, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
