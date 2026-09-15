"""CPU provenance gates and post-inference generated-code evidence only."""
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def validate_environment(binding, prefix, python_version, machine):
    require(binding["schema"] == "instinctflash.nano_compiler_binding.v1", "Unknown compiler binding")
    require(binding["mode"] == "observed" and binding["requests"] == 25, "Compiler study contract differs")
    require(list(python_version[:2]) == binding["python"] == [3, 13], "Compiler wheel needs CPython3.13")
    require(machine == binding["platform_machine"] == "aarch64", "Compiler wheel needs aarch64")
    root = Path(prefix).resolve()
    require(str(root) not in binding["forbidden_original_prefixes"], "Original fresh environment must remain untouched")
    require(str(root) == binding["required_prefix"], "Use the prospectively selected isolated compiler environment")
    require(len(binding["package_files"]) == 348, "Incomplete compiler package binding")
    require(binding["wheel"]["version"] == "3.6.0", "Unexpected compiler version")
    return root


def verify_compiler(binding):
    root = validate_environment(binding, sys.prefix, sys.version_info, platform.machine())
    triton = importlib.metadata.distribution("triton")
    require(triton.version == binding["wheel"]["version"], "Installed Triton version differs")
    files = {}
    for relative, expected in binding["package_files"].items():
        require(Path(relative).parts[0] == "triton" and ".." not in Path(relative).parts, "Unsafe compiler member")
        path = Path(triton.locate_file(relative)).resolve()
        require(path.is_relative_to(root), "Compiler package escapes the isolated environment")
        require(path.is_file() and path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
                f"Compiler payload differs: {relative}")
        files[relative] = ref(path)
    metadata = {}
    for name, expected in binding["distribution_metadata"].items():
        members = [p for p in triton.files or () if str(p).endswith(".dist-info/" + name)]
        require(len(members) == 1, "Compiler distribution metadata is ambiguous")
        path = Path(triton.locate_file(members[0])).resolve()
        require(path.is_relative_to(root) and sha(path) == expected, "Compiler wheel metadata differs")
        metadata[name] = ref(path)
    torch = importlib.metadata.distribution("torch")
    require(torch.version == binding["torch_codecache"]["version"], "Torch package version differs")
    codecache = Path(torch.locate_file(binding["torch_codecache"]["relative"])).resolve()
    require(codecache.is_relative_to(root) and sha(codecache) == binding["torch_codecache"]["sha256"], "Pinned codecache API/source differs")
    return {"status": "exact_public_compiler_payload_verified", "prefix": str(root), "files": files,
            "distribution_metadata": metadata, "torch_codecache": ref(codecache), "public_wheel": binding["wheel"]}


def verify_loaded_compiler(binding, installed, runtime):
    native = runtime["loaded_native_libraries_after_inference"]
    triton = installed["files"]["triton/_C/libtriton.so"]
    require(native.get(triton["path"]) == triton, "Actual loaded Triton compiler differs from bound public wheel")
    packages = {path.split("/site-packages/", 1)[1]: row["sha256"] for path, row in native.items()
                if "/site-packages/" in path}
    for relative, expected in binding["unchanged_loaded_package_libraries"].items():
        require(packages.get(relative) == expected, f"A non-Triton loaded package library changed: {relative}")
    require(runtime["torch_git_version"] == binding["torch_codecache"]["git_version"], "Actual Torch build changed")
    return {"status": "loaded_compiler_and_control_package_libraries_verified", "compiler": triton,
            "unchanged_control_package_libraries": len(binding["unchanged_loaded_package_libraries"]),
            "scope": "Actual mapped Triton compiler plus original control's mapped package libraries; generated launcher/driver paths are retained separately."}


def prepare_fresh_caches(binding, output, environment):
    root = Path(output).resolve()
    for key, expected in binding["matched_compiler_environment"].items():
        require(environment.get(key) == expected, f"Compiler control environment differs: {key}")
    selected = {}
    for key, relative in binding["fresh_cache_directories"].items():
        path = root / relative
        require(".." not in Path(relative).parts and not Path(relative).is_absolute(), "Unsafe fresh cache path")
        require(environment.get(key) == str(path), f"Explicit fresh compiler cache is required: {key}")
        require(not path.exists() and not path.is_symlink(), "A compiler cache already exists")
        selected[key] = path
    for path in selected.values():
        path.mkdir(parents=True, exist_ok=False)
        require(not list(path.iterdir()), "New compiler cache was not empty")
    return {"status": "created_exact_new_empty_caches", "paths": {k: str(v) for k, v in selected.items()},
            "matched_environment": binding["matched_compiler_environment"], "initial_entries": 0}


def _json_metadata(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_metadata(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if hasattr(value, "_asdict"):
        return _json_metadata(value._asdict())
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "Non-string compiler metadata key")
        return {k: _json_metadata(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_metadata(v) for v in value]
    require(value is None or isinstance(value, (str, int, float, bool)), "Unexpected compiler metadata type")
    return value


def collect_generated(modules, output, *, max_bytes, max_files, allowed_source_root, allowed_kernel_cache_roots=None):
    """Read only already loaded Python modules and materialized launcher data.

    No lazy handle initialization, compiler calls, target queries or kernel
    launch methods are invoked. Selected cached launchers are not claimed to be
    a complete dynamic trace of all executed CUDA kernels.
    """
    require(isinstance(modules, list) and modules, "Pinned PyCodeCache.modules is empty or unavailable")
    root = Path(output)
    root.mkdir(exist_ok=False)
    artifacts, module_rows, kernel_rows = {}, [], []
    total = 0

    def store(data, extension):
        nonlocal total
        require(isinstance(data, bytes), "Generated evidence is not bytes")
        require(extension in ("py", "ttir", "ttgir", "llir", "ptx", "cubin", "json"), "Unexpected generated artifact type")
        digest = hashlib.sha256(data).hexdigest()
        name = digest + "." + extension
        if name not in artifacts:
            require(len(artifacts) < max_files and total + len(data) <= max_bytes, "Generated evidence exceeded its finite bound")
            path = root / name
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            artifacts[name] = {"path": str(path.resolve()), "sha256": digest, "bytes": len(data)}
            total += len(data)
        return artifacts[name]

    seen_modules, seen_kernels = set(), set()
    for module in modules:
        if id(module) in seen_modules:
            continue
        seen_modules.add(id(module))
        source = Path(module.__file__)
        require(source.resolve().is_relative_to(Path(allowed_source_root).resolve()), "Generated module came from an unbound compiler cache")
        require(source.is_file() and source.suffix == ".py", "Loaded generated module has no source file")
        copied = store(source.read_bytes(), "py")
        module_rows.append({"original": ref(source), "copied": copied, "name": str(module.__name__)})
        for symbol, autotuner in vars(module).items():
            if type(autotuner).__module__ != "torch._inductor.runtime.triton_heuristics":
                continue
            for launcher in vars(autotuner).get("launchers", ()):
                kernel = vars(launcher).get("bin")
                if kernel is None or id(kernel) in seen_kernels:
                    continue
                seen_kernels.add(id(kernel))
                fields = vars(kernel)
                if "asm" not in fields or "metadata" not in fields:
                    continue
                metadata = _json_metadata(fields["metadata"])
                row = {"python_module_sha256": copied["sha256"], "symbol": symbol,
                       "kernel_hash": fields.get("hash"), "kernel_name": fields.get("name"),
                       "handles_already_initialized": fields.get("module") is not None,
                       "metadata": metadata, "artifacts": {}}
                if allowed_kernel_cache_roots is not None:
                    group = fields.get("metadata_group")
                    require(isinstance(group, dict) and group, "Selected launcher has no compiled-cache provenance")
                    row["metadata_group"] = {}
                    for name, filename in group.items():
                        path = Path(filename).resolve()
                        require(any(path.is_relative_to(Path(base).resolve()) for base in allowed_kernel_cache_roots),
                                "Selected launcher came from a previous or shared compiler cache")
                        row["metadata_group"][name] = ref(path)
                for kind, value in fields["asm"].items():
                    if kind not in ("ttir", "ttgir", "llir", "ptx", "cubin"):
                        continue
                    row["artifacts"][kind] = store(value.encode() if isinstance(value, str) else value, kind)
                kernel_rows.append(row)
    require(module_rows, "No generated Python source was captured")
    manifest = {"status": "captured_loaded_generated_modules", "modules": module_rows,
                "selected_cached_launchers": kernel_rows, "artifacts": artifacts,
                "bytes": total, "dynamic_kernel_execution_trace": False,
                "scope": "Post-inference copied PyCodeCache.modules source and already materialized selected launcher IR/binaries; no compilation or native calls."}
    with (root / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": manifest["status"], "manifest": ref(root / "manifest.json"),
            "modules": len(module_rows), "selected_cached_launchers": len(kernel_rows),
            "files": len(artifacts), "bytes": total, "dynamic_kernel_execution_trace": False}
