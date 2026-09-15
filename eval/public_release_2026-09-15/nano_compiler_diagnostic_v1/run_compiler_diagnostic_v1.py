"""Prospective Nano diagnostic with a fixed public compiler artifact.

No source installation or GPU scheduling. Run each mode in a fresh, externally
bounded process. Instrumented timings are not benchmark results.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_inputs(binding, matrix, fixture):
    require(binding["schema"] == "instinctflash.nano_conditioning_diagnostic.v1", "Unknown binding")
    require(sha(matrix) == binding["matrix_sha256"], "Matrix differs")
    require(sha(fixture) == binding["fixture_sha256"], "Fixture differs")
    data = json.loads(Path(matrix).read_text())
    cells = [row for row in data["cells"] if row["id"] == "nano-runtime_selected"]
    require(len(cells) == 1 and cells[0] == binding["cell"], "Selected cell differs")
    require(binding["requests"] == 25 and binding["trace_requests"] == [0], "Trace protocol differs")
    require(binding["modes"] == ["observed", "conditioning-bypass"], "Mode list differs")
    require(cells[0]["effective_schedule"] == {"guidance": 3.0, "nfe": {"action": 4, "prefix": 1}, "sampler": "unipc", "shift": 5.0, "steps": 4}, "Native schedule changed")
    return cells[0]


def validate_isolation(flags, environment):
    require(bool(flags.isolated), "Use python -I; installed imports must not resolve from the checkout")
    require(not environment.get("PYTHONPATH"), "PYTHONPATH must be unset")


def verify_sources(binding, resolver=None):
    packages = {"instinctflash": "instinctflash", "benchmarks": "instinctflash",
                "cosmos3_iwm": "cosmos3-iwm", "cosmos_framework": "cosmos-framework"}
    resolver = resolver or (lambda dist, path: importlib.metadata.distribution(dist).locate_file(path))
    verified = {}
    for relative, expected in binding["module_sources"].items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe source path")
        resolved = Path(resolver(packages[path.parts[0]], relative)).resolve()
        require(resolved.is_file() and sha(resolved) == expected, f"Installed source differs: {relative}")
        verified[relative] = ref(resolved)
    return verified


def fingerprint(value, torch, np):
    if isinstance(value, torch.Tensor):
        require(value.layout == torch.strided, "Unsupported tensor layout")
        data = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return {"kind": "tensor", "dtype": str(value.dtype), "shape": list(value.shape),
                "stride": list(value.stride()), "device": str(value.device),
                "value_sha256": hashlib.sha256(data).hexdigest()}
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "Object arrays are not admitted")
        return {"kind": "numpy", "dtype": value.dtype.str, "shape": list(value.shape),
                "value_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), "Non-string dictionary key")
        return {key: fingerprint(value[key], torch, np) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item, torch, np) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    require(value is None or isinstance(value, (str, int, float, bool)), f"Unsupported fingerprint value: {type(value)}")
    return value


def rng_state(torch, np, service):
    def digest(value):
        return hashlib.sha256(encoded(fingerprint(value, torch, np))).hexdigest()
    return {"torch_cpu": digest(torch.get_rng_state()),
            "torch_cuda": digest(torch.cuda.get_rng_state(0)),
            "numpy_global": digest(np.random.get_state()),
            "python_global": digest(random.getstate()),
            "native_request_stream": digest(service._rng.bit_generator.state)}


def numeric_flags(torch):
    return {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
            "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            "cudnn_version": torch.backends.cudnn.version()}


def graph_dispatch(original_call, state, graph, args, kwargs):
    """The only treatment: bypass graph instances owned by the active slot."""
    cache = getattr(state.service, "_ifl_conditioning_cache", None)
    active = getattr(cache, "active", None)
    owned = active is not None and any(graph is item for item in active.get("graphs", {}).values())
    if not owned:
        return original_call(graph, *args, **kwargs)
    state.graph_before(graph)
    try:
        if state.mode == "conditioning-bypass":
            graph.stats["diagnostic_bypasses"] = graph.stats.get("diagnostic_bypasses", 0) + 1
            return graph.original(*args, **kwargs)
        return original_call(graph, *args, **kwargs)
    finally:
        state.graph_after(graph)


class Trace:
    def __init__(self, output, mode, torch, np):
        self.output, self.mode, self.torch, self.np = output, mode, torch, np
        self.service = None
        self.request = -1
        self.velocity = -1
        self.stream = (output / "trace.jsonl").open("x")
        self.events = 0
        self.completed_requests = 0
        self.hooks = []
        self.restores = []

    def emit(self, kind, **values):
        row = {"event": self.events, "kind": kind, "request": self.request,
               "velocity": self.velocity, **values}
        self.stream.write(encoded(row).decode() + "\n")
        self.events += 1

    def digest(self, value):
        return fingerprint(value, self.torch, self.np)

    def rng(self):
        return rng_state(self.torch, self.np, self.service)

    def patch(self, owner, attr, replacement):
        self.restores.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, replacement)

    def graph_before(self, graph):
        self.emit("graph_before", layer=graph.name, rng=self.rng(),
                  free_cuda_bytes=self.torch.cuda.mem_get_info()[0],
                  stats=dict(graph.stats))

    def graph_after(self, graph):
        self.emit("graph_after", layer=graph.name, rng=self.rng(), stats=dict(graph.stats))

    @staticmethod
    def pack(value):
        return {key: value[key] for key in ("causal_seq", "full_only_seq", "sample_offsets", "_causal_indices", "_full_indices") if key in value}

    def install(self, loop):
        require(self.service is None, "More than one policy was constructed")
        self.service = service = loop._service
        original_infer = service.infer

        def infer(observation):
            self.request += 1
            self.velocity = -1
            require(self.request < 25, "Unexpected extra request")
            self.emit("request_before", input=self.digest(observation), rng=self.rng(),
                      flags=numeric_flags(self.torch))
            try:
                result = original_infer(observation)
                self.emit("request_after", output=self.digest(result), rng=self.rng())
                self.completed_requests += 1
                return result
            finally:
                self.stream.flush()
                os.fsync(self.stream.fileno())

        self.patch(service, "infer", infer)
        next_seed = service._next_seed

        def seed():
            result = next_seed()
            self.emit("native_seed", value=result)
            return result

        self.patch(service, "_next_seed", seed)
        prepare = service.model._prepare_inference_data

        def prepared(*args, **kwargs):
            before = self.rng()
            result = prepare(*args, **kwargs)
            self.emit("prepared_noise", rng_before=before, rng_after=self.rng(),
                      tokens=self.digest(result[2:4]), noise=self.digest(result[4]),
                      condition_reference=self.digest(result[5]), condition_mask=self.digest(result[6]))
            return result

        self.patch(service.model, "_prepare_inference_data", prepared)
        velocity = service.model._get_velocity

        def velocity_forward(**kwargs):
            self.velocity += 1
            if self.request == 0:
                self.emit("velocity_before", noise=self.digest(kwargs["noise_x"]),
                          timestep=self.digest(kwargs["timestep"]), tokens=self.digest(kwargs["text_tokens"]),
                          rng=self.rng())
            result = velocity(**kwargs)
            if self.request == 0:
                self.emit("velocity_after", output=self.digest(result), rng=self.rng())
            return result

        self.patch(service.model, "_get_velocity", velocity_forward)
        for index, layer in enumerate(service.model.net.language_model.model.layers):
            def before(module, args, kwargs, index=index):
                if self.request == 0:
                    require(not self.torch.cuda.is_current_stream_capturing(), "Trace hook entered graph capture")
                    pack = args[0] if args else kwargs["input"]
                    self.emit("layer_before", layer=index, values=self.digest(self.pack(pack)))

            def after(module, args, kwargs, result, index=index):
                if self.request == 0:
                    self.emit("layer_after", layer=index, values=self.digest(self.pack(result[0])))

            self.hooks.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            self.hooks.append(layer.register_forward_hook(after, with_kwargs=True))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        for owner, attr, original in reversed(self.restores):
            setattr(owner, attr, original)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def runtime_provenance(torch, output, compiler_binding, compiler_tools):
    versions = {}
    for name in ("torch", "triton", "numpy", "transformers", "nvidia-cudnn-cu13", "nvidia-cublas-cu13", "cuda-bindings", "cudnn-frontend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    native = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = Path(fields[-1])
        if path.is_file() and ".so" in path.name and any(word in str(path).lower() for word in ("torch", "cudnn", "cublas", "cuda", "triton", "instinctflash")):
            resolved = path.resolve()
            if str(resolved) not in native:
                native[str(resolved)] = ref(resolved)
    from torch._inductor.codecache import PyCodeCache
    generated = compiler_tools.collect_generated(
        PyCodeCache.modules, output / "generated_code",
        max_bytes=compiler_binding["generated_evidence_max_bytes"],
        max_files=compiler_binding["generated_evidence_max_files"],
        allowed_source_root=os.environ["TORCHINDUCTOR_CACHE_DIR"],
        allowed_kernel_cache_roots=[os.environ["TORCHINDUCTOR_CACHE_DIR"], os.environ["TRITON_CACHE_DIR"]])
    generated_status = generated["status"]
    selected = ("CUBLAS_WORKSPACE_CONFIG", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                "TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "CUDA_MODULE_LOADING")
    return {"interpreter": sys.executable, "packages": versions, "torch_git_version": torch.version.git_version,
            "torch_cuda": torch.version.cuda, "torch_build_config": torch.__config__.show(),
            "environment": {key: os.environ.get(key) for key in selected},
            "loaded_native_libraries_after_inference": native,
            "generated_python_modules": generated, "generated_python_status": generated_status,
            "scope": "Native file hashes captured after inference; no claim of historical native binary equality."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("observed",), default="observed")
    parser.add_argument("--compiler-binding", type=Path, required=True)
    parser.add_argument("--compiler-binding-sha256", required=True)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-only", action="store_true")
    checks.add_argument("--cpu-import-check", action="store_true")
    args = parser.parse_args(argv)
    require(sha(args.binding) == args.binding_sha256, "Binding changed")
    binding = json.loads(args.binding.read_text())
    cell = validate_inputs(binding, args.matrix, args.fixture)
    validate_isolation(sys.flags, os.environ)
    sources = verify_sources(binding)
    require(sha(args.compiler_binding) == args.compiler_binding_sha256, "Compiler binding changed")
    compiler_binding = json.loads(args.compiler_binding.read_text())
    require(compiler_binding["original_binding_sha256"] == args.binding_sha256, "Original input/source binding changed")
    parent = Path(__file__).with_name("parent_worker_1214.py")
    require(sha(parent) == compiler_binding["parent_worker_sha256"] == "1214f969447402461256775b98f1e792450677efbbd5306ed5815d5f367be8a7", "Parent diagnostic source differs")
    helper = Path(__file__).with_name("compiler_tools_v1.py")
    require(sha(helper) == "496fd708b6d56dd91a2f33e307b7c0abc4db638da52c166adc705e621fe7b382", "Compiler provenance helper changed before import")
    spec = importlib.util.spec_from_file_location("nano_compiler_tools_v1", helper)
    compiler_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler_tools)
    compiler_before = compiler_tools.verify_compiler(compiler_binding)
    require(not args.output_root.exists(), "Output already exists")
    args.output_root.mkdir(parents=True)
    fresh_caches = compiler_tools.prepare_fresh_caches(compiler_binding, args.output_root, os.environ)
    save(args.output_root / "prospective.json", {"source": ref(__file__), "binding": ref(args.binding),
         "mode": args.mode, "sources": sources, "matrix": ref(args.matrix), "fixture": ref(args.fixture),
         "instrumented_latency_not_benchmark": True, "check_only": args.check_only,
         "cpu_import_check": args.cpu_import_check, "compiler_binding": ref(args.compiler_binding),
         "compiler_before": compiler_before, "compiler_tools": ref(helper), "parent_worker": ref(parent),
         "fresh_compiler_caches": fresh_caches})
    if args.check_only:
        print(json.dumps({"status": "static_inputs_and_installed_sources_passed", "output": str(args.output_root)}))
        return 0
    if args.cpu_import_check:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU import check requires CUDA_VISIBLE_DEVICES empty")
        import torch
        original_lazy_init = torch.cuda._lazy_init
        def denied_cuda_init(*values, **kwargs):
            raise RuntimeError("CUDA initialization is forbidden in the CPU import check")
        torch.cuda._lazy_init = denied_cuda_init
        try:
            from cosmos3_iwm import adapter as cpu_adapter
            from benchmarks.regression import user_e2e
            from torch._inductor.codecache import PyCodeCache
            require(ref(cpu_adapter.__file__) == sources["cosmos3_iwm/adapter.py"], "CPU imported adapter differs from the hashed installed source")
            require(ref(user_e2e.__file__) == sources["benchmarks/regression/user_e2e.py"], "CPU imported harness differs from the hashed installed source")
            require(isinstance(PyCodeCache.modules, list), "Pinned generated module collection unavailable")
            require(not torch.cuda.is_initialized(), "CPU import check initialized CUDA")
            require(compiler_tools.verify_compiler(compiler_binding) == compiler_before, "Compiler changed during CPU imports")
            require(verify_sources(binding) == sources, "Source changed during CPU imports")
            save(args.output_root / "cpu_import_check.json", {
                "status": "exact_source_and_compiler_CPU_imports_passed", "source": ref(__file__),
                "compiler_binding": ref(args.compiler_binding), "compiler": compiler_before,
                "imports": ["torch", "cosmos3_iwm.adapter", "benchmarks.regression.user_e2e", "torch._inductor.codecache.PyCodeCache"],
                "GPU_initialized": False, "model_constructed": False, "requests": 0})
        finally:
            torch.cuda._lazy_init = original_lazy_init
        print(json.dumps({"status": "exact_source_and_compiler_CPU_imports_passed", "output": str(args.output_root)}))
        return 0
    # All source files were hashed before importing the instrumented modules.
    import numpy as np
    import torch
    from cosmos3_iwm import adapter, thor_graphs
    from benchmarks.regression import user_e2e
    trace = Trace(args.output_root, args.mode, torch, np)
    original_build = adapter.Cosmos3PolicyAdapter._build_droid
    original_graph = thor_graphs.LayerGraph.__call__

    def build(owner, *values, **kwargs):
        loop = original_build(owner, *values, **kwargs)
        trace.install(loop)
        return loop

    def graph(instance, *values, **kwargs):
        return graph_dispatch(original_graph, trace, instance, values, kwargs)

    adapter.Cosmos3PolicyAdapter._build_droid = build
    thor_graphs.LayerGraph.__call__ = graph
    started = time.time()
    result = {"status": "failed", "mode": args.mode, "source": ref(__file__),
              "binding": ref(args.binding), "compiler_binding": ref(args.compiler_binding),
              "compiler_tools": ref(helper), "parent_worker": ref(parent),
              "fresh_compiler_caches": fresh_caches, "implementation_overlay": [
                  "Wrap adapter construction to attach trace callbacks after all native installation guards",
                  "Observe active conditioning graphs unchanged; compiler build is fixed to the public PyTorch artifact"],
              "native_receipt_unchanged": True, "task_quality_certified": False,
              "instrumented_latency_not_benchmark": True}
    code = 1
    try:
        code = user_e2e.capture(args.matrix, cell["id"], args.output_root, args.fixture)
        require(code == 0 and trace.completed_requests == 25, "Native capture or request count failed")
        receipt = args.output_root / cell["receipt"]
        row = json.loads(receipt.read_text())
        require(row["ok"] is True and len(row["cases"]) == 25, "Incomplete native receipt")
        graph_stats = row["backend_stats"]["stats"]["conditioning_cache"]["graph_stats"]
        if args.mode == "conditioning-bypass":
            require(not any(item["captures"] or item["replays"] for item in graph_stats), "Bypass captured a conditioning graph")
            require(sum(item.get("diagnostic_bypasses", 0) for item in graph_stats) > 0, "Treatment never executed")
        result.update(status="diagnostic_capture_completed", native_receipt=ref(receipt),
                      action_archive=ref(receipt.with_suffix(".npz")),
                      matches_fresh_selected_archive=sha(receipt.with_suffix(".npz")) == binding["fresh_selected_archive_sha256"],
                      matches_historical_selected_archive=sha(receipt.with_suffix(".npz")) == binding["historical_selected_archive_sha256"])
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        code = 1
    finally:
        adapter.Cosmos3PolicyAdapter._build_droid = original_build
        thor_graphs.LayerGraph.__call__ = original_graph
        trace.close()
        result.update(completed_requests=trace.completed_requests, events=trace.events,
                      elapsed_seconds=time.time() - started, trace=ref(args.output_root / "trace.jsonl"))
        try:
            result["runtime_provenance"] = runtime_provenance(torch, args.output_root, compiler_binding, compiler_tools)
            for key, expected in {**fresh_caches["paths"], **compiler_binding["matched_compiler_environment"]}.items():
                require(result["runtime_provenance"]["environment"].get(key) == expected, "Compiler/cache environment changed during execution")
            result["compiler_after"] = compiler_tools.verify_compiler(compiler_binding)
            require(result["compiler_after"] == compiler_before, "Compiler files changed during diagnostic")
            result["loaded_compiler_verification"] = compiler_tools.verify_loaded_compiler(
                compiler_binding, compiler_before, result["runtime_provenance"])
            require(verify_sources(binding) == sources, "Installed source changed during diagnostic")
        except BaseException as error:
            result.update(status="failed", provenance_error=repr(error))
            code = 1
        save(args.output_root / "diagnostic.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output_root)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
