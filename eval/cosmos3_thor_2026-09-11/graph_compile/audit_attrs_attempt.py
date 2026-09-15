"""Exploratory upstream baseline audit; not a regression certificate.

Run with the selected upstream checkout first on PYTHONPATH and hold the
Thor GPU lock externally. Reuses the recorded-input regression workload.
AUDIT_CFG_INTERVAL=960,1001 selects the explicitly changed CFG schedule.
AUDIT_BATCHED_CFG=1 tests batching both guidance branches (verify numerics).
AUDIT_COMPILE=1 tests upstream compilation and CUDA-graph defaults.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService
from benchmarks.regression import cosmos


def main():
    if len(sys.argv) < 4:
        raise SystemExit("Usage: audit_upstream.py FAMILY ARM OUTPUT [--iterations N]")
    output = Path(sys.argv[3])
    if any(output.with_suffix(suffix).exists() for suffix in (".json", ".npz", ".audit.json")):
        raise SystemExit("Use a fresh output path; existing audit evidence is immutable")
    interval = os.environ.get("AUDIT_CFG_INTERVAL")
    interval = [int(x) for x in interval.split(",")] if interval else None
    batched = os.environ.get("AUDIT_BATCHED_CFG") == "1"
    compile_model = os.environ.get("AUDIT_COMPILE") == "1"
    dynamic = os.environ.get("AUDIT_DYNAMIC", "1") == "1"
    counts = Counter()
    original_setup = RobolabPolicyService._build_setup_args
    original_init = RobolabPolicyService.__init__
    metadata = {"guidance_interval": interval, "diffusion_cache": False,
                "use_torch_compile": compile_model, "compile_dynamic": dynamic, "batched_cfg": batched,
                "qualification": "exploratory"}

    def setup(self, args):
        result = original_setup(self, args).model_copy(update={
            "diffusion_cache": False, "guardrails": False, "use_torch_compile": compile_model, "compile_dynamic": dynamic, "use_cuda_graphs": True})
        metadata["effective_setup"] = {k: getattr(result, k) for k in
                                       ("diffusion_cache", "guardrails", "use_torch_compile", "use_cuda_graphs", "compile_dynamic")}
        return result

    # Grouped action projections in this Torch build cannot be captured.
    # Keep encode/decode compiled, but capture only the MoT decoder blocks.
    import importlib
    import attrs
    vfm = importlib.import_module("cosmos_framework.model.generator.mot.parallelize_vfm_network")
    original_vfm_compile = vfm.apply_compile
    def compile_vfm_without_graphs(model, config):
        metadata["vfm_compile_calls"] = metadata.get("vfm_compile_calls", 0) + 1
        metadata["vfm_use_cuda_graphs"] = False
        return original_vfm_compile(model, attrs.evolve(config, use_cuda_graphs=False))
    vfm.apply_compile = compile_vfm_without_graphs

    def initialize(self, args):
        if interval is not None:
            args = args.model_copy(update={"guidance_interval": tuple(interval)})
        # Override the benchmark's EagerService method on this exploratory
        # instance so the compile experiment actually exercises compilation.
        self._build_setup_args = lambda args: setup(self, args)
        original_init(self, args)
        if os.environ.get("AUDIT_ENGINE_SWIGLU"):
            sys.path.insert(0, str(Path(__file__).resolve().parent / "kernel-reuse"))
            from engine_swiglu import install
            metadata["engine_swiglu"] = install(self, os.environ["AUDIT_ENGINE_SWIGLU"])
        if os.environ.get("AUDIT_ENGINE_ATTENTION"):
            sys.path.insert(0, str(Path(__file__).resolve().parent / "attention-reuse"))
            from engine_attention import install
            metadata["engine_attention"] = install(self, os.environ["AUDIT_ENGINE_ATTENTION"])
        metadata["effective_guidance_interval"] = getattr(self.cfg, "guidance_interval", None)
        if interval is not None and metadata["effective_guidance_interval"] != tuple(interval):
            raise RuntimeError("The selected server did not apply the requested guidance interval")
        original_generate = self.model.generate_samples_from_batch
        def generate(*a, **kw):
            kw["use_batched_cfg"] = batched
            return original_generate(*a, **kw)
        self.model.generate_samples_from_batch = generate
        for name in ("_can_reuse_inference_pack_templates", "_can_reuse_inference_text_kv", "_get_velocity"):
            if not hasattr(self.model, name):
                metadata.setdefault("unavailable_methods", []).append(name)
                continue
            original = getattr(self.model, name)
            def instrument(*a, _name=name, _original=original, **kw):
                result = _original(*a, **kw)
                counts[_name + (":" + str(result) if _name.startswith("_can") else "")] += 1
                return result
            setattr(self.model, name, instrument)

    RobolabPolicyService._build_setup_args = setup
    RobolabPolicyService.__init__ = initialize
    try:
        code = cosmos.main()
    finally:
        metadata["counts"] = dict(counts)
        metadata["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        output.with_suffix(".audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if output.exists():
            report = json.loads(output.read_text())
            report["qualification"] = "exploratory upstream audit; not a regression certificate"
            report["audit"] = metadata
            if os.environ.get("AUDIT_ENGINE_ATTENTION") and not metadata.get("engine_attention", {}).get("calls", 0):
                report["ok"] = False
                report["error"] = "Requested attention candidate executed zero kernel calls"
                code = 1
            # Keep these arms distinct from the unchanged production regression
            # protocol, including experiments that change the guidance schedule.
            report["benchmark_sha256"] = hashlib.sha256(json.dumps({
                "workload": report["benchmark_sha256"],
                "audit": metadata["audit_script_sha256"],
                "interval": interval, "compile": compile_model, "batched": batched, "engine_swiglu": metadata.get("engine_swiglu"),
                "engine_attention": metadata.get("engine_attention"),
            }, sort_keys=True).encode()).hexdigest()
            output.write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
