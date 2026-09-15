"""Exploratory upstream baseline audit; not a regression certificate.

Run with the selected upstream checkout first on PYTHONPATH and hold the
Thor GPU lock externally. Reuses the recorded-input regression workload.
AUDIT_CFG_INTERVAL=960,1001 selects the explicitly changed CFG schedule.
AUDIT_BATCHED_CFG=1 tests batching both guidance branches (verify numerics).
AUDIT_COMPILE=1 tests upstream compilation and CUDA-graph defaults.
"""
from collections import Counter
import json
import os
from pathlib import Path
import sys

from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService
from benchmarks.regression import cosmos


def main():
    output = Path(sys.argv[3])
    interval = os.environ.get("AUDIT_CFG_INTERVAL")
    interval = [int(x) for x in interval.split(",")] if interval else None
    batched = os.environ.get("AUDIT_BATCHED_CFG") == "1"
    compile_model = os.environ.get("AUDIT_COMPILE") == "1"
    counts = Counter()
    original_setup = RobolabPolicyService._build_setup_args
    original_init = RobolabPolicyService.__init__
    metadata = {"guidance_interval": interval, "diffusion_cache": False,
                "use_torch_compile": compile_model, "batched_cfg": batched,
                "qualification": "exploratory"}

    def setup(self, args):
        result = original_setup(self, args).model_copy(update={
            "diffusion_cache": False, "guardrails": False, "use_torch_compile": compile_model})
        metadata["effective_setup"] = {k: getattr(result, k) for k in
                                       ("diffusion_cache", "guardrails", "use_torch_compile", "use_cuda_graphs")}
        return result

    def initialize(self, args):
        if interval is not None:
            args = args.model_copy(update={"guidance_interval": tuple(interval)})
        # Override the benchmark's EagerService method on this exploratory
        # instance so the compile experiment actually exercises compilation.
        self._build_setup_args = lambda args: setup(self, args)
        original_init(self, args)
        metadata["effective_guidance_interval"] = self.cfg.guidance_interval
        original_generate = self.model.generate_samples_from_batch
        def generate(*a, **kw):
            kw["use_batched_cfg"] = batched
            return original_generate(*a, **kw)
        self.model.generate_samples_from_batch = generate
        for name in ("_can_reuse_inference_pack_templates", "_can_reuse_inference_text_kv", "_get_velocity"):
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
        output.with_suffix(".audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
