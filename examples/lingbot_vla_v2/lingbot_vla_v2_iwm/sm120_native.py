"""Instance-owned SM120 migration of Thor's existing Qwen SDPA configuration.

No projection replacement, quantization, MoE rewrite or scheduler change. The
mixed vision/action attention interface remains upstream's own implementation.
"""
from __future__ import annotations


class NativeSDPA:
    def __init__(self, model):
        self._configs = []
        self.sites = []
        seen = set()
        for path, module in model.named_modules():
            if type(module).__name__ not in {"Qwen3VLVisionAttention", "Qwen3VLTextAttention", "Qwen2Attention"}:
                continue
            config = module.config
            old = config._attn_implementation
            if old not in {"eager", "sdpa", "flash_attention_2"}:
                raise ValueError(f"Unqualified Qwen attention implementation at {path}: {old}")
            self.sites.append(path)
            if id(config) not in seen:
                self._configs.append((config, old))
                seen.add(id(config))
        if not self.sites:
            raise ValueError("No audited Qwen attention sites for native SM120 SDPA")
        for config, _ in self._configs:
            config._attn_implementation = "sdpa"

    def report(self):
        return {"profile": "sm120_native_thor_sdpa", "attention_sites": list(self.sites),
                "precision": "native", "quantization": False,
                "scope": "Thor-style Qwen text/vision SDPA; original mixed attention, MoE, schedule and processors",
                "task_quality": "unqualified"}

    def close(self):
        if any(config._attn_implementation != "sdpa" for config, _ in self._configs):
            raise RuntimeError("Native SDPA configuration changed by another owner")
        for config, old in self._configs:
            config._attn_implementation = old
        self._configs.clear()


def selected(plan, capability, mode, *, enabled):
    from instinctflash.planners.planner import Tier
    if not enabled or tuple(capability) != (12, 0) or mode != "static":
        return False
    if getattr(plan, "resolved_gemm_backend", None):
        return False
    if getattr(plan, "tier_ceiling", Tier.BITEXACT) < Tier.NUMERIC:
        raise ValueError("SM120 native SDPA requires explicit tier_ceiling='numeric'")
    if not any(r.name == "graph_capture" and r.applies for r in plan.results):
        raise ValueError("SM120 native SDPA requires the existing NUMERIC capture plan")
    return True
