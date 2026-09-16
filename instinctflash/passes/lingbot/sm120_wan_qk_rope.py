"""P009-A4: BITEXACT SM120 fusion of Wan Q/K RMSNorm + FP64-complex RoPE."""
from __future__ import annotations
from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult,Tier
class SM120WanQKRoPE:
    name="sm120_wan_qk_rope"; requires_capabilities=frozenset({"backbone:wan_va"})
    hardware=HardwareReq(min_capability=(12,0),requires=frozenset({"cuda","sm120_kernels","sm120_stage2_kernels","sm120_stage3_kernels","sm120_qk_rope_kernels"}))
    def evaluate(self,spec:AdapterSpec,deployment:DeploymentSpec)->PassResult:
        device=getattr(deployment,"device",None)
        if device is None: return PassResult(self.name,False,Tier.BITEXACT,"unprobed target: A4 is certified for SM120 exactly")
        if device.capability!=(12,0): return PassResult(self.name,False,Tier.BITEXACT,f"A4 is certified for SM120 exactly; device is sm_{device.capability[0]}{device.capability[1]}")
        return PassResult(self.name,True,Tier.BITEXACT,"A1/A2/A3 and the independent QK-RoPE ABI are present; fused BF16 RMSNorm materialization and FP64-complex RoPE reproduce Torch 2.9 exactly",params={"depends_on":"sm120_wan_stage3","blocks":30,"calls_per_block":2,"certified_rows":[64,480],"hidden_dim":3072,"heads":24,"head_dim":128,"library_feature":"sm120_qk_rope_kernels"},expected_win="RTX 5090 A3 baseline, 42-cycle A-B-B-A: 404.71 -> 396.42 ms (1.0209x), growing 401.27 -> 393.12 ms, saturated 452.30 -> 442.12 ms; 168/168 actions bitwise equal")
__all__=["SM120WanQKRoPE"]
