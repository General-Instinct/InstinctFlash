"""P009-A5: BITEXACT pinned cuBLASLt tactics for two Wan BF16 Linear shapes."""
from __future__ import annotations
from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult,Tier
class SM120WanGEMM:
    name="sm120_wan_gemm";requires_capabilities=frozenset({"backbone:wan_va"})
    hardware=HardwareReq(min_capability=(12,0),requires=frozenset({"cuda","cublas","sm120_kernels","sm120_stage2_kernels","sm120_stage3_kernels","sm120_qk_rope_kernels","sm120_gemm_kernels"}))
    def evaluate(self,spec:AdapterSpec,deployment:DeploymentSpec)->PassResult:
        device=getattr(deployment,"device",None)
        if device is None:return PassResult(self.name,False,Tier.BITEXACT,"unprobed target: A5 tactics are certified for SM120 exactly")
        if device.capability!=(12,0):return PassResult(self.name,False,Tier.BITEXACT,f"A5 is certified for SM120 exactly; device is sm_{device.capability[0]}{device.capability[1]}")
        return PassResult(self.name,True,Tier.BITEXACT,"A1-A4 and the independent pinned-cuBLASLt ABI are present; two no-split-K tactics reproduce Torch 2.9 BF16 addmm+bias exactly",params={"depends_on":"sm120_wan_qk_rope","certified_shapes":[[480,3072,3072],[64,14336,3072]],"split_k":1,"module_sites":274,"library_feature":"sm120_gemm_kernels"},expected_win="RTX 5090 A4 baseline, 42-cycle A-B-B-A: 395.03 -> 388.24 ms (1.0175x); growing saves 5.18 ms, saturated saves 5.09 ms; 168/168 actions bitwise equal")
__all__=["SM120WanGEMM"]
