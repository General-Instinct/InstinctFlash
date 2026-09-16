#!/usr/bin/env python3
"""Offline/lifecycle/worker gates for P009-A5."""
from __future__ import annotations
import os,tempfile
from pathlib import Path
from instinctflash.adapters.lingbot_va import lingbot_va_spec
from instinctflash.backends import sm120_wan_gemm
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import DeviceProfile,KNOWN_FEATURES
from instinctflash.passes.lingbot import default_passes
from instinctflash.passes.lingbot.sm120_wan_gemm import SM120WanGEMM
from instinctflash.planners.planner import Optimizer,PassResult,Plan,Tier
from instinctflash.runtime.sm120_gemm_install import install_sm120_wan_gemm

def device(cap=(12,0),live=True):
 f={"cuda","cublas","sm120_kernels","sm120_stage2_kernels","sm120_stage3_kernels","sm120_qk_rope_kernels"}
 if live:f.add("sm120_gemm_kernels")
 return DeviceProfile(name="synthetic",capability=cap,total_memory=32<<30,features=frozenset(f))
def result(dev):return Optimizer(passes=[SM120WanGEMM()]).compile(lingbot_va_spec(),DeploymentSpec(device=dev),capabilities=frozenset({"backbone:wan_va"})).results[0]

def test_plan_requires_full_chain_and_sm120():
 assert "sm120_gemm_kernels" in KNOWN_FEATURES;assert result(device()).applies
 miss=result(device(live=False));assert not miss.applies and "sm120_gemm_kernels" in miss.reason
 assert not result(device((11,0))).applies;assert not result(None).applies
 assert sm120_wan_gemm.CERTIFIED_CONFIGS=={(480,3072,3072):(21,15,25,0,0),(64,14336,3072):(21,18,12,0,0)}

def test_default_order_a5_after_a4():
 names=[p.name for p in default_passes()];assert names.index("sm120_wan_qk_rope")+1==names.index("sm120_wan_gemm")

def test_library_override_requires_a5_abi():
 old=os.environ.get(sm120_wan_gemm.LIBRARY_ENV);probe=sm120_wan_gemm._library_abi;probe_lt=sm120_wan_gemm._library_cublaslt_version
 try:
  with tempfile.TemporaryDirectory() as d:
   lib=Path(d)/sm120_wan_gemm.LIBRARY_NAME;os.environ[sm120_wan_gemm.LIBRARY_ENV]=str(lib);assert not sm120_wan_gemm.available();lib.touch();assert not sm120_wan_gemm.available();sm120_wan_gemm._library_abi=lambda path:sm120_wan_gemm.ABI_VERSION;sm120_wan_gemm._library_cublaslt_version=lambda path:sm120_wan_gemm.CERTIFIED_CUBLASLT_VERSION;assert sm120_wan_gemm.available();assert sm120_wan_gemm.resolve_library()==lib
 finally:
  sm120_wan_gemm._library_abi=probe;sm120_wan_gemm._library_cublaslt_version=probe_lt
  if old is None:os.environ.pop(sm120_wan_gemm.LIBRARY_ENV,None)
  else:os.environ[sm120_wan_gemm.LIBRARY_ENV]=old

def test_installer_one_shot_thread_scoped():
 calls=[];kernels=iter((object(),object()));orig_k=sm120_wan_gemm.SM120WanGEMMKernels;orig_i=sm120_wan_gemm.install_wan_gemm
 try:
  sm120_wan_gemm.SM120WanGEMMKernels=lambda:next(kernels);sm120_wan_gemm.install_wan_gemm=lambda tr,k:calls.append((tr,k))
  class VA:
   def __init__(self):self.transformer=object()
  assert install_sm120_wan_gemm(object(),VA)==["sm120_wan_gemm"];a=VA();assert calls==[(a.transformer,VA._ifl_sm120_wan_gemm_kernels)];VA();assert len(calls)==1;install_sm120_wan_gemm(object(),VA);b=VA();assert calls[1]==(b.transformer,VA._ifl_sm120_wan_gemm_kernels)
 finally:sm120_wan_gemm.SM120WanGEMMKernels=orig_k;sm120_wan_gemm.install_wan_gemm=orig_i

def test_install_plan_requires_a4():
 from instinctflash.runtime.lingbot_install import install_plan
 r=PassResult("sm120_wan_gemm",True,Tier.BITEXACT,"synthetic")
 try:install_plan(object(),type("VA",(),{}),Plan("x",[r]))
 except RuntimeError as e:assert "requires sm120_wan_qk_rope" in str(e)
 else:raise AssertionError("A5 installed without A4")

def test_native_and_worker_surface():
 root=Path(sm120_wan_gemm.__file__).resolve().parents[2];src=(root/"instinctflash/native/wan_gemm_sm120.cu").read_text()
 for name in ("instinctflash_sm120_wan_gemm_abi_version","wan_gemm_cublaslt_version","cublasLtMatmulAlgoInit","CUBLASLT_ALGO_CONFIG_SPLITK_NUM","split_k = 1","CUBLASLT_EPILOGUE_BIAS","checked.workspaceSize != 0","wan_gemm_bf16"):assert name in src
 cm=(root/"instinctflash/native/CMakeLists.txt").read_text();assert "instinctflash_sm120_wan_gemm" in cm and "CUDA::cublasLt" in cm
 worker=(root/"instinctflash/runtime/lingbot_worker.py").read_text();assert '"--sm120-wan-gemm"' in worker and "--sm120-wan-gemm requires --sm120-wan-qk-rope" in worker

def test_worker_forwards_full_chain():
 from instinctflash.adapters.lingbot_va import LingBotVA
 class E:
  nfe={"video":2,"action":4};guidance={};extra={"base_weights":"/tmp/base","obs_cam_keys":["observation.images.cam_high","observation.images.cam_left_wrist","observation.images.cam_right_wrist"],"height":256,"width":320,"env_type":"robotwin_tshape"}
 class C:path="/tmp/pkg";model_id="x";execution=E()
 class P:applied=[type("R",(),{"name":n})() for n in ("sm120_gated_residual","sm120_wan_stage2","sm120_wan_stage3","sm120_wan_qk_rope","sm120_wan_gemm")]
 adapter=LingBotVA();adapter.materialize=lambda c:"/tmp/composed";keys=("IFL_SM120_KERNEL_LIBRARY","IFL_SM120_STAGE2_LIBRARY","IFL_SM120_STAGE3_LIBRARY","IFL_SM120_QK_ROPE_LIBRARY","IFL_SM120_GEMM_LIBRARY");old={k:os.environ.get(k) for k in keys}
 try:
  for i,k in enumerate(keys,1):os.environ[k]=f"/tmp/a{i}.so"
  argv,env=adapter.worker_command(C(),P(),port=1,python="python",device=None,nfe=None)
 finally:
  for k,v in old.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
 for flag in ("--sm120-gated-residual","--sm120-wan-stage2","--sm120-wan-stage3","--sm120-wan-qk-rope","--sm120-wan-gemm"):assert flag in argv
 assert env["IFL_SM120_GEMM_LIBRARY"]=="/tmp/a5.so"

if __name__=="__main__":
 from run_tests import run_module_tests
 raise SystemExit(run_module_tests(globals()))
