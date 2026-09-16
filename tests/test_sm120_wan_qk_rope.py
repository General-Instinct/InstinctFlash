#!/usr/bin/env python3
"""Offline/lifecycle/worker gates for P009-A4."""
from __future__ import annotations
import os,tempfile
from pathlib import Path
from instinctflash.adapters.lingbot_va import lingbot_va_spec
from instinctflash.backends import sm120_wan_qk_rope
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import DeviceProfile,KNOWN_FEATURES
from instinctflash.passes.lingbot import default_passes
from instinctflash.passes.lingbot.sm120_wan_qk_rope import SM120WanQKRoPE
from instinctflash.planners.planner import Optimizer,PassResult,Plan,Tier
from instinctflash.runtime.sm120_qk_rope_install import install_sm120_wan_qk_rope

def device(cap=(12,0),live=True):
 f={"cuda","sm120_kernels","sm120_stage2_kernels","sm120_stage3_kernels"}
 if live:f.add("sm120_qk_rope_kernels")
 return DeviceProfile(name='synthetic',capability=cap,total_memory=32<<30,features=frozenset(f))
def result(dev): return Optimizer(passes=[SM120WanQKRoPE()]).compile(lingbot_va_spec(),DeploymentSpec(device=dev),capabilities=frozenset({'backbone:wan_va'})).results[0]
def test_plan_requires_full_chain_and_sm120():
 assert 'sm120_qk_rope_kernels' in KNOWN_FEATURES; assert result(device()).applies
 miss=result(device(live=False)); assert not miss.applies and 'sm120_qk_rope_kernels' in miss.reason
 assert not result(device((11,0))).applies; assert not result(None).applies
def test_default_order_a4_after_a3():
 n=[x.name for x in default_passes()]; assert n.index('sm120_wan_stage3')+1==n.index('sm120_wan_qk_rope')
def test_library_override_requires_a4_abi():
 old=os.environ.get(sm120_wan_qk_rope.LIBRARY_ENV); probe=sm120_wan_qk_rope._library_abi
 try:
  with tempfile.TemporaryDirectory() as d:
   lib=Path(d)/sm120_wan_qk_rope.LIBRARY_NAME; os.environ[sm120_wan_qk_rope.LIBRARY_ENV]=str(lib); assert not sm120_wan_qk_rope.available(); lib.touch(); assert not sm120_wan_qk_rope.available(); sm120_wan_qk_rope._library_abi=lambda path:sm120_wan_qk_rope.ABI_VERSION; assert sm120_wan_qk_rope.available(); assert sm120_wan_qk_rope.resolve_library()==lib
 finally:
  sm120_wan_qk_rope._library_abi=probe
  if old is None: os.environ.pop(sm120_wan_qk_rope.LIBRARY_ENV,None)
  else: os.environ[sm120_wan_qk_rope.LIBRARY_ENV]=old
def test_installer_one_shot_thread_scoped():
 calls=[]; ks=iter((object(),object())); ok=sm120_wan_qk_rope.SM120WanQKRoPEKernels; oi=sm120_wan_qk_rope.install_wan_qk_rope
 try:
  sm120_wan_qk_rope.SM120WanQKRoPEKernels=lambda:next(ks); sm120_wan_qk_rope.install_wan_qk_rope=lambda tr,k:calls.append((tr,k))
  class VA:
   def __init__(self): self.transformer=object()
  assert install_sm120_wan_qk_rope(object(),VA)==['sm120_wan_qk_rope']; a=VA(); assert calls==[(a.transformer,VA._ifl_sm120_wan_qk_rope_kernels)]; VA(); assert len(calls)==1; install_sm120_wan_qk_rope(object(),VA); b=VA(); assert len(calls)==2 and calls[1]==(b.transformer,VA._ifl_sm120_wan_qk_rope_kernels)
 finally: sm120_wan_qk_rope.SM120WanQKRoPEKernels=ok; sm120_wan_qk_rope.install_wan_qk_rope=oi
def test_install_plan_requires_a3():
 from instinctflash.runtime.lingbot_install import install_plan
 r=PassResult('sm120_wan_qk_rope',True,Tier.BITEXACT,'synthetic')
 try: install_plan(object(),type('VA',(),{}),Plan('x',[r]))
 except RuntimeError as e: assert 'requires sm120_wan_stage3' in str(e)
 else: raise AssertionError('A4 installed without A3')
def test_native_and_worker_surface():
 root=Path(sm120_wan_qk_rope.__file__).resolve().parents[2]; src=(root/'instinctflash/native/wan_qk_rope_sm120.cu').read_text()
 for n in ('instinctflash_sm120_wan_qk_rope_abi_version','wan_qk_rms_rope_bf16','rsqrtf','double real','__float2bfloat16_rn','dim3(32,4,1)'): assert n in src
 assert 'instinctflash_sm120_wan_qk_rope' in (root/'instinctflash/native/CMakeLists.txt').read_text(); w=(root/'instinctflash/runtime/lingbot_worker.py').read_text(); assert '"--sm120-wan-qk-rope"' in w and '--sm120-wan-qk-rope requires --sm120-wan-stage3' in w
def test_worker_forwards_full_chain():
 from instinctflash.adapters.lingbot_va import LingBotVA
 class E:
  nfe={"video":2,"action":4}; guidance={}; extra={"base_weights":"/tmp/base","obs_cam_keys":["observation.images.cam_high","observation.images.cam_left_wrist","observation.images.cam_right_wrist"],"height":256,"width":320,"env_type":"robotwin_tshape"}
 class C: path='/tmp/pkg'; model_id='x'; execution=E()
 class P: applied=[type('R',(),{'name':n})() for n in ('sm120_gated_residual','sm120_wan_stage2','sm120_wan_stage3','sm120_wan_qk_rope')]
 a=LingBotVA(); a.materialize=lambda c:'/tmp/composed'; keys=('IFL_SM120_KERNEL_LIBRARY','IFL_SM120_STAGE2_LIBRARY','IFL_SM120_STAGE3_LIBRARY','IFL_SM120_QK_ROPE_LIBRARY'); old={k:os.environ.get(k) for k in keys}
 try:
  for i,k in enumerate(keys,1):os.environ[k]=f'/tmp/a{i}.so'
  argv,env=a.worker_command(C(),P(),port=1,python='python',device=None,nfe=None)
 finally:
  for k,v in old.items(): os.environ.pop(k,None) if v is None else os.environ.__setitem__(k,v)
 for f in ('--sm120-gated-residual','--sm120-wan-stage2','--sm120-wan-stage3','--sm120-wan-qk-rope'): assert f in argv
 assert env['IFL_SM120_QK_ROPE_LIBRARY']=='/tmp/a4.so'
if __name__=='__main__':
 from run_tests import run_module_tests
 raise SystemExit(run_module_tests(globals()))
