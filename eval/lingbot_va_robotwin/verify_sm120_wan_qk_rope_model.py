#!/usr/bin/env python3
"""Deterministic real-model arm and reset gate for P009-A4."""
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np,torch
CAMS=("observation.images.cam_high","observation.images.cam_left_wrist","observation.images.cam_right_wrist"); PROMPT="Use the left arm to lift the plastic drink bottle head-up"
def git(repo,*args):
 try:return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.DEVNULL).strip()
 except Exception:return 'unknown'
def frame(rng):return {k:rng.integers(0,256,size=(240,320,3),dtype=np.uint8) for k in CAMS}
def pointers(tr):return {str(i):{str(k):[t.data_ptr() for t in v] for k,v in getattr(b.attn1,'_ifl_qk_rope_buffers',{}).items()} for i,b in enumerate(tr.blocks)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--library',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--actions',type=Path,required=True);ap.add_argument('--cycles',type=int,default=8);ap.add_argument('--candidate',action='store_true');ap.add_argument('--reset-replay',action='store_true');a=ap.parse_args();repo=Path(__file__).resolve().parents[2]
 result={'variant':'a4' if a.candidate else 'a3','git_branch':git(repo,'branch','--show-current'),'git_commit':git(repo,'rev-parse','HEAD'),'git_status':git(repo,'status','--short'),'cycles':[],'status':'running'};rt=None;actions=[]
 try:
  from instinctflash import Runtime
  rt=Runtime.from_pretrained(a.checkpoint,placement='in_process',seed=123,tier_ceiling='numeric',exclude_passes=('cfg_branch_elision','conv_layout_ndhwc'));rt.reset(prompt=PROMPT);tr=rt._backend._impl._server.transformer
  if getattr(tr,'_ifl_wan_stage3_kernels',None) is None:raise RuntimeError('A4 requires A3 baseline')
  qk=getattr(tr,'_ifl_wan_qk_rope_kernels',None)
  if a.candidate and qk is None:
   from instinctflash.backends.sm120_wan_qk_rope import SM120WanQKRoPEKernels,install_wan_qk_rope
   qk=install_wan_qk_rope(tr,SM120WanQKRoPEKernels(a.library))
  if not a.candidate and qk is not None:raise RuntimeError('baseline unexpectedly installed A4')
  begin=0 if qk is None else qk.calls;torch.cuda.reset_peak_memory_stats()
  def run():
   seq=[];records=[];rng=np.random.default_rng(0)
   for c in range(a.cycles):
    nf=1 if c==0 else(4 if c==1 else 8);obs=[frame(rng) for _ in range(nf)];torch.cuda.synchronize();t=time.perf_counter();o=rt.predict({'obs':obs,'prompt':PROMPT,'save_visualization':False});torch.cuda.synchronize();act=np.asarray(o['action'] if isinstance(o,dict) else o);seq.append(act.copy());records.append({'index':c,'frames':nf,'latency_ms':round((time.perf_counter()-t)*1000,3),'finite':bool(np.isfinite(act).all()),'shape':list(act.shape)})
   return seq,records
  actions,result['cycles']=run()
  if a.reset_replay:
   before=pointers(tr);rt.reset(prompt=PROMPT);second,records=run();after=pointers(tr);eq=[np.array_equal(x,y) for x,y in zip(actions,second,strict=True)];result['reset_replay']={'equal_cycles':sum(eq),'all_equal':all(eq),'max_abs':max(float(np.max(np.abs(x-y))) for x,y in zip(actions,second,strict=True)),'buffer_pointers_stable':before==after,'cycles':records}
   if not all(eq) or before!=after:raise RuntimeError('A4 reset replay failed')
  result.update(qk_calls=0 if qk is None else qk.calls-begin,max_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,4),status='pass')
 except Exception as e:result.update(status='error',error_type=type(e).__name__,error=str(e));raise
 finally:
  if rt is not None:rt.close()
  if torch.distributed.is_available() and torch.distributed.is_initialized():torch.distributed.destroy_process_group()
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
  if actions:np.savez_compressed(a.actions,**{f'cycle_{i}':x for i,x in enumerate(actions)})
  print(json.dumps(result,indent=2,sort_keys=True))
 return 0 if result['status']=='pass' else 1
if __name__=='__main__':raise SystemExit(main())
