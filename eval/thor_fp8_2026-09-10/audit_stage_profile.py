"""Diagnostic synchronized stage timings; never substitute these for latency receipts."""
import json,sys,runpy,time,collections
from pathlib import Path
import torch
from instinctflash import Runtime
root=Path('/home/guanming/ifl_eval/thor_fp8_audit_20260910')
family,precision=sys.argv[1:3];records=[];current=-1;patched=set();graphs={}
def wrap(obj,name,label):
 key=(id(obj),name)
 if key in patched:return
 original=getattr(obj,name);patched.add(key)
 def measured(*args,**kwargs):
  if torch.cuda.is_current_stream_capturing():return original(*args,**kwargs)
  torch.cuda.synchronize();start=time.perf_counter()
  result=original(*args,**kwargs)
  torch.cuda.synchronize();records.append(dict(call=current,stage=label,ms=1000*(time.perf_counter()-start)))
  return result
 setattr(obj,name,measured)
original_predict=Runtime.predict
installed=False

def predict(self,*args,**kwargs):
 global current,installed
 current+=1
 if not installed:
  loop=self._backend._loop if precision=='fp8' else self._backend._impl
  if family=='groot':
   base=loop._native_loop if precision=='fp8' else loop
   model=base._policy.model
   wrap(model.backbone,'forward','native_camera_language_backbone')
   wrap(model.action_head,'get_action','action_head_total')
   if precision=='fp8':
    wrap(type(loop._runner),'__call__','fp8_vlsa')
    wrap(loop._frontend,'infer','bf16_dit_and_state')
  if family in ('vla4','vla2') and precision=='fp8':
   gen=loop._server.vla.model
   vision=gen.vision if family=='vla4' else gen.engine.vision
   wrap(type(vision),'__call__','bf16_vision')
   wrap(gen.frontend if family=='vla4' else gen.engine,'infer_staged','language_action_pipeline' if family=='vla4' else 'vision_language_action_pipeline')
  if family=='pi05' and precision=='native':wrap(loop._p.model,'sample_actions','native_model')
  # Graph replay timing is inclusive and may overlap parent regions.
  original_replay=torch.cuda.CUDAGraph.replay
  def replay(graph,*a,**k):
   key=id(graph)
   if key not in graphs:graphs[key]='cuda_graph_'+str(len(graphs))
   torch.cuda.synchronize();start=time.perf_counter();result=original_replay(graph,*a,**k);torch.cuda.synchronize()
   records.append(dict(call=current,stage=graphs[key],ms=1000*(time.perf_counter()-start)))
   return result
  torch.cuda.CUDAGraph.replay=replay
  installed=True
 return original_predict(self,*args,**kwargs)
Runtime.predict=predict
sys.argv=[str(root.parent/'thor_fp8_20260910/source-v3/benchmark.py'),family,precision,str(root/f'{family}-{precision}.json'),'--iterations','4']
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 out={'family':family,'precision':precision,'records':records,'scope':'Synchronized instrumented stage diagnostic; overlapping parent and graph times must not be added. Not a speed table measurement.'}
 (root/f'{family}-{precision}-stages.json').write_text(json.dumps(out,indent=2)+'\n')
