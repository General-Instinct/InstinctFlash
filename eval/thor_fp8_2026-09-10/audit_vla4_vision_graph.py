"""Same loaded FP8 policy, A/B/A BF16 vision graph diagnostic; no default changes."""
import fcntl,json,runpy,sys,torch,numpy as np
from pathlib import Path
from instinctflash import Runtime
r=Path('/home/guanming/ifl_eval/thor_fp8_audit_20260910');bench=r.parent/'thor_fp8_20260910/source-v3/benchmark.py'
r=r/'vision-graph-v3';r.mkdir(exist_ok=True)
lock=open('/tmp/thor_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
factory=Runtime.from_pretrained;close=Runtime.close;held=[]
def reuse(*args,**kwargs):
 if not held:held.append(factory(*args,**kwargs))
 return held[0]
Runtime.from_pretrained=staticmethod(reuse);Runtime.close=lambda self:None
report={'scope':'A/B/A on one loaded policy and calibrated FP8 weights; only BF16 vision graph replay changes. Short diagnostic, not a simulator certificate.'}
try:
 for phase in ('before','graph','after'):
  if phase=='graph':
   vision=held[0]._backend._loop._server.vla.model.vision
   cls=type(vision);original=cls.__call__;cache={}
   layout_original=vision.layout
   vision.layout=tuple(v.to(device='cuda') if i==1 else tuple(v.detach().cpu().tolist()) if i in (2,3) else v for i,v in enumerate(vision.layout))
   def graphed(self,patches):
    if 'graph' not in cache:
     buf=patches.clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(stream):
      for _ in range(3):original(self,buf)
     stream.synchronize();graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph,stream=stream):out=original(self,buf)
     torch.cuda.synchronize();cache.update(graph=graph,input=buf,output=out)
    cache['input'].copy_(patches);cache['graph'].replay();return cache['output']
   cls.__call__=graphed
  if phase=='after':
   cls.__call__=original;vision.layout=layout_original
  sys.argv=[str(bench),'vla4','fp8',str(r/f'vision-ablation-{phase}.json'),'--iterations','8']
  runpy.run_path(str(bench),run_name='__main__')
  d=json.loads((r/f'vision-ablation-{phase}.json').read_text());report[phase]={'p50_ms':d['p50_ms']}
 arrays=[np.load(r/f'vision-ablation-{phase}.npz')['actions'] for phase in ('before','graph','after')]
 report['before_after_equal']=np.array_equal(arrays[0],arrays[2]);report['before_graph_equal']=np.array_equal(arrays[0],arrays[1]);report['max_abs_action_delta']=float(np.max(np.abs(arrays[0]-arrays[1])));report['ok']=True
except BaseException as e:
 import traceback
 report.update(ok=False,error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 if 'original' in globals():cls.__call__=original
 for api in held:close(api)
 (r/'vision-graph-ablation.json').write_text(json.dumps(report,indent=2));print(report)
