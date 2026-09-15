"""Matched Runtime generation timing; run each variant in a fresh process on an idle GPU."""
import argparse,hashlib,io,json,os,sys,time,traceback,random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from instinctflash import Runtime
P=argparse.ArgumentParser();P.add_argument('family',choices=['pi05','vla4','groot','vla2','va','edge','nano','dreamzero']);P.add_argument('--warmup',type=int,default=10);P.add_argument('--arm',choices=['baseline_a','current','baseline_b'],required=True);P.add_argument('precision',choices=['native']);P.add_argument('output',type=Path);P.add_argument('--iterations',type=int,default=12);P.add_argument('--input-archive',type=Path,required=True);a=P.parse_args()
assert not a.output.exists()
MODELS={'pi05':'lerobot/pi05_libero_finetuned_v044','vla4':'robbyant/lingbot-vla-4b-posttrain-robotwin','vla2':'robbyant/lingbot-vla-v2-6b-robotwin','groot':'nvidia/GR00T-N1.7-3B','va':'robbyant/lingbot-va-posttrain-robotwin','va_2v4a':'robbyant/lingbot-va-posttrain-robotwin','edge':'nvidia/Cosmos3-Edge-Policy-DROID','nano':'nvidia/Cosmos3-Nano-Policy-DROID','dreamzero':'GEAR-Dreams/DreamZero-DROID'}
model=MODELS[a.family]
from huggingface_hub import snapshot_download
snapshot=Path(snapshot_download(model,local_files_only=True));revision=snapshot.name
assert torch.cuda.get_device_capability() in {(9,0),(11,0)}, 'this protocol targets H100 or Thor'
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
torch.manual_seed(9173);np.random.seed(9173);random.seed(9173)
report={'family':a.family,'precision':a.precision,'model_id':model,'revision':revision,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'scope':'Matched public Runtime generation latency, synchronized wall time; recorded cameras and synthetic state, no network or simulator. No task-quality certificate.','calls':[],'variant':a.arm,'options':{k:v for k,v in os.environ.items() if k.startswith(('IFL_PI05','IFL_VLA4B','IFL_GROOT','IFL_BENCH'))}}
import shutil
SMI=shutil.which('nvidia-smi') or ('/usr/sbin/nvidia-smi' if Path('/usr/sbin/nvidia-smi').is_file() else None)
if SMI is None:raise RuntimeError('nvidia-smi is required for contention detection')
def competing_processes():
 import subprocess
 uuid=str(torch.cuda.get_device_properties(0).uuid)
 lines=subprocess.check_output([SMI,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
 return [int(line.split(',')[1]) for line in lines if line.split(',')[0].strip()==uuid and int(line.split(',')[1])!=os.getpid()]
report['competing_gpu_processes']=[]
api=None
loader_trim=None
try:
 competitors=competing_processes()
 if competitors:
  report['competing_gpu_processes'].append({'call':'before_load','pids':competitors})
  raise RuntimeError(f'GPU is occupied before model loading: {competitors}')
 if os.environ.get('IFL_BENCH_LOAD_HEAP_TRIM')=='1':
  from load_memory import LoadingHeapTrim
  loader_trim=LoadingHeapTrim(a.output.with_suffix('.load-memory.jsonl')).start()
 options={'nfe':{'video':2,'action':4}} if a.family=='va_2v4a' else {}
 start=time.perf_counter();api=Runtime.from_pretrained(model,revision=revision,precision=a.precision,device='cuda:0',tier_ceiling=os.environ.get('IFL_BENCH_TIER','bitexact'),**options)
 report['arm']=a.arm
 report['framework']='instinctflash-internal' if a.arm=='current' else ('lerobot' if a.family=='pi05' else 'vendor-reference')
 report['framework_source_revision']=os.environ.get('BENCH_FRAMEWORK_REVISION')
 report['history_regime']='early cycles 0-2, one warmup episode' if a.family in ('va','dreamzero') else None
 report['reference_scope']='Direct upstream eager policy versus native Runtime; framework identity is recorded separately'
 if a.family=='dreamzero' and os.environ.get('IFL_BENCH_DREAMZERO_FULL_CHECKPOINT')=='1':
  report['reference_scope']='Upstream native full-checkpoint loading flag, original initialization and forward versus full default Runtime; same weights, dtype, steps and I/O'
 if a.arm!='current':
  from upstream import build
  api=build(a.family, api)
 report['applied_passes_before_load']=[r.name for r in api.plan.results if r.applies]
 api.reset(prompt='pick up the object')
 if loader_trim is not None:report['loading_heap_trim']=loader_trim.close()
 report['setup_seconds']=time.perf_counter()-start
 loop=api._backend._loop if a.precision=='fp8' else api._backend._impl
 inner=loop
 if a.family=='groot' and a.precision=='fp8':inner=loop
 data=np.load(a.input_archive,allow_pickle=True)
 def decode(v):return [np.asarray(Image.open(io.BytesIO(bytes(x))).convert('RGB')) for x in v]
 frames=[decode(data['frame0_0'])]+[decode(x) for x in data['jpeg_0'][:12]]
 def obs(i,cycle):
  ims=frames[1+i%12]
  if a.family=='pi05':return {'observation.images.image':ims[0].transpose(2,0,1).astype(np.float32)/255.0,'observation.images.image2':ims[1].transpose(2,0,1).astype(np.float32)/255.0,'observation.state':np.zeros(8,np.float32)}
  if a.family in ('vla4','vla2'):return dict(zip(('observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist'),ims),**{'observation.state':np.full(14,.05*(i%3),np.float32)})
  if a.family=='groot':
   state={k:np.zeros(n,np.float32) for k,n in inner._state_dims.items()}
   for k,v in state.items():
    if k.endswith('eef_9d'):v[3:9]=[1,0,0,0,1,0]
   return {'images':ims[:2],'state':state}
  if a.family in ('edge','nano'):return {'image':np.asarray(Image.fromarray(ims[0]).resize((640,540))),'state':np.full(8,.01*(i%3),np.float32),'prompt':'pick up the object' if i<8 else 'place the object down'}
  if a.family.startswith('va'):
   keys=('observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist')
   indices=[0] if cycle==0 else list(range(1,5)) if cycle==1 else list(range(5,13))
   return {'obs':[dict(zip(keys,frames[j])) for j in indices]}
  keys=('observation/exterior_image_0_left','observation/exterior_image_1_left','observation/wrist_image_left')
  indices=[0] if cycle==0 else list(range(1,5))
  o={k:np.stack([frames[j][v] for j in indices]) for v,k in enumerate(keys)}
  o.update({'observation/joint_position':np.zeros(7,np.float32),'observation/gripper_position':np.zeros(1,np.float32)});return o
 outputs=[];queued_outputs=[]
 history=a.family.startswith('va') or a.family=='dreamzero'
 total=(1+max(1,(a.iterations+2)//3))*3 if history else a.warmup+a.iterations
 for i in range(total):
  competitors=competing_processes()
  if competitors:report["competing_gpu_processes"].append({"call":i,"pids":competitors})
  cycle=i%3 if history else 0
  if not history or cycle==0:api.reset(prompt='pick up the object' if i<8 else 'place the object down')
  observation=obs(i,cycle)
  torch.manual_seed(1300+cycle if history else 707+i);np.random.seed(1300+cycle if history else 707+i);random.seed(1300+cycle if history else 707+i)
  torch.cuda.synchronize();start=time.perf_counter()
  feedback={'executed_action':data['actions_0'][cycle % len(data['actions_0'])].copy()} if a.family.startswith('va') else {}
  action=np.asarray(api.predict(observation,**feedback)['action'])
  torch.cuda.synchronize();ms=1000*(time.perf_counter()-start)
  assert action.size and np.isfinite(action).all(),action.shape
  phase=('warmup' if i<3 else 'measured') if history else ('warmup' if i<a.warmup else 'measured')
  row={'i':i,'cycle':cycle,'phase':phase,'ms':ms,'shape':list(action.shape)}
  report['calls'].append(row);outputs.append(action.copy());print(row,flush=True)
  if a.family=='pi05' and a.precision=='native':
   queue=inner._p._action_queue
   queued_outputs.append(torch.stack(list(queue)).float().cpu().numpy() if queue else np.empty((0,),np.float32))
 stats=getattr(loop,'backend_stats',{})
 if callable(stats):stats=stats()
 def packed_tensors(root):
  seen=set();found=[]
  def walk(obj,path,depth=0):
   if id(obj) in seen or depth>128:return
   seen.add(id(obj))
   if isinstance(obj,torch.Tensor):
    if obj.dtype==torch.float8_e4m3fn and obj.ndim==2:found.append({'path':path,'shape':list(obj.shape)})
    return
   if isinstance(obj,dict):
    for k,v in obj.items():walk(v,path+'.'+str(k),depth+1)
   elif isinstance(obj,(list,tuple)):
    for i,v in enumerate(obj):walk(v,path+'['+str(i)+']',depth+1)
   elif isinstance(obj,torch.nn.Module) or type(obj).__module__.startswith(('instinctflash','flash_rt','lingbot','groot','cosmos','dreamzero','pi05','eval_utils')):
    if hasattr(obj,'__dict__'):walk(vars(obj),path,depth+1)
  walk(root,'loop');return found
 if a.precision=='fp8':
  report['e4m3_tensors']=packed_tensors(loop)
  assert report['e4m3_tensors'],'FP8 requested but no actual packed E4M3 tensors found'
 if a.family=='pi05':
  report['input_camera_format']='float32_CHW_0_1'
  report['observation_camera_keys']=['observation.images.image','observation.images.image2']
  if a.precision=='fp8':
   report['active_camera_counts']=sorted(loop._frontends)
   assert report['active_camera_counts']==[2]
 report['backend_stats']=stats
 report['graph_stats']=getattr(loop,'graph_stats',{})
 report['physical_gpu']=os.environ.get('CUDA_VISIBLE_DEVICES')
 report['schedule_override']=options.get('nfe')
 report['default_schedule']=dict(api._checkpoint.execution.nfe or {})
 report['guidance']=str(api._checkpoint.execution.guidance)
 report['history_feedback']='fixed recorded actions' if a.family.startswith('va') else 'native policy'
 report['plan']=api.plan.explain()
 report['execution_policy']=api.execution_policy
 report['applied_passes']=[r.name for r in api.plan.results if r.applies]
 report['interpreter']=sys.executable
 report['benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 report['input_archive_sha256']=hashlib.sha256(a.input_archive.read_bytes()).hexdigest()
 if a.precision=='fp8':
  if a.family in ('edge','nano','dreamzero'):
   assert stats.get('fp8_recipe',{}).get('projections'),'FP8 receipt absent'
  else:
   # Thor fused executors expose family-specific graph/weight receipts.
   report['engine_type']=type(loop).__module__+'.'+type(loop).__name__
   declaration=getattr(loop,'declaration',None)
   if declaration is not None:report['engine_declaration']=declaration()

 measured=[r['ms'] for r in report['calls'] if r['phase']=='measured']
 report.update(ok=True,p50_ms=float(np.median(measured)),p95_ms=float(np.percentile(measured,95)),p99_ms=float(np.percentile(measured,99)),min_ms=min(measured),max_ms=max(measured),measured_count=len(measured),peak_allocated_bytes=torch.cuda.max_memory_allocated(),numeric_environment={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark})
 np.savez_compressed(a.output.with_suffix('.npz'),actions=np.stack(outputs),**({'queued_actions':np.stack(queued_outputs)} if queued_outputs else {}));report['actions_sha256']=hashlib.sha256(a.output.with_suffix('.npz').read_bytes()).hexdigest()
except Exception as e:report.update(ok=False,error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 if loader_trim is not None:report['loading_heap_trim']=loader_trim.close()
 if api is not None:api.close()
 report['sources']={str(Path(m.__file__).resolve()):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for name,m in list(sys.modules.items()) if (name.startswith(('instinctflash','pi05_iwm','lingbot_vla','groot_n17_iwm','cosmos3_iwm','dreamzero_iwm','lerobot','gr00t','groot','lingbotvla','cosmos_framework','wan_va','deploy','upstream','load_memory'))) and getattr(m,'__file__',None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}
 a.output.write_text(json.dumps(report,indent=2,default=str)+'\n')
if not report['ok']:raise SystemExit(1)
