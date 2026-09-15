"""Matched full public VLA4 calls, including live vision and robot processors."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from instinctflash import Runtime

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
parser.add_argument('--precision',choices=['native','fp8'],required=True)
parser.add_argument('--calls',type=int,default=32)
args=parser.parse_args();out=args.output
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
if args.calls<2:raise ValueError('at least two calls required')
frames=np.load(Path.home()/'ifl/t3_assets/calib_obs.npz')
def image(key,i):return np.clip((frames[key][i].astype(np.float32)+1)*127.5,0,255).astype(np.uint8)
def observation(i):
 return {'observation.images.cam_high':image('image',3+i%2),
         'observation.images.cam_left_wrist':image('wrist_image',3),
         'observation.images.cam_right_wrist':image('wrist_image',4),
         'observation.state':np.full(14,0.1*(i%2),dtype=np.float32)}
model='robbyant/lingbot-vla-4b-posttrain-robotwin';revision='fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e'
t0=time.perf_counter()
rt=Runtime.from_pretrained(model,revision=revision,precision=args.precision,device='cuda:0')
rt.reset(prompt='pick up the cup')
setup=time.perf_counter()-t0
latencies=[];actions=[]
try:
 for i in range(4+args.calls):
  obs=observation(i);torch.manual_seed(1701+i);np.random.seed(1701+i)
  torch.cuda.synchronize();t0=time.perf_counter()
  result=rt.predict(obs)
  torch.cuda.synchronize();ms=(time.perf_counter()-t0)*1000
  a=result['action'];assert a.shape==(25,14) and np.isfinite(a).all()
  latencies.append(ms);actions.append(a.copy())
 np.savez(out.with_suffix('.npz'),actions=np.stack(actions),latency_ms=latencies)
 loop=rt._backend._loop if args.precision=='fp8' else rt._backend._impl
 if args.precision=='fp8':
  generator=loop._server.vla.model;fe=generator.frontend
  assert fe._l_qkv_w[0].dtype==torch.float8_e4m3fn and fe._e_qkv_w[0].dtype==torch.float8_e4m3fn
  vision_dtypes=sorted({str(p.dtype) for p in generator.vision.model.parameters()})
  assert vision_dtypes==['torch.bfloat16']
  recipe=dict(vision=vision_dtypes,language_qkv=str(fe._l_qkv_w[0].dtype),expert_qkv=str(fe._e_qkv_w[0].dtype))
 else:
  recipe=dict(parameters=sorted({str(p.dtype) for p in loop._server.vla.parameters()}))
  assert all('float8' not in v for v in recipe['parameters'])
 def stats(v):return dict(count=len(v),p50_ms=float(np.percentile(v,50)),p99_ms=float(np.percentile(v,99)),max_ms=float(max(v)))
 result=dict(model_id=model,revision=revision,precision=args.precision,setup_seconds=setup,
             warmup=stats(latencies[:4]),measured=stats(latencies[4:]),recipe=recipe,
             graph_stats=getattr(loop,'graph_stats',{}),native_source_root=os.environ.get('LINGBOT_VLA_ROOT'),
             matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
             actions_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             loop_source_sha256=hashlib.sha256(Path(inspect.getfile(type(loop))).read_bytes()).hexdigest(),
             scope='Public Runtime 25x14 replies, current vision/state/prompt processors; recorded images and synthetic state; no closed-loop quality certificate')
 out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
finally:rt.close()
