"""Finite serialized follow-through for remaining Thor candidates; no automatic retries."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
status={'pid':os.getpid(),'status':'queued','jobs':[]};dest=r/'remaining-thor-status-v1.json';assert not dest.exists()
def save():
 tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(dest)
save();lock=open('/tmp/thor_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
status['status']='running';save()
base=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas')
s=r/'source';fixture=s/'eval/native_total_2026-09-10/fixtures/va_eval_obs.npz'
py='/home/guanming/frt_env/bin/python'
def run(name,cmd,env,timeout):
 out=r/(name+'.json');assert not out.exists();job={'name':name,'status':'running','started':time.time(),'command':cmd};status['jobs'].append(job);save()
 try:
  with (r/(name+'.log')).open('x') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
  receipt=json.loads(out.read_text()) if out.exists() else {}
  job.update(status='complete' if result.returncode==0 and receipt.get('ok',receipt.get('quality_pass',False)) else 'failed',exit_code=result.returncode,p50_ms=receipt.get('p50_ms'),error=receipt.get('error'))
 except Exception as e:job.update(status='failed',error=repr(e))
 job['ended']=time.time();save()
pi_env=dict(base,PYTHONPATH=':'.join([str(r/'lerobot/src'),str(s),str(s/'serving'),str(s/'examples/pi05_vla')]),BENCH_SOURCE_ROOT=str(s),IFL_BENCH_TIER='numeric')
for arm,name in [('baseline_a','lerobot-pi05-nfe1-compiled-v1'),('current','flash-pi05-nfe1-native-v1')]:
 run(name,[py,str(r/'pi05-nfe1-driver/benchmark_runtime.py'),'pi05','native',str(r/(name+'.json')),'--arm',arm,'--warmup','10','--iterations','30','--input-archive',str(fixture)],pi_env,900)
# The transfer's atomic final filename is the readiness gate; never load a partial file.
checkpoint=r/'lerobot-va-checkpoint';model=checkpoint/'model.safetensors'
if model.exists():
 va_env=dict(base,PYTHONPATH=str(r/'lerobot-extra')+':'+str(r/'lerobot/src'))
 for v,a,name in [(25,50,'lerobot-va-default-v1'),(2,4,'lerobot-va-2v4a-v1')]:
  run(name,[py,str(r/'benchmark_lerobot_va.py'),'--checkpoint',str(checkpoint),'--fixture',str(fixture),'--output',str(r/(name+'.json')),'--video-steps',str(v),'--action-steps',str(a)],va_env,2400)
else:status['jobs'].append({'name':'lerobot-va','status':'waiting_for_checkpoint_transfer','path':str(model)})
status['status']='complete' if all(j['status']=='complete' for j in status['jobs']) else 'incomplete';save()
