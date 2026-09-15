"""Run the remaining LeRobot VA cells after the checkpoint transfer completes."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
r=Path(__file__).parent;dest=r/'lerobot-va-status-v1.json';assert not dest.exists()
s={'pid':os.getpid(),'status':'waiting_for_checkpoint','jobs':[]}
def save():
 tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(dest)
save();checkpoint=r/'lerobot-va-checkpoint'
required=['model.safetensors','config.json','policy_preprocessor.json','policy_postprocessor.json','policy_postprocessor_step_0_unnormalizer_processor.safetensors']
start=time.time()
while not all((checkpoint/f).is_file() for f in required):
 if time.time()-start>1800:s['status']='checkpoint_wait_timeout';save();raise SystemExit(1)
 time.sleep(10)
s['status']='queued';save();lock=open('/tmp/thor_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);s['status']='running';save()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(r/'lerobot-extra')+':'+str(r/'lerobot/src'),TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas')
for v,a,name in [(25,50,'lerobot-va-default-v1'),(2,4,'lerobot-va-2v4a-v1')]:
 out=r/(name+'.json');assert not out.exists()
 cmd=['/home/guanming/frt_env/bin/python',str(r/'benchmark_lerobot_va.py'),'--checkpoint',str(checkpoint),'--fixture',str(r/'source/eval/native_total_2026-09-10/fixtures/va_eval_obs.npz'),'--output',str(out),'--video-steps',str(v),'--action-steps',str(a)]
 job={'name':name,'status':'running','started':time.time(),'command':cmd};s['jobs'].append(job);save()
 try:
  with out.with_suffix('.log').open('x') as log:
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:child.wait(timeout=2400)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
  receipt=json.loads(out.read_text()) if out.exists() else {};job.update(status='complete' if child.returncode==0 and receipt.get('ok') else 'failed',exit_code=child.returncode,p50_ms=receipt.get('p50_ms'),error=receipt.get('error'))
 except Exception as e:job.update(status='failed',error=repr(e));save();break
 job['ended']=time.time();save()
 if job['status']!='complete':break
s['status']='complete' if len(s['jobs'])==2 and all(j['status']=='complete' for j in s['jobs']) else 'incomplete';save()
