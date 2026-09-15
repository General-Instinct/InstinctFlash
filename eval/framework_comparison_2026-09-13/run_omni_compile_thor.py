"""Bounded compiled candidates for the three supported Omni models."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
r=Path(__file__).parent;dest=r/'omni-compiled-status-v1.json';assert not dest.exists()
s={'pid':os.getpid(),'status':'queued','jobs':[]}
def save():
 tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(dest)
save();lock=open('/tmp/thor_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);s['status']='running';save()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(r/'omni-thor-compat-v1')+':'+str(r/'cosmos-framework-benchmark-20260913'),TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas')
for family in ['edge','nano','dreamzero']:
 name='omni-'+family+'-compiled-v1';out=r/(name+'.json');assert not out.exists()
 cosmos=family!='dreamzero';model='nvidia/Cosmos3-'+family.title()+'-Policy-DROID' if cosmos else 'GEAR-Dreams/DreamZero-DROID'
 script='benchmark_omni_cosmos_v2.py' if cosmos else 'benchmark_omni_dreamzero_v4.py';deploy='cosmos3_policy_droid.yaml' if cosmos else 'dreamzero.yaml'
 cmd=[str(r/'omni-env/bin/python'),str(r/script),'--model',model,'--deploy',str(r/'vllm-omni-benchmark-20260913/vllm_omni/deploy'/deploy),'--fixture',str(r/'source/eval/native_total_2026-09-10/fixtures/va_eval_obs.npz'),'--compile','--output',str(out)]
 job={'name':name,'status':'running','started':time.time(),'command':cmd};s['jobs'].append(job);save()
 try:
  with out.with_suffix('.log').open('x') as log:
   run=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:run.wait(timeout=1800)
   except subprocess.TimeoutExpired:
    os.killpg(run.pid,signal.SIGTERM)
    try:run.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(run.pid,signal.SIGKILL);run.wait()
    raise
  receipt=json.loads(out.read_text()) if out.exists() else {};job.update(status='complete' if run.returncode==0 and receipt.get('ok') else 'failed',exit_code=run.returncode,p50_ms=receipt.get('p50_ms'),error=receipt.get('error'))
 except Exception as e:job.update(status='failed',error=repr(e));save();break
 job['ended']=time.time();save()
s['status']='complete' if len(s['jobs'])==3 and all(j['status']=='complete' for j in s['jobs']) else 'incomplete';save()
