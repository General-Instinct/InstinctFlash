import os,json,subprocess,time,fcntl
from pathlib import Path
root=Path(__file__).parent;source=root/'source'
lock=open('/tmp/thor_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
status={'status':'running','pid':os.getpid(),'jobs':[]}
def save():
 (root/'omni-cosmos-status.json').write_text(json.dumps(status,indent=2))
for family in ['Edge','Nano']:
 dest=root/('omni-'+family.lower()+'-eager-v1.json');assert not dest.exists()
 job={'model':family,'started':time.time(),'status':'running'};status['jobs'].append(job);save()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONPATH=str(root/'cosmos-framework-benchmark-20260913'),VLLM_WORKER_MULTIPROC_METHOD='spawn')
 with dest.with_suffix('.log').open('x') as log:
  r=subprocess.run([str(root/'omni-env/bin/python'),str(root/'benchmark_omni_cosmos.py'),'--model','nvidia/Cosmos3-'+family+'-Policy-DROID','--deploy',str(root/'vllm-omni-benchmark-20260913/vllm_omni/deploy/cosmos3_policy_droid.yaml'),'--fixture',str(source/'eval/native_total_2026-09-10/fixtures/va_eval_obs.npz'),'--output',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 job.update(status='complete' if r.returncode==0 else 'failed',exit_code=r.returncode,ended=time.time());save()
status['status']='complete' if all(j['status']=='complete' for j in status['jobs']) else 'incomplete';save()
