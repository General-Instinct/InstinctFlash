"""Run matched Thor arms under one exclusive GPU lock; preserve every attempt."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--resume-failed',action='store_true');a=p.parse_args();r=a.root.resolve();version=('source-v3' if (r/'source-v3.json').exists() else 'source-v2') if a.resume_failed else 'source-v1';s=r/version
lock=Path('/tmp/thor_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=json.loads((r/(version+'.json')).read_text())['files'];libs=json.loads((r/'compiled-libraries.json').read_text())
configs={'vla4':('/home/guanming/venv_vla4b/bin/python',{'LINGBOT_VLA_ROOT':'/home/guanming/lingbot-vla-repo'}),'vla2':('/home/guanming/venv_vla2/bin/python',{'LINGBOT_VLA_V2_ROOT':'/home/guanming/lingbot-vla-v2-repo'}),'pi05':('/home/guanming/frt_env/bin/python',{}),'groot':('/home/guanming/thorcol/groot_env/bin/python',{'GR00T_ROOT':'/home/guanming/thorcol/Isaac-GR00T'}),'va':('/home/guanming/venv_va/bin/python',{'LINGBOT_ROOT':'/home/guanming/lingbot-va'}),'va_2v4a':('/home/guanming/venv_va/bin/python',{'LINGBOT_ROOT':'/home/guanming/lingbot-va'}),'edge':('/home/guanming/thorcol/cosmos-framework/.venv/bin/python',{}),'nano':('/home/guanming/thorcol/cosmos-framework/.venv/bin/python',{}),'dreamzero':('/home/guanming/thorcol/dz_env/bin/python',{'DREAMZERO_ROOT':'/home/guanming/thorcol/dreamzero','DYNAMIC_CACHE_SCHEDULE':'false','NUM_DIT_STEPS':'8','ENABLE_TENSORRT':'false'})}
dest=r/'progress.json'
if a.resume_failed:
 status=json.loads(dest.read_text());assert status['status']=='incomplete'
 archive=r/('progress-recovery-1.json' if (r/'progress-initial.json').exists() else 'progress-initial.json');assert not archive.exists();archive.write_bytes(dest.read_bytes())
 status.update(pid=os.getpid(),status='running',resumed=time.time())
else:
 assert not dest.exists();status={'pid':os.getpid(),'status':'running','jobs':[],'started':time.time()}
def save():
 tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(dest)
def preflight():
 for name,digest in {**manifest,**libs}.items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest,name
try:
 for family,(py,extra) in configs.items():
  for precision in ['native','fp8']:
   prior=next((j for j in status['jobs'] if j['family']==family and j['precision']==precision),None)
   if prior and prior['status']=='complete':continue
   preflight();output=r/f'{family}-{precision}.json'
   if prior:
    assert prior['status'] in ('failed','complete')
    for ext in ('.log','.json','.npz'):
     old=output.with_suffix(ext)
     if old.exists():
      saved=r/(old.stem+('.recovery-attempt-1' if version=='source-v3' else '.initial-attempt')+ext);assert not saved.exists();old.rename(saved)
   assert not output.exists()
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0',MASTER_PORT='29740',NO_ALBUMENTATIONS_UPDATE='1',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',PATH=str(Path(py).parent)+':/usr/local/cuda/bin:/home/guanming/.local/bin:/usr/local/bin:/usr/bin:/bin',PYTHONPATH=':'.join([str(s)]+[str(x) for x in (s/'examples').iterdir() if x.is_dir()]),**extra)
   job={'family':family,'precision':precision,'status':'running','started':time.time()}
   if prior:
    job['previous_attempt']=dict(prior);status['jobs'][status['jobs'].index(prior)]=job
   else:status['jobs'].append(job)
   save()
   with output.with_suffix('.log').open('x') as log:
    run=subprocess.run([py,str(s/'benchmark.py'),family,precision,str(output)],env=env,cwd=s,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
   job.update(exit_code=run.returncode,ended=time.time(),status='complete' if run.returncode==0 else 'failed')
   if output.exists():
    report=json.loads(output.read_text());job.update(p50_ms=report.get('p50_ms'),error=report.get('error'))
   save();print(job,flush=True)
 preflight();status['status']='complete' if all(j['status']=='complete' for j in status['jobs']) else 'incomplete'
except BaseException as e:status.update(status='failed',error=repr(e));raise
finally:save()
