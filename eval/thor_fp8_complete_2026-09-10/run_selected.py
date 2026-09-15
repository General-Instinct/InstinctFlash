"""Remeasure selected complete pairs from a new immutable source snapshot."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('source');p.add_argument('families',nargs='+');a=p.parse_args();r=a.root.resolve();s=r/a.source
lock=Path('/tmp/thor_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
manifest=json.loads((r/(a.source+'.json')).read_text())['files'];libs=json.loads((r/'compiled-libraries.json').read_text())
# Match the original sweep's model environments exactly.
original=(r/'run_sweep.py').read_text();start=original.index('configs=');end=original.index('\ndest=',start)
namespace={};exec(original[start:end],{},namespace);configs=namespace['configs']
dest=r/'progress.json';status=json.loads(dest.read_text());assert status['status'] in ('complete','incomplete')
archive=r/('progress-before-'+a.source+'.json');assert not archive.exists();archive.write_bytes(dest.read_bytes())
status.update(status='running',pid=os.getpid(),resumed=time.time())
def save():
 tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(dest)
def verify():
 for name,digest in {**manifest,**libs}.items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest,name
try:
 for family in a.families:
  py,extra=configs[family]
  for precision in ('native','fp8'):
   verify();output=r/f'{family}-{precision}.json'
   for ext in ('.json','.log','.npz'):
    old=output.with_suffix(ext)
    if old.exists():
     saved=r/(old.stem+'.before-'+a.source+ext);assert not saved.exists();old.rename(saved)
   prior=next(j for j in status['jobs'] if j['family']==family and j['precision']==precision)
   job=dict(family=family,precision=precision,status='running',source=a.source,started=time.time(),previous_attempt=prior)
   status['jobs'][status['jobs'].index(prior)]=job;save()
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0',MASTER_PORT='29740',NO_ALBUMENTATIONS_UPDATE='1',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',PATH=str(Path(py).parent)+':/usr/local/cuda/bin:/home/guanming/.local/bin:/usr/local/bin:/usr/bin:/bin',PYTHONPATH=':'.join([str(s)]+[str(x) for x in (s/'examples').iterdir() if x.is_dir()]),**extra)
   with output.with_suffix('.log').open('x') as log:
    run=subprocess.run([py,str(s/'benchmark.py'),family,precision,str(output)],env=env,cwd=s,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
   d=json.loads(output.read_text()) if output.exists() else {}
   job.update(status='complete' if run.returncode==0 and d.get('ok') else 'failed',ended=time.time(),exit_code=run.returncode,p50_ms=d.get('p50_ms'),error=d.get('error'));save();print(job,flush=True)
 verify();status['status']='complete' if all(j['status']=='complete' for j in status['jobs']) else 'incomplete'
except BaseException as e:status.update(status='failed',error=repr(e));raise
finally:save()
