"""Serial per-GPU pairs; independent lanes use separate, initially idle H100s."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('families',nargs='+');a=p.parse_args()
r=a.root.resolve();s=r/'source-v1';manifest=json.loads((r/'source-v1.json').read_text())['files']
configs={'pi05':('/home/ubuntu/tools/pi05env313/bin/python',{}),'vla4':('/home/ubuntu/lingbot-vla-repo/.venv/bin/python',{'LINGBOT_VLA_ROOT':'/home/ubuntu/lingbot-vla-repo'}),'vla2':('/home/ubuntu/lingbot-vla-v2-repo/.venv/bin/python',{'LINGBOT_VLA_V2_ROOT':'/home/ubuntu/lingbot-vla-v2-repo'}),'groot':('/home/ubuntu/Isaac-GR00T/.venv/bin/python',{'GR00T_ROOT':'/home/ubuntu/Isaac-GR00T'}),'edge':('/home/ubuntu/cosmos-framework/.venv/bin/python',{}),'nano':('/home/ubuntu/cosmos-framework/.venv/bin/python',{}),'dreamzero':('/home/ubuntu/dreamzero-repo/.venv/bin/python',{'DREAMZERO_ROOT':'/home/ubuntu/dreamzero-repo','DYNAMIC_CACHE_SCHEDULE':'false','NUM_DIT_STEPS':'8','ENABLE_TENSORRT':'false'}),'va':('/home/ubuntu/.venv-lingbot/bin/python',{'LINGBOT_ROOT':'/home/ubuntu/lingbot-va'}),'va_2v4a':('/home/ubuntu/.venv-lingbot/bin/python',{'LINGBOT_ROOT':'/home/ubuntu/lingbot-va'})}
status={'gpu':a.gpu,'status':'running','jobs':[]}
status_path=r/f'lane-{a.gpu}.json';assert not status_path.exists()
def save():
 t=status_path.with_suffix('.tmp');t.write_text(json.dumps(status,indent=2)+'\n');t.replace(status_path)
try:
 for family in a.families:
  for precision in ['native','fp8']:
   for name,digest in manifest.items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest,name
   py,extra=configs[family]
   output=r/f'{family}-{precision}.json';assert not output.exists()
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0',MASTER_PORT=str(29600+int(a.gpu)),NO_ALBUMENTATIONS_UPDATE='1',PYTHONPATH=':'.join([str(s)]+[str(x) for x in (s/'examples').iterdir() if x.is_dir()]),**extra)
   job={'family':family,'precision':precision,'status':'running','started':time.time()};status['jobs'].append(job);save()
   command=[py,str(s/'benchmark.py'),family,precision,str(output)]
   with output.with_suffix('.log').open('x') as log:done=subprocess.run(command,env=env,cwd=s,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
   job.update(exit_code=done.returncode,ended=time.time(),status='complete' if done.returncode==0 else 'failed');save()
   if done.returncode==0:
    report=json.loads(output.read_text());assert report['ok']
    job['p50_ms']=report['p50_ms'];save();print(job,flush=True)
   # A failed model is recorded; do not leave the other authorized families untested.
 status['status']='complete' if all(j['status']=='complete' for j in status['jobs']) else 'incomplete'
except BaseException as e:status.update(status='failed',error=repr(e));raise
finally:save()
