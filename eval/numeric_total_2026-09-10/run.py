"""Serialized A/B/A native coverage; each arm runs in a fresh process."""
import argparse, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--device',choices=['h100','thor'],required=True);p.add_argument('--gpu',required=True);p.add_argument('--families',nargs='+',required=True);a=p.parse_args()
r=a.root.resolve();s=r/'source';out=r/(a.device+'-'+a.gpu);out.mkdir(exist_ok=True)
lock=open('/tmp/thor_gpu.lock' if a.device=='thor' else '/tmp/ifl-native-total-gpu'+a.gpu+'.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if a.device=='h100':
 home=Path('/home/ubuntu');configs={
 'pi05':('pi05env312/bin/python',{}),
 'vla4':('lingbot-vla-repo/.venv/bin/python',{'LINGBOT_VLA_ROOT':str(home/'lingbot-vla-repo')}),
 'groot':('Isaac-GR00T/.venv/bin/python',{'GR00T_ROOT':str(home/'Isaac-GR00T')}),
 'vla2':('lingbot-vla-v2-repo/.venv/bin/python',{'LINGBOT_VLA_V2_ROOT':str(home/'lingbot-vla-v2-repo')}),
 'edge':('cosmos-framework/.venv/bin/python',{}),'nano':('cosmos-framework/.venv/bin/python',{}),
 'dreamzero':('dreamzero-repo/.venv/bin/python',{'DREAMZERO_ROOT':str(home/'dreamzero-repo')}),
 'va':('.venv-lingbot/bin/python',{'LINGBOT_ROOT':str(home/'lingbot-va')})}
 fixture=s/'eval/native_total_2026-09-10/fixtures/va_eval_obs.npz'
else:
 home=Path('/home/guanming');configs={
 'pi05':('frt_env/bin/python',{}),
 'vla4':('venv_vla4b/bin/python',{'LINGBOT_VLA_ROOT':str(home/'lingbot-vla-repo')}),
 'groot':('thorcol/groot_env/bin/python',{'GR00T_ROOT':str(home/'thorcol/Isaac-GR00T')}),
 'vla2':('venv_vla2/bin/python',{'LINGBOT_VLA_V2_ROOT':str(home/'lingbot-vla-v2-repo')}),
 'edge':('thorcol/cosmos-framework/.venv/bin/python',{}),'nano':('thorcol/cosmos-framework/.venv/bin/python',{}),
 'dreamzero':('thorcol/dz_env/bin/python',{'DREAMZERO_ROOT':str(home/'thorcol/dreamzero')}),
 'va':('venv_va/bin/python',{'LINGBOT_ROOT':str(home/'lingbot-va')})}
 fixture=s/'eval/native_total_2026-09-10/fixtures/va_eval_obs.npz'
manifest=json.loads((r/'source.json').read_text())
def verify():
 for name,digest in manifest.items():
  assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest,name
status={'pid':os.getpid(),'status':'running','started':time.time(),'jobs':[]}
progress=out/'progress.json';assert not progress.exists()
def save():
 tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(progress)
try:
 for family in a.families:
  py,extra=configs[family];py=str(home/py)
  arms=['baseline_a','current']+(['current_fewstep']+(['current_fewstep_fp8'] if a.device=='thor' else []) if family=='va' else [])+['baseline_b']
  for arm in arms:
   verify();dest=out/(family+'-'+arm+'.json');assert not dest.exists()
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0',MASTER_PORT=str(29810+int(a.gpu)),NO_ALBUMENTATIONS_UPDATE='1',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',PATH=str(Path(py).parent)+':/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/usr/sbin:/bin',PYTHONPATH=':'.join([str(s),str(s/'serving')]+[str(x) for x in (s/'examples').iterdir() if x.is_dir()]),DYNAMIC_CACHE_SCHEDULE='false',NUM_DIT_STEPS='8',ENABLE_TENSORRT='false',IFL_BENCH_TIER='numeric' if arm.startswith('current') else 'bitexact',**extra)
   job={'family':family,'arm':arm,'status':'running','started':time.time()};status['jobs'].append(job);save()
   try:
    with dest.with_suffix('.log').open('x') as log:
     run=subprocess.run([py,str(s/'eval/numeric_total_2026-09-10/benchmark.py'),family,'fp8' if arm=='current_fewstep_fp8' else 'native',str(dest),'--arm',arm,'--iterations','20','--input-archive',str(fixture)],cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    receipt=json.loads(dest.read_text()) if dest.exists() else {}
    job.update(status='complete' if run.returncode==0 and receipt.get('ok') else 'failed',exit_code=run.returncode,p50_ms=receipt.get('p50_ms'),error=receipt.get('error'))
   except Exception as error:job.update(status='failed',error=repr(error))
   job['ended']=time.time();save();print(job,flush=True)
 verify();status['status']='complete' if all(j['status']=='complete' for j in status['jobs']) else 'incomplete'
finally:save()
