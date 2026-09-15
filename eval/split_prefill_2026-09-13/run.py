"""Bounded Edge A / split-prefill / B screen on Thor, no schedule change."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
source=r/'source'
driver=source/'benchmarks/regression/cosmos_numeric.py'
code=driver.read_text()
needle='            torch.cuda.synchronize()\n            start = time.perf_counter()'
assert code.count(needle)==1
code=code.replace(needle, "            region_owner = api._backend._impl._service._ifl_generation_regions\n"
    "            counters_before = dict(region_owner.stats)\n" + needle)
needle="            report['calls'].append(row)"
assert code.count(needle)==1
code=code.replace(needle, "            row['region_counts'] = {k: region_owner.stats[k] - counters_before[k]\n"
    "                for k in ('compiled_calls', 'compiled_prefill_calls', 'prefill_checks', 'eager_checks')}\n" + needle)
driver.write_text(code)
python='/home/guanming/thorcol/cosmos-framework/.venv/bin/python'
status=r/'live_status.json';assert not status.exists()
s={'status':'queued','pid':os.getpid(),'jobs':[]}
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(status)
save()
env=dict(os.environ,PATH='/home/guanming/thorcol/cosmos-framework/.venv/bin:/home/guanming/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/usr/sbin:/bin',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',PYTHONPATH=f'{source}:{source}/examples/cosmos3_policy',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas')
with open('/tmp/thor_gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);s['status']='running';save()
 for family in ('edge',):
  for arm in ('reference-a','split-prefill','reference-b'):
   out=r/f'{family}-{arm}.json';assert not out.exists()
   script=source/'benchmarks/regression/cosmos_numeric.py'
   env['IFL_COSMOS3_GEN_REGIONS']='1'
   env['IFL_COSMOS3_SPLIT_PREFILL']='1' if arm=='split-prefill' else '0'
   cmd=[python,str(script),family,'current',str(out),'--tier-ceiling','numeric','--iterations','10']
   job={'family':family,'arm':arm,'command':cmd,'status':'running','started':time.time()};s['jobs'].append(job);save()
   with out.with_suffix('.log').open('x') as log:
    child=subprocess.Popen(cmd,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job['pid']=child.pid;save()
    try:child.wait(timeout=900)
    except subprocess.TimeoutExpired:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     job['timeout']=True
   receipt=json.loads(out.read_text()) if out.exists() else {}
   job.update(status='complete' if child.returncode==0 and receipt.get('ok') else 'failed',exit_code=child.returncode,error=receipt.get('error'),ended=time.time());save()
   # A compiler rejection is retained. The other model remains independently testable.
 s['status']='complete' if all(j['status']=='complete' for j in s['jobs']) else 'completed_with_rejections';save()
