"""Monitor Thor sweep; verify and publish completed results automatically."""
import fcntl,json,subprocess,time,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
raw=Path('/home/ubuntu/ifl_eval/thor_fp8_20260910');remote='/home/guanming/ifl_eval/thor_fp8_20260910';host='guanming@100.68.159.80'
lock=(raw/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(value):
 p=raw/'monitor-status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def run(command,log):
 with (raw/log).open('a') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)
while True:
 status={'heartbeat':time.time()}
 try:
  data=subprocess.check_output(['ssh','-o','BatchMode=yes',host,'cat '+remote+'/progress.json'],text=True,timeout=20)
  progress=json.loads(data);(raw/'progress.json').write_text(data)
  status.update(state=progress['status'],completed=sum(j['status']=='complete' for j in progress['jobs']),failed=[j for j in progress['jobs'] if j['status']=='failed'])
  write(status)
  if progress['status']=='complete':
   if not (raw/'published.json').exists():
    status['state']='verifying';write(status)
    run(['ssh','-o','BatchMode=yes',host,'test -f '+remote+'/source-final.json || python3 '+remote+'/verify_source.py'],'final-source-check.log')
    run(['scp',host+':'+remote+'/*.json',str(raw)],'fetch-json.log')
    run(['scp',host+':'+remote+'/*.npz',str(raw)],'fetch-actions.log')
    run([str(repo/'.venv-dev/bin/python'),str(repo/'eval/thor_fp8_2026-09-10/publish.py'),str(raw)],'publish.log')
    (raw/'published.json').write_text(json.dumps({'ok':True,'time':time.time()})+'\n')
   status['state']='published';write(status);break
  if progress['status']=='incomplete' and not (raw/'recovery-started.json').exists():
   assert all(j['family'] in ('va','va_2v4a','edge','nano','dreamzero') for j in progress['jobs'] if j['status']=='failed')
   run(['ssh','-o','BatchMode=yes',host,'test -f '+remote+'/va-cache-repair.json'],'recovery-preflight.log')
   (raw/'recovery-started.json').write_text(json.dumps({'reason':'VA cache repaired; post-timing inventory depth fixed in source-v2; retry unsuccessful arms; validate identical inference/timing code across source versions','time':time.time()})+'\n')
   run(['ssh','-o','BatchMode=yes',host,'nohup python3 -u '+remote+'/run_sweep.py '+remote+' --resume-failed > '+remote+'/recovery.log 2>&1 < /dev/null &'],'recovery-launch.log')
   time.sleep(5)
   continue
  if progress['status'] in ('incomplete','failed'):
   status['state']='requires_recovery';write(status);break
 except Exception as e:
  status.update(state='monitor_error',error=repr(e));write(status)
  # Never repeatedly mutate a table after a partially completed publish stage.
  if (raw/'publish.log').exists():break
 time.sleep(30)
