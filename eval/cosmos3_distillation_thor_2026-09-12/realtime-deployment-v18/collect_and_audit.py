"""Observe the existing V18 deployment only; collect and audit, never launch jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
p.add_argument('--coordinator',type=Path,default=Path('/home/ubuntu/ifl_cosmos_quality_preflight/v18_deployment_followthrough_v2/live_status.json'))
p.add_argument('--coordinator-pid',type=int,default=1789651)
p.add_argument('--remote',default='/home/guanming/ifl_eval/cosmos_distill_20260912/realtime-deployment-v18-v1/results')
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
source=Path(__file__).resolve().parent
coordinator=a.coordinator
host='guanming@100.68.159.80';remote=a.remote
status=dict(status='waiting_coordinator',coordinator_pid=a.coordinator_pid,task_quality_certified=False)
def save():
    status['updated_unix']=time.time();temp=a.output/'status.tmp'
    temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(a.output/'live_status.json')
def ssh(command):return subprocess.check_output(['ssh',host,command],text=True,timeout=30)
try:
    save();deadline=time.monotonic()+10800
    while True:
        c=json.loads(coordinator.read_text())
        if c['status']=='failed':raise RuntimeError('Coordinator failed; no restart')
        if c['status']=='thor_launched_unqualified':break
        os.kill(a.coordinator_pid,0)
        if time.monotonic()>deadline:raise TimeoutError('Coordinator observation budget exhausted')
        time.sleep(15)
    status.update(status='observing_thor',thor_pid=c['active']['pid']);save()
    while True:
        try:
            payload=ssh('cat '+remote+'/live_status.json 2>/dev/null || true')
            if not payload.strip():
                ssh('kill -0 '+str(int(c['active']['pid'])))
                if time.monotonic()>deadline:raise TimeoutError('Remote startup observation expired')
                time.sleep(2);continue
            r=json.loads(payload)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            status.update(status='observation_retry',last_observation_error=repr(error));save()
            if time.monotonic()>deadline:raise TimeoutError('Observation unavailable; worker state unknown')
            time.sleep(15);continue
        status.update(status='observing_thor',remote_status=r);save()
        if r['status']=='success':break
        if r['status']=='failed':raise RuntimeError('Thor runner failed; preserve remote partial receipts')
        try:ssh('kill -0 '+str(int(c['active']['pid'])))
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            status.update(status='observation_retry',last_observation_error=repr(error));save()
        if time.monotonic()>deadline:raise TimeoutError('Thor observation budget exhausted')
        time.sleep(30)
    for seed in (12031,12032):
        dest=a.output/f'seed{seed}';dest.mkdir()
        subprocess.run(['scp',host+':'+remote+f'/seed{seed}/*.json',host+':'+remote+f'/seed{seed}/*.npz',str(dest)+'/'],check=True)
        prepared=Path(f'/home/ubuntu/ifl_cosmos_deployment/v18-cfg1-seed{seed}-v1-preparation.json')
        package=json.loads(prepared.read_text())
        for arm in ('native','cudnn','combined'):
            report=json.loads((dest/f'{arm}.json').read_text())
            assert report['checkpoint_manifest_sha256']==package['manifest_sha256']
            assert report['benchmark_sha256']==c['sources']['benchmark.py']
            assert report['cache_helper_sha256']==c['sources']['persistent_cache.py']
        comparison=json.loads((dest/'comparison.json').read_text())
        for name,digest in comparison['files'].items():
            assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest
    subprocess.run([sys.executable,str(source/'audit_contract.py'),str(a.output),str(a.output/'contract_audit.json')],check=True)
    status.update(status='deployment_screen_audited',task_quality_certified=False);save()
except BaseException as error:
    status.update(status='observer_stopped',error=repr(error));save();raise
