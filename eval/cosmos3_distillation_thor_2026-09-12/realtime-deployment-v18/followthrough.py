"""One-shot CPU preparation/transfer after producer gates; launch one Thor study."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('receipt_dir',type=Path)
a=p.parse_args();a.receipt_dir=a.receipt_dir.resolve();a.receipt_dir.mkdir(parents=True,exist_ok=False)
os.nice(10)
source=Path(__file__).resolve().parent
producer=Path('/home/ubuntu/InstinctCompress/results/cosmos3_droid/nano_dmd_v18')
completion=producer/'training_execution_completion.json'
local=Path('/home/ubuntu/ifl_cosmos_deployment')
host='guanming@100.68.159.80'
remote='/home/guanming/ifl_eval/cosmos_distill_20260912'
remote_study=remote+'/realtime-deployment-v18-v1'
source_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('prepare.py','benchmark.py','run_all.py','persistent_cache.py','prompt_contract.py')}
status=dict(status='waiting_producer',supervisor=1769044,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=source_hashes,stages=[])
def save():
    status['updated_at']=datetime.now(timezone.utc).isoformat()
    temp=a.receipt_dir/'status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(a.receipt_dir/'live_status.json')
def run(stage,command):
    for name,digest in source_hashes.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen pipeline source changed: {name}')
    with (a.receipt_dir/f'{stage}.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        status.update(status='running',active=dict(stage=stage,pid=child.pid));save()
        code=child.wait()
    status['stages'].append(dict(stage=stage,exit_code=code));save()
    if code:raise RuntimeError(f'{stage} failed with {code}; no automatic retry')
try:
    save();deadline=time.monotonic()+3600
    while not completion.exists():
        os.kill(1769044,0)
        if time.monotonic()>deadline:raise TimeoutError('Producer observation budget exhausted; no restart')
        time.sleep(15)
    assert json.loads(completion.read_text())['status']=='success'
    status['producer_completion_sha256']=hashlib.sha256(completion.read_bytes()).hexdigest();save()
    for seed in (12031,12032):
        export=producer/f'training/nano_sde1_cfg1_seed{seed}/export_roundtrip'
        destination=local/f'v18-cfg1-seed{seed}-v1'
        run(f'prepare_{seed}',[sys.executable,str(source/'prepare.py'),str(export),str(destination),'--completion',str(completion)])
    run('remote_directory',['ssh',host,'mkdir '+shlex.quote(remote_study)])
    scripts=['benchmark.py','run_all.py','persistent_cache.py','prompt_contract.py']
    run('stage_sources',['scp',*[str(source/name) for name in scripts],host+':'+remote_study+'/'])
    for seed in (12031,12032):
        name=f'student-v18-cfg1-seed{seed}-v1'
        # Fresh paths, checksum matching, and final manifest validation. Existing
        # identical frozen native files may be hardlinked; never edited in place.
        run(f'reserve_{seed}',['ssh',host,'mkdir '+shlex.quote(remote+'/'+name)])
        run(f'transfer_{seed}',['rsync','-a','--checksum','--link-dest='+remote+'/nano-original-native-v1',
            str(local/f'v18-cfg1-seed{seed}-v1')+'/',host+':'+remote+'/'+name+'/'])
    env=['env','CUDA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=4','HF_HUB_OFFLINE=1','PYTHONHASHSEED=0',
         'TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas','TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas',
         'PATH=/home/guanming/thorcol/cosmos-framework/.venv/bin:/home/guanming/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin',
         f'PYTHONPATH={remote}/flash:{remote}/compress:{remote}/flash/examples/cosmos3_policy:{remote}/realtime-deployment-v2',
         '/home/guanming/thorcol/cosmos-framework/.venv/bin/python',remote_study+'/run_all.py',
         remote+'/student-v18-cfg1-seed12031-v1',remote+'/student-v18-cfg1-seed12032-v1',remote_study+'/results',
         '--fixture',remote+'/flash/eval/native_total_2026-09-10/fixtures/va_eval_obs.npz']
    command='nohup '+shlex.join(env)+' > '+shlex.quote(remote_study+'/runner.log')+' 2>&1 < /dev/null & echo $!'
    launched=subprocess.run(['ssh',host,command],check=True,text=True,capture_output=True)
    pid=int(launched.stdout.strip())
    status.update(status='thor_launched_unqualified',active=dict(host=host,pid=pid),
                  remote_live_status=remote_study+'/results/live_status.json');save()
except BaseException as error:
    status.update(status='failed',error=repr(error),active=None);save();raise
