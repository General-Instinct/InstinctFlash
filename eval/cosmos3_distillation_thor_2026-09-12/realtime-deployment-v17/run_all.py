"""Serialize both V17 seed native/cuDNN pairs; persist actual child handles."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('study_root',type=Path)
p.add_argument('results',type=Path)
p.add_argument('--fixture',type=Path,required=True)
a=p.parse_args();a.results.mkdir(parents=True,exist_ok=False)
expected={12031:'1eec9424c306fb663ee0986fc84b7051988dc6faecb5238de21c7229eef21261',
          12032:'b5aede0d0cc6354fac1c9116cc592daee495caa55b99b417432a76b35b966aa9'}
status=dict(status='waiting_for_lock',completed=[],active=None)
def save():
    status['updated_unix']=time.time()
    temp=a.results/'live_status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(a.results/'live_status.json')
save()
try:
    with open('/tmp/thor_gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for seed,digest in expected.items():
            checkpoint=a.study_root/f'student-v17-cfg1-seed{seed}-v3'
            assert hashlib.sha256((checkpoint/'instinctcompress_manifest.json').read_bytes()).hexdigest()==digest
            output=a.results/f'seed{seed}';output.mkdir()
            for arm in ('native','cudnn'):
                command=[sys.executable,str(Path(__file__).with_name('benchmark_candidate.py')),
                    str(checkpoint),str(output/f'{arm}.json'),'--attention',arm,
                    '--fixture',str(a.fixture),'--allow-unqualified']
                with (output/f'{arm}.log').open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    status.update(status='running',active=dict(seed=seed,attention=arm,pid=child.pid));save()
                    try: code=child.wait(timeout=2400)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise
                    status['completed'].append(dict(seed=seed,attention=arm,pid=child.pid,exit_code=code));save()
                    if code:raise RuntimeError(f'{seed} {arm} exited {code}')
            subprocess.run([sys.executable,str(Path(__file__).with_name('compare.py')),str(output)],check=True)
        status.update(status='success',active=None);save()
except BaseException as error:
    status.update(status='failed',error=repr(error),active=None);save();raise
