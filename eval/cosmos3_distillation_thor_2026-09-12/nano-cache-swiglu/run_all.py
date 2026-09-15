"""Verify each cached module on changed requests before timing cross-request reuse."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
p.add_argument('--fixture',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
status=dict(status='waiting_for_lock',completed=[],active=None)
def save():
    status['updated_unix']=time.time()
    temp=a.output/'status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(a.output/'live_status.json')
save()
try:
    with open('/tmp/thor_gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for mode in ('off','reuse'):
            command=[sys.executable,str(Path(__file__).with_name('benchmark.py')),str(a.checkpoint),
                str(a.output/f'{mode}.json'),'--attention','cudnn','--allow-unqualified',
                '--fixture',str(a.fixture),'--cache-mode','reuse']
            if mode == 'reuse':command.append('--swiglu')
            with (a.output/f'{mode}.log').open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                status.update(status='running',active=dict(mode=mode,pid=child.pid));save()
                try:code=child.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise
                status['completed'].append(dict(mode=mode,pid=child.pid,exit_code=code));save()
                if code:raise RuntimeError(f'{mode} exited {code}')
        reports={mode:json.loads((a.output/f'{mode}.json').read_text()) for mode in ('off','reuse')}
        arrays={mode:np.load(a.output/f'{mode}.npz')['actions'] for mode in reports}
        comparisons={}
        for mode in reports:
            r=reports[mode];assert r['ok'] and arrays[mode].shape==(36,32,8) and np.isfinite(arrays[mode]).all()
            for key in ('fixture_sha256','checkpoint_manifest_sha256','benchmark_sha256','cache_helper_sha256',
                        'warmup_requests','measured_requests','declared_execution','first_request_native_branch_clocks','prompt_contract','torch','cuda','cudnn'):
                assert r[key]==reports['off'][key],key
            delta=np.abs(arrays[mode].astype(np.float64)-arrays['off'].astype(np.float64))
            comparisons[mode]=dict(byte_identical=arrays[mode].tobytes()==arrays['off'].tobytes(),maxabs=float(delta.max()),meanabs=float(delta.mean()))
        result=dict(arm_labels={'off':'persistent cache','reuse':'persistent cache plus shared BF16 SwiGLU'},category='SCREEN',task_quality_certified=False,trained_student=False,
            comparisons=comparisons,p50_ms={m:r['p50_ms'] for m,r in reports.items()},
            p95_ms={m:r['p95_ms'] for m,r in reports.items()},speedup=reports['off']['p50_ms']/reports['reuse']['p50_ms'],
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.suffix in ('.json','.npz') and p.name!='live_status.json'})
        (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        assert all(x['byte_identical'] for x in comparisons.values()),'Cross-request action bytes changed'
        status.update(status='success',active=None);save();print(json.dumps(result),flush=True)
except BaseException as error:
    status.update(status='failed',error=repr(error),active=None);save();raise
