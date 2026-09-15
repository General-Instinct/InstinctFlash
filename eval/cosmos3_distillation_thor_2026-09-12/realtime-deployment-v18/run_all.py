"""Both final Nano seeds: native, cuDNN, and cache/SwiGLU; no quality admission."""
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
p.add_argument('seed12031',type=Path);p.add_argument('seed12032',type=Path)
p.add_argument('output',type=Path);p.add_argument('--fixture',type=Path,required=True)
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
status=dict(status='waiting_lock',completed=[],active=None)
def save():
    status['updated_unix']=time.time()
    tmp=a.output/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(a.output/'live_status.json')
save()
try:
    with open('/tmp/thor_gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for seed in (12031,12032):
            checkpoint=getattr(a,f'seed{seed}').resolve()
            output=a.output/f'seed{seed}';output.mkdir()
            for arm in ('native','cudnn','combined'):
                command=[sys.executable,str(Path(__file__).with_name('benchmark.py')),str(checkpoint),
                         str(output/f'{arm}.json'),'--fixture',str(a.fixture.resolve()),'--allow-unqualified',
                         '--attention','native' if arm=='native' else 'cudnn']
                if arm=='combined':command+=['--cache-mode','reuse','--swiglu']
                with (output/f'{arm}.log').open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    status.update(status='running',active=dict(seed=seed,arm=arm,pid=child.pid));save()
                    try:code=child.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise
                    status['completed'].append(dict(seed=seed,arm=arm,pid=child.pid,exit_code=code));save()
                    if code:raise RuntimeError(f'{seed}/{arm} exited {code}; no retry')
            reports={arm:json.loads((output/f'{arm}.json').read_text()) for arm in ('native','cudnn','combined')}
            arrays={arm:np.load(output/f'{arm}.npz')['actions'] for arm in reports}
            for arm,r in reports.items():
                assert r['ok'] and r['trained_student'] and r['student_provenance']['training_seed']==seed
                assert arrays[arm].shape==(36,32,8) and np.isfinite(arrays[arm]).all()
                for key in ('fixture_sha256','checkpoint_manifest_sha256','student_provenance',
                            'benchmark_sha256','cache_helper_sha256','declared_execution',
                            'first_request_native_branch_clocks','prompt_contract','warmup_requests',
                            'measured_requests','torch','cuda','cudnn'):
                    assert r[key]==reports['native'][key],key
            comparisons={}
            for left,right in (('native','cudnn'),('cudnn','combined')):
                delta=np.abs(arrays[left].astype(np.float64)-arrays[right].astype(np.float64))
                comparisons[f'{left}_to_{right}']=dict(maxabs=float(delta.max()),meanabs=float(delta.mean()),
                    byte_identical=arrays[left].tobytes()==arrays[right].tobytes(),
                    speedup=reports[left]['p50_ms']/reports[right]['p50_ms'])
            result=dict(category='SCREEN',trained_student=True,task_quality_certified=False,
                seed=seed,comparisons=comparisons,p50_ms={k:r['p50_ms'] for k,r in reports.items()},
                p95_ms={k:r['p95_ms'] for k,r in reports.items()},
                files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.iterdir() if f.suffix in ('.json','.npz')})
            (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
            assert comparisons['cudnn_to_combined']['byte_identical'],'Combined optimization changed action bytes'
        status.update(status='success',active=None);save()
except BaseException as error:
    status.update(status='failed',error=repr(error),active=None);save();raise
