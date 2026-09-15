"""Matched original Nano SDE1 cuDNN BF16 SwiGLU ablation under Thor lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--fixture', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
with open('/tmp/thor_gpu.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for arm in ('baseline', 'swiglu'):
        output = a.output.resolve()/f'{arm}.json'
        command = [sys.executable, str(Path(__file__).with_name('benchmark_original.py')),
            str(a.checkpoint.resolve()), str(output), '--allow-unqualified',
            '--attention', 'cudnn', '--fixture', str(a.fixture.resolve())]
        if arm == 'swiglu':
            command.append('--swiglu')
        with output.with_suffix('.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=2400, env=os.environ.copy())
    reports = {arm:json.loads((a.output/f'{arm}.json').read_text()) for arm in ('baseline','swiglu')}
    arrays = {arm:np.load(a.output/f'{arm}.npz')['actions'] for arm in reports}
    for arm in reports:
        assert reports[arm]['ok'] and arrays[arm].shape == (16,32,8) and np.isfinite(arrays[arm]).all()
    for key in ('checkpoint_manifest_sha256','fixture_sha256','declared_execution','first_request_native_branch_clocks','prompt_contract','torch','cuda','cudnn'):
        assert reports['baseline'][key] == reports['swiglu'][key], key
    delta = np.abs(arrays['baseline'].astype(np.float64)-arrays['swiglu'].astype(np.float64))
    result = dict(category='SCREEN', trained_student=False, task_quality_certified=False,
        action_byte_identical=arrays['baseline'].tobytes()==arrays['swiglu'].tobytes(),
        maxabs=float(delta.max()), meanabs=float(delta.mean()),
        p50_ms={arm:reports[arm]['p50_ms'] for arm in reports},
        p95_ms={arm:reports[arm]['p95_ms'] for arm in reports},
        speedup=reports['baseline']['p50_ms']/reports['swiglu']['p50_ms'],
        artifacts=[identity(a.output/f'{arm}.{ext}') for arm in reports for ext in ('json','npz')])
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
