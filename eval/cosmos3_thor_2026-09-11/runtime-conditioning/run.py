import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
family = sys.argv[2]
variants = sys.argv[3:] or ['baseline_a', 'cached', 'baseline_b']
lock = open('/tmp/thor_gpu.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
py = '/home/guanming/thorcol/cosmos-framework/.venv/bin/python'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
    PYTHONHASHSEED='0', HF_HUB_OFFLINE='1', TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
    TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',
    PYTHONPATH=f'{root}:/home/guanming/thorcol/cosmos-framework:{root}/examples/cosmos3_policy',
    PATH=f'{Path(py).parent}:/home/guanming/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin')
for variant in variants:
    assert variant in ('baseline_a', 'cached', 'verify', 'baseline_b')
    env['IFL_COSMOS3_CONDITIONING_CACHE'] = '1' if variant in ('cached', 'verify') else '0'
    output = root / f'{family}-{variant}.json'
    assert not output.exists()
    with output.with_suffix('.log').open('x') as log:
        proc = subprocess.run([py, str(root/'eval/cosmos3_thor_2026-09-11/runtime-conditioning/benchmark.py'),
            family, 'current', str(output), '--iterations', '10'],
            env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    r = json.loads(output.read_text()) if output.exists() else {}
    print(dict(variant=variant, exit_code=proc.returncode,
        p50_ms=statistics.median(c['ms'] for c in r['calls'] if c['phase']=='measured') if r.get('ok') else None,
        error=r.get('error')), flush=True)
    if proc.returncode:
        raise SystemExit(proc.returncode)
