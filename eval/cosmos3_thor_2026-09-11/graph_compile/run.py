"""Fresh-process compile-shape ablation on the existing BF16 numerical lane."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import statistics
import sys

root = Path(sys.argv[1]).resolve()
family = sys.argv[2]
assert family in ('edge','nano')
lock = open('/tmp/thor_gpu.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
py = '/home/guanming/thorcol/cosmos-framework/.venv/bin/python'
base = '/home/guanming/ifl_eval/cosmos3_baseline_audit_20260911'
source = base + '/source'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
    PYTHONHASHSEED='0', HF_HUB_OFFLINE='1', AUDIT_COMPILE='1', AUDIT_BATCHED_CFG='0',
    AUDIT_ENGINE_ATTENTION='/home/guanming/ifl_eval/cosmos_attention_reuse_20260911/bf16_fmha.so',
    TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',
    PYTHONPATH=f'{root}:{base}/upstream-2b6c9a7:{source}:{source}/examples/cosmos3_policy',
    PATH=f'{Path(py).parent}:/home/guanming/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin')
for key in ('AUDIT_CFG_INTERVAL','AUDIT_ENGINE_SWIGLU'):
    env.pop(key,None)
for variant in ('graphs',):
    output=root/f'{family}-{variant}.json'
    assert not output.exists()
    env['AUDIT_DYNAMIC']='0' if variant=='static' else '1'
    with output.with_suffix('.log').open('x') as log:
        result=subprocess.run([py,str(root/'audit_upstream.py'),family,'baseline_a',str(output),'--iterations','10'],
            cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    report=json.loads(output.read_text()) if output.exists() else {}
    print(dict(family=family,variant=variant,exit_code=result.returncode,p50_ms=statistics.median(c['ms'] for c in report['calls'] if c['phase']=='measured') if report.get('ok') else None,error=report.get('error')),flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
