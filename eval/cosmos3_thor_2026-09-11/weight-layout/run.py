import fcntl,json,os,statistics,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
lock=open('/tmp/thor_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
py='/home/guanming/thorcol/cosmos-framework/.venv/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',HF_HUB_OFFLINE='1',
    TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',
    PYTHONPATH=f'{root}:/home/guanming/thorcol/cosmos-framework:{root}/examples/cosmos3_policy',
    PATH=f'{Path(py).parent}:/home/guanming/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin')
for variant in ('baseline_a','packed','baseline_b'):
    env['IFL_WEIGHT_LAYOUT']='1' if variant=='packed' else '0'
    output=root/f'nano-{variant}.json';assert not output.exists()
    with output.with_suffix('.log').open('x') as log:
        proc=subprocess.run([py,str(root/'eval/cosmos3_thor_2026-09-11/weight-layout/benchmark.py'),
            'nano','current',str(output),'--iterations','10'],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    r=json.loads(output.read_text()) if output.exists() else {}
    print(dict(variant=variant,exit_code=proc.returncode,
        p50_ms=statistics.median(c['ms'] for c in r['calls'] if c['phase']=='measured') if r.get('ok') else None,error=r.get('error')),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)
