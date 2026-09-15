"""Sequential, fresh-process campaign; failures retained, no retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
arms = ['stock10', 'capture10', 'engine16', 'engine8', 'stock50', 'capture50']
env = dict(os.environ)
env['PYTHONPATH'] = str(root / 'source') + ':' + str(Path.home() / 'ifl_eval/native_qualification_20260906/final-execution')
env['PYTHONUNBUFFERED'] = '1'
env['OMP_NUM_THREADS'] = '4'
env['HF_HUB_OFFLINE'] = '1'
env['TRANSFORMERS_OFFLINE'] = '1'
checkpoint = Path.home() / '.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/8e174154ef5f6c60a8da12ae99c303d8963138c1'
status = []
for repeat in range(3):
    order = arms[2*repeat:] + arms[:2*repeat]
    for arm in order:
        label = f'{arm}.{repeat}'
        output = root / 'results' / (label + '.json')
        log = output.with_suffix('.log')
        if output.exists() or log.exists():
            raise RuntimeError(f'refusing to overwrite {label}')
        command = [sys.executable, str(root / 'source/probe_pi05.py'), '--arm', arm,
                   '--repeat', str(repeat), '--checkpoint', str(checkpoint),
                   '--assets', str(Path.home() / 'ifl/t3_assets/thor_assets.json'),
                   '--frames', str(Path.home() / 'ifl/t3_assets/calib_obs.npz'),
                   '--output', str(output)]
        print('START', label, flush=True)
        start = time.time()
        with log.open('x') as stream:
            try:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        status.append(dict(label=label, command=command, exit_code=code, elapsed_s=time.time()-start))
        (root / 'status.json').write_text(json.dumps(status, indent=2))
        print('DONE', label, code, flush=True)
(root / 'complete.json').write_text(json.dumps(dict(attempts=len(status), failures=sum(x['exit_code'] != 0 for x in status))))
