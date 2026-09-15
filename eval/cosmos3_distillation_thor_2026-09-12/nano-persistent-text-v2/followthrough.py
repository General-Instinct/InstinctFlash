"""Run one fresh latency pair only after the exact strict stress runner succeeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
receipt=root/'followthrough-status.json'
assert not receipt.exists()
def save(**kwargs):
    kwargs['updated_unix']=time.time()
    temporary=receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(kwargs,indent=2)+'\n');temporary.replace(receipt)
try:
    save(status='waiting',supervisor=155103)
    deadline=time.monotonic()+1800
    while True:
        state=json.loads((root/'results/live_status.json').read_text())
        if state['status']=='success':break
        if state['status']=='failed':raise RuntimeError('Stress failed; no latency launch')
        os.kill(155103,0) # Missing exact supervisor is terminal, never a restart trigger.
        if time.monotonic()>deadline:raise TimeoutError('Stress observation budget exhausted')
        time.sleep(10)
    comparison=json.loads((root/'results/comparison.json').read_text())
    assert all(row['byte_identical'] for row in comparison['comparisons'].values())
    reuse=json.loads((root/'results/reuse.json').read_text())
    history=reuse['persistent_cache']['graph_lifetime_stats']
    assert reuse['ok'] and history and not any(g['rejected'] for g in history)
    command=[sys.executable,str(root/'run_all.py'),str(root.parent/'nano-original-sde1-cost-v1'),
             str(root/'latency-results'),'--fixture',str(root.parent/'flash/eval/native_total_2026-09-10/fixtures/va_eval_obs.npz')]
    with (root/'latency-runner.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        save(status='running',pid=child.pid)
        code=child.wait()
        if code:raise RuntimeError(f'Latency runner exit {code}; no retry')
    save(status='success',exit_code=0)
except BaseException as error:
    save(status='failed',error=repr(error));raise
