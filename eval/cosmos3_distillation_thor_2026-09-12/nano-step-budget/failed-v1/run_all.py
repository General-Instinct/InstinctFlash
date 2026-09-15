"""Prepare matched original Nano packages and serialize four Thor cost arms."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from prepare import prepare, digest

source = Path(__file__).resolve().parent
root = source.parent
status = root / 'nano-step-budget-live-v1.json'
assert not status.exists()
state = dict(status='running', pid=os.getpid(), phase='prepare_original', completed=[])

def write():
    state['updated_at_unix'] = time.time()
    tmp = status.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2)+'\n')
    tmp.replace(status)

write()
try:
    cached = Path('/home/guanming/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano-Policy-DROID/snapshots/6706d7680581c255ff61e0f3bb49d90eac55c79e')
    original = root / 'nano-original-native-v1'
    expected = json.loads((source/'inventory_h100.json').read_text())
    observed = json.loads((source/'inventory_thor.json').read_text())
    aa = {x['relative_path']:x for x in expected['files']}
    bb = {x['relative_path']:x for x in observed['files']}
    assert aa.keys() == bb.keys() and [k for k in aa if aa[k] != bb[k]] == ['README.md']
    original.mkdir(exist_ok=False)
    for item in expected['files']:
        relative = item['relative_path']
        target = original/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == 'README.md':
            shutil.copyfile(source/'source_README.md', target)
        else:
            os.link((cached/relative).resolve(), target)
    for steps in (1, 2):
        package = root/f'nano-original-sde{steps}-cost-v1'
        state.update(phase='prepare_package', steps=steps)
        write()
        manifest = prepare(original, source/'inventory_h100.json', package, steps)
        state['manifest_sha256'] = digest(package/'budget_manifest.json')
        output = root/f'results-nano-original-sde{steps}-cost-v1'
        command = [sys.executable, str(source/'run_pair.py'), str(package), str(output),
                   '--fixture', str(root/'flash/eval/native_total_2026-09-10/fixtures/va_eval_obs.npz')]
        with output.with_suffix('.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            state.update(phase='paired_benchmark', child_pid=child.pid, output=str(output))
            write()
            while child.poll() is None:
                time.sleep(5)
                write()
            if child.returncode:
                raise RuntimeError(f'SDE{steps} pair failed: {child.returncode}')
        subprocess.run([sys.executable, str(source/'compare.py'), str(output)], check=True)
        state['completed'].append(dict(steps=steps, output=str(output), files=len(manifest['files'])))
        write()
    state.update(status='success', phase='complete')
    write()
except BaseException as error:
    state.update(status='failed_or_interrupted', error=str(error))
    write()
    raise
