"""Run three frozen original controls serially; never restart incomplete banks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
source = Path(__file__).resolve().parent
status = root / 'original-controls-live-v2.json'
assert not status.exists()
plan = json.loads((source / 'plan.json').read_text())
state = dict(status='running', pid=os.getpid(), completed=[], active=None)


def write():
    state['updated_at_unix'] = time.time()
    temporary = status.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(status)


write()
try:
    for config in plan['configurations']:
        output = root / f'control-{config}-v2'
        assert not output.exists()
        args = [sys.executable, str(source / 'run_control.py'), config, str(output)]
        with output.with_suffix('.log').open('x') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            state['active'] = dict(configuration=config, pid=child.pid, output=str(output), argv=args)
            write()
            print(json.dumps(state['active']), flush=True)
            while child.poll() is None:
                time.sleep(5)
                write()
            if child.returncode:
                raise RuntimeError(f'{config}: exit {child.returncode}')
        report = json.loads((output / 'report.json').read_text())
        assert report['status'] == 'success' and len(report['requests']) == 128
        state['completed'].append(dict(configuration=config, output=str(output)))
        state['active'] = None
        write()
    state['status'] = 'success'
    write()
except BaseException as error:
    state.update(status='failed_or_interrupted', error=str(error))
    write()
    raise
