"""Wait on the existing capture queue and export each completed control once."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
source = Path(__file__).resolve().parent
status = root / 'controls-postprocess-live-v2.json'
assert not status.exists()
validator = root / 'quality-orchestration-v1/validate_capture.py'
assert validator.is_file()
state = dict(status='running', pid=os.getpid(), completed=[], active=None)


def write():
    state['updated_at_unix'] = time.time()
    temporary = status.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(status)


write()
try:
    for config in ('original_unipc4', 'original_sde1_cfg1', 'original_sde1_cfg4'):
        capture, output = root / f'control-{config}-v2', root / f'control-compact-{config}-v2'
        assert not output.exists()
        state['active'] = dict(configuration=config, phase='waiting_capture')
        write()
        while not (capture / 'report.json').exists():
            running = json.loads((root / 'original-controls-live-v2.json').read_text())
            assert running['status'] == 'running', running
            process = subprocess.run(['ps', '-p', str(running['pid']), '-o', 'stat='], capture_output=True, text=True)
            assert process.returncode == 0 and process.stdout.strip() and not process.stdout.strip().startswith('Z')
            time.sleep(15)
            write()
        assert json.loads((capture / 'report.json').read_text())['status'] == 'success'
        args = [sys.executable, str(source / 'export_control.py'), str(capture), str(output),
                '--source', str(root / 'realtime-controls-v2'),
                '--student-source', str(root / 'realtime-quality-v1'),
                '--student-validator', str(validator), '--study', str(root)]
        with output.with_suffix('.log').open('x') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            state['active'] = dict(configuration=config, phase='validate_and_pair', pid=child.pid, argv=args)
            write()
            while child.poll() is None:
                time.sleep(5)
                write()
            if child.returncode:
                raise RuntimeError(f'{config}: postprocess exit {child.returncode}')
        state['completed'].append(dict(configuration=config, output=str(output)))
        state['active'] = None
        write()
        print(json.dumps(state['completed'][-1]), flush=True)
    state['status'] = 'success'
    write()
except BaseException as error:
    state.update(status='failed_or_interrupted', error=str(error))
    write()
    raise
