"""Four finite serialized Thor jobs; stop on failure, no automatic retry."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

root = Path(__file__).resolve().parent
source = root / 'source'
sys.path.insert(0, str(source / 'benchmarks' / 'regression'))
from runtime_bundle import verify_manifest

status_path = root / 'live_status.json'
assert not status_path.exists(), 'This run already has a supervisor receipt'
status = dict(status='queued', pid=os.getpid(), jobs=[], quality_certified=False)

def terminate(signum, frame):
    raise SystemExit(f'Supervisor received signal {signum}')

signal.signal(signal.SIGTERM, terminate)

def save():
    path = status_path.with_suffix('.tmp')
    path.write_text(json.dumps(status, indent=2) + '\n')
    path.replace(status_path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    verify_manifest(source, json.loads((root / 'source_manifest.json').read_text()))
    for filename, expected in json.loads((root / 'driver_manifest.json').read_text()).items():
        assert sha(root / filename) == expected, filename
    for filename, expected in json.loads((root / 'external_manifest.json').read_text()).items():
        assert sha(Path(filename)) == expected, filename

def gpu_pids():
    return subprocess.check_output(['/usr/sbin/nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).strip()

def stop(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()

python = '/home/guanming/thorcol/dz_env/bin/python'
fixture = '/home/guanming/ifl_eval/four_frameworks_20260913/source/eval/native_total_2026-09-10/fixtures/va_eval_obs.npz'
env = {k: v for k, v in os.environ.items() if not k.startswith('IFL_')}
env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
    PYTHONUNBUFFERED='1', PYTHONHASHSEED='0', MASTER_PORT='29814',
    NO_ALBUMENTATIONS_UPDATE='1', DYNAMIC_CACHE_SCHEDULE='false', NUM_DIT_STEPS='8',
    ENABLE_TENSORRT='false', TORCHDYNAMO_DISABLE='0',
    TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
    TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',
    DREAMZERO_ROOT='/home/guanming/thorcol/dreamzero',
    PATH=str(Path(python).parent)+':/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/usr/sbin:/bin',
    PYTHONPATH=':'.join([str(source), str(source/'serving')]
        + [str(p) for p in sorted((source/'examples').iterdir()) if p.is_dir()]))

save()
try:
    with open('/tmp/thor_gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
        assert not gpu_pids(), 'Thor is occupied'
        status.update(status='running', started=time.time(),
            manifests={name: sha(root/name) for name in
                ('source_manifest.json', 'driver_manifest.json', 'external_manifest.json')})
        save()
        for precision, schedule in [('native', 'dynamic'), ('native', 'checkpoint'),
                                    ('fp8', 'dynamic'), ('fp8', 'checkpoint')]:
            verify()
            assert not gpu_pids(), 'Unexpected competing Thor process'
            output = root / f'{precision}-{schedule}.json'
            assert not output.exists()
            command = [python, str(root/'benchmark.py'), precision, schedule, str(output),
                       '--input-archive', fixture]
            job = dict(precision=precision, schedule=schedule, status='running',
                       command=command, started=time.time())
            status['jobs'].append(job)
            save()
            with output.with_suffix('.log').open('x') as log:
                child = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                job['pid'] = child.pid
                save()
                try:
                    child.wait(timeout=2700)
                except BaseException:
                    stop(child)
                    raise
                finally:
                    job['exit_code'] = child.returncode
                    job['ended'] = time.time()
            receipt = json.loads(output.read_text()) if output.exists() else {}
            job['status'] = 'passed' if child.returncode == 0 and receipt.get('status') == 'passed' else 'failed'
            job['error'] = receipt.get('error')
            save()
            if job['status'] != 'passed':
                raise RuntimeError(f'{precision}/{schedule} failed; queue stopped')
            if schedule == 'dynamic':
                assert receipt.get('native_dynamic_parity', {}).get('passed') is True
        verify()
        status.update(status='passed', ended=time.time())
        save()
        artifacts = {p.name: sha(p) for p in sorted(root.iterdir())
                     if p.is_file() and p.name not in ('live_status.json', 'completion.json', 'supervisor.log')}
        (root/'completion.json').write_text(json.dumps(dict(status='passed', artifacts=artifacts,
            fresh_jobs=4, expected_requests=72, quality_certified=False,
            claim='Bounded latency SCREEN and parity against the same dynamic native policy'), indent=2)+'\n')
except BaseException:
    if status['jobs'] and status['jobs'][-1]['status'] == 'running':
        status['jobs'][-1].update(status='failed', ended=time.time(),
                                 error=traceback.format_exc())
    status.update(status='failed', error=traceback.format_exc(), ended=time.time())
    save()
    raise
