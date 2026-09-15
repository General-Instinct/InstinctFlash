"""One owned Thor capture; a failed run is retained and never retried here."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    task = Path(__file__).resolve().parent
    base = task.parent
    environment = Path('/dev/shm/ifl_va_native_2v4a_env_20260915_v1')
    site = environment / 'lib/python3.12/site-packages'
    output = task / 'run_native_v1'
    output.mkdir(exist_ok=False)
    status = {'status': 'running', 'started_unix': time.time(), 'pid': os.getpid()}
    lock = open('/tmp/thor_gpu.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader'], text=True).strip()
        status['initial_gpu_processes'] = apps
        if apps:
            raise RuntimeError('Thor GPU has competing processes')
        restored = task / 'environment_restore_v1/completion.json'
        status['environment_restore_sha256'] = sha(restored)
        status['benchmark_overlay'] = []
        for filename in ('native_reference.py', 'user_e2e.py'):
            source = task / filename
            target = site / 'benchmarks/regression' / filename
            assert target.stat().st_nlink == 1, 'Benchmark file must be an independent copy'
            row = {'source': str(source), 'installed': str(target),
                   'before_sha256': sha(target), 'source_sha256': sha(source)}
            target.write_bytes(source.read_bytes())
            assert sha(target) == row['source_sha256']
            status['benchmark_overlay'].append(row)
        fixture = base / 'va_both_fp8_prepared_v2/inputs/recorded_inputs_v1.npz'
        assert sha(fixture) == '37843e22fa6dd9a2abf2bae390ccb8e5c4319446e3d3fab5d0d477860065f411'
        env = dict(os.environ)
        for key in list(env):
            if key.startswith('IFL_') or key in ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES'):
                env.pop(key)
        temporary = Path('/dev/shm/ifl_va_native_2v4a_run_20260915_v1')
        temporary.mkdir(exist_ok=False)
        additions = {'LINGBOT_ROOT': str(base / 'vendors_v1/va'),
                     'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                     'HF_HUB_DISABLE_TELEMETRY': '1', 'TMPDIR': str(temporary),
                     'PYTHONDONTWRITEBYTECODE': '1'}
        env.update(additions)
        command = [str(environment / 'bin/python'), '-I', '-B', '-m',
                   'benchmarks.regression.user_e2e', '--matrix', str(task / 'matrix_v1.json'),
                   '--cell', 'va-eager_native-2v4a', '--output-root', str(output),
                   '--fixture', str(fixture)]
        status.update(command=command, environment=additions,
                      matrix_sha256=sha(task / 'matrix_v1.json'), fixture_sha256=sha(fixture))
        (task / 'live_status_v1.json').write_text(json.dumps(status, indent=2) + '\n')
        with (task / 'native_capture_v1.log').open('x') as log:
            done = subprocess.run(command, cwd=output, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=1800)
        status['returncode'] = done.returncode
        receipt = output / 'cells/va-eager_native-2v4a/receipt.json'
        status['receipt_sha256'] = sha(receipt)
        captured = json.loads(receipt.read_text())
        if done.returncode or not captured['ok']:
            raise RuntimeError('Native capture failed; evidence retained')
        status.update(status='passed', total_calls=len(captured['calls']))
    except Exception as error:
        status.update(status='failed', error=repr(error))
    finally:
        status['completed_unix'] = time.time()
        with (task / 'capture_completion_v1.json').open('x') as stream:
            json.dump(status, stream, indent=2)
            stream.write('\n')
        lock.close()
    return 0 if status['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
