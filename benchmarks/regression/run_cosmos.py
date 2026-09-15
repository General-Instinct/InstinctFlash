"""Serialize fresh-process A/B/A Cosmos regression arms on a dedicated Thor."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.regression.compare import compare


def require_conditioning_evidence(report):
    backend = report.get('backend_stats', {})
    cache = backend.get('conditioning_cache') or {}
    graphs = cache.get('graph_stats', [])
    retired = cache.get('retired_graph_stats') or {}
    if (not (backend.get('conditioning_cache_status') or {}).get('admitted')
            or cache.get('disabled', True) or cache.get('rejected')
            or cache.get('verified_tensors', 0) <= 0
            or sum(g.get('replays', 0) for g in graphs) + retired.get('replays', 0) <= 0
            or retired.get('rejection_count', 0) or retired.get('rejection_samples')
            or any(g.get('rejected') for g in graphs)):
        raise ValueError('Conditioning cache was not admitted, verified and replayed without rejection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--families', nargs='+', choices=['edge', 'nano'], default=['edge', 'nano'])
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--candidate', choices=['default', 'pointwise', 'graphs', 'reuse', 'conditioning'], default='default')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    status = {'status': 'RUNNING', 'started': time.time(), 'jobs': [], 'comparisons': {}}
    def save():
        temp = args.output / 'progress.tmp'
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(args.output / 'progress.json')
    save()
    lock = open('/tmp/thor_gpu.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        status.update(status='BUSY', ended=time.time()); save(); return 2
    try:
        env = dict(os.environ, OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1',
                   PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='0',
                   TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                   TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas')
        # Never inherit experimental precision/schedule overrides into regression.
        for name in list(env):
            if name.startswith('IFL_COSMOS3_'):
                del env[name]
        env['PYTHONPATH'] = ':'.join([str(root), str(root / 'examples/cosmos3_policy')])
        env['PATH'] = str(Path(args.python).resolve().parent) + ':' + os.environ.get('PATH', '')
        for family in args.families:
            complete = True
            for arm in ['baseline_a', 'current', 'baseline_b']:
                output = args.output / f'{family}-{arm}.json'
                current_env = dict(env)
                if arm == 'current' and args.candidate != 'default':
                    current_env.update(IFL_COSMOS3_EXACT_POINTWISE='1',
                                       IFL_COSMOS3_LAYER_GRAPHS='1' if args.candidate in ('graphs', 'conditioning') else '0')
                if arm == 'current' and args.candidate == 'conditioning':
                    current_env['IFL_COSMOS3_CONDITIONING_CACHE'] = '1'
                if arm == 'current' and args.candidate in ('reuse', 'graphs', 'conditioning') and family == 'nano':
                    current_env['IFL_COSMOS3_NANO_ACTION_ONLY'] = '1'
                job = {'family': family, 'arm': arm, 'started': time.time(), 'status': 'RUNNING'}
                status['jobs'].append(job); save()
                with output.with_suffix('.log').open('x') as log:
                    try:
                        result = subprocess.run([args.python, '-m', 'benchmarks.regression.cosmos', family,
                            arm, str(output), '--iterations', str(args.iterations)], cwd=root, env=current_env,
                            stdout=log, stderr=subprocess.STDOUT, timeout=900)
                        job.update(exit_code=result.returncode, status='PASS' if result.returncode == 0 else 'FAIL')
                    except subprocess.TimeoutExpired:
                        job.update(status='FAIL', error='Timeout')
                job['ended'] = time.time(); save()
                complete &= job['status'] == 'PASS'
            if complete:
                try:
                    if args.candidate == 'conditioning':
                        require_conditioning_evidence(json.loads((args.output / f'{family}-current.json').read_text()))
                    comparison = compare(*(args.output / f'{family}-{arm}.json' for arm in ('baseline_a', 'current', 'baseline_b')))
                except (ValueError, KeyError, OSError) as error:
                    comparison = {'status': 'INVALID', 'error': str(error)}
            else:
                comparison = {'status': 'INVALID', 'error': 'An arm failed'}
            (args.output / f'{family}-comparison.json').write_text(json.dumps(comparison, indent=2))
            status['comparisons'][family] = comparison; save()
        status['status'] = 'PASS' if all(c['status'] == 'PASS' for c in status['comparisons'].values()) else 'FAIL'
    except Exception as error:
        status.update(status='FAIL', error=repr(error))
    finally:
        status['ended'] = time.time(); save()
        lock.close()
    return 0 if status['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
