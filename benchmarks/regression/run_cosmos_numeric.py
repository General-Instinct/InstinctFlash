"""Paired native BF16 numerical screen on Thor; never a task-quality certificate."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def compare(directory, family, *, min_speedup=1.1, max_abs_action_delta=None):
    reports, arrays = {}, {}
    for arm in ('bitexact', 'numeric'):
        path = directory / f'{family}-{arm}.json'
        report = json.loads(path.read_text())
        if not report['ok'] or report['competing_gpu_processes']:
            raise ValueError(f'{arm}: failed or contended run')
        archive = path.with_suffix('.npz')
        if hashlib.sha256(archive.read_bytes()).hexdigest() != report['actions_sha256']:
            raise ValueError('Action archive digest mismatch')
        with np.load(archive) as data:
            arrays[arm] = data['actions'].copy()
        if arrays[arm].shape != (len(report['cases']), 32, 8) or not np.isfinite(arrays[arm]).all():
            raise ValueError('Incomplete or nonfinite actions')
        cases, calls = report['cases'], report['calls']
        if (len(cases) < 16 or len(calls) != len(cases)
                or [c['i'] for c in cases] != list(range(len(cases)))
                or any(call['i'] != case['i'] or call['phase'] != case['phase']
                       or not np.isfinite(call['ms']) or call['ms'] <= 0
                       for case, call in zip(cases, calls))):
            raise ValueError('Incomplete or invalid request timings')
        policy = report['execution_policy']
        if policy['category'] != arm.upper() or policy['precision'] != 'native' or policy['changed_schedule']:
            raise ValueError('Unexpected execution policy')
        reports[arm] = report
    ref, cand = reports['bitexact'], reports['numeric']
    for key in ('family', 'model_id', 'revision', 'precision', 'torch', 'device',
                'input_archive_sha256', 'reference_constructor_sha256', 'benchmark_sha256',
                'cases', 'effective_schedule', 'numeric_environment', 'runtime_versions'):
        if ref[key] != cand[key]:
            raise ValueError(f'Unmatched protocol: {key}')
    # The numeric module is imported only by the candidate; every common source
    # must still be identical, and no other candidate-only module is allowed.
    a, b = ref['sources'], cand['sources']
    if any(b.get(k) != v for k, v in a.items()) or any(
            not k.endswith('/cosmos3_iwm/numeric_attention.py') for k in b.keys() - a.keys()):
        raise ValueError('Unmatched source')
    attention = cand['backend_stats'].get('numeric_attention')
    if not attention or attention['eligible_python_calls'] <= 0 or attention['backend'] != 'cudnn':
        raise ValueError('Numeric backend did not execute')
    for report in reports.values():
        stats = report['backend_stats']
        cache = stats['conditioning_cache']
        if (not stats['conditioning_cache_status']['admitted'] or cache['disabled']
                or cache['rejected'] or cache['verified_tensors'] <= 0
                or not sum(g['replays'] for g in cache['graph_stats'])
                or any(g['rejected'] for g in cache['graph_stats'])):
            raise ValueError('Conditioning cache was not qualified and replayed')
    if arrays['bitexact'].dtype != arrays['numeric'].dtype:
        raise ValueError('Action dtype mismatch')
    rows = {}
    for arm, report in reports.items():
        times = [c['ms'] for c in report['calls'] if c['phase'] == 'measured']
        if len(times) < 10:
            raise ValueError('Insufficient measured requests')
        rows[arm] = dict(p50_ms=statistics.median(times), p95_ms=float(np.percentile(times, 95)))
    delta = np.abs(arrays['numeric'].astype(np.float64) - arrays['bitexact'].astype(np.float64))
    speedup = rows['bitexact']['p50_ms'] / rows['numeric']['p50_ms']
    performance_ok = speedup >= min_speedup
    delta_ok = max_abs_action_delta is None or float(delta.max()) <= max_abs_action_delta
    return dict(validation_complete=True, category='SCREEN', family=family, rows=rows,
                status='PASS' if performance_ok and delta_ok else 'REGRESSION',
                min_speedup=min_speedup, max_abs_action_delta_limit=max_abs_action_delta,
                speedup=rows['bitexact']['p50_ms'] / rows['numeric']['p50_ms'],
                max_abs_action_delta=float(delta.max()), mean_abs_action_delta=float(delta.mean()),
                per_dimension_max_abs=delta.max(axis=(0, 1)).tolist(),
                task_quality_certified=False,
                scope='One matched pair; no drift bracket or closed-loop quality certificate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--family', choices=['edge', 'nano'], action='append')
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--min-speedup', type=float, default=1.1)
    parser.add_argument('--max-abs-action-delta', type=float)

    args = parser.parse_args()
    if not np.isfinite(args.min_speedup) or args.min_speedup <= 0:
        parser.error('--min-speedup must be finite and positive')
    if args.max_abs_action_delta is not None and (
            not np.isfinite(args.max_abs_action_delta) or args.max_abs_action_delta < 0):
        parser.error('--max-abs-action-delta must be finite and nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('IFL_COSMOS3_')}
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
               PYTHONHASHSEED='0', IFL_COSMOS3_CONDITIONING_CACHE='1',
               PYTHONPATH=f'{ROOT}:{ROOT}/examples/cosmos3_policy:' + env.get('PYTHONPATH', ''))
    for key in ('TRITON_PTXAS_PATH', 'TRITON_PTXAS_BLACKWELL_PATH'):
        env.setdefault(key, '/usr/local/cuda/bin/ptxas')
    with open('/tmp/thor_gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for family in args.family or ['edge', 'nano']:
            result_path = args.output / f'{family}-comparison.json'
            result_path.write_text(json.dumps(dict(validation_complete=False,
                status='INVALID', task_quality_certified=False)) + '\n')
            if not args.compare_only:
                for arm in ('bitexact', 'numeric'):
                    output = args.output.resolve() / f'{family}-{arm}.json'
                    with output.with_suffix('.log').open('x') as log:
                        subprocess.run([sys.executable, '-m', 'benchmarks.regression.cosmos_numeric',
                                        family, 'current', str(output), '--iterations', '10',
                                        '--tier-ceiling', arm], env=env, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=True, timeout=2400)
            result = compare(args.output, family, min_speedup=args.min_speedup,
                             max_abs_action_delta=args.max_abs_action_delta)
            result_path.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            if result['status'] != 'PASS':
                raise SystemExit(1)


if __name__ == '__main__':
    main()
