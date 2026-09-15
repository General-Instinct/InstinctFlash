"""Validate the fixed-KV experiment against guarded conditioning Runtime."""
import hashlib
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('family', choices=('edge', 'nano'))
parser.add_argument('--minimum-speedup', type=float, default=1.0)
parser.add_argument('--maximum-reference-drift', type=float, default=0.05)
args = parser.parse_args()
assert np.isfinite(args.minimum_speedup) and args.minimum_speedup > 0
assert np.isfinite(args.maximum_reference_drift) and args.maximum_reference_drift >= 0
root, family = args.root, args.family
output = root / f'{family}-comparison.json'
# An invalid new run must never leave an older passing receipt behind.
output.write_text(json.dumps(dict(passed=False, status='validation_incomplete')) + '\n')
variants = ('baseline_a', 'cached', 'baseline_b')
reports = {v: json.loads((root / f'{family}-{v}.json').read_text()) for v in variants}
arrays = {}
for variant, report in reports.items():
    assert report['ok'] and not report['competing_gpu_processes'], variant
    for field in ('family', 'arm', 'model_id', 'revision', 'precision', 'torch',
                  'device', 'input_archive_sha256', 'reference_constructor_sha256',
                  'benchmark_sha256', 'cases', 'effective_schedule',
                  'numeric_environment', 'sources', 'fixed_kv_sources'):
        assert report[field] == reports['baseline_a'][field], (variant, field)
    path = root / f'{family}-{variant}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['actions_sha256']
    with np.load(path) as data:
        arrays[variant] = data['actions'].copy()
    assert arrays[variant].shape == (len(report['cases']), 32, 8)
    assert np.isfinite(arrays[variant]).all()
    fixed = report['fixed_kv']
    if variant.startswith('baseline'):
        assert not fixed
    else:
        assert len(fixed) == 1 and fixed[0]['prefills'] > 0 and fixed[0]['updates'] > 0
    stats = report['backend_stats']['conditioning_cache']
    assert stats['requests'] == len(report['cases'])
    assert stats['prefills'] == 2 * stats['requests']
    assert stats['decodes'] == 6 * stats['requests']
    assert stats['modules'] > 0 and stats['module_hits'] > 0
    assert not stats['disabled'] and not stats['rejected'] and stats['admitted_branches'] > 0
    assert stats['verified_tensors'] > 0
    assert report['backend_stats']['conditioning_cache_status']['admitted']
    assert stats['verify'] == (variant == 'verify')
    if variant == 'verify':
        assert stats['verified_tensors'] >= stats['modules'] * stats['decodes']
    else:
        assert stats['graph_stats'] and sum(g['replays'] for g in stats['graph_stats']) > 0
        assert not any(g['rejected'] for g in stats['graph_stats'])

reference = arrays['baseline_a']
rows = []
for variant, report in reports.items():
    actions = arrays[variant]
    assert actions.dtype == reference.dtype
    delta = np.abs(actions.astype(float) - reference.astype(float))
    rows.append(dict(variant=variant,
        p50_ms=statistics.median(c['ms'] for c in report['calls'] if c['phase'] == 'measured'),
        p95_ms=float(np.percentile([c['ms'] for c in report['calls'] if c['phase'] == 'measured'], 95)),
        byte_equal_to_baseline_a=np.array_equal(actions.view(np.uint8), reference.view(np.uint8)),
        max_abs_delta=float(delta.max()), mean_abs_delta=float(delta.mean())))
result = dict(family=family, rows=rows,
    speedup_vs_a=rows[0]['p50_ms'] / rows[1]['p50_ms'],
    speedup_vs_b=rows[2]['p50_ms'] / rows[1]['p50_ms'],
    qualification='Fixed interleaved KV storage vs guarded conditioning Runtime; no task-quality certificate')
result['reference_drift'] = max(rows[0]['p50_ms'], rows[2]['p50_ms']) / min(rows[0]['p50_ms'], rows[2]['p50_ms']) - 1
result['action_bytes_passed'] = all(row['byte_equal_to_baseline_a'] for row in rows)
result['performance_passed'] = (
    result['reference_drift'] <= args.maximum_reference_drift and
    min(result['speedup_vs_a'], result['speedup_vs_b']) >= args.minimum_speedup and
    rows[1]['p95_ms'] <= max(rows[0]['p95_ms'], rows[2]['p95_ms']) * 1.10)
result['thresholds'] = dict(minimum_speedup=args.minimum_speedup,
    maximum_reference_drift=args.maximum_reference_drift, maximum_p95_regression=0.10)
result['passed'] = result['action_bytes_passed'] and result['performance_passed']
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if not result['passed']:
    raise SystemExit('Fixed-KV action/performance gate failed; inspect the saved comparison')
