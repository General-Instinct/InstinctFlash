"""Compare all three native arms without converting variability into a tolerance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1] / 'native_optimization_2026-09-10/compare.py'
spec = importlib.util.spec_from_file_location('paired_native_compare', BASE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def compare_family(root, family):
    root = Path(root)
    paths = {arm: root / f'{family}-{arm}.json'
             for arm in ('baseline_a', 'current', 'baseline_b')}
    reports = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    failures = {arm: r.get('error', 'incomplete') for arm, r in reports.items()
                if not r.get('ok')}
    if failures:
        return {'family': family, 'status': 'FAILED', 'failures': failures}
    for arm, path in paths.items():
        actual = hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()
        if actual != reports[arm]['actions_sha256']:
            raise ValueError(f'Action artifact hash mismatch: {path}')
    pairs = {name: module.compare(paths[left], paths[right]) for name, left, right in (
        ('baseline_current', 'baseline_a', 'current'),
        ('baseline_repeat', 'baseline_a', 'baseline_b'),
        ('repeat_current', 'baseline_b', 'current'))}
    for name, left, right in (
        ('baseline_current', 'baseline_a', 'current'),
        ('baseline_repeat', 'baseline_a', 'baseline_b'),
        ('repeat_current', 'baseline_b', 'current')):
        with np.load(paths[left].with_suffix('.npz')) as x, np.load(paths[right].with_suffix('.npz')) as y:
            a, b = x['actions'], y['actions']
            pairs[name]['matching_calls'] = sum(
                u.dtype == v.dtype and u.shape == v.shape and
                np.isfinite(u).all() and np.isfinite(v).all() and u.tobytes() == v.tobytes()
                for u, v in zip(a, b))
            pairs[name]['matching_calls'] = int(pairs[name]['matching_calls'])
            pairs[name]['total_calls'] = len(a)
    source_drift = {}
    for arm in ('current', 'baseline_b'):
        a, b = reports['baseline_a']['sources'], reports[arm]['sources']
        source_drift[arm] = [p for p in a.keys() & b.keys() if a[p] != b[p]]
    invalid = any(p['protocol_differences'] for p in pairs.values()) or any(source_drift.values())
    exact = all(p['bitexact'] for p in pairs.values()) and not invalid
    status = ('INVALID_PROTOCOL' if invalid else 'BITEXACT' if exact else
              'BASELINE_VARIABLE' if not pairs['baseline_repeat']['bitexact'] else 'ACTION_MISMATCH')
    current = reports['current']
    return dict(family=family, device=current['device'], status=status,
                baseline_p50_ms=reports['baseline_a']['p50_ms'],
                current_p50_ms=current['p50_ms'],
                repeated_baseline_p50_ms=reports['baseline_b']['p50_ms'],
                speedup_range=sorted([p['speedup'] for name, p in pairs.items()
                                      if name != 'baseline_repeat']),
                applied_passes=current['applied_passes'],
                baseline_applied_passes=reports['baseline_a']['applied_passes'],
                unchanged_transform_plan=(current['applied_passes'] == reports['baseline_a']['applied_passes']),
                execution_policy=current['execution_policy'], source_drift=source_drift,
                backend_stats=current.get('backend_stats', {}),
                reference_scope=current.get('reference_scope'),
                comparisons=pairs,
                scope='Recorded camera inputs with synthetic states; finite action bytes, no tolerance or task-quality claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('family')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare_family(args.root, args.family)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparisons', 'execution_policy')}, indent=2))
    raise SystemExit(0 if result['status'] == 'BITEXACT' else 1)
