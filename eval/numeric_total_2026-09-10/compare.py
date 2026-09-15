"""Total speed and action-delta screens with explicitly bounded protocol changes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'native_optimization_2026-09-10/compare.py'
spec = importlib.util.spec_from_file_location('numeric_pair', SOURCE)
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


def allowed_changes(family, arm):
    allowed = {}
    if family == 'pi05' and arm == 'current':
        common = {'cudnn_tf32': False, 'cudnn_benchmark': False}
        allowed['numeric_environment'] = [dict(matmul_tf32=False, **common),
                                          dict(matmul_tf32=True, **common)]
    if family == 'va' and arm in ('current_fewstep', 'current_fewstep_fp8'):
        allowed['schedule_override'] = [None, {'video': 2, 'action': 4}]
        allowed['default_schedule'] = [{'video': 25, 'action': 50}, {'video': 2, 'action': 4}]
        if arm == 'current_fewstep_fp8':
            allowed['precision'] = ['native', 'fp8']
    return allowed


def compare_profile(root, family, arm='current'):
    root = Path(root)
    paths = {a: root / f'{family}-{a}.json' for a in ('baseline_a', arm, 'baseline_b')}
    reports = {a: json.loads(p.read_text()) for a, p in paths.items()}
    failures = {a: d.get('error', 'incomplete') for a, d in reports.items() if not d.get('ok')}
    if failures:
        return {'family': family, 'arm': arm, 'status': 'FAILED', 'failures': failures}
    for a, p in paths.items():
        assert hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest() == reports[a]['actions_sha256'], p
    comparisons = {}
    invalid = {}
    for name, left, right in [('reference_repeat', 'baseline_a', 'baseline_b'),
                              ('reference_candidate', 'baseline_a', arm),
                              ('repeat_candidate', 'baseline_b', arm)]:
        result = pair.compare(paths[left], paths[right])
        allowed = {} if name == 'reference_repeat' else allowed_changes(family, arm)
        unexpected = {k: v for k, v in result['protocol_differences'].items() if allowed.get(k) != v}
        result['declared_protocol_changes'] = {k: v for k, v in result['protocol_differences'].items() if k not in unexpected}
        result['unexpected_protocol_changes'] = unexpected
        source_a, source_b = reports[left]['sources'], reports[right]['sources']
        drift = [p for p in source_a.keys() & source_b.keys() if source_a[p] != source_b[p]]
        result['source_drift'] = drift
        with pair.np.load(paths[left].with_suffix('.npz')) as x, pair.np.load(paths[right].with_suffix('.npz')) as y:
            same_keys = set(x.files) == set(y.files)
        arrays_ok = same_keys and bool(result['arrays']) and all(x['finite'] and x['max_abs_delta'] is not None for x in result['arrays'])
        if unexpected or drift or not arrays_ok:
            invalid[name] = {'protocol': unexpected, 'source_drift': drift, 'finite_same_geometry': arrays_ok}
        comparisons[name] = result
    current = reports[arm]
    speedups = [comparisons[k]['speedup'] for k in ('reference_candidate', 'repeat_candidate')]
    changed_schedule = arm in ('current_fewstep', 'current_fewstep_fp8')
    return dict(family=family, arm=arm, device=current['device'],
                status='INVALID_PROTOCOL' if invalid else 'OPERATING_POINT_SCREEN' if changed_schedule else 'NUMERIC_SCREEN',
                invalid=invalid, baseline_p50_ms=reports['baseline_a']['p50_ms'],
                current_p50_ms=current['p50_ms'], repeated_baseline_p50_ms=reports['baseline_b']['p50_ms'],
                speedup_range=sorted(speedups), reference_repeat_bitexact=comparisons['reference_repeat']['bitexact'],
                applied_passes=current['applied_passes'], execution_policy=current['execution_policy'],
                pass_results=current['pass_results'], graph_stats=current.get('graph_stats', {}),
                backend_stats=current.get('backend_stats', {}), comparisons=comparisons,
                scope='Original default schedule is the denominator. Action deltas are a recorded-input screen, not a task-quality certificate or a new numerical tolerance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('family', choices=['va', 'vla2', 'pi05'])
    parser.add_argument('--arm', default='current', choices=['current', 'current_fewstep', 'current_fewstep_fp8'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = compare_profile(args.root, args.family, args.arm)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparisons', 'pass_results', 'graph_stats', 'backend_stats', 'execution_policy')}, indent=2))
    raise SystemExit(1 if result['status'] in ('INVALID_PROTOCOL', 'FAILED') else 0)
