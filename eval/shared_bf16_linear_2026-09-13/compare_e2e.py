"""Validate A/candidate/B coverage and paired full actions; no task certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('results', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
names = ['reference-a', 'fused-up', 'reference-b']
paths = [a.results / f'edge-{name}.json' for name in names]
reports = [json.loads(path.read_text()) for path in paths]
assert all(r['ok'] and not r['competing_gpu_processes'] for r in reports)
for field in ('cases', 'revision', 'effective_schedule', 'numeric_environment'):
    assert all(r[field] == reports[0][field] for r in reports), field
arrays = [np.load(path.with_suffix('.npz'))['actions'] for path in paths]
assert all(x.shape == (16, 32, 8) and np.isfinite(x).all() for x in arrays)
for r, path in zip(reports, paths):
    assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == r['actions_sha256']
stats = reports[1]['backend_stats']['generation_regions']
assert stats['split_prefill'] and not stats['rejected']
assert stats['prefill_checks'] >= 56
assert len(stats['fused_linear']) == 28
assert all(x['eligible'] for x in stats['fused_linear'])
assert all(region['state'] == 'ready' for region in stats['regions'])
coverage = {}
for name, r in zip(names, reports):
    expected = 224
    expected_prefill = 56
    rows = [c for c in r['calls'] if c['phase'] == 'measured']
    assert len(rows) == 10
    assert all(c['region_counts']['compiled_calls'] == expected for c in rows), name
    assert all(c['region_counts']['compiled_prefill_calls'] == expected_prefill for c in rows), name
    assert all(c['region_counts']['prefill_checks'] == c['region_counts']['eager_checks'] == 0 for c in rows)
    coverage[name] = dict(measured_requests=10, compiled_layers_per_request=expected,
                         compiled_prefill_layers_per_request=expected_prefill)
latency = [float(np.median([c['ms'] for c in r['calls'] if c['phase'] == 'measured'])) for r in reports]
delta = np.abs(arrays[0].astype(float) - arrays[1].astype(float))
repeat_delta = np.abs(arrays[0].astype(float) - arrays[2].astype(float))
result = dict(status='complete_numerical_latency_screen', quality_certified=False, default_enabled=False,
    latency_p50_ms=dict(zip(names, latency)),
    latency_reduction_percent_vs_a=100 * (1 - latency[1] / latency[0]),
    latency_reduction_percent_vs_b=100 * (1 - latency[1] / latency[2]),
    reference_ab_byte_equal=arrays[0].dtype == arrays[2].dtype and arrays[0].tobytes() == arrays[2].tobytes(),
    reference_ab_max_abs=float(repeat_delta.max()),
    candidate_byte_equal_to_a=arrays[0].dtype == arrays[1].dtype and arrays[0].tobytes() == arrays[1].tobytes(),
    candidate_action_mae=float(delta.mean()), candidate_action_max_abs=float(delta.max()),
    candidate_joint_mae=float(delta[..., :7].mean()), candidate_gripper_mae=float(delta[..., 7:].mean()),
    coverage=coverage, prefill_extraction_checks=stats['prefill_checks'],
    total_compiled_prefill_layers=stats['compiled_prefill_calls'],
    schedule=reports[0]['effective_schedule'],
    receipts=[dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths])
result['regression_thresholds'] = dict(exact_actions_required=True,
    minimum_paired_latency_reduction_percent=3.0)
result['regression_passed'] = (result['reference_ab_byte_equal']
    and result['candidate_byte_equal_to_a']
    and min(result['latency_reduction_percent_vs_a'], result['latency_reduction_percent_vs_b']) >= 3.0)
a.output.write_text(json.dumps(result, indent=2) + '\n')
assert result['regression_passed'], 'Shared BF16 numerical/performance regression failed; see receipt' 
print(json.dumps(result, indent=2))
