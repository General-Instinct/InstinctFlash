"""Check Nano extraction gates and paired native full-action/latency evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
a = p.parse_args()
paths = [a.root / f'nano-{n}.json' for n in ('reference-a', 'capacity-cache', 'split-prefill', 'reference-b')]
r = [json.loads(p.read_text()) for p in paths]
assert all(x['ok'] and not x['competing_gpu_processes'] for x in r)
for field in ('cases', 'revision', 'effective_schedule', 'numeric_environment', 'benchmark_sha256'):
    assert all(x[field] == r[0][field] for x in r)
arrays = [np.load(p.with_suffix('.npz'))['actions'] for p in paths]
for p, report, array in zip(paths, r, arrays):
    assert hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest() == report['actions_sha256']
    assert array.shape == (16, 32, 8) and np.isfinite(array).all()
    assert report['backend_stats']['timestep_cache']['hits'] > 0
measured = [[c for c in x['calls'] if c['phase'] == 'measured'] for x in r]
assert all(len(rows) == 10 for rows in measured)
for i, rows in enumerate(measured):
    expected = dict(compiled_calls=288 if i == 2 else 216,
                    compiled_prefill_calls=72 if i == 2 else 0,
                    prefill_checks=0, eager_checks=0)
    assert all(c['region_counts'] == expected for c in rows), (i, expected)
stats = r[2]['backend_stats']['generation_regions']
assert stats['split_prefill'] and not stats['rejected'] and stats['prefill_checks'] >= 72
assert all(x['state'] == 'ready' for x in stats['regions'])
assert arrays[0].tobytes() == arrays[3].tobytes(), 'Unstable reference actions'
assert arrays[0].dtype == arrays[1].dtype and arrays[0].tobytes() == arrays[1].tobytes(), 'Capacity-only action change'
for i in (1,2):
    cache = r[i]['backend_stats']['timestep_cache']
    assert cache['max_bytes'] == 256*1024*1024 and cache['evictions'] == 0
delta = np.abs(arrays[2].astype(float)-arrays[0].astype(float))
medians = [statistics.median(c['ms'] for c in rows) for rows in measured]
result = dict(status='complete_numeric_latency_screen', quality_certified=False,
    latency_p50_ms=dict(zip(('reference-a','capacity-cache','split-prefill','reference-b'), medians)),
    capacity_reduction_percent_vs_references=[100*(1-medians[1]/medians[i]) for i in (0,3)],
    split_reduction_percent_vs_capacity=100*(1-medians[2]/medians[1]),
    combined_reduction_percent_vs_references=[100*(1-medians[2]/medians[i]) for i in (0,3)],
    caches=[x['backend_stats']['timestep_cache'] for x in r],
    reference_actions_byte_equal=True,
    candidate_actions_byte_equal=arrays[0].dtype == arrays[2].dtype and arrays[0].tobytes() == arrays[2].tobytes(),
    action_mae=float(delta.mean()), action_max_abs=float(delta.max()),
    action_joint_mae=float(delta[...,:7].mean()), action_gripper_mae=float(delta[...,7:].mean()),
    native_prefill_checks=stats['prefill_checks'], native_eager_checks=stats['eager_checks'],
    measured_gen_calls_before=216, measured_gen_calls_after=288,
    schedule=r[0]['effective_schedule'],
    receipts=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
(a.root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
