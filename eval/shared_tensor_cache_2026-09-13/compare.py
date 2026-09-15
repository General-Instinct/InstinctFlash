"""Full-action/cache/coverage paired regression. No task-success certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
a = p.parse_args()
result = {}
for family in ('edge', 'nano'):
    names = ['reference-a', 'exact-cache', 'reference-b']
    paths = [a.root / f'{family}-{name}.json' for name in names]
    reports = [json.loads(path.read_text()) for path in paths]
    assert all(r['ok'] and not r['competing_gpu_processes'] for r in reports)
    for field in ('cases', 'revision', 'effective_schedule', 'numeric_environment', 'benchmark_sha256'):
        assert all(r[field] == reports[0][field] for r in reports), field
    arrays = [np.load(path.with_suffix('.npz'))['actions'] for path in paths]
    assert all(x.shape == (16, 32, 8) and np.isfinite(x).all() for x in arrays)
    for r, path in zip(reports, paths):
        assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == r['actions_sha256']
    equal = [x.dtype == arrays[0].dtype and x.tobytes() == arrays[0].tobytes() for x in arrays]
    cache = reports[1]['backend_stats']['timestep_cache']
    assert cache['hits'] > 0 and cache['bytes'] <= cache['max_bytes']
    assert reports[0]['backend_stats']['timestep_cache'] is None
    assert reports[2]['backend_stats']['timestep_cache'] is None
    measured = [[c for c in r['calls'] if c['phase'] == 'measured'] for r in reports]
    assert all(len(rows) == 10 for rows in measured)
    coverage = [rows[0]['region_counts'] for rows in measured]
    assert all(all(c['region_counts'] == coverage[0] for c in rows) for rows in measured)
    assert coverage[0]['compiled_calls'] > 0
    assert coverage[0]['prefill_checks'] == coverage[0]['eager_checks'] == 0
    medians = [statistics.median(c['ms'] for c in rows) for rows in measured]
    result[family] = dict(action_bytes_equal=all(equal),
        max_action_delta=max(float(np.abs(x.astype(float)-arrays[0].astype(float)).max()) for x in arrays),
        latency_p50_ms=dict(zip(names, medians)),
        reduction_percent_vs_references=[100*(1-medians[1]/medians[i]) for i in (0, 2)],
        measured_region_counts=coverage[0], cache=cache,
        schedule=reports[0]['effective_schedule'], quality_certified=False,
        receipts=[dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
(a.root / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
assert all(r['action_bytes_equal'] for r in result.values()), 'Action regression failed; retain evidence'
