"""Validate Edge original-weight budget receipts and the matched attention pair."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
a = p.parse_args()
output = a.directory / 'summary.json'
output.write_text(json.dumps(dict(validation_complete=False)) + '\n')
reports, arrays, rows = {}, {}, []
for name, cfg, attention in (
    ('s1-cfg1-native', 1., 'native'),
    ('s1-cfg1-cudnn', 1., 'cudnn'),
    ('s1-cfg4-cudnn', 4., 'cudnn'),
):
    path = a.directory / f'{name}.json'
    r = json.loads(path.read_text())
    assert r['ok'] and r['trained_student'] is False
    assert r['category'] == 'OPERATING-POINT' and not r['task_quality_certified']
    assert r['checkpoint'] == 'nvidia/Cosmos3-Edge-Policy-DROID'
    assert (r['requested_steps'], r['requested_guidance'], r['attention']) == (1, cfg, attention)
    assert r['experimental_override'] == dict(guidance=cfg, action_steps=1, sampler='native UniPC')
    assert r['actual_velocity_branches'] == 16 * (1 if cfg == 1 else 2)
    assert r['execution_policy']['precision'] == 'native'
    archive = path.with_suffix('.npz')
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == r['actions_sha256']
    with np.load(archive) as data:
        x = data['actions'].copy()
    assert x.shape == (16, 32, 8) and np.isfinite(x).all()
    assert len(r['calls']) == 16 and [c['i'] for c in r['calls']] == list(range(16))
    assert [c['phase'] for c in r['calls']] == ['warmup'] * 6 + ['measured'] * 10
    assert all(np.isfinite(c['ms']) and c['ms'] > 0 for c in r['calls'])
    if attention == 'cudnn':
        assert r['backend_stats']['numeric_attention']['eligible_python_calls'] > 0
    times = [c['ms'] for c in r['calls'][6:]]
    p50, p95 = float(np.median(times)), float(np.percentile(times, 95))
    assert np.isclose(p50, r['p50_ms']) and np.isclose(p95, r['p95_ms'])
    rows.append(dict(name=name, p50_ms=p50, p95_ms=p95, max_ms=max(times),
        branches_per_request=1 if cfg == 1 else 2,
        conditional_15hz_budgets=[dict(executed_actions=n, budget_ms=1000*n/15,
            measured_misses=sum(t > 1000*n/15 for t in times)) for n in (8, 16, 32)]))
    reports[name], arrays[name] = r, x

ref = reports['s1-cfg1-native']
for r in reports.values():
    for key in ('revision', 'checkpoint_index_sha256', 'benchmark_sha256',
                'fixture_sha256', 'torch', 'cuda', 'cudnn'):
        assert r[key] == ref[key], key
    assert all(r['sources'].get(k) == v for k, v in ref['sources'].items())
cand = reports['s1-cfg1-cudnn']
assert cand['sources'] == reports['s1-cfg4-cudnn']['sources']
assert arrays['s1-cfg1-native'].dtype == arrays['s1-cfg1-cudnn'].dtype
delta = np.abs(arrays['s1-cfg1-native'].astype(np.float64)
               - arrays['s1-cfg1-cudnn'].astype(np.float64))
result = dict(validation_complete=True, trained_student=False, task_quality_certified=False,
    rows=rows, cfg1_attention_pair=dict(category='SCREEN',
        speedup=ref['p50_ms']/cand['p50_ms'], max_abs_action_delta=float(delta.max()),
        mean_abs_action_delta=float(delta.mean()),
        per_dimension_max_abs=delta.max(axis=(0, 1)).tolist()),
    scope='Original Edge weights, changed UniPC schedule/guidance. Not a trained SDE student, quality certificate, or closed-loop realtime claim.')
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
