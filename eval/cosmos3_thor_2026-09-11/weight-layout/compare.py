"""Validate a matched native / packed / native Cosmos action comparison."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

root = Path(sys.argv[1])
variants = ('baseline_a', 'packed', 'baseline_b')
reports = {v: json.loads((root / f'nano-{v}.json').read_text()) for v in variants}
arrays = {}
for v, report in reports.items():
    assert report['ok'] and not report['competing_gpu_processes'], v
    for field in ('family', 'arm', 'model_id', 'revision', 'precision', 'torch',
                  'device', 'input_archive_sha256', 'reference_constructor_sha256',
                  'benchmark_sha256', 'cases', 'effective_schedule',
                  'numeric_environment', 'sources', 'weight_layout_script_sha256'):
        assert report[field] == reports['baseline_a'][field], (v, field)
    path = root / f'nano-{v}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['actions_sha256']
    with np.load(path) as data:
        arrays[v] = data['actions'].copy()
    assert np.isfinite(arrays[v]).all(), v
    assert arrays[v].shape == (len(report['cases']), 32, 8), v
    layout = report['weight_layout']
    if v == 'packed':
        assert layout['installed'] == 72 and layout['live_layout_passed']
        assert len(layout['weights']) == 72
        assert all(w['old_stride'] == [4096, 1] and w['new_stride'] == [1, 12288]
                   for w in layout['weights'])
    else:
        assert not layout['weights'] and not layout.get('installed', 0)

reference = arrays['baseline_a']
rows = []
for v, report in reports.items():
    actions = arrays[v]
    assert actions.dtype == reference.dtype
    delta = np.abs(actions.astype(float) - reference.astype(float))
    rows.append(dict(
        variant=v,
        p50_ms=statistics.median(c['ms'] for c in report['calls'] if c['phase'] == 'measured'),
        byte_equal_to_baseline_a=np.array_equal(actions.view(np.uint8), reference.view(np.uint8)),
        max_abs_delta=float(delta.max()), mean_abs_delta=float(delta.mean()),
        shape=list(actions.shape),
    ))
result = dict(rows=rows, speedup_vs_a=rows[0]['p50_ms'] / rows[1]['p50_ms'],
              speedup_vs_b=rows[2]['p50_ms'] / rows[1]['p50_ms'],
              qualification='Finite action-byte comparison on recorded inputs; no general checkpoint or closed-loop quality certificate')
(root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if not rows[2]['byte_equal_to_baseline_a']:
    raise SystemExit('Baseline action bytes did not reproduce')
