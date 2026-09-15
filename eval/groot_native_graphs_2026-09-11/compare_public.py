"""Validate the shipping Runtime path against its capture kill-switch controls."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
reports = {v: json.loads((root / (v + '.json')).read_text()) for v in ('baseline_a', 'current', 'baseline_b')}
reference = None
rows = []
for v, report in reports.items():
    assert report['ok'] and not report['competing_gpu_processes'], v
    for key in ('revision', 'precision', 'torch', 'device', 'benchmark_sha256',
                'input_archive_sha256', 'default_schedule', 'guidance',
                'numeric_environment', 'execution_policy', 'measured_count'):
        assert report[key] == reports['baseline_a'][key], (v, key)
    with np.load(root / (v + '.npz')) as data:
        actions = data['actions'].copy()
    assert hashlib.sha256((root / (v + '.npz')).read_bytes()).hexdigest() == report['actions_sha256']
    assert np.isfinite(actions).all()
    if reference is None:
        reference = actions
    assert actions.shape == reference.shape and actions.dtype == reference.dtype
    assert np.array_equal(actions.view(np.uint8), reference.view(np.uint8)), v
    stats = report['backend_stats']
    assert stats['captured'] == (v == 'current'), (v, stats)
    assert stats['action_nfe'] == 4
    if v == 'current':
        assert stats['graph_captures'] > 0 and stats['graph_replays'] > 0
    rows.append(dict(variant=v, p50_ms=report['p50_ms'], finite_byte_equal=True,
        action_shape=list(actions.shape), graph_captures=stats['graph_captures'],
        graph_replays=stats['graph_replays']))
result = dict(rows=rows, speedup_vs_a=reports['baseline_a']['p50_ms']/reports['current']['p50_ms'],
    speedup_vs_b=reports['baseline_b']['p50_ms']/reports['current']['p50_ms'],
    scope='Native Runtime, existing DiT graph only; unchanged checkpoint/precision/four steps; tested actions, no simulator certificate')
(root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
