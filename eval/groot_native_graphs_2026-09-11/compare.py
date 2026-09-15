"""Compare every saved action chunk and matched-protocol field."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
variants = ('baseline_a', 'graph_only', 'fused_graph', 'baseline_b')
reports = {v: json.loads((root / (v + '.json')).read_text()) for v in variants}
arrays = {}
for v, report in reports.items():
    assert report['ok'] and not report['competing_gpu_processes'], v
    path = root / (v + '.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['actions_sha256']
    with np.load(path) as data:
        arrays[v] = data['actions'].copy()
    assert np.isfinite(arrays[v]).all()
    for key in ('revision', 'precision', 'torch', 'device', 'benchmark_sha256',
                'candidate_sha256', 'input_archive_sha256', 'default_schedule',
                'guidance', 'numeric_environment', 'execution_policy', 'measured_count'):
        assert report[key] == reports['baseline_a'][key], (v, key)
    if v in ('graph_only', 'fused_graph'):
        assert report['norm_graphs'] and all(g['replays'] and not g['disabled'] and g['self_check']['passed'] for g in report['norm_graphs'])
reference = arrays['baseline_a']
rows = []
for v in variants:
    a = arrays[v]
    assert a.shape == reference.shape and a.dtype == reference.dtype
    byte_equal = np.array_equal(a.view(np.uint8), reference.view(np.uint8))
    rows.append(dict(variant=v, p50_ms=reports[v]['p50_ms'],
        action_shape=list(a.shape), finite_byte_equal=byte_equal,
        max_abs_delta=float(np.max(np.abs(a.astype(float) - reference.astype(float)))),
        speedup_vs_baseline_a=reports['baseline_a']['p50_ms'] / reports[v]['p50_ms'],
        speedup_vs_baseline_b=reports['baseline_b']['p50_ms'] / reports[v]['p50_ms'],
        norm_graphs=len(reports[v]['norm_graphs']),
        norm_replays=sum(g['replays'] for g in reports[v]['norm_graphs'])))
result = dict(scope='Recorded camera inputs and synthetic state, full action arrays; no simulator certificate',
    rows=rows, report_sha256={v: hashlib.sha256((root / (v + '.json')).read_bytes()).hexdigest() for v in variants})
(root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if not all(r['finite_byte_equal'] for r in rows):
    raise SystemExit(1)
