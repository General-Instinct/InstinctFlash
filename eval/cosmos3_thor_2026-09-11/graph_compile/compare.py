"""Recompute the selective-graph ablation from archived finite action arrays."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

root = Path(sys.argv[1])
reference_path = root.parent.parent / 'static_compile/receipts/edge-dynamic_a.json'

def load(path):
    report = json.loads(path.read_text())
    assert report['ok'] and not report['competing_gpu_processes']
    actions_path = path.with_suffix('.npz')
    assert hashlib.sha256(actions_path.read_bytes()).hexdigest() == report['actions_sha256']
    with np.load(actions_path) as data:
        actions = data['actions'].copy()
    assert np.isfinite(actions).all()
    assert actions.shape == (len(report['cases']), 32, 8)
    return report, actions

reference, ref_actions = load(reference_path)
rows = []
for variant in ('padded', 'unpadded'):
    report, actions = load(root / variant / 'edge-graphs.json')
    for field in ('family', 'revision', 'precision', 'torch', 'device',
                  'input_archive_sha256', 'reference_constructor_sha256', 'cases',
                  'effective_schedule', 'numeric_environment'):
        assert report[field] == reference[field], (variant, field)
    audit = report['audit']
    assert audit['effective_setup']['use_cuda_graphs']
    assert audit['effective_setup']['use_torch_compile']
    assert not audit['effective_setup']['diffusion_cache']
    assert not audit['vfm_use_cuda_graphs']
    assert audit['engine_attention']['library_sha256'] == reference['audit']['engine_attention']['library_sha256']
    assert audit['counts']['cuda_graph_replays'] > 0
    assert actions.dtype == ref_actions.dtype
    rows.append(dict(variant=variant,
        p50_ms=statistics.median(c['ms'] for c in report['calls'] if c['phase'] == 'measured'),
        byte_equal_to_dynamic_a=np.array_equal(actions.view(np.uint8), ref_actions.view(np.uint8)),
        max_abs_delta=float(np.abs(actions.astype(float) - ref_actions.astype(float)).max()),
        graph_replays=audit['counts']['cuda_graph_replays'],
        attention_host_calls=audit['engine_attention']['calls'],
        shapes=audit['engine_attention']['shapes']))
(root / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
