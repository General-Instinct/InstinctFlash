"""Compare saved actions, protocol identity and graph admission; no inferred certificates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    a, b = (json.loads(p.read_text()) for p in (baseline, candidate))
    fields = ('family', 'precision', 'model_id', 'revision', 'device', 'torch',
              'input_archive_sha256', 'schedule_override', 'default_schedule',
              'guidance', 'numeric_environment', 'benchmark_sha256')
    differences = {k: [a.get(k), b.get(k)] for k in fields if a.get(k) != b.get(k)}
    call_identity = lambda report: [{k: row.get(k) for k in ('i', 'cycle', 'phase', 'shape')} for row in report.get('calls', [])]
    if call_identity(a) != call_identity(b):
        differences['call_identity'] = [call_identity(a), call_identity(b)]
    if a.get('competing_gpu_processes') or b.get('competing_gpu_processes'):
        differences['gpu_contention'] = [a.get('competing_gpu_processes'), b.get('competing_gpu_processes')]
    arrays = []
    with np.load(baseline.with_suffix('.npz')) as x, np.load(candidate.with_suffix('.npz')) as y:
        same_keys = set(x.files) == set(y.files)
        for k in sorted(set(x.files) & set(y.files)):
            left, right = x[k], y[k]
            geometry = left.shape == right.shape and left.dtype == right.dtype
            finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
            arrays.append(dict(name=k, shape=list(left.shape), finite=finite,
                               bitexact=geometry and finite and left.tobytes() == right.tobytes(),
                               max_abs_delta=float(np.abs(left.astype(np.float64)-right.astype(np.float64)).max()) if geometry and finite and left.size else None))
    stats = b.get('graph_stats') or b.get('backend_stats') or {}
    return dict(baseline=str(baseline), candidate=str(candidate),
                baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                protocol_differences=differences,
                bitexact=bool(a.get('ok') and b.get('ok') and not differences and same_keys and arrays and all(x['bitexact'] for x in arrays)),
                arrays=arrays, candidate_stats=stats,
                baseline_p50_ms=a['p50_ms'], candidate_p50_ms=b['p50_ms'],
                speedup=a['p50_ms']/b['p50_ms'],
                scope='Recorded inputs only; inspect candidate_stats for actual admission. No task-quality certificate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['bitexact'] else 1)
