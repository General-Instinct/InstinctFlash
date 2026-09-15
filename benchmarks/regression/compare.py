"""Fail-closed A/B/A regression gate for complete action chunks.

Action equality and performance have separate verdicts. A noisy reference or
contended GPU is inconclusive, never a passing performance certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL_FIELDS = (
    'family', 'model_id', 'revision', 'device', 'torch', 'precision',
    'input_archive_sha256', 'default_schedule', 'guidance', 'numeric_environment',
    'reference_constructor_sha256', 'benchmark_sha256', 'cases', 'effective_schedule', 'runtime_versions',
)


def _load(path):
    path = Path(path)
    data = json.loads(path.read_text())
    missing = [key for key in (*PROTOCOL_FIELDS, 'calls', 'actions_sha256', 'ok', 'competing_gpu_processes', 'sources') if key not in data]
    if missing or not data.get('ok'):
        raise ValueError(f'{path}: missing fields {missing} or failed execution')
    archive = path.with_suffix('.npz')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != data['actions_sha256']:
        raise ValueError(f'{path}: action archive checksum mismatch')
    with np.load(archive, allow_pickle=False) as arrays:
        actions = arrays['actions'].copy()
    if actions.ndim != 3 or actions.shape[1:] != (32, 8):
        raise ValueError(f'{path}: expected complete 32x8 DROID action chunks')
    if actions.size == 0 or len(actions) != len(data['calls']) or not np.isfinite(actions).all():
        raise ValueError(f'{path}: empty, incomplete or non-finite actions')
    if len(data['cases']) != len(data['calls']):
        raise ValueError(f'{path}: incomplete request sequence')
    if any(not isinstance(case, dict) or call.get('i') != case.get('i') or call.get('phase') != case.get('phase')
           for case, call in zip(data['cases'], data['calls'])):
        raise ValueError(f'{path}: timing rows do not match request cases')
    measured = [row['ms'] for row in data['calls'] if row['phase'] == 'measured']
    if len(measured) < 10 or not all(math.isfinite(x) and x > 0 for x in measured):
        raise ValueError(f'{path}: at least 10 finite positive measured latencies required')
    return data, actions, np.asarray(measured)


def compare(baseline_a, candidate, baseline_b, *, latency_tolerance=0.05, drift_tolerance=0.05):
    for value in (latency_tolerance, drift_tolerance):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError('Tolerances must be finite fractions in [0, 1)')
    loaded = [_load(path) for path in (baseline_a, candidate, baseline_b)]
    reports, arrays, times = zip(*loaded)
    differences = [field for field in PROTOCOL_FIELDS if any(r[field] != reports[0][field] for r in reports[1:])]
    shared_sources = set.intersection(*(set(r['sources']) for r in reports))
    if not shared_sources or any(len({r['sources'][p] for r in reports}) != 1 for p in shared_sources):
        differences.append('shared_source_hashes')
    if differences:
        raise ValueError(f'Incomparable protocols: {differences}')
    def exact(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    reference_exact = exact(arrays[0], arrays[2])
    candidate_exact = exact(arrays[0], arrays[1])
    quantiles = [{'p50_ms': float(np.median(t)), 'p95_ms': float(np.percentile(t, 95)),
                  'max_ms': float(max(t)), 'samples': len(t)} for t in times]
    reference_p50 = [quantiles[i]['p50_ms'] for i in (0, 2)]
    drift = max(reference_p50) / min(reference_p50) - 1
    contention = any(r['competing_gpu_processes'] for r in reports)
    performance = 'INCONCLUSIVE' if contention or drift > drift_tolerance else (
        'PASS' if quantiles[1]['p50_ms'] <= min(reference_p50) * (1 + latency_tolerance)
        and quantiles[1]['p95_ms'] <= max(quantiles[i]['p95_ms'] for i in (0, 2)) * 1.10 else 'REGRESSION')
    correctness = 'UNSTABLE_REFERENCE' if not reference_exact else ('PASS' if candidate_exact else 'ACTION_MISMATCH')
    shape_equal = arrays[0].shape == arrays[1].shape
    return {
        'status': 'PASS' if correctness == performance == 'PASS' else 'FAIL',
        'correctness': correctness, 'performance': performance,
        'reference_bitexact': reference_exact, 'candidate_bitexact': candidate_exact,
        'max_abs_action_delta': float(np.max(np.abs(arrays[0].astype(np.float64) - arrays[1].astype(np.float64)))) if shape_equal else None,
        'baseline_a': quantiles[0], 'candidate': quantiles[1], 'baseline_b': quantiles[2],
        'speedup_range': [v / quantiles[1]['p50_ms'] for v in sorted(reference_p50)],
        'reference_drift': drift, 'gpu_contention': contention,
        'latency_tolerance': latency_tolerance, 'p95_tolerance': 0.10, 'drift_tolerance': drift_tolerance,
        'scope': 'Recorded-input action regression; not closed-loop task success.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_a', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('baseline_b', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.baseline_a, args.candidate, args.baseline_b)
    except (ValueError, KeyError, OSError) as error:
        result = {'status': 'INVALID', 'error': str(error)}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
