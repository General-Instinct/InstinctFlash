"""Compare normalized performance with an explicitly accepted, immutable baseline."""
import hashlib
import json
from .compare import PROTOCOL_FIELDS


def record(report, comparison):
    if comparison['status'] != 'PASS':
        raise ValueError('Only a passing paired run can become a historical baseline')
    protocol = {k: report[k] for k in PROTOCOL_FIELDS}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return {'protocol_sha256': fingerprint,
            'p50_ratio': comparison['candidate']['p50_ms'] / min(comparison[k]['p50_ms'] for k in ('baseline_a', 'baseline_b')),
            'p95_ratio': comparison['candidate']['p95_ms'] / max(comparison[k]['p95_ms'] for k in ('baseline_a', 'baseline_b'))}


def compare_history(current, accepted):
    if current.keys() != accepted.keys():
        return {'status': 'INVALID', 'error': 'Model coverage changed; explicit rebaseline required'}
    rows = {}
    for family in current:
        a, b = accepted[family], current[family]
        if a['protocol_sha256'] != b['protocol_sha256']:
            rows[family] = {'status': 'INVALID', 'error': 'Protocol changed; explicit rebaseline required'}
        else:
            p50 = b['p50_ratio'] / a['p50_ratio']
            p95 = b['p95_ratio'] / a['p95_ratio']
            rows[family] = {'status': 'PASS' if p50 <= 1.05 and p95 <= 1.10 else 'REGRESSION',
                            'p50_relative_to_accepted': p50, 'p95_relative_to_accepted': p95}
    return {'status': 'PASS' if all(r['status'] == 'PASS' for r in rows.values()) else 'FAIL', 'models': rows}
