"""Numeric screening must not certify silent fallback or mismatched protocols."""
import hashlib
import json

import numpy as np
import pytest

from benchmarks.regression.run_cosmos_numeric import compare


def receipts(root):
    for arm in ('bitexact', 'numeric'):
        path = root / f'edge-{arm}.json'
        np.savez_compressed(path.with_suffix('.npz'), actions=np.full((16, 32, 8),
                            .125 if arm == 'numeric' else 0, np.float32))
        report = {key: 'same' for key in (
            'family', 'model_id', 'revision', 'precision', 'torch', 'device',
            'input_archive_sha256', 'reference_constructor_sha256', 'benchmark_sha256',
            'effective_schedule', 'numeric_environment', 'runtime_versions')}
        report.update(ok=True, competing_gpu_processes=[], sources={'vendor.py': 'fixed'},
                      cases=[{'i': i, 'phase': 'warmup' if i < 6 else 'measured'} for i in range(16)],
                      calls=[{'i': i, 'phase': 'warmup' if i < 6 else 'measured', 'ms': 5 if arm == 'numeric' else 10} for i in range(16)],
                      execution_policy=dict(category=arm.upper(), precision='native', changed_schedule={}),
                      actions_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(),
                      backend_stats=dict(numeric_attention=dict(eligible_python_calls=1, backend='cudnn'),
                        conditioning_cache_status=dict(admitted=True),
                        conditioning_cache=dict(disabled=False, rejected=0, verified_tensors=1,
                            graph_stats=[dict(replays=1, rejected=0)])))
        path.write_text(json.dumps(report))
    return root / 'edge-numeric.json'


def test_screen_and_explicit_regression_bounds(tmp_path):
    receipts(tmp_path)
    result = compare(tmp_path, 'edge')
    assert result['status'] == 'PASS' and result['category'] == 'SCREEN'
    assert not result['task_quality_certified']
    assert compare(tmp_path, 'edge', min_speedup=3)['status'] == 'REGRESSION'
    assert compare(tmp_path, 'edge', max_abs_action_delta=.1)['status'] == 'REGRESSION'


@pytest.mark.parametrize('change', ['fallback', 'source', 'schedule', 'precision', 'hash', 'timings'])
def test_false_green_rejected(tmp_path, change):
    path = receipts(tmp_path)
    report = json.loads(path.read_text())
    if change == 'fallback':
        report['backend_stats']['numeric_attention']['eligible_python_calls'] = 0
    elif change == 'source':
        report['sources']['vendor.py'] = 'changed'
    elif change == 'schedule':
        report['execution_policy']['changed_schedule'] = {'action': 2}
    elif change == 'precision':
        report['execution_policy']['precision'] = 'fp8'
    elif change == 'timings':
        report['calls'][-1]['i'] = 0
    else:
        report['actions_sha256'] = 'bad'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare(tmp_path, 'edge')


def test_failed_recheck_cannot_leave_stale_pass(tmp_path, monkeypatch):
    import sys
    from benchmarks.regression.run_cosmos_numeric import main
    result = tmp_path / 'edge-comparison.json'
    result.write_text('{"status": "PASS", "validation_complete": true}')
    monkeypatch.setattr(sys, 'argv', ['screen', str(tmp_path), '--family', 'edge', '--compare-only'])
    with pytest.raises(FileNotFoundError):
        main()
    assert json.loads(result.read_text())['status'] == 'INVALID'
