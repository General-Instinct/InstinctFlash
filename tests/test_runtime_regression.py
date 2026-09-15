"""Regression gates reject false-green or incomparable benchmark artifacts."""
import hashlib
import json
import numpy as np
import pytest
from benchmarks.regression.compare import compare, PROTOCOL_FIELDS


def receipts(tmp_path):
    paths = []
    for arm in ('a', 'candidate', 'b'):
        path = tmp_path / f'{arm}.json'
        np.savez_compressed(path.with_suffix('.npz'), actions=np.zeros((12, 32, 8), np.float32))
        report = {k: 'same' for k in PROTOCOL_FIELDS}
        report.update(ok=True, sources={'upstream.py': 'fixed'}, cases=[{'i': i, 'phase': 'measured'} for i in range(12)], competing_gpu_processes=[],
            calls=[{'i': i, 'phase': 'measured', 'ms': 10} for i in range(12)],
            actions_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())
        path.write_text(json.dumps(report)); paths.append(path)
    return paths


def edit(path, **kwargs):
    if 'calls' in kwargs:
        kwargs['calls'] = [dict(row, i=i) for i, row in enumerate(kwargs['calls'])]
    doc = json.loads(path.read_text()); doc.update(kwargs); path.write_text(json.dumps(doc))


def test_matching_actions_and_speed_pass(tmp_path):
    assert compare(*receipts(tmp_path))['status'] == 'PASS'


def test_slow_candidate_fails_independently_of_exactness(tmp_path):
    paths = receipts(tmp_path)
    edit(paths[1], calls=[{'phase': 'measured', 'ms': 12} for _ in range(12)])
    result = compare(*paths)
    assert result['correctness'] == 'PASS' and result['performance'] == 'REGRESSION'


def test_drift_is_inconclusive(tmp_path):
    paths = receipts(tmp_path)
    edit(paths[2], calls=[{'phase': 'measured', 'ms': 12} for _ in range(12)])
    assert compare(*paths)['performance'] == 'INCONCLUSIVE'


def test_contention_is_not_a_pass(tmp_path):
    paths = receipts(tmp_path); edit(paths[1], competing_gpu_processes=[123])
    result = compare(*paths)
    assert result['correctness'] == 'PASS' and result['status'] == 'FAIL'


def test_hash_or_protocol_mismatch_is_invalid(tmp_path):
    paths = receipts(tmp_path); edit(paths[1], revision='other')
    with pytest.raises(ValueError, match='protocols'):
        compare(*paths)
    edit(paths[1], revision='same', actions_sha256='wrong')
    with pytest.raises(ValueError, match='checksum'):
        compare(*paths)


def test_last_action_signed_zero_is_a_mismatch(tmp_path):
    paths = receipts(tmp_path)
    x = np.zeros((12, 32, 8), np.float32); x[-1, -1, -1] = -0.0
    np.savez_compressed(paths[1].with_suffix('.npz'), actions=x)
    edit(paths[1], actions_sha256=hashlib.sha256(paths[1].with_suffix('.npz').read_bytes()).hexdigest())
    assert compare(*paths)['correctness'] == 'ACTION_MISMATCH'


def test_missing_metadata_or_nan_is_invalid(tmp_path):
    paths = receipts(tmp_path)
    doc = json.loads(paths[1].read_text()); del doc['device']; paths[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='missing fields'):
        compare(*paths)


def test_modified_shared_source_invalidates_pair(tmp_path):
    paths = receipts(tmp_path); edit(paths[1], sources={'upstream.py': 'mutated'})
    with pytest.raises(ValueError, match='shared_source_hashes'):
        compare(*paths)


def test_tail_regression_fails_even_with_unchanged_median(tmp_path):
    paths = receipts(tmp_path)
    edit(paths[1], calls=[{'phase': 'measured', 'ms': 10 if i < 10 else 100} for i in range(12)])
    assert compare(*paths)['performance'] == 'REGRESSION'


def test_history_catches_lost_speedup_still_faster_than_upstream():
    from benchmarks.regression.history import compare_history
    previous = {'edge': {'protocol_sha256': 'fixed', 'p50_ratio': .7, 'p95_ratio': .7}}
    current = {'edge': {'protocol_sha256': 'fixed', 'p50_ratio': .9, 'p95_ratio': .9}}
    assert compare_history(current, previous)['models']['edge']['status'] == 'REGRESSION'
    current['edge']['protocol_sha256'] = 'changed'
    assert compare_history(current, previous)['models']['edge']['status'] == 'INVALID'


def test_deployment_rejects_shell_metacharacters_before_ssh(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, str(root / 'benchmarks/regression/deploy_thor.py'),
        '--host', 'host;bad', '--remote-root', '/tmp/results', '--python', '/env/bin/python',
        '--output-root', str(tmp_path)], capture_output=True, text=True)
    assert run.returncode == 2 and 'Invalid SSH host' in run.stderr


def test_failed_deployment_replaces_stale_green_status(tmp_path, monkeypatch):
    from benchmarks.regression import deploy_thor
    import sys
    (tmp_path / 'latest.json').write_text('{"status":"PASS","exit_code":0}')
    monkeypatch.setattr(sys, 'argv', ['deploy', '--host', 'host', '--remote-root', '/tmp/results',
                                    '--python', '/env/bin/python', '--output-root', str(tmp_path)])
    def fail(args):
        raise OSError('Connection failed')
    monkeypatch.setattr(deploy_thor, 'run_once', fail)
    assert deploy_thor.main() == 1
    assert json.loads((tmp_path / 'latest.json').read_text())['status'] == 'ERROR'


def test_matching_truncated_chunks_cannot_pass(tmp_path):
    paths = receipts(tmp_path)
    for path in paths:
        np.savez_compressed(path.with_suffix('.npz'), actions=np.zeros((12, 16, 8), np.float32))
        edit(path, actions_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='complete 32x8'):
        compare(*paths)
