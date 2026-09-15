"""A repeated reference must gate any advertised native exactness result."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'native_coverage_compare', ROOT / 'eval/native_coverage_2026-09-10/compare.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def receipts(tmp_path, values=(1.0, 1.0, 1.0)):
    for arm, value in zip(('baseline_a', 'current', 'baseline_b'), values):
        path = tmp_path / f'edge-{arm}.json'
        np.savez_compressed(path.with_suffix('.npz'), actions=np.full((3, 2, 8), value, np.float32))
        report = dict(ok=True, family='edge', device='test', p50_ms=10,
                      applied_passes=[], execution_policy={}, sources={'model.py': 'unchanged'},
                      calls=[dict(i=i, cycle=0, phase='measured', shape=[2, 8]) for i in range(3)],
                      actions_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())
        path.write_text(json.dumps(report))


def test_exact_control_is_marked_unchanged(tmp_path):
    receipts(tmp_path)
    result = module.compare_family(tmp_path, 'edge')
    assert result['status'] == 'BITEXACT'
    assert result['unchanged_transform_plan'] is True


@pytest.mark.parametrize('values,status', [
    ((1, 1, 2), 'BASELINE_VARIABLE'),
    ((1, 2, 1), 'ACTION_MISMATCH'),
])
def test_reference_repeat_prevents_false_exactness(tmp_path, values, status):
    receipts(tmp_path, values)
    assert module.compare_family(tmp_path, 'edge')['status'] == status


def test_modified_action_artifact_is_rejected(tmp_path):
    receipts(tmp_path)
    np.savez_compressed(tmp_path / 'edge-current.npz', actions=np.zeros((3, 2, 8), np.float32))
    with pytest.raises(ValueError, match='hash mismatch'):
        module.compare_family(tmp_path, 'edge')


@pytest.mark.parametrize('field,value', [('competing_gpu_processes', [123]),
                                        ('sources', {'model.py': 'changed'})])
def test_contention_and_source_drift_invalidate_pair(tmp_path, field, value):
    receipts(tmp_path)
    path = tmp_path / 'edge-current.json'
    report = json.loads(path.read_text()); report[field] = value
    path.write_text(json.dumps(report))
    assert module.compare_family(tmp_path, 'edge')['status'] == 'INVALID_PROTOCOL'


def test_cli_cannot_report_success_for_a_variable_reference(tmp_path):
    receipts(tmp_path, (1, 1, 2))
    output = tmp_path / 'comparison.json'
    run = subprocess.run(
        [sys.executable, str(ROOT / 'eval/native_coverage_2026-09-10/compare.py'),
         str(tmp_path), 'edge', '--output', str(output)], capture_output=True, text=True)
    assert run.returncode == 1
    assert json.loads(output.read_text())['status'] == 'BASELINE_VARIABLE'
