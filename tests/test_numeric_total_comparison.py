"""Numerical screens may change only the explicitly requested computation."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('numeric_total', ROOT / 'eval/numeric_total_2026-09-10/compare.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def receipts(root, family, arm, *, bad_schedule=False, missing_queue=False):
    for variant in ('baseline_a', arm, 'baseline_b'):
        current = variant == arm
        path = root / f'{family}-{variant}.json'
        arrays = {'actions': np.ones((3, 8), np.float32) * (0.99 if current else 1)}
        if family == 'pi05' and not (current and missing_queue):
            arrays['queued_actions'] = np.zeros((3, 49, 1, 8), np.float32)
        np.savez_compressed(path.with_suffix('.npz'), **arrays)
        schedule = {'video': 2, 'action': 4} if current else None
        if bad_schedule and current:
            schedule = {'video': 1, 'action': 2}
        path.write_text(json.dumps(dict(ok=True, family=family, device='test',
            p50_ms=5 if current else 100, applied_passes=[], pass_results=[],
            execution_policy={}, sources={}, calls=[], model_id='same', revision='same',
            precision='native', schedule_override=schedule if family == 'va' else None,
            default_schedule={'video': 25, 'action': 50} if family == 'va' else {'action': 10},
            numeric_environment={'matmul_tf32': current and family == 'pi05',
                                 'cudnn_tf32': False, 'cudnn_benchmark': False},
            actions_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())))


def test_few_steps_use_original_default_schedule_denominator(tmp_path):
    receipts(tmp_path, 'va', 'current_fewstep')
    result = comparison.compare_profile(tmp_path, 'va', 'current_fewstep')
    assert result['status'] == 'OPERATING_POINT_SCREEN'
    assert result['speedup_range'] == [20, 20]
    assert result['reference_repeat_bitexact']
    assert not result['comparisons']['reference_candidate']['bitexact']


def test_undeclared_step_change_is_not_accepted_as_numeric(tmp_path):
    receipts(tmp_path, 'va', 'current_fewstep', bad_schedule=True)
    result = comparison.compare_profile(tmp_path, 'va', 'current_fewstep')
    assert result['status'] == 'INVALID_PROTOCOL'
    assert 'schedule_override' in result['invalid']['reference_candidate']['protocol']


def test_numeric_screen_cannot_drop_pi05_queued_actions(tmp_path):
    receipts(tmp_path, 'pi05', 'current', missing_queue=True)
    result = comparison.compare_profile(tmp_path, 'pi05')
    assert result['status'] == 'INVALID_PROTOCOL'


def test_tf32_permission_does_not_permit_cudnn_benchmark_change(tmp_path):
    receipts(tmp_path, 'pi05', 'current')
    path = tmp_path / 'pi05-current.json'
    data = json.loads(path.read_text())
    data['numeric_environment']['cudnn_benchmark'] = True
    path.write_text(json.dumps(data))
    result = comparison.compare_profile(tmp_path, 'pi05')
    assert result['status'] == 'INVALID_PROTOCOL'
