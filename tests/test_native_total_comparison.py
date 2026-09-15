"""Total acceleration evidence must include buffered actions and native history."""
import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pi05_equal_first_action_does_not_hide_changed_queue(tmp_path):
    compare = load('total_compare', 'eval/native_coverage_2026-09-10/compare.py')
    for arm in ('baseline_a', 'current', 'baseline_b'):
        path = tmp_path / f'pi05-{arm}.json'
        queue = np.zeros((3, 49, 1, 32), np.float32)
        if arm == 'current':
            queue[2, 48, 0, 31] = 0.01
        np.savez_compressed(path.with_suffix('.npz'),
                            actions=np.zeros((3, 8), np.float32), queued_actions=queue)
        path.write_text(json.dumps(dict(ok=True, family='pi05', device='test', p50_ms=10,
            applied_passes=[], execution_policy={}, sources={}, calls=[],
            actions_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())))
    result = compare.compare_family(tmp_path, 'pi05')
    assert result['status'] == 'ACTION_MISMATCH'
    assert result['comparisons']['baseline_current']['matching_calls'] == 3
    assert not result['comparisons']['baseline_current']['bitexact']


def test_reference_commits_recorded_feedback_after_prediction_without_loading_runtime():
    upstream = load('total_upstream', 'eval/native_total_2026-09-10/upstream.py')
    events = []
    class Loop:
        def predict(self, observation):
            events.append(('predict', observation))
            return {'action': 'predicted'}
        def commit(self, observation, action):
            events.append(('commit', observation, action))
    class Plan:
        results = []
        def without(self, *args):
            return self
    class Declaration:
        _checkpoint = object()
        plan = Plan()
        def reset(self, **kwargs):
            raise AssertionError('Reference must not load the Runtime backend')
    reference = upstream.Reference(Declaration(), Loop())
    observation = {'obs': 'recorded'}
    assert reference.predict(observation, executed_action='executed') == {'action': 'predicted'}
    assert events == [('predict', observation), ('commit', observation, 'executed')]


def test_loading_heap_trim_preserves_live_values_rng_and_stops(tmp_path):
    import ctypes
    import time
    import torch
    import pytest
    if not hasattr(ctypes.CDLL(None), 'malloc_trim'):
        pytest.skip('glibc heap trimming unavailable')
    memory = load('total_loading_memory', 'eval/native_total_2026-09-10/load_memory.py')
    value = torch.randn(128, 128)
    before = value.clone()
    rng = torch.get_rng_state().clone()
    trace = tmp_path / 'loading.jsonl'
    worker = memory.LoadingHeapTrim(trace, interval=0.001).start()
    deadline = time.monotonic() + 1
    while not worker.samples and time.monotonic() < deadline:
        time.sleep(0.005)
    summary = worker.close()
    assert summary['samples'] >= 1
    assert torch.equal(value, before)
    assert torch.equal(torch.get_rng_state(), rng)
    size = trace.stat().st_size
    time.sleep(0.01)
    assert trace.stat().st_size == size
    assert worker.close() == summary
