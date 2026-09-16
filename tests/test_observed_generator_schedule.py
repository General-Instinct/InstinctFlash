"""The benchmark must inspect loaded samplers, including generic FP8 wrappers."""
from types import SimpleNamespace as NS
import sys

import pytest

from benchmarks.regression.user_e2e import dreamzero_contract, observed_schedule, queue_length


def api(loop, wrapped):
    if wrapped:
        def execution_recipe():
            raise AssertionError('FP8 execution metadata is not a sampler declaration')
        loop = NS(inner=loop, declaration=execution_recipe)
    return NS(_backend=NS(_impl=loop))


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('video,action', [(25, 50), (2, 4)])
def test_va_reads_actual_server_steps_and_guidance(wrapped, video, action):
    config = NS(num_inference_steps=video, action_num_inference_steps=action,
                guidance_scale=5.0, action_guidance_scale=1.0)
    model = api(NS(_server=NS(job_config=config)), wrapped)
    expected = {'video': video, 'action': action}
    assert observed_schedule(model, 'va', expected) == expected
    config.num_inference_steps += 1
    with pytest.raises(AssertionError):
        observed_schedule(model, 'va', expected)
    config.num_inference_steps = video
    config.guidance_scale = 3.0
    with pytest.raises(AssertionError):
        observed_schedule(model, 'va', expected)


def test_va_thor_frontend_declaration_remains_supported():
    declaration = {'steps': {'video': 2, 'action': 4, 'kv_refresh': 2},
                   'guidance': {'video': ('cfg', 5.0), 'action': ('positive_only', 1.0)}}
    model = api(NS(declaration=lambda: declaration), False)
    assert observed_schedule(model, 'va', declaration['steps']) == declaration['steps']


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('dynamic', [False, True])
def test_dreamzero_observes_the_native_head_under_fp8(monkeypatch, wrapped, dynamic):
    mask = (True,) * 16
    declaration = {'steps': {'video_action': 16, 'kv_commit': 1},
                   'guidance': {'video_action': ('cfg', 5.0)},
                   'dit_step_mask': mask, 'dynamic_cache_schedule': dynamic}
    head = object()

    def actual_head(value):
        assert value is head
        return declaration

    monkeypatch.setitem(sys.modules, 'dreamzero_iwm.adapter',
                        NS(SHIPPED_DIT_MASK=mask, _head_declaration=actual_head))
    native = NS(_wrapper=NS(_policy=NS(trained_model=NS(action_head=head))),
                declaration=lambda: declaration)
    model = api(native, wrapped)
    assert observed_schedule(model, 'dreamzero', {'video_action': 16, 'kv_commit': 1}) == {'video_action': 16}
    assert dreamzero_contract(model, dynamic=dynamic) == declaration
    declaration['dynamic_cache_schedule'] = not dynamic
    with pytest.raises(AssertionError):
        dreamzero_contract(model, dynamic=dynamic)
    declaration['steps']['video_action'] = 8
    with pytest.raises(AssertionError):
        observed_schedule(model, 'dreamzero', {'video_action': 16, 'kv_commit': 1})


@pytest.mark.parametrize('wrapped', [False, True])
def test_public_queue_observation_tracks_the_actual_queue(wrapped):
    queue = [1, 2]
    model = api(NS(_p=NS(_action_queue=queue)), wrapped)
    assert queue_length(model) == 2
    queue.pop()
    assert queue_length(model) == 1


def test_thor_pi05_engine_queue_still_uses_its_own_storage():
    assert queue_length(api(NS(_queue=[1]), False)) == 1
