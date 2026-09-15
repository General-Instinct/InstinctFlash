"""Raw control observation must preserve native SDE endpoint arithmetic."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import test_cosmos3_action_padding as fixtures
from instinct_compress.models.cosmos3_realtime_rollout_v16 import DeploymentGrid, rollout_deployment
from capture_control import capture_control

setup = fixtures.setup


@pytest.mark.parametrize('guidance,kind,steps', [(cfg, 'fixed', n) for cfg in (1., 4.) for n in (1, 2, 4)] + [(4., 'unipc', 4)])
def test_original_control_native_endpoint(setup, monkeypatch, guidance, kind, steps):
    from cosmos_framework.data.generator.action import action_processing
    from cosmos_framework.scripts import action_policy_server_robolab as server
    condition, model, role = setup
    role.guidance = guidance
    record = SimpleNamespace(raw_action_dim=8, action_normalizer=None)
    monkeypatch.setattr(action_processing, 'get_action_processing_records', lambda batch: [record])
    monkeypatch.setattr(server, '_build_data_batch_from_sample', lambda sample: {})
    fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, 'fork_rng', lambda devices=None: fork(devices=[]))
    if kind == 'fixed':
        expected = rollout_deployment(role, condition, DeploymentGrid(steps, guidance), grad_last=False, receipt={})

    def generate(batch, *, sampler, seed, guidance, num_steps, shift, **kwargs):
        plans, clean, conditional, unconditional, noise, reference, mask = model._prepare_inference_data()
        def callback(state, time):
            def branch(tokens):
                return model._get_velocity(noise_x=state, timestep=time, text_tokens=tokens,
                                           sequence_plans=plans, gen_data_clean=clean)
            c = branch(conditional)
            if guidance == 1.:
                return c
            u = branch(unconditional)
            return [uv + guidance * (cv - uv) for cv, uv in zip(c, u, strict=True)]
        with role._bound_heads():
            if kind == 'fixed':
                # The actual model replaces shift with zero for FixedStepSampler.
                result = sampler(callback, noise, seed=seed, shift=0., num_steps=num_steps,
                                 condition_reference=reference, condition_mask=mask)
            else:
                # Native UniPC does not accept condition_reference/condition_mask.
                result = sampler(callback, noise, seed=seed, shift=shift, num_steps=num_steps)
        return dict(action=[result[0][condition.action_offset:].reshape(33, 64)], endpoint=torch.stack(result))

    model.generate_samples_from_batch = generate
    if kind == 'unipc':
        from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
        model.sampler = UniPCSampler(tensor_kwargs={'device': torch.device('cpu')})
        expected = generate({}, sampler=model.sampler, seed=list(condition.seeds), guidance=guidance,
                            num_steps=4, shift=1.)['endpoint']
    service = SimpleNamespace(model=model, cfg=SimpleNamespace(history_length=1),
                              _lock=nullcontext(), _build_sample=lambda x: x)
    config = dict(weight='original', kind=kind, times=[1. - i / steps for i in range(steps + 1)], steps=steps,
                  guidance=guidance, shift=1., action_padding='zero' if kind=='fixed' else 'native')
    arrays, fields, trace = capture_control(service,
        dict(image=np.zeros((2, 2, 3), np.uint8), state=np.zeros(8, np.float32), prompt='fixture'),
        np.zeros((32, 8), np.float32), condition.seeds[0], config)
    assert np.array_equal(fields['endpoint'], expected.numpy())
    assert trace['observers_restored']
    if kind == 'fixed':
        assert trace['action_padding_intervention']['hooks_restored']
    else:
        assert trace['action_padding_intervention'] is None
    assert model.generate_samples_from_batch is generate
    assert np.array_equal(arrays['model_action'], expected.numpy()[0, condition.action_offset:].reshape(33, 64)[1:, :8])


@pytest.mark.parametrize('steps,times', [
    (2, [1., 0.]), (2, [1., .5]), (2, [1., .25, 0.]),
    (True, [1., 0.]), (2., [1., .5, 0.]),
])
def test_reject_incomplete_or_mismatched_grid_before_generation(steps, times):
    config = dict(kind='fixed', steps=steps, times=times, guidance=1., action_padding='zero')
    with pytest.raises(AssertionError):
        capture_control(None, None, None, 1, config)
