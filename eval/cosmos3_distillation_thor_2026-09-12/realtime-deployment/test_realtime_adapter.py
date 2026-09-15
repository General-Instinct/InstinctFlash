"""Integration checks using Compress's native CPU fixture and rollout oracle."""
from types import SimpleNamespace

import pytest
import torch

import test_cosmos3_action_padding as fixtures
from instinct_compress.flash.cosmos3 import _install_sampler
from instinct_compress.flash.cosmos3_action_padding import action_sampling_contract
from instinct_compress.models.cosmos3_realtime_rollout_v16 import DeploymentGrid, rollout_deployment
from realtime_adapter import Cosmos3RealtimeAdapter, GRIDS, RealtimePaddingExecution, realtime_contract

setup = fixtures.setup


def checkpoint(steps, guidance):
    return SimpleNamespace(execution=SimpleNamespace(
        nfe={'prefix': 1, 'action': steps}, guidance={'action': {'mode': 'cfg', 'scale': guidance}},
        extra=dict(action_padding='zero', action_dim=8, action_chunk_size=32, domain_name='droid_lerobot',
            sampling=dict(kind='rectified_flow_fixed_step', sample_type='sde',
                sigmas=GRIDS[steps].copy(), num_train_timesteps=1000.))))


@pytest.mark.parametrize('steps', [1, 2, 4])
@pytest.mark.parametrize('guidance', [1., 4.])
def test_native_serving_matches_training_rollout(setup, steps, guidance):
    condition, model, role = setup
    role.guidance = guidance
    deployment = DeploymentGrid(steps, guidance)
    receipt = {}
    expected = rollout_deployment(role, condition, deployment, grad_last=False, receipt=receipt)
    model.queries.clear()
    model.config.fixed_step_sampler_config = SimpleNamespace(t_list=GRIDS[steps][:-1], sample_type='sde')
    declaration = checkpoint(steps, guidance)
    contract, _, _ = realtime_contract(declaration)
    spec = Cosmos3RealtimeAdapter().spec_for_checkpoint(declaration)
    assert spec.guidance['action'].scale == guidance
    assert next(p for p in spec.phases if p.name == 'action').nfe == steps

    def generate(*, sampler, guidance, num_steps):
        plans, clean, conditional, unconditional, noise, reference, mask = model._prepare_inference_data()
        def velocity(state, time):
            def branch(tokens):
                return model._get_velocity(noise_x=state, timestep=time, text_tokens=tokens,
                    sequence_plans=plans, gen_data_clean=clean)
            c = branch(conditional)
            if guidance == 1.:
                return c
            u = branch(unconditional)
            return [uv + guidance * (cv - uv) for cv, uv in zip(c, u, strict=True)]
        with role._bound_heads(), torch.no_grad():
            return sampler(velocity, noise, seed=list(condition.seeds), num_steps=num_steps,
                condition_reference=reference, condition_mask=mask)

    model.generate_samples_from_batch = generate
    sampler = _install_sampler(model, contract)
    before = model.generate_samples_from_batch
    prepare_before = model._prepare_inference_data
    padding = RealtimePaddingExecution(model, steps, guidance)
    with torch.random.fork_rng():
        actual = model.generate_samples_from_batch(guidance=guidance, num_steps=steps)
    assert torch.equal(torch.stack(actual), expected)
    assert sampler.calls == 1 and sampler.velocity_evaluations == steps
    assert padding.calls == 1 and padding.branches == receipt['network_branches']
    assert padding.last_audit.hooks_restored
    assert model._prepare_inference_data is prepare_before and '_get_velocity' not in vars(model)
    with pytest.raises(ValueError, match='exact declared'):
        model.generate_samples_from_batch(guidance=3., num_steps=steps)
    model.fail_at = len(model.queries) + 1
    with pytest.raises(RuntimeError, match='deliberate'), torch.random.fork_rng():
        model.generate_samples_from_batch(guidance=guidance, num_steps=steps)
    assert padding.calls == 1 and padding.last_audit is None
    assert model._prepare_inference_data is prepare_before and '_get_velocity' not in vars(model)
    padding.restore()
    assert model.generate_samples_from_batch is before


@pytest.mark.parametrize('bad', ['exit', 'truncated', 'cfg3', 'cfg_bool', 'scale', 'nfe_bool', 'padding'])
def test_rejects_ambiguous_or_changed_contract(bad):
    cp = checkpoint(2, 1.)
    if bad == 'exit':
        cp.execution.extra['sampling']['sigmas'] = [1., .75, 0.]
    elif bad == 'truncated':
        cp.execution.nfe['action'] = 1
    elif bad == 'cfg3':
        cp.execution.guidance['action']['scale'] = 3.
    elif bad == 'cfg_bool':
        cp.execution.guidance['action']['scale'] = True
    elif bad == 'scale':
        cp.execution.extra['sampling']['num_train_timesteps'] = 999.
    elif bad == 'nfe_bool':
        cp.execution.nfe['prefix'] = True
    elif bad == 'padding':
        cp.execution.extra['action_padding'] = 'native'
    with pytest.raises(ValueError):
        realtime_contract(cp)


def test_historical_contract_and_fp8_rejection_unchanged():
    action_sampling_contract(checkpoint(4, 4.))
    for steps, guidance in [(1, 1.), (1, 4.), (2, 4.), (4, 1.)]:
        with pytest.raises(ValueError):
            action_sampling_contract(checkpoint(steps, guidance))
    with pytest.raises(ValueError, match='native precision only'):
        Cosmos3RealtimeAdapter().build_fp8(checkpoint(1, 1.))
