"""Native original-weight controls using the exact historical generation arguments."""
from contextlib import nullcontext
import functools

import numpy as np
import torch

import producer_capture as native


def capture_control(service, observation, measured_action, seed, config):
    from cosmos_framework.data.generator.action.action_processing import get_action_processing_records
    from cosmos_framework.scripts.action_policy_server_robolab import _build_data_batch_from_sample
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
    from instinct_compress.models.cosmos3_action_padding import zero_action_padding

    model = service.model
    assert config['weight'] == 'original' and service.cfg.history_length == 1
    state = np.asarray(observation['state'], np.float32)
    sample = {'observation/image': observation['image'].copy(),
              'observation/joint_position': state[:7], 'observation/gripper_position': state[7:],
              'prompt': observation['prompt']}
    batch = _build_data_batch_from_sample(service._build_sample(sample))
    records = get_action_processing_records(batch)
    assert len(records) == 1 and records[0].raw_action_dim == 8
    sampler = model.sampler if config['kind'] == 'unipc' else FixedStepSampler(
        list(config['times']), sample_type='sde', num_train_timesteps=1000.)
    method = 'forward' if config['kind'] == 'unipc' else '__call__'
    observer = native.NativeCapture(model, sampler, method)
    cls = type(sampler)
    original = getattr(cls, method)
    effective = {}

    @functools.wraps(original)
    def sampling(owner, callback, noise, *args, **kwargs):
        if owner is sampler:
            # UniPC's velocity callback closes over the prepared conditioning;
            # only FixedStepSampler receives conditioning as sampler arguments.
            reference = kwargs['condition_reference'] if config['kind'] == 'fixed' else observer.prepared[5]
            mask = kwargs['condition_mask'] if config['kind'] == 'fixed' else observer.prepared[6]
            effective['reference'] = torch.stack(reference).detach().clone()
            effective['preserve_mask'] = torch.stack(mask).detach().clone()
        return original(owner, callback, noise, *args, **kwargs)

    generation = dict(seed=[int(seed)], guidance=float(config['guidance']),
                      num_steps=config['steps'], shift=config['shift'], guidance_interval=None,
                      normalize_cfg=False, skip_text_tokens_for_cfg=False)
    padding_context = zero_action_padding(model) if config['action_padding'] == 'zero' else nullcontext(None)
    setattr(cls, method, sampling)
    try:
        with service._lock, torch.random.fork_rng(devices=[0]), torch.inference_mode(), \
                padding_context as padding, observer.instrument():
            samples = model.generate_samples_from_batch(batch, sampler=sampler, **generation)
    finally:
        intact = getattr(cls, method) is sampling
        setattr(cls, method, original)
        if not intact:
            raise RuntimeError('Native sampler observer was replaced')
    trace = observer.finalize()
    assert (trace['sampler_calls'], trace['preparation_calls']) == (1, 1)
    assert trace['sampler_denoiser_callbacks'] == config['steps']
    assert trace['model_velocity_branch_calls'] == config['steps'] * (1 if config['guidance'] == 1 else 2)
    sampler_shift = 0. if config['kind'] == 'fixed' else config['shift']
    assert trace['effective_sampler_arguments'] == dict(num_steps=config['steps'], shift=sampler_shift, seed=[seed])
    if config['kind'] == 'fixed':
        assert trace['callback_timesteps'] == [1000.]
    if padding is not None:
        assert padding.hooks_restored
    endpoint = torch.stack(observer.latents).detach().float()
    offset = endpoint.shape[1] - 33 * 64
    normalized = endpoint[0, offset:].reshape(33, 64)[1:, :8].cpu().numpy().copy()
    action = samples['action'][0][1:, :8].detach().float().cpu().numpy().copy()
    action[:, -1] = 1. - action[:, -1]
    arrays = dict(action=action, model_action=normalized,
                  measured_action=np.asarray(measured_action, np.float32).copy(),
                  model_measured_action=native.normalize_recorded(measured_action, records[0], endpoint.device))
    assert all(v.dtype == np.float32 and v.shape == (32, 8) and np.isfinite(v).all() for v in arrays.values())
    fields = dict(endpoint=endpoint.cpu().numpy(),
                  initial_noise=torch.stack(observer.initial).detach().float().cpu().numpy(),
                  **{k: v.float().cpu().numpy() for k, v in effective.items()})
    assert all(v.shape == fields['endpoint'].shape and np.isfinite(v).all() for v in fields.values())
    if config['action_padding'] == 'zero':
        assert np.all(fields['endpoint'][:, offset:].reshape(1, 33, 64)[:, :, 8:] == 0)
    trace.update(generation_arguments=generation, normalizer=native.normalizer_metadata(records[0]),
                 action_offset=offset, action_shape=[33, 64], observers_restored=True,
                 conditioning_observation='sampler_arguments' if config['kind'] == 'fixed' else 'native_preparation_callback_context',
                 action_padding_intervention=None if padding is None else padding.report())
    return arrays, fields, trace
