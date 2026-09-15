"""Observe the installed public runtime, preserving its sampler and processing."""
import functools

import numpy as np
import torch

import producer_capture as native
from request_seed import predict_with_request_seed


def capture_request(runtime, observation, measured_action, seed):
    from cosmos_framework.data.generator.action.action_processing import get_action_processing_records

    loop = runtime._backend._impl
    service, sampler = loop._native_loop._service, loop._sampler
    model = service.model
    observer = native.NativeCapture(model, sampler, '__call__')
    cls = type(sampler)
    original_call = cls.__call__
    original_generate = model.generate_samples_from_batch
    generate_owned = 'generate_samples_from_batch' in vars(model)
    effective, records = {}, []

    @functools.wraps(original_call)
    def sampling(owner, callback, noise, *args, **kwargs):
        if owner is sampler:
            effective['reference'] = torch.stack(kwargs['condition_reference']).detach().clone()
            effective['preserve_mask'] = torch.stack(kwargs['condition_mask']).detach().clone()
        return original_call(owner, callback, noise, *args, **kwargs)

    @functools.wraps(original_generate)
    def generate(batch, *args, **kwargs):
        batch_records = get_action_processing_records(batch)
        if len(batch_records) != 1:
            raise ValueError('Require one actual native DROID action processing record')
        records.append(batch_records[0])
        return original_generate(batch, *args, **kwargs)

    cls.__call__, model.generate_samples_from_batch = sampling, generate
    try:
        with torch.random.fork_rng(devices=[0]), observer.instrument():
            result, seed_receipt = predict_with_request_seed(runtime, observation, seed)
    finally:
        intact = cls.__call__ is sampling and model.generate_samples_from_batch is generate
        cls.__call__ = original_call
        if generate_owned:
            model.generate_samples_from_batch = original_generate
        else:
            del model.generate_samples_from_batch
        if not intact:
            raise RuntimeError('Quality observers were replaced during prediction')
    if len(records) != 1 or observer.latents is None:
        raise RuntimeError('Expected one complete native generation and normalizer record')
    trace = observer.finalize()
    endpoint = torch.stack(observer.latents).detach()
    offset = endpoint.shape[1] - 33*64
    if offset < 0:
        raise ValueError('Native endpoint has no complete 33x64 action block')
    normalized = endpoint[0, offset:].reshape(33,64)[1:,:8].float().cpu().numpy().copy()
    arrays = dict(action=np.asarray(result['action'],np.float32).copy(), model_action=normalized,
        measured_action=np.asarray(measured_action,np.float32).copy(),
        model_measured_action=native.normalize_recorded(measured_action,records[0],endpoint.device))
    if any(x.shape != (32,8) or x.dtype != np.float32 or not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('Expected finite full raw/model action and target arrays')
    fields = dict(endpoint=endpoint.float().cpu().numpy(),
        initial_noise=torch.stack(observer.initial).detach().float().cpu().numpy(),
        **{k:v.float().cpu().numpy() for k,v in effective.items()})
    normalizer = native.normalizer_metadata(records[0])
    expected_branches = 1 if service.cfg.guidance == 1. else 2
    if (trace['sampler_calls'],trace['preparation_calls'],trace['sampler_denoiser_callbacks'],
            trace['model_velocity_branch_calls'],trace['callback_timesteps']) != (1,1,1,expected_branches,[1000.]):
        raise ValueError('Observed execution differs from complete SDE1 and literal CFG')
    if trace['effective_sampler_arguments']['seed'] != [int(seed)]:
        raise ValueError('Actual sampler seed differs from the frozen quality request')
    if not loop._padding.last_audit.hooks_restored:
        raise RuntimeError('Padding hooks were not restored')
    trace.update(seed_receipt=seed_receipt,normalizer=normalizer,action_offset=offset,
        action_shape=[33,64],effective_mask_observed=True,observers_restored=True,
        action_padding_intervention=loop._padding.last_audit.report())
    return arrays, fields, trace
