"""Experimental complete-grid SDE serving for the V16 distillation study.

Registration is explicit. This does not admit students to production, infer
their quality, or relax the historical four-step/CFG4 backbone.
"""
from __future__ import annotations

import dataclasses
import functools

from instinct_compress.flash.cosmos3 import Cosmos3FixedStepAdapter, sampling_contract
from instinct_compress.flash.cosmos3_action_padding import (
    PaddingExecution, _ActionFixedStepLoop, validate_padding_sidecar,
)

BACKBONE = 'cosmos3_policy_action_realtime_v16'
GRIDS = {1: [1., 0.], 2: [1., .5, 0.], 4: [1., .75, .5, .25, 0.]}


def realtime_contract(checkpoint, nfe=None):
    contract = sampling_contract(checkpoint, nfe)
    steps = len(contract['sigmas']) - 1
    if steps not in GRIDS or contract['sigmas'] != GRIDS[steps]:
        raise ValueError('Require a complete declared V16 one/two/four-step grid')
    if contract['num_train_timesteps'] != 1000.:
        raise ValueError('V16 native timestep scale must be 1000')
    for value in (*dict(checkpoint.execution.nfe or {}).values(), *dict(nfe or {}).values()):
        if type(value) is not int:
            raise ValueError('NFE values must be integer counts, not booleans or floats')
    guidance = checkpoint.execution.guidance.get('action')
    if not isinstance(guidance, dict) or set(guidance) != {'mode', 'scale'}:
        raise ValueError('Declare literal CFG1 or CFG4')
    scale = guidance['scale']
    if guidance['mode'] != 'cfg' or isinstance(scale, bool) or scale not in (1., 4.):
        raise ValueError('Declare literal CFG1 or CFG4')
    extra = checkpoint.execution.extra or {}
    if (extra.get('action_padding'), extra.get('action_dim'),
            extra.get('action_chunk_size'), extra.get('domain_name')) != (
            'zero', 8, 32, 'droid_lerobot'):
        raise ValueError('Require zero-padded native DROID 32x8 action contract')
    return contract, steps, float(scale)


class RealtimePaddingExecution(PaddingExecution):
    """Project native inputs/transitions, checking the declared branch budget."""

    def __init__(self, model, steps, guidance):
        from instinct_compress.models.cosmos3_action_padding import zero_action_padding

        self.model = model
        self.own = 'generate_samples_from_batch' in vars(model)
        self.original = model.generate_samples_from_batch
        self.calls = self.branches = 0
        self.last_audit = None
        self.steps, self.guidance = steps, guidance
        expected_branches = steps * (1 if guidance == 1. else 2)

        @functools.wraps(self.original)
        def generate(*args, **kwargs):
            if kwargs.get('guidance') != guidance or kwargs.get('num_steps') != steps:
                raise ValueError('Generation must use the exact declared V16 guidance and step count')
            self.last_audit = None
            with zero_action_padding(model) as audit:
                result = self.original(*args, **kwargs)
            if (len(audit.original_preparations) != 1
                    or len(audit.branches) != expected_branches or not audit.hooks_restored):
                raise RuntimeError('Native preparation/CFG branch count or padding hook restoration differs')
            self.last_audit = audit
            self.calls += 1
            self.branches += len(audit.branches)
            return result

        self.installed = generate
        model.generate_samples_from_batch = generate


class Cosmos3RealtimeAdapter(Cosmos3FixedStepAdapter):
    def spec(self):
        spec = super().spec()
        return dataclasses.replace(spec, notes={**spec.notes, 'backbone': BACKBONE,
            'action_padding': 'zero', 'qualification': 'experimental V16 deployment'})

    def spec_for_checkpoint(self, checkpoint):
        _, _, guidance = realtime_contract(checkpoint)
        spec = super().spec_for_checkpoint(checkpoint)
        return dataclasses.replace(spec, guidance={**spec.guidance,
            'action': dataclasses.replace(spec.guidance['action'], scale=guidance)})

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None, seed=None):
        contract, steps, guidance = realtime_contract(checkpoint, nfe)
        validate_padding_sidecar(checkpoint)
        loop = super().build_in_process(checkpoint, plan, device=device, nfe=nfe, seed=seed)
        try:
            service = loop._native_loop._service
            if service.cfg.guidance != guidance or service.cfg.num_steps != steps:
                raise RuntimeError('Loaded service differs from the complete V16 deployment declaration')
            padding = RealtimePaddingExecution(service.model, steps, guidance)
            return _RealtimeLoop(loop._native_loop, loop._sampler, contract, padding)
        except BaseException:
            loop.close()
            raise


class _RealtimeLoop(_ActionFixedStepLoop):
    def declaration(self):
        return {**super().declaration(), 'backbone': BACKBONE,
            'literal_guidance': self._padding.guidance,
            'evidence': 'Complete declared native SDE grid and zero-padding branch audit; quality unqualified'}
