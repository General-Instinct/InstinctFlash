"""CPU native-sampler fixture checks the public quality observation boundary."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import test_cosmos3_action_padding as fixtures
from instinct_compress.flash.cosmos3 import _install_sampler
from instinct_compress.models.cosmos3_realtime_rollout_v16 import DeploymentGrid, rollout_deployment
from realtime_adapter import RealtimePaddingExecution
from capture_request import capture_request

setup = fixtures.setup


@dataclass(frozen=True)
class Config:
    guidance: float
    seed: int = 0
    deterministic_seed: bool = False


@pytest.mark.parametrize('guidance',[1.,4.])
def test_capture_full_native_fields_and_restoration(setup,monkeypatch,guidance):
    from cosmos_framework.data.generator.action import action_processing
    condition,model,role=setup; role.guidance=guidance
    record=SimpleNamespace(raw_action_dim=8,action_normalizer=None)
    monkeypatch.setattr(action_processing,'get_action_processing_records',lambda batch:[record])
    fork=torch.random.fork_rng
    monkeypatch.setattr(torch.random,'fork_rng',lambda devices=None:fork(devices=[]))
    expected=rollout_deployment(role,condition,DeploymentGrid(1,guidance),grad_last=False,receipt={})
    model.config.fixed_step_sampler_config=SimpleNamespace(t_list=[1.],sample_type='sde')
    contract=dict(sigmas=[1.,0.],sample_type='sde',num_train_timesteps=1000.)
    def generate(batch,*,sampler,seed,guidance,num_steps):
        plans,clean,conditional,unconditional,noise,reference,mask=model._prepare_inference_data()
        def callback(state,time):
            def branch(tokens):
                return model._get_velocity(noise_x=state,timestep=time,text_tokens=tokens,
                    sequence_plans=plans,gen_data_clean=clean)
            c=branch(conditional)
            if guidance==1.:return c
            u=branch(unconditional)
            return [uv+guidance*(cv-uv) for cv,uv in zip(c,u,strict=True)]
        with role._bound_heads(),torch.no_grad():
            result=sampler(callback,noise,seed=seed,num_steps=num_steps,
                condition_reference=reference,condition_mask=mask)
        return dict(action=[result[0][condition.action_offset:].reshape(33,64)])
    model.generate_samples_from_batch=generate
    sampler=_install_sampler(model,contract)
    padding=RealtimePaddingExecution(model,1,guidance)
    service=SimpleNamespace(model=model,cfg=Config(guidance),_rng=np.random.default_rng(0))
    def predict(obs):
        result=model.generate_samples_from_batch({},seed=[service.cfg.seed],
            guidance=service.cfg.guidance,num_steps=1)
        raw=result['action'][0][1:,:8].float().numpy().copy();raw[:,-1]=1-raw[:,-1]
        return dict(action=raw)
    loop=SimpleNamespace(_native_loop=SimpleNamespace(_service=service),_sampler=sampler,_padding=padding)
    runtime=SimpleNamespace(_backend=SimpleNamespace(_impl=loop),predict=predict)
    original_generate=model.generate_samples_from_batch;original_call=type(sampler).__call__
    arrays,fields,trace=capture_request(runtime,{},np.zeros((32,8),np.float32),condition.seeds[0])
    assert np.array_equal(fields['endpoint'],expected.numpy())
    assert arrays['model_action'].shape==(32,8)
    assert trace['effective_sampler_arguments']['seed']==list(condition.seeds)
    assert trace['model_velocity_branch_calls']==(1 if guidance==1 else 2)
    assert np.count_nonzero(fields['preserve_mask'][:,condition.action_offset:].reshape(1,33,64)[:,:,8:] != 1)==0
    assert trace['normalizer']['raw_action_dim']==8 and trace['observers_restored']
    assert model.generate_samples_from_batch is original_generate and type(sampler).__call__ is original_call
    assert service.cfg==Config(guidance)
    padding.restore()
