"""Seed pairing must leave normal inference's RNG stream and config intact."""
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from request_seed import predict_with_request_seed


@dataclass(frozen=True)
class Config:
    seed: int = 123
    deterministic_seed: bool = False


def setup_runtime(fail=False):
    # Execute the actual vendor method without importing its GPU/model loaders.
    path=Path('/home/ubuntu/cosmos-framework/cosmos_framework/scripts/action_policy_server_robolab.py')
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RobolabPolicyService')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_next_seed')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    def generate(**kwargs):
        if fail: raise RuntimeError('native failure')
        return kwargs['seed'][0]
    service=SimpleNamespace(cfg=Config(),_rng=np.random.default_rng(123),
        model=SimpleNamespace(generate_samples_from_batch=generate))
    def predict(obs):
        seed=namespace['_next_seed'](service)
        return service.model.generate_samples_from_batch(seed=[seed])
    loop=SimpleNamespace(_native_loop=SimpleNamespace(_service=service),_padding=object(),_sampler=object())
    return SimpleNamespace(_backend=SimpleNamespace(_impl=loop),predict=predict), service


def test_direct_seed_differs_from_base_stream_and_restores():
    runtime,service=setup_runtime()
    config=service.cfg; original=service.model.generate_samples_from_batch
    reference=np.random.default_rng(123)
    for seed in (7,91,2**32-1):
        result,receipt=predict_with_request_seed(runtime,{},seed)
        assert result==seed and receipt['actual_generation_seeds']==[[seed]]
        assert service.cfg is config and service.model.generate_samples_from_batch is original
    assert service._rng.bit_generator.state==reference.bit_generator.state
    assert runtime.predict({}) != 123  # Normal Runtime seed is a stream initializer.


def test_native_failure_restores_config_binding_and_rng():
    runtime,service=setup_runtime(fail=True)
    config=service.cfg; original=service.model.generate_samples_from_batch
    state=service._rng.bit_generator.state
    with pytest.raises(RuntimeError,match='native failure'):
        predict_with_request_seed(runtime,{},7)
    assert service.cfg is config and service.model.generate_samples_from_batch is original
    assert service._rng.bit_generator.state==state


@pytest.mark.parametrize('seed',[True,-1,2**32,1.5,'7'])
def test_invalid_seed_rejected_before_touching_runtime(seed):
    with pytest.raises(ValueError): predict_with_request_seed(None,{},seed)
