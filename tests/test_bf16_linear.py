from types import SimpleNamespace
import pytest
import torch
from instinctflash.backends.bf16_linear import BF16LinearActivation, supports, _validate
from instinctflash.planners.planner import Tier


def test_permission_precedes_device_or_library_access():
    with pytest.raises(ValueError, match='numeric'):
        BF16LinearActivation(None, None, activation_name='relu2',
                             plan=SimpleNamespace(tier_ceiling=Tier.BITEXACT))


@pytest.mark.parametrize('change', [dict(activation='silu'), dict(activation='gelu'),
    dict(gated=True), dict(bias=True), dict(capability=(9, 0)), dict(dtype=torch.float16),
    dict(shape=(1, 9216, 2048)), dict(shape=(3093, 8192, 2048))])
def test_dispatch_excludes_unqualified_arithmetic_and_shapes(change):
    args = dict(activation='relu2', gated=False, bias=False, capability=(11, 0),
                dtype=torch.bfloat16, shape=(3093, 9216, 2048))
    assert supports(**args)
    args.update(change)
    assert not supports(**args)


@pytest.mark.parametrize('activation,name', [(torch.nn.functional.gelu, 'gelu'),
    (torch.nn.functional.silu, 'silu'), (lambda x: x.relu().square(), 'relu2')])
def test_fallback_preserves_original_math_bias_and_training(activation, name):
    linear = torch.nn.Linear(4, 8)
    wrapped = BF16LinearActivation(linear, activation, activation_name=name,
                                  plan=SimpleNamespace(tier_ceiling=Tier.NUMERIC))
    x = torch.randn(3, 4, requires_grad=True)
    expected, actual = activation(linear(x)), wrapped(x)
    assert torch.equal(expected, actual)
    assert torch.equal(torch.autograd.grad(expected.sum(), x)[0],
                       torch.autograd.grad(actual.sum(), x)[0])
    assert not wrapped.report()['eligible']


def test_raw_op_rejects_cpu_before_pointer_launch():
    with pytest.raises(ValueError, match='CUDA'):
        _validate(torch.empty(3, 4), torch.empty(8, 4))


def test_fallback_does_not_resolve_optional_library(monkeypatch):
    monkeypatch.setenv('IFL_BF16_KERNEL_LIBRARY', '/not/a/library.so')
    wrapped = BF16LinearActivation(torch.nn.Linear(4,8), torch.relu, activation_name='relu',
                                  plan=SimpleNamespace(tier_ceiling=Tier.NUMERIC))
    assert wrapped(torch.zeros(1,4)).shape == (1,8)
