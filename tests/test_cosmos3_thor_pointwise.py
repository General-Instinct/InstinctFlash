"""Actual BF16 rounding, shape changes and safe fallbacks for Thor fusions."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/cosmos3_policy'))

@pytest.fixture
def kernels():
    pytest.importorskip('triton')
    from cosmos3_iwm import exact_pointwise
    return exact_pointwise


def test_checked_falls_back_permanently_on_byte_difference(kernels):
    stats = {'checks': 0, 'calls': 0, 'rejected': [], 'errors': []}
    calls = []
    def bad(x):
        calls.append(1)
        return -x
    gate = kernels.Checked(lambda x: x.clone(), bad, lambda x: True, stats, 'test')
    x = torch.tensor([0.0])
    assert torch.equal(gate(x).view(torch.uint8), x.view(torch.uint8))
    gate(x)
    assert len(calls) == 1 and stats['rejected'] == ['test']


def test_checked_rechecks_changed_geometry(kernels):
    stats = {'checks': 0, 'calls': 0, 'rejected': [], 'errors': []}
    gate = kernels.Checked(lambda x: x.clone(), lambda x: x.clone(), lambda x: True, stats, 'test')
    for n in [2, 2, 3]:
        gate(torch.arange(n, dtype=torch.float32))
    assert stats['checks'] == 2 and stats['calls'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_thor_pointwise_preserves_eager_rounding(kernels):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip('Thor kernel qualification')
    from benchmarks.regression.verify_cosmos_kernels import verify_pointwise
    verify_pointwise(kernels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_thor_swiglu_rounding_probe(kernels):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip('Thor kernel qualification')
    from benchmarks.regression.verify_cosmos_kernels import verify_swiglu
    verify_swiglu(kernels)


def test_checked_tracks_keyword_tensor_shapes(kernels):
    stats = {'checks': 0, 'calls': 0, 'rejected': [], 'errors': []}
    gate = kernels.Checked(lambda x: x.clone(), lambda x: x.clone(), lambda x: True, stats, 'test')
    gate(x=torch.ones(2)); gate(x=torch.ones(3)); gate(x=torch.ones(3))
    assert stats['checks'] == 2 and stats['calls'] == 1
