"""Shared fusion contracts, including paths that must retain eager execution."""
import pytest
import torch

pytest.importorskip('triton')
from instinctflash.backends import bf16_pointwise as kernels


def test_cpu_swiglu_preserves_autograd_and_noncontiguous_inputs():
    gate = torch.randn(7, 11, requires_grad=True)
    up = torch.randn(7, 11, requires_grad=True)
    out = kernels.swiglu(gate.t(), up.t())
    ref = torch.nn.functional.silu(gate.t()) * up.t()
    assert torch.equal(out, ref)
    actual = torch.autograd.grad(out.sum(), (gate, up))
    expected = torch.autograd.grad(ref.sum(), (gate, up))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize('round_first', [True, False])
def test_norm_declares_rounding_boundary(round_first):
    torch.manual_seed(14)
    x = torch.randn(17, 32).bfloat16()
    w = torch.randn(32).bfloat16()
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    ref = w * normalized.bfloat16() if round_first else (w.float() * normalized).bfloat16()
    assert torch.equal(kernels.norm(x, w, 1e-6, round_first).view(torch.int16), ref.view(torch.int16))


def test_partial_rope_keeps_tail_and_accepts_distinct_head_counts():
    q = torch.randn(3, 4, 16).bfloat16()
    k = torch.randn(3, 2, 16).bfloat16()
    cos = torch.ones(3, 8).bfloat16()
    sin = torch.zeros_like(cos)
    qo, ko = kernels.rope(q, k, cos, sin)
    assert torch.equal(qo, q) and torch.equal(ko, k)
    with pytest.raises(ValueError, match='geometry'):
        kernels.rope(q, k, cos[:, :7], sin[:, :7])


def test_empty_inputs_fall_back():
    x = torch.empty(0, 16, dtype=torch.bfloat16)
    assert kernels.swiglu(x, x).shape == x.shape
    assert kernels.relu2(x).shape == x.shape
    assert kernels.norm(x, torch.ones(16).bfloat16(), 1e-6, True).shape == x.shape


@pytest.mark.parametrize('round_first', [True, False])
def test_residual_norm_preserves_sum_and_does_not_alias(round_first):
    x = torch.randn(3, 32).bfloat16()
    r = torch.randn_like(x)
    w = torch.randn(32).bfloat16()
    before_x, before_r = x.clone(), r.clone()
    actual, added = kernels.residual_norm(x, r, w, 1e-6, round_first)
    expected = kernels.norm(x + r, w, 1e-6, round_first)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(added.view(torch.int16), (x + r).view(torch.int16))
    added.zero_()
    assert torch.equal(x, before_x) and torch.equal(r, before_r)
    with pytest.raises(ValueError, match='Residual'):
        kernels.residual_norm(x, r[:1], w, 1e-6, round_first)
